/*
 * Copyright 2011-2021 Blender Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

CCL_NAMESPACE_BEGIN

#if 0
/* Volume Stack
 *
 * This is an array of object/shared ID's that the current segment of the path
 * is inside of. */

ccl_device void volume_stack_init(INTEGRATOR_STATE_ARGS,
                                  const ShaderData *stack_sd,
                                  ccl_addr_space const PathState *state,
                                  ccl_addr_space const Ray *ray,
                                  ccl_addr_space VolumeStack *stack)
{
  /* NULL ray happens in the baker, does it need proper initialization of
   * camera in volume?
   */
  if (!kernel_data.cam.is_inside_volume || ray == NULL) {
    /* Camera is guaranteed to be in the air, only take background volume
     * into account in this case.
     */
    if (kernel_data.background.volume_shader != SHADER_NONE) {
      stack[0].shader = kernel_data.background.volume_shader;
      stack[0].object = PRIM_NONE;
      stack[1].shader = SHADER_NONE;
    }
    else {
      stack[0].shader = SHADER_NONE;
    }
    return;
  }

  kernel_assert(state->flag & PATH_RAY_CAMERA);

  Ray volume_ray = *ray;
  volume_ray.t = FLT_MAX;

  const uint visibility = (state->flag & PATH_RAY_ALL_VISIBILITY);
  int stack_index = 0, enclosed_index = 0;

#  ifdef __VOLUME_RECORD_ALL__
  Intersection hits[2 * VOLUME_STACK_SIZE + 1];
  uint num_hits = scene_intersect_volume_all(
      kg, &volume_ray, hits, 2 * VOLUME_STACK_SIZE, visibility);
  if (num_hits > 0) {
    int enclosed_volumes[VOLUME_STACK_SIZE];
    Intersection *isect = hits;

    qsort(hits, num_hits, sizeof(Intersection), intersections_compare);

    for (uint hit = 0; hit < num_hits; ++hit, ++isect) {
      shader_setup_from_ray(kg, stack_sd, isect, &volume_ray);
      if (stack_sd->flag & SD_BACKFACING) {
        bool need_add = true;
        for (int i = 0; i < enclosed_index && need_add; ++i) {
          /* If ray exited the volume and never entered to that volume
           * it means that camera is inside such a volume.
           */
          if (enclosed_volumes[i] == stack_sd->object) {
            need_add = false;
          }
        }
        for (int i = 0; i < stack_index && need_add; ++i) {
          /* Don't add intersections twice. */
          if (stack[i].object == stack_sd->object) {
            need_add = false;
            break;
          }
        }
        if (need_add && stack_index < VOLUME_STACK_SIZE - 1) {
          stack[stack_index].object = stack_sd->object;
          stack[stack_index].shader = stack_sd->shader;
          ++stack_index;
        }
      }
      else {
        /* If ray from camera enters the volume, this volume shouldn't
         * be added to the stack on exit.
         */
        enclosed_volumes[enclosed_index++] = stack_sd->object;
      }
    }
  }
#  else
  int enclosed_volumes[VOLUME_STACK_SIZE];
  int step = 0;

  while (stack_index < VOLUME_STACK_SIZE - 1 && enclosed_index < VOLUME_STACK_SIZE - 1 &&
         step < 2 * VOLUME_STACK_SIZE) {
    Intersection isect;
    if (!scene_intersect_volume(kg, &volume_ray, &isect, visibility)) {
      break;
    }

    shader_setup_from_ray(kg, stack_sd, &isect, &volume_ray);
    if (stack_sd->flag & SD_BACKFACING) {
      /* If ray exited the volume and never entered to that volume
       * it means that camera is inside such a volume.
       */
      bool need_add = true;
      for (int i = 0; i < enclosed_index && need_add; ++i) {
        /* If ray exited the volume and never entered to that volume
         * it means that camera is inside such a volume.
         */
        if (enclosed_volumes[i] == stack_sd->object) {
          need_add = false;
        }
      }
      for (int i = 0; i < stack_index && need_add; ++i) {
        /* Don't add intersections twice. */
        if (stack[i].object == stack_sd->object) {
          need_add = false;
          break;
        }
      }
      if (need_add) {
        stack[stack_index].object = stack_sd->object;
        stack[stack_index].shader = stack_sd->shader;
        ++stack_index;
      }
    }
    else {
      /* If ray from camera enters the volume, this volume shouldn't
       * be added to the stack on exit.
       */
      enclosed_volumes[enclosed_index++] = stack_sd->object;
    }

    /* Move ray forward. */
    volume_ray.P = ray_offset(stack_sd->P, -stack_sd->Ng);
    ++step;
  }
#  endif
  /* stack_index of 0 means quick checks outside of the kernel gave false
   * positive, nothing to worry about, just we've wasted quite a few of
   * ticks just to come into conclusion that camera is in the air.
   *
   * In this case we're doing the same above -- check whether background has
   * volume.
   */
  if (stack_index == 0 && kernel_data.background.volume_shader == SHADER_NONE) {
    stack[0].shader = kernel_data.background.volume_shader;
    stack[0].object = OBJECT_NONE;
    stack[1].shader = SHADER_NONE;
  }
  else {
    stack[stack_index].shader = SHADER_NONE;
  }
}
#endif

template<typename StackReadOp, typename StackWriteOp>
ccl_device void volume_stack_enter_exit(INTEGRATOR_STATE_ARGS,
                                        const ShaderData *sd,
                                        StackReadOp stack_read,
                                        StackWriteOp stack_write)
{
  /* todo: we should have some way for objects to indicate if they want the
   * world shader to work inside them. excluding it by default is problematic
   * because non-volume objects can't be assumed to be closed manifolds */
  if (!(sd->flag & SD_HAS_VOLUME)) {
    return;
  }

  if (sd->flag & SD_BACKFACING) {
    /* Exit volume object: remove from stack. */
    for (int i = 0;; i++) {
      VolumeStack entry = stack_read(i);
      if (entry.shader == SHADER_NONE) {
        break;
      }

      if (entry.object == sd->object) {
        /* Shift back next stack entries. */
        do {
          entry = stack_read(i + 1);
          stack_write(i, entry);
          i++;
        } while (entry.shader != SHADER_NONE);

        return;
      }
    }
  }
  else {
    /* Enter volume object: add to stack. */
    int i;
    for (i = 0;; i++) {
      VolumeStack entry = stack_read(i);
      if (entry.shader == SHADER_NONE) {
        break;
      }

      /* Already in the stack? then we have nothing to do. */
      if (entry.object == sd->object) {
        return;
      }
    }

    /* If we exceed the stack limit, ignore. */
    if (i >= VOLUME_STACK_SIZE - 1) {
      return;
    }

    /* Add to the end of the stack. */
    const VolumeStack new_entry = {sd->object, sd->shader};
    const VolumeStack empty_entry = {OBJECT_NONE, SHADER_NONE};
    stack_write(i, new_entry);
    stack_write(i + 1, empty_entry);
  }
}

ccl_device void volume_stack_enter_exit(INTEGRATOR_STATE_ARGS, const ShaderData *sd)
{
  volume_stack_enter_exit(
      INTEGRATOR_STATE_PASS,
      sd,
      [=](const int i) { return integrator_state_read_volume_stack(INTEGRATOR_STATE_PASS, i); },
      [=](const int i, const VolumeStack entry) {
        integrator_state_write_volume_stack(INTEGRATOR_STATE_PASS, i, entry);
      });
}

ccl_device void shadow_volume_stack_enter_exit(INTEGRATOR_STATE_ARGS, const ShaderData *sd)
{
  volume_stack_enter_exit(
      INTEGRATOR_STATE_PASS,
      sd,
      [=](const int i) {
        return integrator_state_read_shadow_volume_stack(INTEGRATOR_STATE_PASS, i);
      },
      [=](const int i, const VolumeStack entry) {
        integrator_state_write_shadow_volume_stack(INTEGRATOR_STATE_PASS, i, entry);
      });
}

/* Clean stack after the last bounce.
 *
 * It is expected that all volumes are closed manifolds, so at the time when ray
 * hits nothing (for example, it is a last bounce which goes to environment) the
 * only expected volume in the stack is the world's one. All the rest volume
 * entries should have been exited already.
 *
 * This isn't always true because of ray intersection precision issues, which
 * could lead us to an infinite non-world volume in the stack, causing render
 * artifacts.
 *
 * Use this function after the last bounce to get rid of all volumes apart from
 * the world's one after the last bounce to avoid render artifacts.
 */
ccl_device_inline void volume_stack_clean(INTEGRATOR_STATE_ARGS)
{
  if (kernel_data.background.volume_shader != SHADER_NONE) {
    /* Keep the world's volume in stack. */
    INTEGRATOR_STATE_ARRAY_WRITE(volume_stack, 1, shader) = SHADER_NONE;
  }
  else {
    INTEGRATOR_STATE_ARRAY_WRITE(volume_stack, 0, shader) = SHADER_NONE;
  }
}

CCL_NAMESPACE_END
