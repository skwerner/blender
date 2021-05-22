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

#include "kernel/kernel_differential.h"
#include "kernel/kernel_light.h"
#include "kernel/kernel_path_state.h"
#include "kernel/kernel_projection.h"
#include "kernel/kernel_shadow_catcher.h"

#include "kernel/geom/geom.h"

#include "kernel/bvh/bvh.h"

CCL_NAMESPACE_BEGIN

ccl_device_forceinline int integrator_intersect_next_shader(INTEGRATOR_STATE_ARGS,
                                                            const Intersection *ccl_restrict isect)
{
  /* Find shader from intersection. */
  const int shader = intersection_get_shader(kg, isect);

  /* Optional AO bounce termination. */
  if (path_state_ao_bounce(INTEGRATOR_STATE_PASS)) {
    const int flags = kernel_tex_fetch(__shaders, shader).flags;

    if (flags & (SD_HAS_TRANSPARENT_SHADOW | SD_HAS_EMISSION)) {
      INTEGRATOR_STATE_WRITE(path, flag) |= PATH_RAY_TERMINATE_AFTER_TRANSPARENT;
    }
    else {
      return SHADER_NONE;
    }
  }

  /* Load random number state. */
  RNGState rng_state;
  path_state_rng_load(INTEGRATOR_STATE_PASS, &rng_state);

  /* We perform path termination in this kernel to avoid launching shade_surface
   * and evaluating the shader when not needed. Only for emission and transparent
   * surfaces in front of emission do we need to evaluate the shader, since we
   * perform MIS as part of indirect rays. */
  const float probability = path_state_continuation_probability(INTEGRATOR_STATE_PASS);

  if (probability != 1.0f) {
    const float terminate = path_state_rng_1D(kg, &rng_state, PRNG_TERMINATE);

    if (probability == 0.0f || terminate >= probability) {
      const int flags = kernel_tex_fetch(__shaders, shader).flags;

      if (flags & (SD_HAS_TRANSPARENT_SHADOW | SD_HAS_EMISSION)) {
        /* Mark path to be terminated right after shader evaluation. */
        INTEGRATOR_STATE_WRITE(path, flag) |= PATH_RAY_TERMINATE_IMMEDIATE;
      }
      else {
        return SHADER_NONE;
      }
    }
  }

  return shader;
}

ccl_device void integrator_intersect_closest(INTEGRATOR_STATE_ARGS)
{
  /* Read ray from integrator state into local memory. */
  Ray ray ccl_optional_struct_init;
  integrator_state_read_ray(INTEGRATOR_STATE_PASS, &ray);
  kernel_assert(ray.t != 0.0f);

  uint visibility = path_state_ray_visibility(INTEGRATOR_STATE_PASS);

  /* Trick to use short AO rays to approximate indirect light at the end of the path. */
  if (path_state_ao_bounce(INTEGRATOR_STATE_PASS)) {
    ray.t = kernel_data.background.ao_distance;
  }

  /* Scene Intersection. */
  Intersection isect ccl_optional_struct_init;
  bool hit = scene_intersect(kg, &ray, visibility, &isect);

  /* TODO: remove this and do it in the various intersection functions instead. */
  if (!hit) {
    isect.prim = PRIM_NONE;
  }

  /* Light intersection for MIS. */
  if (kernel_data.integrator.use_lamp_mis && !(INTEGRATOR_STATE(path, flag) & PATH_RAY_CAMERA)) {
    /* NOTE: if we make lights visible to camera rays, we'll need to initialize
     * these in the path_state_init. */
    const int last_prim = INTEGRATOR_STATE(isect, prim);
    const int last_object = INTEGRATOR_STATE(isect, object);
    const int last_type = INTEGRATOR_STATE(isect, type);

    hit = lights_intersect(kg, &ray, &isect, last_prim, last_object, last_type) || hit;
  }

  /* Write intersection result into global integrator state memory. */
  integrator_state_write_isect(INTEGRATOR_STATE_PASS, &isect);

#ifdef __VOLUME__
  if (INTEGRATOR_STATE_ARRAY(volume_stack, 0, object) != OBJECT_NONE) {
    /* Continue with volume kernel if we are inside a volume, regardless
     * if we hit anything. */
    INTEGRATOR_PATH_NEXT(INTERSECT_CLOSEST, SHADE_VOLUME);
    return;
  }
#endif

  if (hit) {
    /* Hit a surface, continue with light or surface kernel. */
    if (isect.type & PRIMITIVE_LAMP) {
      INTEGRATOR_PATH_NEXT(INTERSECT_CLOSEST, SHADE_LIGHT);
      return;
    }
    else {
      /* Hit a surface, continue with surface kernel unless terminated. */
      const int shader = integrator_intersect_next_shader(INTEGRATOR_STATE_PASS, &isect);
      if (shader != SHADER_NONE) {
        INTEGRATOR_PATH_SET_SORT_KEY(shader);
        INTEGRATOR_PATH_NEXT(INTERSECT_CLOSEST, SHADE_SURFACE);

        const int object_flags = intersection_get_object_flags(kg, &isect);
        kernel_shadow_catcher_split(INTEGRATOR_STATE_PASS, object_flags);
        return;
      }
      else {
        INTEGRATOR_PATH_TERMINATE(INTERSECT_CLOSEST);
        return;
      }
    }
  }
  else {
    /* Nothing hit, continue with background kernel. */
    INTEGRATOR_PATH_NEXT(INTERSECT_CLOSEST, SHADE_BACKGROUND);
    return;
  }
}

CCL_NAMESPACE_END
