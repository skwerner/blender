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

ccl_device_forceinline uint integrate_intersect_shadow_visibility(INTEGRATOR_STATE_CONST_ARGS)
{
  const uint32_t path_flag = INTEGRATOR_STATE(path, flag);
#ifdef __SHADOW_TRICKS__
  return (path_flag & PATH_RAY_SHADOW_CATCHER) ? PATH_RAY_SHADOW_NON_CATCHER : PATH_RAY_SHADOW;
#else
  return PATH_RAY_SHADOW;
#endif
}

ccl_device bool integrate_intersect_shadow_opaque(INTEGRATOR_STATE_ARGS,
                                                  const Ray *ray,
                                                  const uint visibility)
{
  PROFILING_INIT(kg, PROFILING_SCENE_INTERSECT);

  Intersection isect;
  const bool opaque_hit = scene_intersect(kg, ray, visibility & PATH_RAY_SHADOW_OPAQUE, &isect);

  if (!opaque_hit) {
    /* Null terminator for array, no transparent hits to shade. */
    INTEGRATOR_STATE_ARRAY_WRITE(shadow_isect, 0, prim) = PRIM_NONE;
  }

  return opaque_hit;
}

#ifdef __TRANSPARENT_SHADOWS__
ccl_device bool integrate_intersect_shadow_transparent(INTEGRATOR_STATE_ARGS,
                                                       const Ray *ray,
                                                       const uint visibility)
{
  /* TODO: add mechanism to detect and continue tracing if max_hits exceeded. */
  PROFILING_INIT(kg, PROFILING_SCENE_INTERSECT);

  Intersection isect[INTEGRATOR_SHADOW_ISECT_SIZE];
  const uint max_hits = INTEGRATOR_SHADOW_ISECT_SIZE;
  uint num_hits;
  const bool opaque_hit = scene_intersect_shadow_all(
      kg, ray, isect, visibility, max_hits, &num_hits);

  if (!opaque_hit) {
    if (num_hits > 0) {
      sort_intersections(isect, num_hits);

      /* Write intersection result into global integrator state memory. */
      for (int hit = 0; hit < num_hits; hit++) {
        integrator_state_write_shadow_isect(INTEGRATOR_STATE_PASS, &isect[hit], hit);
      }
    }

    INTEGRATOR_STATE_WRITE(shadow_path, num_hits) = num_hits;
  }

  return opaque_hit;
}
#endif

ccl_device void integrator_intersect_shadow(INTEGRATOR_STATE_ARGS)
{
  /* Read ray from integrator state into local memory. */
  Ray ray ccl_optional_struct_init;
  integrator_state_read_shadow_ray(INTEGRATOR_STATE_PASS, &ray);

  /* Compute visibility. */
  const uint visibility = integrate_intersect_shadow_visibility(INTEGRATOR_STATE_PASS);

#ifdef __TRANSPARENT_SHADOWS__
  /* TODO: compile different kernels depending on this? Especially for OptiX
   * conditional trace calls are bad. */
  const bool opaque_hit =
      (kernel_data.integrator.transparent_shadows) ?
          integrate_intersect_shadow_transparent(INTEGRATOR_STATE_PASS, &ray, visibility) :
          integrate_intersect_shadow_opaque(INTEGRATOR_STATE_PASS, &ray, visibility);
#else
  const bool opaque_hit = integrate_intersect_shadow_opaque(
      INTEGRATOR_STATE_PASS, &ray, visibility);
#endif

  if (opaque_hit) {
    /* Hit an opaque surface, shadow path ends here. */
    INTEGRATOR_SHADOW_PATH_TERMINATE(INTERSECT_SHADOW);
    return;
  }
  else {
    /* Hit nothing or transparent surfaces, continue to shadow kernel
     * for shading and render buffer output.
     *
     * TODO: could also write to render buffer directly if no transparent shadows?
     * Could save a kernel execution for the common case. */
    INTEGRATOR_SHADOW_PATH_NEXT(INTERSECT_SHADOW, SHADE_SHADOW);
    return;
  }
}

CCL_NAMESPACE_END
