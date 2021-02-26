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
    sort_intersections(isect, num_hits);

    /* Write intersection result into global integrator state memory. */
    for (int hit = 0; hit < num_hits; hit++) {
      INTEGRATOR_STATE_ARRAY_WRITE(shadow_isect, hit, t) = isect[hit].t;
      INTEGRATOR_STATE_ARRAY_WRITE(shadow_isect, hit, u) = isect[hit].u;
      INTEGRATOR_STATE_ARRAY_WRITE(shadow_isect, hit, v) = isect[hit].v;
      INTEGRATOR_STATE_ARRAY_WRITE(shadow_isect, hit, object) = isect[hit].object;
      INTEGRATOR_STATE_ARRAY_WRITE(shadow_isect, hit, prim) = isect[hit].prim;
      INTEGRATOR_STATE_ARRAY_WRITE(shadow_isect, hit, type) = isect[hit].type;
#  ifdef __EMBREE__
      INTEGRATOR_STATE_ARRAY_WRITE(shadow_isect, hit, Ng) = isect[hit].Ng;
#  endif
    }

    /* Null terminator for array. */
    if (num_hits < INTEGRATOR_SHADOW_ISECT_SIZE) {
      INTEGRATOR_STATE_ARRAY_WRITE(shadow_isect, num_hits, prim) = PRIM_NONE;
    }
  }

  return opaque_hit;
}
#endif

ccl_device void integrator_intersect_shadow(INTEGRATOR_STATE_ARGS)
{
  /* Only execute if shadow ray needs to be traced. */
  if (INTEGRATOR_SHADOW_PATH_IS_TERMINATED) {
    return;
  }

  /* Read ray from integrator state into local memory. */
  Ray ray;
  ray.P = INTEGRATOR_STATE(shadow_ray, P);
  ray.D = INTEGRATOR_STATE(shadow_ray, D);
  ray.t = INTEGRATOR_STATE(shadow_ray, t);
  ray.time = INTEGRATOR_STATE(shadow_ray, time);
  ray.dP = differential3_zero();
  ray.dD = differential3_zero();

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
    INTEGRATOR_SHADOW_PATH_TERMINATE;
    return;
  }
  else {
    /* Hit nothing or transparent surfaces, continue to shadow kernel
     * for shading and render buffer output.
     *
     * TODO: could also write to render buffer directly if no transparent shadows?
     * Could save a kernel execution for the common case. */
    INTEGRATOR_SHADOW_PATH_NEXT(shadow);
    return;
  }
}

CCL_NAMESPACE_END
