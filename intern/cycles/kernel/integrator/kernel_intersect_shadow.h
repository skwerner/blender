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

ccl_device_forceinline bool intersect_shadow_scene(INTEGRATOR_STATE_CONST_ARGS,
                                                   const Ray *ray,
                                                   Intersection *isect)
{
  PROFILING_INIT(kg, PROFILING_SCENE_INTERSECT);
  const uint32_t path_flag = INTEGRATOR_STATE(path, flag);
#ifdef __SHADOW_TRICKS__
  const uint visibility = (path_flag & PATH_RAY_SHADOW_CATCHER) ? PATH_RAY_SHADOW_NON_CATCHER :
                                                                  PATH_RAY_SHADOW;
#else
  const uint visibility = PATH_RAY_SHADOW;
#endif

  /* TODO: transparent shadows. */
  return scene_intersect(kg, ray, visibility & PATH_RAY_SHADOW_OPAQUE, isect);
}

ccl_device void kernel_integrate_intersect_shadow(INTEGRATOR_STATE_ARGS)
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

  /* TODO: this means light casts no shadow. */
  kernel_assert(ray.t != 0.0f);

  /* Scene Intersection. */
  Intersection isect;
  const bool hit = intersect_shadow_scene(INTEGRATOR_STATE_PASS, &ray, &isect);
  if (!hit) {
    isect.prim = PRIM_NONE;
  }

  /* Write intersection result into global integrator state memory. */
  INTEGRATOR_STATE_ARRAY_WRITE(shadow_isect, 0, t) = isect.t;
  INTEGRATOR_STATE_ARRAY_WRITE(shadow_isect, 0, u) = isect.u;
  INTEGRATOR_STATE_ARRAY_WRITE(shadow_isect, 0, v) = isect.v;
  INTEGRATOR_STATE_ARRAY_WRITE(shadow_isect, 0, Ng) = isect.Ng;
  INTEGRATOR_STATE_ARRAY_WRITE(shadow_isect, 0, object) = isect.object;
  INTEGRATOR_STATE_ARRAY_WRITE(shadow_isect, 0, prim) = isect.prim;
  INTEGRATOR_STATE_ARRAY_WRITE(shadow_isect, 0, type) = isect.type;

#if INTEGRATOR_SHADOW_ISECT_SIZE > 1
  /* Null terminator for array. */
  INTEGRATOR_STATE_ARRAY_WRITE(shadow_isect, 1, object) = OBJECT_NONE;
  INTEGRATOR_STATE_ARRAY_WRITE(shadow_isect, 1, prim) = PRIM_NONE;
  INTEGRATOR_STATE_ARRAY_WRITE(shadow_isect, 1, type) = PRIMITIVE_NONE;
#endif

  const bool shadow_opaque = true;
  if (hit && shadow_opaque) {
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
