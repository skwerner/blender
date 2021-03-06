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
#include "kernel/kernel_projection.h"
#include "kernel/kernel_random.h"

#include "kernel/geom/geom.h"

#include "kernel/bvh/bvh.h"

CCL_NAMESPACE_BEGIN

ccl_device_forceinline bool intersect_closest_scene(INTEGRATOR_STATE_CONST_ARGS,
                                                    const Ray *ray,
                                                    Intersection *isect)
{
  PROFILING_INIT(kg, PROFILING_SCENE_INTERSECT);
  const uint visibility = path_state_ray_visibility(INTEGRATOR_STATE_PASS);

  /* TODO */
#if 0
  /* Trick to use short AO rays to approximate indirect light at the end of the path. */
  if (path_state_ao_bounce(INTEGRATOR_STATE_PASS)) {
    visibility = PATH_RAY_SHADOW;
    ray->t = kernel_data.background.ao_distance;
  }
#endif

  return scene_intersect(kg, ray, visibility, isect);
}

ccl_device void integrator_intersect_closest(INTEGRATOR_STATE_ARGS)
{
  /* Only execute if path is active. */
  if (INTEGRATOR_PATH_IS_TERMINATED) {
    return;
  }

  /* Read ray from integrator state into local memory. */
  Ray ray;
  ray.P = INTEGRATOR_STATE(ray, P);
  ray.D = INTEGRATOR_STATE(ray, D);
  ray.t = INTEGRATOR_STATE(ray, t);
  ray.time = INTEGRATOR_STATE(ray, time);
  ray.dP = differential3_zero();
  ray.dD = differential3_zero();

  kernel_assert(ray.t != 0.0f);

  /* Scene Intersection. */
  Intersection isect;
  const bool hit = intersect_closest_scene(INTEGRATOR_STATE_PASS, &ray, &isect);
  if (!hit) {
    isect.prim = PRIM_NONE;
  }

  /* Write intersection result into global integrator state memory. */
  INTEGRATOR_STATE_WRITE(isect, t) = isect.t;
  INTEGRATOR_STATE_WRITE(isect, u) = isect.u;
  INTEGRATOR_STATE_WRITE(isect, v) = isect.v;
  INTEGRATOR_STATE_WRITE(isect, object) = isect.object;
  INTEGRATOR_STATE_WRITE(isect, prim) = isect.prim;
  INTEGRATOR_STATE_WRITE(isect, type) = isect.type;
#ifdef __EMBREE__
  INTEGRATOR_STATE_WRITE(isect, Ng) = isect.Ng;
#endif

#ifdef __VOLUME__
  if (INTEGRATOR_STATE_ARRAY(volume_stack, 0, object) != OBJECT_NONE) {
    /* Continue with volume kernel if we are inside a volume, regardless
     * if we hit anything. */
    INTEGRATOR_PATH_NEXT(INTERSECT_CLOSEST, SHADE_VOLUME);
    return;
  }
#endif

  if (hit) {
    /* Hit a surface, continue with surface kernel. */
    INTEGRATOR_PATH_NEXT(INTERSECT_CLOSEST, SHADE_SURFACE);
    return;
  }
  else {
    /* Nothing hit, continue with background kernel. */
    INTEGRATOR_PATH_NEXT(INTERSECT_CLOSEST, SHADE_BACKGROUND);
    return;
  }
}

CCL_NAMESPACE_END
