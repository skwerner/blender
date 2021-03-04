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

ccl_device void integrator_intersect_subsurface(INTEGRATOR_STATE_ARGS)
{
  /* Only execute if path is active and ray was marked for subsurface scattering. */
  if (INTEGRATOR_PATH_IS_TERMINATED || !(INTEGRATOR_STATE(path, flag) & PATH_RAY_SUBSURFACE)) {
    return;
  }

#ifdef __SUBSURFACE__
  /* Subsurface scattering to find exit point. */
  const float3 throughput = INTEGRATOR_STATE(path, throughput);

  /* We're done if no exit point found. */
  const bool exit_point_found = false;
  if (!exit_point_found) {
    INTEGRATOR_PATH_TERMINATE(intersect_subsurface);
    return;
  }

  /* Direct lighting. */
  const bool direct_lighting = false;
  if (direct_lighting) {
    /* Generate shadow ray. */
    INTEGRATOR_STATE_WRITE(shadow_ray, P) = make_float3(0.0f, 0.0f, 0.0f);
    INTEGRATOR_STATE_WRITE(shadow_ray, D) = make_float3(0.0f, 0.0f, 1.0f);
    INTEGRATOR_STATE_WRITE(shadow_ray, t) = FLT_MAX;
    INTEGRATOR_STATE_WRITE(shadow_ray, time) = 0.0f;

    /* Copy entire state and volume stack */
    INTEGRATOR_STATE_WRITE(shadow_path, throughput) = INTEGRATOR_STATE(path, throughput);

    /* Branch of shadow kernel. */
    INTEGRATOR_SHADOW_PATH_INIT(intersect_shadow);
  }

  /* Sample BSDF and continue path. */
  INTEGRATOR_STATE_WRITE(ray, P) = make_float3(0.0f, 0.0f, 0.0f);
  INTEGRATOR_STATE_WRITE(ray, D) = make_float3(0.0f, 0.0f, 1.0f);
  INTEGRATOR_STATE_WRITE(ray, t) = FLT_MAX;
  INTEGRATOR_STATE_WRITE(ray, time) = 0.0f;
  INTEGRATOR_STATE_WRITE(path, throughput) = throughput;

  /* Mark subsurface scattering as done. */
  INTEGRATOR_STATE_WRITE(path, flag) &= ~PATH_RAY_SUBSURFACE;

  /* Queue intersect_closest kernel. */
  INTEGRATOR_PATH_NEXT(intersect_subsurface, intersect_closest);
#endif /* __SUBSURFACE__ */
}

CCL_NAMESPACE_END
