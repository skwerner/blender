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

ccl_device void kernel_integrate_surface(INTEGRATOR_STATE_ARGS, ccl_global float *render_buffer)
{
  kernel_assert(INTEGRATOR_STATE(isect, object) != OBJECT_NONE);

  /* Evaluate shader. */

  /* Subsurface scattering does scattering, direct and indirect light in own kernel. */
  const bool subsurface = false;
  if (subsurface) {
    INTEGRATOR_STATE_WRITE(subsurface, albedo) = make_float3(1.0f, 1.0f, 1.0f);
    INTEGRATOR_FLOW_QUEUE(subsurface);
    return;
  }

  /* Direct lighting. */
  const float3 throughput = INTEGRATOR_STATE(path, throughput);
  const bool direct_lighting = false;
  if (direct_lighting) {
    /* Generate shadow ray. */
    INTEGRATOR_STATE_WRITE(shadow_ray, P) = make_float3(0.0f, 0.0f, 0.0f);
    INTEGRATOR_STATE_WRITE(shadow_ray, D) = make_float3(0.0f, 0.0f, 1.0f);
    INTEGRATOR_STATE_WRITE(shadow_ray, t) = FLT_MAX;
    INTEGRATOR_STATE_WRITE(shadow_ray, time) = 0.0f;

    /* Copy entire path state. */
    INTEGRATOR_STATE_COPY(shadow_path, path);
    INTEGRATOR_STATE_COPY(shadow_volume_stack, volume_stack);

    /* Branch of shadow kernel. */
    INTEGRATOR_FLOW_SHADOW_QUEUE(intersect_shadow);
  }

  const bool end_path = true;
  if (end_path) {
    /* End path. */
    INTEGRATOR_FLOW_END;
    return;
  }
  else {
    /* Sample BSDF and continue path. */
    INTEGRATOR_STATE_WRITE(ray, P) = make_float3(0.0f, 0.0f, 0.0f);
    INTEGRATOR_STATE_WRITE(ray, D) = make_float3(0.0f, 0.0f, 1.0f);
    INTEGRATOR_STATE_WRITE(ray, t) = FLT_MAX;
    INTEGRATOR_STATE_WRITE(ray, time) = 0.0f;
    INTEGRATOR_STATE_WRITE(path, throughput) = throughput;

    /* Queue intersect_closest kernel. */
    INTEGRATOR_FLOW_QUEUE(intersect_closest);
    return;
  }
}

CCL_NAMESPACE_END
