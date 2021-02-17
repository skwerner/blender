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

ccl_device void kernel_integrate_generate_camera_rays(INTEGRATOR_STATE_ARGS)
{
  /* Initialize path state */
  INTEGRATOR_STATE_WRITE(path, x) = 0;
  INTEGRATOR_STATE_WRITE(path, y) = 0;
  INTEGRATOR_STATE_WRITE(path, sample) = 0;
  INTEGRATOR_STATE_WRITE(path, depth) = 0;
  INTEGRATOR_STATE_WRITE(path, rng) = 0;
  INTEGRATOR_STATE_WRITE(path, ray_pdf) = 1.0f;
  INTEGRATOR_STATE_WRITE(path, throughput) = make_float3(1.0f, 1.0f, 1.0f);

  INTEGRATOR_STATE_ARRAY_WRITE(volume_stack, 0, object) = OBJECT_NONE;
  INTEGRATOR_STATE_ARRAY_WRITE(volume_stack, 0, shader) = SHADER_NONE;

  /* Generate camera ray. */
  INTEGRATOR_STATE_WRITE(ray, P) = make_float3(0.0f, 0.0f, 0.0f);
  INTEGRATOR_STATE_WRITE(ray, D) = make_float3(0.0f, 0.0f, 1.0f);
  INTEGRATOR_STATE_WRITE(ray, t) = FLT_MAX;
  INTEGRATOR_STATE_WRITE(ray, time) = 0.0f;

  /* Continue with intersect_closest kernel. */
  INTEGRATOR_FLOW_QUEUE(intersect_closest);
}

CCL_NAMESPACE_END
