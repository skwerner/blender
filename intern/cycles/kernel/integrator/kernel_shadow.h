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

CCL_NAMESPACE_BEGIN

ccl_device void kernel_integrate_shadow(INTEGRATOR_STATE_ARGS, ccl_global float *render_buffer)
{
  kernel_assert(INTEGRATOR_STATE_ARRAY(shadow_isect, 0, object) != OBJECT_NONE);

  /* Modify throughput. */
  const float3 throughput = INTEGRATOR_STATE(path, throughput);

  const bool shadow_isect_done = true;
  if (shadow_isect_done) {
    const bool shadow_opaque = true;
    if (!shadow_opaque) {
      /* Write to render buffer. */
      render_buffer[0] = 0.0f;
    }
  }
  else {
    /* Continue shadow path. */
    const float new_t = INTEGRATOR_STATE(shadow_ray, t);
    INTEGRATOR_STATE_WRITE(shadow_ray, t) = new_t;
    INTEGRATOR_STATE_WRITE(path, throughput) = throughput;

    /* Queue intersect_shadow kernel. */
  }
}

CCL_NAMESPACE_END
