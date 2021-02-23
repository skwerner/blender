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

ccl_device void kernel_integrate_shadow(INTEGRATOR_STATE_ARGS,
                                        ccl_global float *ccl_restrict render_buffer)
{
  /* Only execute if anything was hit, otherwise path must have been terminated. */
  if (INTEGRATOR_SHADOW_PATH_IS_TERMINATED) {
    return;
  }

  kernel_assert(INTEGRATOR_STATE_ARRAY(shadow_isect, 0, prim) != PRIM_NONE);

  /* Modify throughput. */
  const bool shadow_isect_done = true;
  if (shadow_isect_done) {
    const float3 L = INTEGRATOR_STATE(shadow_light, L);
    kernel_accum_light(INTEGRATOR_STATE_PASS, L, render_buffer);

    INTEGRATOR_SHADOW_PATH_TERMINATE;
    return;
  }
  else {
    /* Continue shadow path. */
    const float new_t = INTEGRATOR_STATE(shadow_ray, t);
    INTEGRATOR_STATE_WRITE(shadow_ray, t) = new_t;
    INTEGRATOR_STATE_WRITE(shadow_path, throughput) = INTEGRATOR_STATE(shadow_path, throughput);

    INTEGRATOR_SHADOW_PATH_NEXT(intersect_shadow);
    return;
  }
}

CCL_NAMESPACE_END
