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

#include "kernel/kernel_write_passes.h"

CCL_NAMESPACE_BEGIN

ccl_device void kernel_integrate_background(INTEGRATOR_STATE_ARGS,
                                            ccl_global float *ccl_restrict render_buffer)
{
  kernel_assert(INTEGRATOR_STATE(isect, object) == OBJECT_NONE);

  /* Placeholder. */
  const float3 L = make_float3(0.0f, 0.0f, 0.0f);
  const float alpha = 0.0f;

  if (kernel_data.film.pass_flag & PASSMASK(COMBINED)) {
    kernel_write_pass_float4(render_buffer, make_float4(L.x, L.y, L.z, alpha));
  }

  /* Path ends here. */
  INTEGRATOR_FLOW_END;
}

CCL_NAMESPACE_END
