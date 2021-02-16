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

ccl_device void kernel_integrate_background(INTEGRATOR_STATE_ARGS, ccl_global float *render_buffer)
{
  kernel_assert(INTEGRATOR_STATE(isect, object) == OBJECT_NONE);

  /* Write to render buffer. */
  render_buffer[0] = 0.0f;

  /* Path ends here. */
  INTEGRATOR_FLOW_END;
}

CCL_NAMESPACE_END
