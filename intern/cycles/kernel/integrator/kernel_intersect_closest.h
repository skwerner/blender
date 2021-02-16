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

ccl_device void kernel_integrate_intersect_closest(INTEGRATOR_STATE_ARGS)
{
  kernel_assert(INTEGRATOR_STATE(ray, t) != 0.0f);

  /* Scene ray intersection. */
  INTEGRATOR_STATE_WRITE(isect, t) = 0.0f;
  INTEGRATOR_STATE_WRITE(isect, u) = 0.0f;
  INTEGRATOR_STATE_WRITE(isect, v) = 0.0f;
  INTEGRATOR_STATE_WRITE(isect, Ng) = make_float3(0.0f, 0.0f, 0.0f);
  INTEGRATOR_STATE_WRITE(isect, object) = OBJECT_NONE;
  INTEGRATOR_STATE_WRITE(isect, prim) = PRIM_NONE;
  INTEGRATOR_STATE_WRITE(isect, type) = PRIMITIVE_NONE;

#ifdef __VOLUME__
  if (INTEGRATOR_STATE_ARRAY(volume_stack, 0, object) != OBJECT_NONE) {
    /* Continue with volume kernel if we are inside a volume, regardless
     * if we hit anything. */
    INTEGRATOR_FLOW_QUEUE(volume);
    return;
  }
#endif

  if (INTEGRATOR_STATE(isect, object) == OBJECT_NONE) {
    /* Nothing hit, continue with background kernel. */
    INTEGRATOR_FLOW_QUEUE(background);
    return;
  }
  else {
    /* Hit a surface continue with surface kernel. */
    INTEGRATOR_FLOW_QUEUE(surface);
    return;
  }
}

CCL_NAMESPACE_END
