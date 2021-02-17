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

ccl_device void kernel_integrate_intersect_shadow(INTEGRATOR_STATE_ARGS)
{
  kernel_assert(INTEGRATOR_STATE(shadow_ray, t) != 0.0f);

  /* Scene ray intersection. */
  INTEGRATOR_STATE_ARRAY_WRITE(shadow_isect, 0, t) = 0.0f;
  INTEGRATOR_STATE_ARRAY_WRITE(shadow_isect, 0, u) = 0.0f;
  INTEGRATOR_STATE_ARRAY_WRITE(shadow_isect, 0, v) = 0.0f;
  INTEGRATOR_STATE_ARRAY_WRITE(shadow_isect, 0, object) = OBJECT_NONE;
  INTEGRATOR_STATE_ARRAY_WRITE(shadow_isect, 0, prim) = PRIM_NONE;
  INTEGRATOR_STATE_ARRAY_WRITE(shadow_isect, 0, type) = PRIMITIVE_NONE;

#if INTEGRATOR_SHADOW_ISECT_SIZE > 1
  /* Null terminator for array. */
  INTEGRATOR_STATE_ARRAY_WRITE(shadow_isect, 1, object) = OBJECT_NONE;
  INTEGRATOR_STATE_ARRAY_WRITE(shadow_isect, 1, prim) = PRIM_NONE;
  INTEGRATOR_STATE_ARRAY_WRITE(shadow_isect, 1, type) = PRIMITIVE_NONE;
#endif

  const bool shadow_opaque = true;
  if (INTEGRATOR_STATE_ARRAY(shadow_isect, 0, object) != OBJECT_NONE && shadow_opaque) {
    /* Hit an opaque surface, shadow path ends here. */
    INTEGRATOR_FLOW_SHADOW_END;
    return;
  }
  else {
    /* Hit nothing or transparent surfaces, continue to shadow kernel
     * for shading and render buffer output.
     *
     * TODO: could also write to render buffer directly if no transparent shadows?
     * Could save a kernel execution for the common case. */
    INTEGRATOR_FLOW_SHADOW_QUEUE(shadow);
    return;
  }
}

CCL_NAMESPACE_END
