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

#include "kernel/integrator/integrator_state_util.h"
#include "kernel/kernel_path_state.h"

CCL_NAMESPACE_BEGIN

/* Initialize shadow catcher state.
 * Is to be called from the very beginning, for example, from `integrator_init_from_camera()`. */
ccl_device_inline void kernel_shadow_catcher_state_init(INTEGRATOR_STATE_ARGS)
{
  /* When there are shadow catchers in the scene half of the states will be provisioned for use for
   * the split state. Those are to be initialized here to have no kernels scheduled for until split
   * actually happens.
   *
   * When there are no shadow catchers in the scene all path states are used for path tracing, and
   * none of the split states are to be modified here (as it will interfere with rendering. */

  if (!kernel_data.integrator.has_shadow_catcher) {
    return;
  }

  path_state_init_queues(INTEGRATOR_SHADOW_CATCHER_STATE_PASS);
}

/* Check whether current surface bounce is where path is to be split for the shadow catcher. */
ccl_device_inline bool kernel_shadow_catcher_is_path_split_bounce(INTEGRATOR_STATE_ARGS,
                                                                  const int object_flag)
{
#ifdef __SHADOW_CATCHER__
  if (!kernel_data.integrator.has_shadow_catcher) {
    return false;
  }

  /* Check the flag first, avoiding fetches form global memory. */
  if ((object_flag & SD_OBJECT_SHADOW_CATCHER) == 0) {
    return false;
  }

  const int path_flag = INTEGRATOR_STATE(path, flag);

  if ((path_flag & PATH_RAY_CAMERA) == 0) {
    /* Split only on primary rays, secondary bounces are to treat shadow catcher as a regular
     * object. */
    return false;
  }

  if (path_flag & PATH_RAY_SHADOW_CATCHER_PASS) {
    return false;
  }

  return true;
#else
  (void)object_flag;
  return false;
#endif
}

ccl_device void kernel_shadow_catcher_split(INTEGRATOR_STATE_ARGS, const int object_flag)
{
#ifdef __SHADOW_CATCHER__

  if (!kernel_shadow_catcher_is_path_split_bounce(INTEGRATOR_STATE_PASS, object_flag)) {
    return;
  }

  /* The split is to be done. Mark the current state as such, so that it stops contributing to the
   * shadow catcher matte pass, but keeps contributing to the combined pass. */
  INTEGRATOR_STATE_WRITE(path, flag) |= PATH_RAY_SHADOW_CATCHER_HIT;

  /* Split new state from the current one. This new state will only track contribution of shadow
   * catcher objects ignoring non-catcher objects. */

  integrator_state_copy_to_shadow_catcher(INTEGRATOR_STATE_PASS);

  INTEGRATOR_SHADOW_CATCHER_STATE_WRITE(path, flag) |= PATH_RAY_SHADOW_CATCHER_PASS;

#endif
}

#ifdef __SHADOW_CATCHER__

ccl_device_forceinline bool kernel_shadow_catcher_is_matte_path(INTEGRATOR_STATE_CONST_ARGS)
{
  return (INTEGRATOR_STATE(path, flag) & PATH_RAY_SHADOW_CATCHER_HIT) == 0;
}

ccl_device_forceinline bool kernel_shadow_catcher_is_object_pass(INTEGRATOR_STATE_CONST_ARGS)
{
  return INTEGRATOR_STATE(path, flag) & PATH_RAY_SHADOW_CATCHER_PASS;
}

#endif /* __SHADOW_CATCHER__ */

CCL_NAMESPACE_END
