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

#include "kernel/integrator/integrator_subsurface.h"

CCL_NAMESPACE_BEGIN

ccl_device void integrator_intersect_subsurface(INTEGRATOR_STATE_ARGS)
{
#ifdef __SUBSURFACE__
  if (subsurface_random_walk(INTEGRATOR_STATE_PASS)) {
    return;
  }
#endif

  /* TODO: update volume stack. Instead of a special for_subsurface, we could
   * just re-init the volume stack completely, sharing the same kernel as for
   * cameras. */
#if 0
#  ifdef __VOLUME__
  bool need_update_volume_stack = kernel_data.integrator.use_volumes &&
                                  sd->object_flag & SD_OBJECT_INTERSECTS_VOLUME;

  if (need_update_volume_stack) {
    Ray volume_ray = *ray;
    /* Setup ray from previous surface point to the new one. */
    volume_ray.D = normalize_len(hit_ray->P - volume_ray.P, &volume_ray.t);

    kernel_volume_stack_update_for_subsurface(
        kg, emission_sd, &volume_ray, hit_state->volume_stack);
  }
#  endif /* __VOLUME__ */
#endif

  INTEGRATOR_PATH_TERMINATE(DEVICE_KERNEL_INTEGRATOR_INTERSECT_SUBSURFACE);
}

CCL_NAMESPACE_END
