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

#include "kernel/integrator/integrator_init_from_camera.h"
#include "kernel/integrator/integrator_intersect_closest.h"
#include "kernel/integrator/integrator_intersect_shadow.h"
#include "kernel/integrator/integrator_intersect_subsurface.h"
#include "kernel/integrator/integrator_shade_background.h"
#include "kernel/integrator/integrator_shade_shadow.h"
#include "kernel/integrator/integrator_shade_surface.h"
#include "kernel/integrator/integrator_shade_volume.h"

CCL_NAMESPACE_BEGIN

ccl_device void integrator_megakernel(INTEGRATOR_STATE_ARGS,
                                      ccl_global float *ccl_restrict render_buffer)
{
  while (!INTEGRATOR_PATH_IS_TERMINATED) {
    integrator_intersect_closest(INTEGRATOR_STATE_PASS);
    integrator_shade_volume(INTEGRATOR_STATE_PASS, render_buffer);
    integrator_shade_background(INTEGRATOR_STATE_PASS, render_buffer);
    integrator_shade_surface(INTEGRATOR_STATE_PASS, render_buffer);
    integrator_intersect_subsurface(INTEGRATOR_STATE_PASS);

    while (!INTEGRATOR_SHADOW_PATH_IS_TERMINATED) {
      integrator_intersect_shadow(INTEGRATOR_STATE_PASS);
      integrator_shade_shadow(INTEGRATOR_STATE_PASS, render_buffer);
    }
  }
}

CCL_NAMESPACE_END
