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

#include <ostream>  // NOLINT

CCL_NAMESPACE_BEGIN

/* high level identifier of a kernel.
 * Used in the device API to communicate which kernel caller is interested in. */
enum class DeviceKernel {
  INTEGRATOR_INIT_FROM_CAMERA = 0,
  INTEGRATOR_INTERSECT_CLOSEST,
  INTEGRATOR_INTERSECT_SHADOW,
  INTEGRATOR_INTERSECT_SUBSURFACE,
  INTEGRATOR_SHADE_BACKGROUND,
  INTEGRATOR_SHADE_SHADOW,
  INTEGRATOR_SHADE_SURFACE,
  INTEGRATOR_SHADE_VOLUME,
};
const char *device_kernel_as_string(DeviceKernel kernel);
std::ostream &operator<<(std::ostream &os, DeviceKernel kernel);

CCL_NAMESPACE_END
