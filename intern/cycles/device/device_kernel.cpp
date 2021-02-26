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

#include "device/device_kernel.h"

#include "util/util_logging.h"

CCL_NAMESPACE_BEGIN

const char *device_kernel_as_string(DeviceKernel kernel)
{
  switch (kernel) {
    case DeviceKernel::INTEGRATOR_INIT_FROM_CAMERA:
      return "integrator_init_from_camera";
    case DeviceKernel::INTEGRATOR_INTERSECT_CLOSEST:
      return "integrator_intersect_closest";
    case DeviceKernel::INTEGRATOR_INTERSECT_SHADOW:
      return "integrator_intersect_shadow";
    case DeviceKernel::INTEGRATOR_INTERSECT_SUBSURFACE:
      return "integrator_intersect_subsurface";
    case DeviceKernel::INTEGRATOR_SHADE_BACKGROUND:
      return "integrator_shade_background";
    case DeviceKernel::INTEGRATOR_SHADE_SHADOW:
      return "integrator_shade_shadow";
    case DeviceKernel::INTEGRATOR_SHADE_SURFACE:
      return "integrator_shade_surface";
    case DeviceKernel::INTEGRATOR_SHADE_VOLUME:
      return "integrator_shade_volume";
  };
  LOG(FATAL) << "Unhandled kernel " << static_cast<int>(kernel) << ", should never happen.";
  return "UNKNOWN";
}

std::ostream &operator<<(std::ostream &os, DeviceKernel kernel)
{
  os << device_kernel_as_string(kernel);
  return os;
}

CCL_NAMESPACE_END
