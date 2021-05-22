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
    /* Integrator. */
    case DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA:
      return "integrator_init_from_camera";
    case DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST:
      return "integrator_intersect_closest";
    case DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW:
      return "integrator_intersect_shadow";
    case DEVICE_KERNEL_INTEGRATOR_INTERSECT_SUBSURFACE:
      return "integrator_intersect_subsurface";
    case DEVICE_KERNEL_INTEGRATOR_SHADE_BACKGROUND:
      return "integrator_shade_background";
    case DEVICE_KERNEL_INTEGRATOR_SHADE_LIGHT:
      return "integrator_shade_light";
    case DEVICE_KERNEL_INTEGRATOR_SHADE_SHADOW:
      return "integrator_shade_shadow";
    case DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE:
      return "integrator_shade_surface";
    case DEVICE_KERNEL_INTEGRATOR_SHADE_VOLUME:
      return "integrator_shade_volume";
    case DEVICE_KERNEL_INTEGRATOR_MEGAKERNEL:
      return "integrator_megakernel";
    case DEVICE_KERNEL_INTEGRATOR_QUEUED_PATHS_ARRAY:
      return "integrator_queued_paths_array";
    case DEVICE_KERNEL_INTEGRATOR_QUEUED_SHADOW_PATHS_ARRAY:
      return "integrator_queued_shadow_paths_array";
    case DEVICE_KERNEL_INTEGRATOR_ACTIVE_PATHS_ARRAY:
      return "integrator_active_paths_array";
    case DEVICE_KERNEL_INTEGRATOR_TERMINATED_PATHS_ARRAY:
      return "integrator_terminated_paths_array";
    case DEVICE_KERNEL_INTEGRATOR_SORTED_PATHS_ARRAY:
      return "integrator_sorted_paths_array";
    case DEVICE_KERNEL_INTEGRATOR_RESET:
      return "integrator_reset";

    /* Shader evaluation. */
    case DEVICE_KERNEL_SHADER_EVAL_DISPLACE:
      return "shader_eval_displace";
    case DEVICE_KERNEL_SHADER_EVAL_BACKGROUND:
      return "shader_eval_background";

    /* Film. */
    case DEVICE_KERNEL_CONVERT_TO_HALF_FLOAT:
      return "convert_to_half_float";

    /* Adaptive sampling. */
    case DEVICE_KERNEL_ADAPTIVE_SAMPLING_CONVERGENCE_CHECK:
      return "adaptive_sampling_convergence_check";
    case DEVICE_KERNEL_ADAPTIVE_SAMPLING_CONVERGENCE_FILTER_X:
      return "adaptive_sampling_filter_x";
    case DEVICE_KERNEL_ADAPTIVE_SAMPLING_CONVERGENCE_FILTER_Y:
      return "adaptive_sampling_filter_y";

    /* Denoising. */
    case DEVICE_KERNEL_FILTER_CONVERT_TO_RGB:
      return "filter_convert_to_rgb";
    case DEVICE_KERNEL_FILTER_CONVERT_FROM_RGB:
      return "filter_convert_from_rgb";

    /* Generic */
    case DEVICE_KERNEL_PREFIX_SUM:
      return "prefix_sum";

    case DEVICE_KERNEL_NUM:
      break;
  };
  LOG(FATAL) << "Unhandled kernel " << static_cast<int>(kernel) << ", should never happen.";
  return "UNKNOWN";
}

std::ostream &operator<<(std::ostream &os, DeviceKernel kernel)
{
  os << device_kernel_as_string(kernel);
  return os;
}

string device_kernel_mask_as_string(DeviceKernelMask mask)
{
  string str;

  for (uint64_t i = 0; i < sizeof(DeviceKernelMask) * 8; i++) {
    if (mask & (uint64_t(1) << i)) {
      if (!str.empty()) {
        str += " ";
      }
      str += device_kernel_as_string((DeviceKernel)i);
    }
  }

  return str;
}

CCL_NAMESPACE_END
