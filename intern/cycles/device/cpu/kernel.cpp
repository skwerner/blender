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

#include "device/cpu/kernel.h"

// clang-format off
#include "kernel/kernel.h"
#include "kernel/filter/filter.h"
// clang-format on

CCL_NAMESPACE_BEGIN

#define KERNEL_FUNCTIONS(name) \
  KERNEL_NAME_EVAL(cpu, name), KERNEL_NAME_EVAL(cpu_sse2, name), \
      KERNEL_NAME_EVAL(cpu_sse3, name), KERNEL_NAME_EVAL(cpu_sse41, name), \
      KERNEL_NAME_EVAL(cpu_avx, name), KERNEL_NAME_EVAL(cpu_avx2, name)

#define REGISTER_KERNEL(name) name(KERNEL_FUNCTIONS(name))

CPUKernels::CPUKernels()
    : REGISTER_KERNEL(path_trace),
      REGISTER_KERNEL(convert_to_half_float),
      REGISTER_KERNEL(convert_to_byte),
      REGISTER_KERNEL(shader),
      REGISTER_KERNEL(bake),
      REGISTER_KERNEL(filter_divide_shadow),
      REGISTER_KERNEL(filter_get_feature),
      REGISTER_KERNEL(filter_write_feature),
      REGISTER_KERNEL(filter_detect_outliers),
      REGISTER_KERNEL(filter_combine_halves),
      REGISTER_KERNEL(filter_nlm_calc_difference),
      REGISTER_KERNEL(filter_nlm_blur),
      REGISTER_KERNEL(filter_nlm_calc_weight),
      REGISTER_KERNEL(filter_nlm_update_output),
      REGISTER_KERNEL(filter_nlm_normalize),
      REGISTER_KERNEL(filter_construct_transform),
      REGISTER_KERNEL(filter_nlm_construct_gramian),
      REGISTER_KERNEL(filter_finalize),
      REGISTER_KERNEL(background),
      REGISTER_KERNEL(generate_camera_rays),
      REGISTER_KERNEL(intersect_closest),
      REGISTER_KERNEL(intersect_shadow),
      REGISTER_KERNEL(shadow),
      REGISTER_KERNEL(subsurface),
      REGISTER_KERNEL(surface),
      REGISTER_KERNEL(volume)
{
}

#undef REGISTER_KERNEL
#undef KERNEL_FUNCTIONS

CCL_NAMESPACE_END
