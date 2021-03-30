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

#include "device/cpu/kernel_function.h"
#include "util/util_types.h"

CCL_NAMESPACE_BEGIN

struct KernelGlobals;
struct IntegratorState;
struct TileInfo;

class CPUKernels {
 public:
  using IntegratorFunction =
      CPUKernelFunction<void (*)(const KernelGlobals *kg, IntegratorState *state)>;
  using IntegratorShadeFunction = CPUKernelFunction<void (*)(
      const KernelGlobals *kg, IntegratorState *state, ccl_global float *render_buffer)>;
  using IntegratorInitFunction = CPUKernelFunction<void (*)(
      const KernelGlobals *kg, IntegratorState *state, KernelWorkTile *tile)>;
  using ShaderEvalFunction = CPUKernelFunction<void (*)(
      const KernelGlobals *kg, const KernelShaderEvalInput *, float4 *, const int)>;

  CPUKernelFunction<void (*)(const KernelGlobals *, float *, int, int, int, int, int)> path_trace;
  CPUKernelFunction<void (*)(const KernelGlobals *, uchar4 *, float *, float, int, int, int, int)>
      convert_to_half_float;
  CPUKernelFunction<void (*)(const KernelGlobals *, uchar4 *, float *, float, int, int, int, int)>
      convert_to_byte;
  ShaderEvalFunction shader_eval_displace;
  ShaderEvalFunction shader_eval_background;
  CPUKernelFunction<void (*)(const KernelGlobals *, float *, int, int, int, int, int)> bake;

  CPUKernelFunction<void (*)(
      int, TileInfo *, int, int, float *, float *, float *, float *, float *, int *, int, int)>
      filter_divide_shadow;
  CPUKernelFunction<void (*)(
      int, TileInfo *, int, int, int, int, float *, float *, float, int *, int, int)>
      filter_get_feature;
  CPUKernelFunction<void (*)(int, int, int, int *, float *, float *, int, int *)>
      filter_write_feature;
  CPUKernelFunction<void (*)(int, int, float *, float *, float *, float *, int *, int)>
      filter_detect_outliers;
  CPUKernelFunction<void (*)(int, int, float *, float *, float *, float *, int *, int)>
      filter_combine_halves;

  CPUKernelFunction<void (*)(
      int, int, float *, float *, float *, float *, int *, int, int, int, float, float)>
      filter_nlm_calc_difference;
  CPUKernelFunction<void (*)(float *, float *, int *, int, int)> filter_nlm_blur;
  CPUKernelFunction<void (*)(float *, float *, int *, int, int)> filter_nlm_calc_weight;
  CPUKernelFunction<void (*)(
      int, int, float *, float *, float *, float *, float *, int *, int, int, int)>
      filter_nlm_update_output;
  CPUKernelFunction<void (*)(float *, float *, int *, int)> filter_nlm_normalize;

  CPUKernelFunction<void (*)(
      float *, TileInfo *, int, int, int, float *, int *, int *, int, int, bool, int, float)>
      filter_construct_transform;
  CPUKernelFunction<void (*)(int,
                             int,
                             int,
                             float *,
                             float *,
                             float *,
                             int *,
                             float *,
                             float3 *,
                             int *,
                             int *,
                             int,
                             int,
                             int,
                             int,
                             bool)>
      filter_nlm_construct_gramian;
  CPUKernelFunction<void (*)(int, int, int, float *, int *, float *, float3 *, int *, int)>
      filter_finalize;

  IntegratorInitFunction integrator_init_from_camera;
  IntegratorFunction integrator_intersect_closest;
  IntegratorFunction integrator_intersect_shadow;
  IntegratorFunction integrator_intersect_subsurface;
  IntegratorShadeFunction integrator_shade_background;
  IntegratorShadeFunction integrator_shade_light;
  IntegratorShadeFunction integrator_shade_shadow;
  IntegratorShadeFunction integrator_shade_surface;
  IntegratorShadeFunction integrator_shade_volume;
  IntegratorShadeFunction integrator_megakernel;

  CPUKernels();
};

CCL_NAMESPACE_END
