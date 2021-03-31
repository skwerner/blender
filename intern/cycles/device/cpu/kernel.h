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
  using ConvertToHalfFloatFunction = CPUKernelFunction<void (*)(const KernelGlobals *kg,
                                                                uchar4 *rgba,
                                                                float *buffer,
                                                                float sample_scale,
                                                                int x,
                                                                int y,
                                                                int offset,
                                                                int stride)>;

  ConvertToHalfFloatFunction convert_to_half_float;
  ShaderEvalFunction shader_eval_displace;
  ShaderEvalFunction shader_eval_background;
  CPUKernelFunction<void (*)(const KernelGlobals *, float *, int, int, int, int, int)> bake;

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
