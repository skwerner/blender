/*
 * Copyright 2011-2013 Blender Foundation
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

/* Templated common declaration part of all CPU kernels. */

void KERNEL_FUNCTION_FULL_NAME(path_trace)(
    const KernelGlobals *kg, float *buffer, int sample, int x, int y, int offset, int stride);

void KERNEL_FUNCTION_FULL_NAME(convert_to_byte)(const KernelGlobals *kg,
                                                uchar4 *rgba,
                                                float *buffer,
                                                float sample_scale,
                                                int x,
                                                int y,
                                                int offset,
                                                int stride);

void KERNEL_FUNCTION_FULL_NAME(convert_to_half_float)(const KernelGlobals *kg,
                                                      uchar4 *rgba,
                                                      float *buffer,
                                                      float sample_scale,
                                                      int x,
                                                      int y,
                                                      int offset,
                                                      int stride);

void KERNEL_FUNCTION_FULL_NAME(shader_eval_background)(const KernelGlobals *kg,
                                                       const KernelShaderEvalInput *input,
                                                       float4 *output,
                                                       const int offset);
void KERNEL_FUNCTION_FULL_NAME(shader_eval_displace)(const KernelGlobals *kg,
                                                     const KernelShaderEvalInput *input,
                                                     float4 *output,
                                                     const int offset);
void KERNEL_FUNCTION_FULL_NAME(bake)(
    const KernelGlobals *kg, float *buffer, int sample, int x, int y, int offset, int stride);

/* ********************************************************************************************* */
/* *                            *** The new split kernel ***                                   * */
/* ********************************************************************************************* */

#define KERNEL_INTEGRATOR_FUNCTION(name) \
  void KERNEL_FUNCTION_FULL_NAME(integrator_##name)(const KernelGlobals *ccl_restrict kg, \
                                                    IntegratorState *state)

#define KERNEL_INTEGRATOR_SHADE_FUNCTION(name) \
  void KERNEL_FUNCTION_FULL_NAME(integrator_##name)(const KernelGlobals *ccl_restrict kg, \
                                                    IntegratorState *state, \
                                                    ccl_global float *render_buffer)

#define KERNEL_INTEGRATOR_INIT_FUNCTION(name) \
  void KERNEL_FUNCTION_FULL_NAME(integrator_##name)( \
      const KernelGlobals *ccl_restrict kg, IntegratorState *state, KernelWorkTile *tile)

KERNEL_INTEGRATOR_INIT_FUNCTION(init_from_camera);
KERNEL_INTEGRATOR_FUNCTION(intersect_closest);
KERNEL_INTEGRATOR_FUNCTION(intersect_shadow);
KERNEL_INTEGRATOR_FUNCTION(intersect_subsurface);
KERNEL_INTEGRATOR_SHADE_FUNCTION(shade_background);
KERNEL_INTEGRATOR_SHADE_FUNCTION(shade_light);
KERNEL_INTEGRATOR_SHADE_FUNCTION(shade_shadow);
KERNEL_INTEGRATOR_SHADE_FUNCTION(shade_surface);
KERNEL_INTEGRATOR_SHADE_FUNCTION(shade_volume);
KERNEL_INTEGRATOR_SHADE_FUNCTION(megakernel);

#undef KERNEL_INTEGRATOR_FUNCTION

#undef KERNEL_ARCH
