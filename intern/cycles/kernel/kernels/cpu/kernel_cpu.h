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
    KernelGlobals *kg, float *buffer, int sample, int x, int y, int offset, int stride);

void KERNEL_FUNCTION_FULL_NAME(convert_to_byte)(KernelGlobals *kg,
                                                uchar4 *rgba,
                                                float *buffer,
                                                float sample_scale,
                                                int x,
                                                int y,
                                                int offset,
                                                int stride);

void KERNEL_FUNCTION_FULL_NAME(convert_to_half_float)(KernelGlobals *kg,
                                                      uchar4 *rgba,
                                                      float *buffer,
                                                      float sample_scale,
                                                      int x,
                                                      int y,
                                                      int offset,
                                                      int stride);

void KERNEL_FUNCTION_FULL_NAME(shader)(KernelGlobals *kg,
                                       uint4 *input,
                                       float4 *output,
                                       int type,
                                       int filter,
                                       int i,
                                       int offset,
                                       int sample);

void KERNEL_FUNCTION_FULL_NAME(bake)(
    KernelGlobals *kg, float *buffer, int sample, int x, int y, int offset, int stride);

/* ********************************************************************************************* */
/* *                            *** The new split kernel ***                                   * */
/* ********************************************************************************************* */

#define KERNEL_INTEGRATOR_FUNCTION(name) \
  void KERNEL_FUNCTION_FULL_NAME(name)(const KernelGlobals *kg, IntegratorState *state)

KERNEL_INTEGRATOR_FUNCTION(generate_camera_rays);
KERNEL_INTEGRATOR_FUNCTION(intersect_closest);
KERNEL_INTEGRATOR_FUNCTION(intersect_shadow);
KERNEL_INTEGRATOR_FUNCTION(shadow);
KERNEL_INTEGRATOR_FUNCTION(subsurface);
KERNEL_INTEGRATOR_FUNCTION(surface);
KERNEL_INTEGRATOR_FUNCTION(volume);

#undef KERNEL_INTEGRATOR_FUNCTION

#undef KERNEL_ARCH
