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

/* Templated common implementation part of all CPU kernels.
 *
 * The idea is that particular .cpp files sets needed optimization flags and
 * simply includes this file without worry of copying actual implementation over.
 */

// clang-format off
#include "kernel/kernel_compat_cpu.h"

#ifndef KERNEL_STUB
#    include "kernel/kernel_math.h"
#    include "kernel/kernel_types.h"

#    include "kernel/kernel_globals.h"

#    include "kernel/integrator/kernel_background.h"
#    include "kernel/integrator/kernel_generate_camera_rays.h"
#    include "kernel/integrator/kernel_intersect_closest.h"
#    include "kernel/integrator/kernel_intersect_shadow.h"
#    include "kernel/integrator/kernel_shadow.h"
#    include "kernel/integrator/kernel_subsurface.h"
#    include "kernel/integrator/kernel_surface.h"
#    include "kernel/integrator/kernel_volume.h"

#    include "kernel/kernel_color.h"
#    include "kernel/kernels/cpu/kernel_cpu_image.h"
#    include "kernel/kernel_film.h"
#    include "kernel/kernel_path.h"
#    include "kernel/kernel_path_branched.h"
#    include "kernel/kernel_bake.h"
#else
#  define STUB_ASSERT(arch, name) \
    assert(!(#name " kernel stub for architecture " #arch " was called!"))
#endif   /* KERNEL_STUB */
// clang-format on

CCL_NAMESPACE_BEGIN

/* Path Tracing */

void KERNEL_FUNCTION_FULL_NAME(path_trace)(
    KernelGlobals *kg, float *buffer, int sample, int x, int y, int offset, int stride)
{
#ifdef KERNEL_STUB
  STUB_ASSERT(KERNEL_ARCH, path_trace);
#else
#  ifdef __BRANCHED_PATH__
  if (kernel_data.integrator.branched) {
    kernel_branched_path_trace(kg, buffer, sample, x, y, offset, stride);
  }
  else
#  endif
  {
    kernel_path_trace(kg, buffer, sample, x, y, offset, stride);
  }
#endif /* KERNEL_STUB */
}

/* Film */

void KERNEL_FUNCTION_FULL_NAME(convert_to_byte)(KernelGlobals *kg,
                                                uchar4 *rgba,
                                                float *buffer,
                                                float sample_scale,
                                                int x,
                                                int y,
                                                int offset,
                                                int stride)
{
#ifdef KERNEL_STUB
  STUB_ASSERT(KERNEL_ARCH, convert_to_byte);
#else
  kernel_film_convert_to_byte(kg, rgba, buffer, sample_scale, x, y, offset, stride);
#endif /* KERNEL_STUB */
}

void KERNEL_FUNCTION_FULL_NAME(convert_to_half_float)(KernelGlobals *kg,
                                                      uchar4 *rgba,
                                                      float *buffer,
                                                      float sample_scale,
                                                      int x,
                                                      int y,
                                                      int offset,
                                                      int stride)
{
#ifdef KERNEL_STUB
  STUB_ASSERT(KERNEL_ARCH, convert_to_half_float);
#else
  kernel_film_convert_to_half_float(kg, rgba, buffer, sample_scale, x, y, offset, stride);
#endif /* KERNEL_STUB */
}

/* Bake */

void KERNEL_FUNCTION_FULL_NAME(bake)(
    KernelGlobals *kg, float *buffer, int sample, int x, int y, int offset, int stride)
{
#ifdef KERNEL_STUB
  STUB_ASSERT(KERNEL_ARCH, bake);
#else
#  ifdef __BAKING__
  kernel_bake_evaluate(kg, buffer, sample, x, y, offset, stride);
#  endif
#endif /* KERNEL_STUB */
}

/* Shader Evaluate */

void KERNEL_FUNCTION_FULL_NAME(shader)(KernelGlobals *kg,
                                       uint4 *input,
                                       float4 *output,
                                       int type,
                                       int filter,
                                       int i,
                                       int offset,
                                       int sample)
{
#ifdef KERNEL_STUB
  STUB_ASSERT(KERNEL_ARCH, shader);
#else
  if (type == SHADER_EVAL_DISPLACE) {
    kernel_displace_evaluate(kg, input, output, i);
  }
  else {
    kernel_background_evaluate(kg, input, output, i);
  }
#endif /* KERNEL_STUB */
}

/* ********************************************************************************************* */
/* *                            *** The new split kernel ***                                   * */
/* ********************************************************************************************* */

#ifdef KERNEL_STUB
#  define DEFINE_INTEGRATOR_KERNEL(name) \
    void KERNEL_FUNCTION_FULL_NAME(name)(const KernelGlobals * /*kg*/, \
                                         IntegratorState * /*state*/) \
    { \
      STUB_ASSERT(KERNEL_ARCH, name); \
    }
#else
#  define DEFINE_INTEGRATOR_KERNEL(name) \
    void KERNEL_FUNCTION_FULL_NAME(name)(const KernelGlobals *kg, IntegratorState *state) \
    { \
      kernel_integrate_##name(kg, state); \
    }
#endif

#ifdef KERNEL_STUB
#  define DEFINE_INTEGRATOR_OUTPUT_KERNEL(name) \
    void KERNEL_FUNCTION_FULL_NAME(name)(const KernelGlobals * /*kg*/, \
                                         IntegratorState * /*state*/, \
                                         ccl_global float *render_buffer) \
    { \
      STUB_ASSERT(KERNEL_ARCH, name); \
    }
#else
#  define DEFINE_INTEGRATOR_OUTPUT_KERNEL(name) \
    void KERNEL_FUNCTION_FULL_NAME(name)( \
        const KernelGlobals *kg, IntegratorState *state, ccl_global float *render_buffer) \
    { \
      kernel_integrate_##name(kg, state, render_buffer); \
    }
#endif

DEFINE_INTEGRATOR_OUTPUT_KERNEL(background)
DEFINE_INTEGRATOR_KERNEL(generate_camera_rays)
DEFINE_INTEGRATOR_KERNEL(intersect_closest)
DEFINE_INTEGRATOR_KERNEL(intersect_shadow)
DEFINE_INTEGRATOR_OUTPUT_KERNEL(shadow)
DEFINE_INTEGRATOR_KERNEL(subsurface)
DEFINE_INTEGRATOR_OUTPUT_KERNEL(surface)
DEFINE_INTEGRATOR_OUTPUT_KERNEL(volume)

#undef DEFINE_INTEGRATOR_KERNEL

#undef KERNEL_STUB
#undef STUB_ASSERT
#undef KERNEL_ARCH

CCL_NAMESPACE_END
