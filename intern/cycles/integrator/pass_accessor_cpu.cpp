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

#include "integrator/pass_accessor_cpu.h"

#include "util/util_logging.h"
#include "util/util_tbb.h"

// clang-format off
#include "kernel/device/cpu/compat.h"
#include "kernel/device/cpu/globals.h"
#include "kernel/kernel_types.h"
#include "kernel/kernel_film.h"
// clang-format on

CCL_NAMESPACE_BEGIN

/* --------------------------------------------------------------------
 * Kernel processing.
 */

void PassAccessorCPU::init_kernel_film_convert(KernelFilmConvert *kfilm_convert,
                                               const RenderBuffers *render_buffers) const
{
  const BufferParams &params = render_buffers->params;
  const PassInfo &pass_info = Pass::get_info(pass_access_info_.type);

  kfilm_convert->pass_offset = pass_access_info_.offset;

  kfilm_convert->pass_use_exposure = pass_info.use_exposure;
  kfilm_convert->pass_use_filter = pass_info.use_filter;

  kfilm_convert->pass_divide = params.get_pass_offset(pass_info.divide_type);

  kfilm_convert->pass_combined = params.get_pass_offset(PASS_COMBINED);
  kfilm_convert->pass_sample_count = params.get_pass_offset(PASS_SAMPLE_COUNT);
  kfilm_convert->pass_motion_weight = params.get_pass_offset(PASS_MOTION_WEIGHT);
  kfilm_convert->pass_shadow_catcher = params.get_pass_offset(PASS_SHADOW_CATCHER);
  kfilm_convert->pass_shadow_catcher_matte = params.get_pass_offset(PASS_SHADOW_CATCHER_MATTE);

  if (pass_info.use_filter) {
    kfilm_convert->scale = 1.0f / num_samples_;
  }
  else {
    kfilm_convert->scale = 1.0f;
  }

  if (pass_info.use_exposure) {
    kfilm_convert->exposure = exposure_;
  }
  else {
    kfilm_convert->exposure = 1.0f;
  }

  kfilm_convert->scale_exposure = kfilm_convert->scale * kfilm_convert->exposure;

  kfilm_convert->use_approximate_shadow_catcher = pass_access_info_.use_approximate_shadow_catcher;
}

template<typename Processor>
inline void PassAccessorCPU::run_get_pass_kernel_processor(const RenderBuffers *render_buffers,
                                                           float *pixels,
                                                           const Processor &processor) const
{
  KernelFilmConvert kfilm_convert;
  init_kernel_film_convert(&kfilm_convert, render_buffers);

  const BufferParams &params = render_buffers->params;

  const float *buffer_data = render_buffers->buffer.data();

  tbb::parallel_for(0, params.height, [&](int y) {
    int64_t pixel_index = y * params.width;
    for (int x = 0; x < params.width; ++x, ++pixel_index) {
      const int64_t input_pixel_offset = pixel_index * params.pass_stride;
      const float *buffer = buffer_data + input_pixel_offset;
      float *pixel = pixels + pixel_index * num_components_;

      processor(&kfilm_convert, buffer, pixel);
    }
  });
}

/* --------------------------------------------------------------------
 * Pass accessors.
 */

#define DEFINE_PASS_ACCESSOR(pass) \
  void PassAccessorCPU::get_pass_##pass(const RenderBuffers *render_buffers, float *pixels) const \
  { \
    run_get_pass_kernel_processor(render_buffers, pixels, film_get_pass_pixel_##pass); \
  }

/* Float (scalar) passes. */
DEFINE_PASS_ACCESSOR(depth)
DEFINE_PASS_ACCESSOR(mist)
DEFINE_PASS_ACCESSOR(sample_count)
DEFINE_PASS_ACCESSOR(float)

/* Float3 passes. */
DEFINE_PASS_ACCESSOR(shadow3)
DEFINE_PASS_ACCESSOR(divide_even_color)
DEFINE_PASS_ACCESSOR(float3)

/* Float4 passes. */
DEFINE_PASS_ACCESSOR(shadow4)
DEFINE_PASS_ACCESSOR(motion)
DEFINE_PASS_ACCESSOR(cryptomatte)
DEFINE_PASS_ACCESSOR(denoising_color)
DEFINE_PASS_ACCESSOR(shadow_catcher)
DEFINE_PASS_ACCESSOR(shadow_catcher_matte_with_shadow)
DEFINE_PASS_ACCESSOR(float4)

#undef DEFINE_PASS_ACCESSOR

CCL_NAMESPACE_END
