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

#include "integrator/pass_accessor.h"
#include "render/buffers.h"

CCL_NAMESPACE_BEGIN

/* Pass accessor implementation for CPU side. */
class PassAccessorCPU : public PassAccessor {
 public:
  using PassAccessor::PassAccessor;

 protected:
  template<typename Processor>
  inline void run_get_pass_processor(const RenderBuffers *render_buffers,
                                     float *pixels,
                                     const Processor &processor) const
  {
    const BufferParams &params = render_buffers->params;

    const float *buffer_data = render_buffers->buffer.data();
    const float *input_pixels = buffer_data + pass_access_info_.offset;

    const int pass_stride = params.pass_stride;
    const int64_t num_pixels = int64_t(params.width) * params.height;

    for (int64_t i = 0; i < num_pixels; i++) {
      const int64_t input_pixel_offset = i * pass_stride;
      const float *in = input_pixels + input_pixel_offset;
      float *pixel = pixels + i * num_components_;

      processor(i, in, pixel);
    }
  }

  template<typename Processor>
  inline void run_get_pass_processor(const RenderBuffers *render_buffers,
                                     const int pass_offset_a,
                                     float *pixels,
                                     const Processor &processor) const
  {
    const BufferParams &params = render_buffers->params;

    const float *buffer_data = render_buffers->buffer.data();
    const float *input_pixels = buffer_data + pass_access_info_.offset;
    const float *input_pixels_pass_a = buffer_data + pass_offset_a;

    const int pass_stride = params.pass_stride;
    const int64_t num_pixels = int64_t(params.width) * params.height;

    for (int64_t i = 0; i < num_pixels; i++) {
      const int64_t input_pixel_offset = i * pass_stride;
      const float *in = input_pixels + input_pixel_offset;
      const float *in_pass_a = input_pixels_pass_a + input_pixel_offset;
      float *pixel = pixels + i * num_components_;

      processor(i, in, in_pass_a, pixel);
    }
  }

  template<typename Processor>
  inline void run_get_pass_processor(const RenderBuffers *render_buffers,
                                     const int pass_offset_a,
                                     const int pass_offset_b,
                                     float *pixels,
                                     const Processor &processor) const
  {
    const BufferParams &params = render_buffers->params;

    const float *buffer_data = render_buffers->buffer.data();
    const float *input_pixels = buffer_data + pass_access_info_.offset;
    const float *input_pixels_pass_a = buffer_data + pass_offset_a;
    const float *input_pixels_pass_b = buffer_data + pass_offset_b;

    const int pass_stride = params.pass_stride;
    const int64_t num_pixels = int64_t(params.width) * params.height;

    for (int64_t i = 0; i < num_pixels; i++) {
      const int64_t input_pixel_offset = i * pass_stride;
      const float *in = input_pixels + input_pixel_offset;
      const float *in_pass_a = input_pixels_pass_a + input_pixel_offset;
      const float *in_pass_b = input_pixels_pass_b + input_pixel_offset;
      float *pixel = pixels + i * num_components_;

      processor(i, in, in_pass_a, in_pass_b, pixel);
    }
  }

#define DECLARE_PASS_ACCESSOR(pass) \
  virtual void get_pass_##pass(const RenderBuffers *render_buffers, float *pixels) const override;

  /* Float (scalar) passes. */
  DECLARE_PASS_ACCESSOR(depth)
  DECLARE_PASS_ACCESSOR(mist)
  DECLARE_PASS_ACCESSOR(sample_count)
  DECLARE_PASS_ACCESSOR(float)

  /* Float3 passes. */
  DECLARE_PASS_ACCESSOR(shadow3)
  DECLARE_PASS_ACCESSOR(divide_even_color)
  DECLARE_PASS_ACCESSOR(float3)

  /* Float4 passes. */
  DECLARE_PASS_ACCESSOR(shadow4)
  DECLARE_PASS_ACCESSOR(motion)
  DECLARE_PASS_ACCESSOR(cryptomatte)
  DECLARE_PASS_ACCESSOR(denoising_color)
  DECLARE_PASS_ACCESSOR(shadow_catcher)
  DECLARE_PASS_ACCESSOR(shadow_catcher_matte_with_shadow)
  DECLARE_PASS_ACCESSOR(float4)

#undef DECLARE_PASS_ACCESSOR
};

CCL_NAMESPACE_END
