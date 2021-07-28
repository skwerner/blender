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

#include "integrator/pass_accessor.h"

#include "render/background.h"
#include "render/buffers.h"
#include "render/film.h"
#include "util/util_logging.h"

// clang-format off
#include "kernel/device/cpu/compat.h"
#include "kernel/kernel_types.h"
// clang-format on

CCL_NAMESPACE_BEGIN

/* --------------------------------------------------------------------
 * Pass input information.
 */

PassAccessor::PassAccessInfo::PassAccessInfo(const Pass &pass,
                                             const Film &film,
                                             const Background &background,
                                             const vector<Pass> &passes)
    : type(pass.type),
      mode(pass.mode),
      offset(Pass::get_offset(passes, pass)),
      use_approximate_shadow_catcher(film.get_use_approximate_shadow_catcher()),
      use_approximate_shadow_catcher_background(use_approximate_shadow_catcher &&
                                                !background.get_transparent())
{
}

/* --------------------------------------------------------------------
 * Pass destination.
 */

PassAccessor::Destination::Destination(float *pixels, int num_components)
    : pixels(pixels), num_components(num_components)
{
}

PassAccessor::Destination::Destination(const PassType pass_type, half4 *pixels)
    : Destination(pass_type)
{
  pixels_half_rgba = pixels;
}

PassAccessor::Destination::Destination(const PassType pass_type)
{
  const PassInfo pass_info = Pass::get_info(pass_type);
  num_components = pass_info.num_components;
}

/* --------------------------------------------------------------------
 * Pass source.
 */

PassAccessor::Source::Source(const float *pixels, int num_components)
    : pixels(pixels), num_components(num_components)
{
}

/* --------------------------------------------------------------------
 * Pass accessor.
 */

PassAccessor::PassAccessor(const PassAccessInfo &pass_access_info, float exposure, int num_samples)
    : pass_access_info_(pass_access_info), exposure_(exposure), num_samples_(num_samples)
{
}

bool PassAccessor::get_render_tile_pixels(const RenderBuffers *render_buffers,
                                          const Destination &destination) const
{
  if (render_buffers == nullptr || render_buffers->buffer.data() == nullptr) {
    return false;
  }

  return get_render_tile_pixels(render_buffers, render_buffers->params, destination);
}

static void pad_pixels(const BufferParams &buffer_params,
                       const PassAccessor::Destination &destination,
                       const int src_num_components)
{
  /* When requesting a single channel pass as RGBA, or RGB pass as RGBA,
   * fill in the additional components for convenience. */
  const int dest_num_components = destination.num_components;

  if (src_num_components >= dest_num_components) {
    return;
  }

  const size_t size = buffer_params.width * buffer_params.height;
  if (destination.pixels) {
    float *pixel = destination.pixels;

    for (size_t i = 0; i < size; i++, pixel += dest_num_components) {
      if (dest_num_components >= 3 && src_num_components == 1) {
        pixel[1] = pixel[0];
        pixel[2] = pixel[0];
      }
      if (dest_num_components >= 4) {
        pixel[3] = 1.0f;
      }
    }
  }

  if (destination.pixels_half_rgba) {
    const half one = float_to_half(1.0f);
    half4 *pixel = destination.pixels_half_rgba;

    for (size_t i = 0; i < size; i++, pixel++) {
      if (dest_num_components >= 3 && src_num_components == 1) {
        pixel[0].y = pixel[0].x;
        pixel[0].z = pixel[0].x;
      }
      if (dest_num_components >= 4) {
        pixel[0].w = one;
      }
    }
  }
}

bool PassAccessor::get_render_tile_pixels(const RenderBuffers *render_buffers,
                                          const BufferParams &buffer_params,
                                          const Destination &destination) const
{
  if (render_buffers == nullptr || render_buffers->buffer.data() == nullptr) {
    return false;
  }

  const PassType type = pass_access_info_.type;
  const PassMode mode = pass_access_info_.mode;
  const PassInfo pass_info = Pass::get_info(type);

  DCHECK_LE(pass_info.num_components, destination.num_components)
      << "Number of components mismatch for " << pass_type_as_string(type);

  if (pass_info.num_components == 1) {
    if (mode == PassMode::DENOISED) {
      /* Denoised passes store their final pixels, no need in special calculation. */
      get_pass_float(render_buffers, buffer_params, destination);
    }
    else if (type == PASS_RENDER_TIME) {
      /* TODO(sergey): Needs implementation. */
    }
    else if (type == PASS_DEPTH) {
      get_pass_depth(render_buffers, buffer_params, destination);
    }
    else if (type == PASS_MIST) {
      get_pass_mist(render_buffers, buffer_params, destination);
    }
    else if (type == PASS_SAMPLE_COUNT) {
      get_pass_sample_count(render_buffers, buffer_params, destination);
    }
    else {
      get_pass_float(render_buffers, buffer_params, destination);
    }
  }
  else if (pass_info.num_components == 3) {
    if (mode == PassMode::DENOISED) {
      /* Denoised passes store their final pixels, no need in special calculation. */
      get_pass_float3(render_buffers, buffer_params, destination);
    }
    else if (pass_info.divide_type != PASS_NONE) {
      /* RGB lighting passes that need to divide out color */
      get_pass_divide_even_color(render_buffers, buffer_params, destination);
    }
    else if (type == PASS_SHADOW_CATCHER) {
      get_pass_shadow_catcher(render_buffers, buffer_params, destination);
    }
    else {
      /* RGB/vector */
      get_pass_float3(render_buffers, buffer_params, destination);
    }
  }
  else if (pass_info.num_components == 4) {
    if (type == PASS_SHADOW_CATCHER_MATTE && pass_access_info_.use_approximate_shadow_catcher) {
      /* Denoised matte with shadow needs to do calculation (will use denoised shadow catcher pass
       * to approximate shadow with). */
      get_pass_shadow_catcher_matte_with_shadow(render_buffers, buffer_params, destination);
    }
    else if (mode == PassMode::DENOISED) {
      /* Denoised passes store their final pixels, no need in special calculation. */
      if (type == PASS_COMBINED || type == PASS_SHADOW_CATCHER ||
          type == PASS_SHADOW_CATCHER_MATTE) {
        get_pass_combined(render_buffers, buffer_params, destination);
      }
      else {
        get_pass_float4(render_buffers, buffer_params, destination);
      }
    }
    else if (type == PASS_MOTION) {
      get_pass_motion(render_buffers, buffer_params, destination);
    }
    else if (type == PASS_CRYPTOMATTE) {
      get_pass_cryptomatte(render_buffers, buffer_params, destination);
    }
    else if (type == PASS_COMBINED || type == PASS_SHADOW_CATCHER_MATTE) {
      get_pass_combined(render_buffers, buffer_params, destination);
    }
    else {
      get_pass_float4(render_buffers, buffer_params, destination);
    }
  }

  pad_pixels(buffer_params, destination, pass_info.num_components);

  return true;
}

void PassAccessor::init_kernel_film_convert(KernelFilmConvert *kfilm_convert,
                                            const BufferParams &buffer_params,
                                            const Destination &destination) const
{
  const PassMode mode = pass_access_info_.mode;
  const PassInfo &pass_info = Pass::get_info(pass_access_info_.type);

  kfilm_convert->pass_offset = pass_access_info_.offset;
  kfilm_convert->pass_stride = buffer_params.pass_stride;

  kfilm_convert->pass_use_exposure = pass_info.use_exposure;
  kfilm_convert->pass_use_filter = pass_info.use_filter;

  /* TODO(sergey): Some of the passes needs to become denoised when denoised pass is accessed. */

  kfilm_convert->pass_divide = buffer_params.get_pass_offset(pass_info.divide_type);

  kfilm_convert->pass_combined = buffer_params.get_pass_offset(PASS_COMBINED);
  kfilm_convert->pass_sample_count = buffer_params.get_pass_offset(PASS_SAMPLE_COUNT);
  kfilm_convert->pass_adaptive_aux_buffer = buffer_params.get_pass_offset(
      PASS_ADAPTIVE_AUX_BUFFER);
  kfilm_convert->pass_motion_weight = buffer_params.get_pass_offset(PASS_MOTION_WEIGHT);
  kfilm_convert->pass_shadow_catcher = buffer_params.get_pass_offset(PASS_SHADOW_CATCHER, mode);
  kfilm_convert->pass_shadow_catcher_sample_count = buffer_params.get_pass_offset(
      PASS_SHADOW_CATCHER_SAMPLE_COUNT);
  kfilm_convert->pass_shadow_catcher_matte = buffer_params.get_pass_offset(
      PASS_SHADOW_CATCHER_MATTE, mode);

  /* Background is not denoised, so always use noisy pass. */
  kfilm_convert->pass_background = buffer_params.get_pass_offset(PASS_BACKGROUND);

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
  kfilm_convert->use_approximate_shadow_catcher_background =
      pass_access_info_.use_approximate_shadow_catcher_background;
  kfilm_convert->show_active_pixels = pass_access_info_.show_active_pixels;

  kfilm_convert->num_components = destination.num_components;
  kfilm_convert->pixel_stride = destination.pixel_stride ? destination.pixel_stride :
                                                           destination.num_components;

  kfilm_convert->is_denoised = (mode == PassMode::DENOISED);
}

bool PassAccessor::set_render_tile_pixels(RenderBuffers *render_buffers, const Source &source)
{
  if (render_buffers == nullptr || render_buffers->buffer.data() == nullptr) {
    return false;
  }

  const PassType type = pass_access_info_.type;
  const PassInfo pass_info = Pass::get_info(type);

  const BufferParams &buffer_params = render_buffers->params;

  float *buffer_data = render_buffers->buffer.data();
  const int size = buffer_params.width * buffer_params.height;

  const int out_stride = buffer_params.pass_stride;
  const int in_stride = source.num_components;
  const int num_components_to_copy = min(source.num_components, pass_info.num_components);

  float *out = buffer_data + pass_access_info_.offset;
  const float *in = source.pixels + source.offset * in_stride;

  for (int i = 0; i < size; i++, out += out_stride, in += in_stride) {
    memcpy(out, in, sizeof(float) * num_components_to_copy);
  }

  return true;
}

CCL_NAMESPACE_END
