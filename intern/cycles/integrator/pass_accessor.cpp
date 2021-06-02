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

#include "render/buffers.h"
#include "util/util_logging.h"

CCL_NAMESPACE_BEGIN

/* --------------------------------------------------------------------
 * Pass input information.
 */

PassAccessor::PassAccessInfo::PassAccessInfo(const Pass &pass,
                                             const Film &film,
                                             const vector<Pass> &passes)
    : type(pass.type),
      offset(Pass::get_offset(passes, pass)),
      use_approximate_shadow_catcher(film.get_use_approximate_shadow_catcher())
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
    : pixels_half_rgba(pixels)
{
  const PassInfo pass_info = Pass::get_info(pass_type);

  if (pass_info.divide_type != PASS_NONE) {
    /* Divide is used for colors, which has 3 destination components.
     * The passes which use division are stored as aligned float4 internally, and there is no
     * implementation of divide_even_color for float4. So we force it here.
     * The rest of the aligned float3 passes should be fine, because they have float4
     * implementation. */
    num_components = 3;
  }
  else {
    num_components = pass_info.num_components;
  }
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

bool PassAccessor::get_render_tile_pixels(const RenderBuffers *render_buffers,
                                          const BufferParams &buffer_params,
                                          const Destination &destination) const
{
  if (render_buffers == nullptr || render_buffers->buffer.data() == nullptr) {
    return false;
  }

  const PassType type = pass_access_info_.type;
  const PassInfo pass_info = Pass::get_info(type);

  if (destination.num_components == 1) {
    DCHECK_EQ(pass_info.num_components, destination.num_components)
        << "Number of components mismatch for pass type " << pass_info.type;

    /* Scalar */
    if (type == PASS_RENDER_TIME) {
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
  else if (destination.num_components == 3) {
    if (pass_info.is_unaligned) {
      DCHECK_EQ(pass_info.num_components, 3)
          << "Number of components mismatch for pass type " << pass_info.type;
    }
    else {
      DCHECK_EQ(pass_info.num_components, 4)
          << "Number of components mismatch for pass type " << pass_info.type;
    }

    /* RGBA */
    if (type == PASS_SHADOW) {
      get_pass_shadow3(render_buffers, buffer_params, destination);
    }
    else if (pass_info.divide_type != PASS_NONE) {
      /* RGB lighting passes that need to divide out color */
      get_pass_divide_even_color(render_buffers, buffer_params, destination);
    }
    else {
      /* RGB/vector */
      get_pass_float3(render_buffers, buffer_params, destination);
    }
  }
  else if (destination.num_components == 4) {
    DCHECK_EQ(pass_info.num_components, 4)
        << "Number of components mismatch for pass type " << pass_info.type;

    /* RGBA */
    if (type == PASS_SHADOW) {
      get_pass_shadow4(render_buffers, buffer_params, destination);
    }
    else if (type == PASS_MOTION) {
      get_pass_motion(render_buffers, buffer_params, destination);
    }
    else if (type == PASS_CRYPTOMATTE) {
      get_pass_cryptomatte(render_buffers, buffer_params, destination);
    }
    else if (type == PASS_DENOISING_COLOR) {
      get_pass_denoising_color(render_buffers, buffer_params, destination);
    }
    else if (type == PASS_SHADOW_CATCHER) {
      get_pass_shadow_catcher(render_buffers, buffer_params, destination);
    }
    else if (type == PASS_SHADOW_CATCHER_MATTE &&
             pass_access_info_.use_approximate_shadow_catcher) {
      get_pass_shadow_catcher_matte_with_shadow(render_buffers, buffer_params, destination);
    }
    else {
      get_pass_float4(render_buffers, buffer_params, destination);
    }
  }

  return true;
}

#if 0
/* TODO(sergey): Need to be converted to a kernel-based processing if it will be used. */
bool PassAccessor::set_pass_rect(PassType type, int components, float *pixels, int samples)
{
  if (buffer.data() == NULL) {
    return false;
  }

  int pass_offset = 0;

  for (size_t j = 0; j < params.passes.size(); j++) {
    Pass &pass = params.passes[j];

    if (pass.type != type) {
      pass_offset += pass.components;
      continue;
    }

    float *out = buffer.data() + pass_offset;
    const int pass_stride = params.passes_size;
    const int size = params.width * params.height;

    DCHECK_EQ(pass.components, components)
        << "Number of components mismatch for pass " << pass.name;

    for (int i = 0; i < size; i++, out += pass_stride, pixels += components) {
      if (pass.filter) {
        /* Scale by the number of samples, inverse of what we do in get_render_tile_pixels.
         * A better solution would be to remove the need for set_pass_rect entirely,
         * and change baking to bake multiple objects in a tile at once. */
        for (int j = 0; j < components; j++) {
          out[j] = pixels[j] * samples;
        }
      }
      else {
        /* For non-filtered passes just straight copy, these may contain non-float data. */
        memcpy(out, pixels, sizeof(float) * components);
      }
    }

    return true;
  }

  return false;
}
#endif

CCL_NAMESPACE_END
