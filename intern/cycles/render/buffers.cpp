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

#include <stdlib.h>

#include "device/device.h"
#include "render/buffers.h"

#include "util/util_foreach.h"
#include "util/util_hash.h"
#include "util/util_math.h"
#include "util/util_opengl.h"
#include "util/util_time.h"
#include "util/util_types.h"

CCL_NAMESPACE_BEGIN

/* Buffer Params */

BufferParams::BufferParams()
{
  width = 0;
  height = 0;

  full_x = 0;
  full_y = 0;
  full_width = 0;
  full_height = 0;
}

void BufferParams::update_passes(vector<Pass> &passes)
{
  update_offset_stride();

  pass_sample_count = PASS_UNUSED;

  pass_denoising_color = PASS_UNUSED;
  pass_denoising_normal = PASS_UNUSED;
  pass_denoising_albedo = PASS_UNUSED;

  pass_stride = 0;
  for (const Pass &pass : passes) {
    switch (pass.type) {
      case PASS_SAMPLE_COUNT:
        pass_sample_count = pass_stride;
        break;

      case PASS_DENOISING_COLOR:
        pass_denoising_color = pass_stride;
        break;
      case PASS_DENOISING_NORMAL:
        pass_denoising_normal = pass_stride;
        break;
      case PASS_DENOISING_ALBEDO:
        pass_denoising_albedo = pass_stride;
        break;

      default:
        break;
    }

    pass_stride += pass.components;
  }

  pass_stride = align_up(pass_stride, 4);
}

void BufferParams::update_offset_stride()
{
  offset = -(full_x + full_y * width);
  stride = width;
}

bool BufferParams::modified(const BufferParams &params) const
{
  return !(width == params.width && height == params.height && full_x == params.full_x &&
           full_y == params.full_y && full_width == params.full_width &&
           full_height == params.full_height && offset == params.offset &&
           stride == params.stride && pass_stride == params.pass_stride &&
           pass_sample_count == params.pass_sample_count &&
           pass_denoising_color == params.pass_denoising_color &&
           pass_denoising_normal == params.pass_denoising_normal &&
           pass_denoising_albedo == params.pass_denoising_albedo);
}

/* Render Buffer Task */

RenderTile::RenderTile()
{
  x = 0;
  y = 0;
  w = 0;
  h = 0;

  sample = 0;
  start_sample = 0;
  num_samples = 0;
  resolution = 0;

  offset = 0;
  stride = 0;

  buffer = 0;

  buffers = NULL;
  stealing_state = NO_STEALING;
}

/* Render Buffers */

RenderBuffers::RenderBuffers(Device *device)
    : buffer(device, "RenderBuffers", MEM_READ_WRITE),
      map_neighbor_copied(false),
      render_time(0.0f)
{
}

RenderBuffers::~RenderBuffers()
{
  buffer.free();
}

void RenderBuffers::reset(const BufferParams &params_)
{
  DCHECK(params_.pass_stride != -1);

  params = params_;

  /* re-allocate buffer */
  buffer.alloc(params.width * params.pass_stride, params.height);
  buffer.zero_to_device();
}

void RenderBuffers::zero()
{
  buffer.zero_to_device();
}

bool RenderBuffers::copy_from_device()
{
  DCHECK(params.pass_stride != -1);

  if (!buffer.device_pointer)
    return false;

  buffer.copy_from_device(0, params.width * params.pass_stride, params.height);

  return true;
}

namespace {

/* Helper class which takes care of calculating sample scale and exposure scale for render passes,
 * taking adaptive sampling into account. */
class Scaler {
 public:
  Scaler(const RenderBuffers *render_buffers,
         const Pass &pass,
         const float *pass_buffer,
         const int sample,
         const float exposure)
      : pass_(pass),
        pass_stride_(render_buffers->params.pass_stride),
        sample_inv_(1.0f / sample),
        exposure_(exposure),
        sample_count_pass_(get_sample_count_pass(render_buffers))
  {
    /* Special trick to only scale the samples count pass with the sample scale. Otherwise the pass
     * becomes a uniform 1.0. */
    if (sample_count_pass_ == pass_buffer) {
      sample_count_pass_ = nullptr;
    }

    /* Pre-calculate values when adaptive sampling is not used. */
    if (!sample_count_pass_) {
      scale_ = pass.filter ? sample_inv_ : 1.0f;
      scale_exposure_ = pass.exposure ? scale_ * exposure_ : scale_;
    }
  }

  inline float scale(const int pixel_index) const
  {
    if (!sample_count_pass_) {
      return scale_;
    }

    return (pass_.filter) ? 1.0f / (sample_count_pass_[pixel_index * pass_stride_]) : 1.0f;
  }

  inline float scale_exposure(const int pixel_index) const
  {
    if (!sample_count_pass_) {
      return scale_exposure_;
    }

    float scale, scale_exposure;
    scale_and_scale_exposure(pixel_index, scale, scale_exposure);

    return scale_exposure;
  }

  inline void scale_and_scale_exposure(int pixel_index, float &scale, float &scale_exposure) const
  {
    if (!sample_count_pass_) {
      scale = scale_;
      scale_exposure = scale_exposure_;
      return;
    }

    scale = this->scale(pixel_index);
    scale_exposure = (pass_.exposure) ? scale * exposure_ : scale;
  }

 protected:
  const float *get_sample_count_pass(const RenderBuffers *render_buffers)
  {
    if (render_buffers->params.pass_sample_count == -1) {
      return nullptr;
    }

    return render_buffers->buffer.data() + render_buffers->params.pass_sample_count;
  }

  const Pass &pass_;
  const int pass_stride_;

  const float sample_inv_ = 1.0f;
  const float exposure_ = 1.0f;

  const float *sample_count_pass_ = nullptr;

  float scale_ = 0.0f;
  float scale_exposure_ = 0.0f;
};

int find_pass_offset_by_type(const vector<Pass> &passes, const PassType type)
{
  int pass_offset = 0;
  for (const Pass &color_pass : passes) {
    if (color_pass.type == type) {
      return pass_offset;
    }
    pass_offset += color_pass.components;
  }
  return PASS_UNUSED;
}

} /* namespace */

bool RenderBuffers::get_pass_rect(const vector<Pass> &passes,
                                  const string &name,
                                  float exposure,
                                  int sample,
                                  int components,
                                  float *pixels)
{
  if (buffer.data() == NULL) {
    return false;
  }

  int pass_offset = 0;
  for (const Pass &pass : passes) {
    /* Pass is identified by both type and name, multiple of the same type
     * may exist with a different name. */
    if (pass.name != name) {
      pass_offset += pass.components;
      continue;
    }

    const PassType type = pass.type;

    const float *in = buffer.data() + pass_offset;
    const int pass_stride = params.pass_stride;

    const Scaler scaler(this, pass, in, sample, exposure);

    const int size = params.width * params.height;

    if (components == 1 && type == PASS_RENDER_TIME) {
      /* Render time is not stored by kernel, but measured per tile. */
      const float val = (float)(1000.0 * render_time / (params.width * params.height * sample));
      for (int i = 0; i < size; i++, pixels++) {
        pixels[0] = val;
      }
    }
    else if (components == 1) {
      DCHECK_EQ(pass.components, components)
          << "Number of components mismatch for pass " << pass.name;

      /* Scalar */
      if (type == PASS_DEPTH) {
        for (int i = 0; i < size; i++, in += pass_stride, pixels++) {
          const float f = *in;
          pixels[0] = (f == 0.0f) ? 1e10f : f * scaler.scale_exposure(i);
        }
      }
      else if (type == PASS_MIST) {
        for (int i = 0; i < size; i++, in += pass_stride, pixels++) {
          const float f = *in;
          /* Note that we accumulate 1 - mist in the kernel to avoid having to
           * track the mist values in the integrator state. */
          pixels[0] = saturate(1.0f - f * scaler.scale_exposure(i));
        }
      }
#ifdef WITH_CYCLES_DEBUG
      else if (type == PASS_BVH_TRAVERSED_NODES || type == PASS_BVH_TRAVERSED_INSTANCES ||
               type == PASS_BVH_INTERSECTIONS || type == PASS_RAY_BOUNCES) {
        for (int i = 0; i < size; i++, in += pass_stride, pixels++) {
          const float f = *in;
          pixels[0] = f * scaler.scale_exposure(i);
        }
      }
#endif
      else {
        for (int i = 0; i < size; i++, in += pass_stride, pixels++) {
          const float f = *in;
          pixels[0] = f * scaler.scale_exposure(i);
        }
      }
    }
    else if (components == 3) {
      if (pass.is_unaligned) {
        DCHECK_EQ(pass.components, 3) << "Number of components mismatch for pass " << pass.name;
      }
      else {
        DCHECK_EQ(pass.components, 4) << "Number of components mismatch for pass " << pass.name;
      }

      /* RGBA */
      if (type == PASS_SHADOW) {
        for (int i = 0; i < size; i++, in += pass_stride, pixels += 3) {
          const float weight = in[3];
          const float weight_inv = (weight > 0.0f) ? 1.0f / weight : 1.0f;

          const float3 shadow = make_float3(in[0], in[1], in[2]) * weight_inv;

          pixels[0] = shadow.x;
          pixels[1] = shadow.y;
          pixels[2] = shadow.z;
        }
      }
      else if (pass.divide_type != PASS_NONE) {
        /* RGB lighting passes that need to divide out color */
        const int pass_divide = find_pass_offset_by_type(passes, pass.divide_type);
        DCHECK_NE(pass_divide, PASS_UNUSED);

        const float *in_divide = buffer.data() + pass_divide;

        for (int i = 0; i < size; i++, in += pass_stride, in_divide += pass_stride, pixels += 3) {
          const float3 f = make_float3(in[0], in[1], in[2]);
          const float3 f_divide = make_float3(in_divide[0], in_divide[1], in_divide[2]);
          const float3 f_divided = safe_divide_even_color(f * exposure, f_divide);

          pixels[0] = f_divided.x;
          pixels[1] = f_divided.y;
          pixels[2] = f_divided.z;
        }
      }
      else {
        /* RGB/vector */
        for (int i = 0; i < size; i++, in += pass_stride, pixels += 3) {
          const float scale_exposure = scaler.scale_exposure(i);
          const float3 f = make_float3(in[0], in[1], in[2]) * scale_exposure;

          pixels[0] = f.x;
          pixels[1] = f.y;
          pixels[2] = f.z;
        }
      }
    }
    else if (components == 4) {
      DCHECK_EQ(pass.components, components)
          << "Number of components mismatch for pass " << pass.name;

      /* RGBA */
      if (type == PASS_SHADOW) {
        for (int i = 0; i < size; i++, in += pass_stride, pixels += 4) {
          const float weight = in[3];
          const float weight_inv = (weight > 0.0f) ? 1.0f / weight : 1.0f;

          const float3 shadow = make_float3(in[0], in[1], in[2]) * weight_inv;

          pixels[0] = shadow.x;
          pixels[1] = shadow.y;
          pixels[2] = shadow.z;
          pixels[3] = 1.0f;
        }
      }
      else if (type == PASS_MOTION) {
        /* need to normalize by number of samples accumulated for motion */
        const int pass_motion_weight = find_pass_offset_by_type(passes, PASS_MOTION_WEIGHT);
        DCHECK_NE(pass_motion_weight, PASS_UNUSED);

        const float *in_weight = buffer.data() + pass_motion_weight;

        for (int i = 0; i < size; i++, in += pass_stride, in_weight += pass_stride, pixels += 4) {
          const float weight = in_weight[0];
          const float weight_inv = (weight > 0.0f) ? 1.0f / weight : 0.0f;

          const float4 motion = make_float4(in[0], in[1], in[2], in[3]) * weight_inv;

          pixels[0] = motion.x;
          pixels[1] = motion.y;
          pixels[2] = motion.z;
          pixels[3] = motion.w;
        }
      }
      else if (type == PASS_CRYPTOMATTE) {
        for (int i = 0; i < size; i++, in += pass_stride, pixels += 4) {
          const float scale = scaler.scale(i);

          const float4 f = make_float4(in[0], in[1], in[2], in[3]);
          /* x and z contain integer IDs, don't rescale them.
             y and w contain matte weights, they get scaled. */
          pixels[0] = f.x;
          pixels[1] = f.y * scale;
          pixels[2] = f.z;
          pixels[3] = f.w * scale;
        }
      }
      else if (type == PASS_DENOISING_COLOR) {
        const int pass_combined = find_pass_offset_by_type(passes, PASS_COMBINED);
        DCHECK_NE(pass_combined, PASS_UNUSED);

        const float *in_combined = buffer.data() + pass_combined;

        /* Special code which converts noisy image pass from RGB to RGBA using alpha from the
         * combined pass. */
        for (int i = 0; i < size;
             i++, in += pass_stride, in_combined += pass_stride, pixels += 4) {
          float scale, scale_exposure;
          scaler.scale_and_scale_exposure(i, scale, scale_exposure);

          const float3 color = make_float3(in[0], in[1], in[2]) * scale_exposure;
          const float transparency = in_combined[3] * scale;

          pixels[0] = color.x;
          pixels[1] = color.y;
          pixels[2] = color.z;

          pixels[3] = saturate(1.0f - transparency);
        }
      }
      else {
        for (int i = 0; i < size; i++, in += pass_stride, pixels += 4) {
          float scale, scale_exposure;
          scaler.scale_and_scale_exposure(i, scale, scale_exposure);

          /* Note that 3rd channel contains transparency = 1 - alpha at this point. */
          const float3 color = make_float3(in[0], in[1], in[2]) * scale_exposure;
          const float transparency = in[3] * scale;

          pixels[0] = color.x;
          pixels[1] = color.y;
          pixels[2] = color.z;

          /* Clamp since alpha might end up outside of 0..1 due to Russian roulette. */
          pixels[3] = saturate(1.0f - transparency);
        }
      }
    }

    return true;
  }

  return false;
}

#if 0
bool RenderBuffers::set_pass_rect(PassType type, int components, float *pixels, int samples)
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
        /* Scale by the number of samples, inverse of what we do in get_pass_rect.
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
