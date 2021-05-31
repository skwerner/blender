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

CCL_NAMESPACE_BEGIN

namespace {

/* Helper class which takes care of calculating sample scale and exposure scale for render passes,
 * taking adaptive sampling into account. */
class Scaler {
 public:
  Scaler(const RenderBuffers *render_buffers,
         const PassType &pass_type,
         const int num_samples,
         const float exposure)
      : pass_info_(Pass::get_info(pass_type)),
        pass_stride_(render_buffers->params.pass_stride),
        num_samples_inv_(1.0f / num_samples),
        exposure_(exposure),
        sample_count_pass_(get_sample_count_pass(render_buffers))
  {
    /* Pre-calculate values when adaptive sampling is not used. */
    if (!sample_count_pass_) {
      scale_ = pass_info_.use_filter ? num_samples_inv_ : 1.0f;
      scale_exposure_ = pass_info_.use_exposure ? scale_ * exposure_ : scale_;
    }
  }

  inline float scale(const int pixel_index) const
  {
    if (!sample_count_pass_) {
      return scale_;
    }

    return (pass_info_.use_filter) ? 1.0f / (sample_count_pass_[pixel_index * pass_stride_]) :
                                     1.0f;
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
    scale_exposure = (pass_info_.use_exposure) ? scale * exposure_ : scale;
  }

 protected:
  const uint *get_sample_count_pass(const RenderBuffers *render_buffers)
  {
    const int pass_sample_count = render_buffers->params.get_pass_offset(PASS_SAMPLE_COUNT);
    if (pass_sample_count == PASS_UNUSED) {
      return nullptr;
    }

    return reinterpret_cast<const uint *>(render_buffers->buffer.data()) + pass_sample_count;
  }

  const PassInfo pass_info_;
  const int pass_stride_;

  const float num_samples_inv_ = 1.0f;
  const float exposure_ = 1.0f;

  const uint *sample_count_pass_ = nullptr;

  float scale_ = 0.0f;
  float scale_exposure_ = 0.0f;
};

} /* namespace */

static float4 shadow_catcher_calc_pixel(const float scale,
                                        const float scale_exposure,
                                        const float *in_combined,
                                        const float *in_catcher,
                                        const float *in_matte)
{
  const float3 color_catcher = make_float3(in_catcher[0], in_catcher[1], in_catcher[2]) *
                               scale_exposure;

  const float3 color_combined = make_float3(in_combined[0], in_combined[1], in_combined[2]) *
                                scale_exposure;

  const float3 color_matte = make_float3(in_matte[0], in_matte[1], in_matte[2]) * scale_exposure;

  const float transparency = in_combined[3] * scale;
  const float alpha = saturate(1.0f - transparency);

  const float3 shadow_catcher = safe_divide_even_color(color_combined,
                                                       color_catcher + color_matte);

  /* Restore pre-multipled nature of the color, avoiding artifacts on the edges.
   * Makes sense since the division of premultiplied color's "removes" alpha from the
   * result. */
  const float3 pixel = (1.0f - alpha) * one_float3() + alpha * shadow_catcher;

  return make_float4(pixel.x, pixel.y, pixel.z, 1.0f);
}

static float4 shadow_catcher_calc_matte_with_shadow(const float scale,
                                                    const float scale_exposure,
                                                    const float *in_combined,
                                                    const float *in_catcher,
                                                    const float *in_matte)
{
  /* The approximation of the shadow is 1 - average(shadow_catcher_pass). A better approximation
   * is possible.
   *
   * The matte is alpha-overed onto the shadow (which is kind of alpha-overing shadow onto footage,
   * and then alpha-overing synthetic objects on top). */

  const float4 shadow_catcher = shadow_catcher_calc_pixel(
      scale, scale_exposure, in_combined, in_catcher, in_matte);

  const float3 color_matte = make_float3(in_matte[0], in_matte[1], in_matte[2]) * scale_exposure;

  const float transparency = in_matte[3] * scale;
  const float alpha = saturate(1.0f - transparency);

  return make_float4(color_matte[0],
                     color_matte[1],
                     color_matte[2],
                     (1.0f - alpha) * (1.0f - average(float4_to_float3(shadow_catcher))) + alpha);
}

/* --------------------------------------------------------------------
 * Float (scalar) passes.
 */

void PassAccessorCPU::get_pass_depth(const RenderBuffers *render_buffers, float *pixels) const
{
  const Scaler scaler(render_buffers, pass_access_info_.type, num_samples_, exposure_);

  run_get_pass_processor(
      render_buffers, pixels, [scaler](const int pixel_index, const float *in, float *pixel) {
        const float f = *in;
        pixel[0] = (f == 0.0f) ? 1e10f : f * scaler.scale_exposure(pixel_index);
      });
}

void PassAccessorCPU::get_pass_mist(const RenderBuffers *render_buffers, float *pixels) const
{
  const Scaler scaler(render_buffers, pass_access_info_.type, num_samples_, exposure_);

  run_get_pass_processor(
      render_buffers, pixels, [scaler](const int pixel_index, const float *in, float *pixel) {
        const float f = *in;
        /* Note that we accumulate 1 - mist in the kernel to avoid having to
         * track the mist values in the integrator state. */
        pixel[0] = saturate(1.0f - f * scaler.scale_exposure(pixel_index));
      });
}

void PassAccessorCPU::get_pass_sample_count(const RenderBuffers *render_buffers,
                                            float *pixels) const
{
  /* TODO(sergey): Consider normalizing into the [0..1] range, so that it is possible to see
   * meaningful value when adaptive sampler stopped rendering image way before the maximum
   * number of samples was reached (for examples when number of samples is set to 0 in
   * viewport). */

  const float scale = 1.0f / num_samples_;

  run_get_pass_processor(
      render_buffers, pixels, [scale](const int /*pixel_index*/, const float *in, float *pixel) {
        const float f = *in;
        pixel[0] = __float_as_uint(f) * scale;
      });
}

void PassAccessorCPU::get_pass_float(const RenderBuffers *render_buffers, float *pixels) const
{
  const Scaler scaler(render_buffers, pass_access_info_.type, num_samples_, exposure_);

  run_get_pass_processor(
      render_buffers, pixels, [scaler](const int pixel_index, const float *in, float *pixel) {
        const float f = *in;
        pixel[0] = f * scaler.scale_exposure(pixel_index);
      });
}

/* --------------------------------------------------------------------
 * Float3 passes.
 */
void PassAccessorCPU::get_pass_shadow3(const RenderBuffers *render_buffers, float *pixels) const
{
  run_get_pass_processor(
      render_buffers, pixels, [](const int /*pixel_index*/, const float *in, float *pixel) {
        const float weight = in[3];
        const float weight_inv = (weight > 0.0f) ? 1.0f / weight : 1.0f;

        const float3 shadow = make_float3(in[0], in[1], in[2]) * weight_inv;

        pixel[0] = shadow.x;
        pixel[1] = shadow.y;
        pixel[2] = shadow.z;
      });
}

void PassAccessorCPU::get_pass_divide_even_color(const RenderBuffers *render_buffers,
                                                 float *pixels) const
{
  const PassInfo pass_info = Pass::get_info(pass_access_info_.type);

  const int pass_divide = render_buffers->params.get_pass_offset(pass_info.divide_type);
  DCHECK_NE(pass_divide, PASS_UNUSED);

  run_get_pass_processor(
      render_buffers,
      pass_divide,
      pixels,
      [exposure = exposure_](
          const int /*pixel_index*/, const float *in, const float *in_divide, float *pixel) {
        const float3 f = make_float3(in[0], in[1], in[2]);
        const float3 f_divide = make_float3(in_divide[0], in_divide[1], in_divide[2]);
        const float3 f_divided = safe_divide_even_color(f * exposure, f_divide);

        pixel[0] = f_divided.x;
        pixel[1] = f_divided.y;
        pixel[2] = f_divided.z;
      });
}

void PassAccessorCPU::get_pass_float3(const RenderBuffers *render_buffers, float *pixels) const
{
  const Scaler scaler(render_buffers, pass_access_info_.type, num_samples_, exposure_);

  run_get_pass_processor(
      render_buffers, pixels, [scaler](const int pixel_index, const float *in, float *pixel) {
        const float scale_exposure = scaler.scale_exposure(pixel_index);
        const float3 f = make_float3(in[0], in[1], in[2]) * scale_exposure;

        pixel[0] = f.x;
        pixel[1] = f.y;
        pixel[2] = f.z;
      });
}

/* --------------------------------------------------------------------
 * Float4 passes.
 */

void PassAccessorCPU::get_pass_shadow4(const RenderBuffers *render_buffers, float *pixels) const
{
  run_get_pass_processor(
      render_buffers, pixels, [](const int /*pixel_index*/, const float *in, float *pixel) {
        const float weight = in[3];
        const float weight_inv = (weight > 0.0f) ? 1.0f / weight : 1.0f;

        const float3 shadow = make_float3(in[0], in[1], in[2]) * weight_inv;

        pixel[0] = shadow.x;
        pixel[1] = shadow.y;
        pixel[2] = shadow.z;
        pixel[3] = 1.0f;
      });
}

void PassAccessorCPU::get_pass_motion(const RenderBuffers *render_buffers, float *pixels) const
{
  /* Need to normalize by number of samples accumulated for motion. */
  const int pass_motion_weight = render_buffers->params.get_pass_offset(PASS_MOTION_WEIGHT);
  DCHECK_NE(pass_motion_weight, PASS_UNUSED);

  run_get_pass_processor(
      render_buffers,
      pass_motion_weight,
      pixels,
      [](const int /*pixel_index*/, const float *in, const float *in_weight, float *pixel) {
        const float weight = in_weight[0];
        const float weight_inv = (weight > 0.0f) ? 1.0f / weight : 0.0f;

        const float4 motion = make_float4(in[0], in[1], in[2], in[3]) * weight_inv;

        pixel[0] = motion.x;
        pixel[1] = motion.y;
        pixel[2] = motion.z;
        pixel[3] = motion.w;
      });
}

void PassAccessorCPU::get_pass_cryptomatte(const RenderBuffers *render_buffers,
                                           float *pixels) const
{
  const Scaler scaler(render_buffers, pass_access_info_.type, num_samples_, exposure_);

  run_get_pass_processor(
      render_buffers, pixels, [scaler](const int pixel_index, const float *in, float *pixel) {
        const float scale = scaler.scale(pixel_index);

        const float4 f = make_float4(in[0], in[1], in[2], in[3]);
        /* x and z contain integer IDs, don't rescale them.
         * y and w contain matte weights, they get scaled. */
        pixel[0] = f.x;
        pixel[1] = f.y * scale;
        pixel[2] = f.z;
        pixel[3] = f.w * scale;
      });
}

void PassAccessorCPU::get_pass_denoising_color(const RenderBuffers *render_buffers,
                                               float *pixels) const
{
  /* Special code which converts noisy image pass from RGB to RGBA using alpha from the combined
   * pass. */

  const int pass_combined = render_buffers->params.get_pass_offset(PASS_COMBINED);
  DCHECK_NE(pass_combined, PASS_UNUSED);

  const Scaler scaler(render_buffers, pass_access_info_.type, num_samples_, exposure_);

  run_get_pass_processor(
      render_buffers,
      pass_combined,
      pixels,
      [&scaler](const int pixel_index, const float *in, const float *in_combined, float *pixel) {
        float scale, scale_exposure;
        scaler.scale_and_scale_exposure(pixel_index, scale, scale_exposure);

        const float3 color = make_float3(in[0], in[1], in[2]) * scale_exposure;
        const float transparency = in_combined[3] * scale;

        pixel[0] = color.x;
        pixel[1] = color.y;
        pixel[2] = color.z;

        pixel[3] = saturate(1.0f - transparency);
      });
}

void PassAccessorCPU::get_pass_shadow_catcher(const RenderBuffers *render_buffers,
                                              float *pixels) const
{
  /* For the shadow catcher pass we divide combined pass by the shadow catcher.
   *
   * The non-obvious trick here is that we add matte pass to the shadow catcher, so that we
   * avoid division by zero. This solves artifacts around edges of the artificial object.
   *
   * Another trick we do here is to alpha-over the pass on top of white. and ignore the alpha.
   * This way using transparent film to render artificial objects will be easy to be combined
   * with a backdrop. */

  const int pass_combined = render_buffers->params.get_pass_offset(PASS_COMBINED);
  const int pass_matte = render_buffers->params.get_pass_offset(PASS_SHADOW_CATCHER_MATTE);

  DCHECK_NE(pass_combined, PASS_UNUSED);
  DCHECK_NE(pass_matte, PASS_UNUSED);

  const Scaler scaler(render_buffers, pass_access_info_.type, num_samples_, exposure_);

  run_get_pass_processor(render_buffers,
                         pass_combined,
                         pass_matte,
                         pixels,
                         [&scaler](const int pixel_index,
                                   const float *in_catcher,
                                   const float *in_combined,
                                   const float *in_matte,
                                   float *pixel) {
                           float scale, scale_exposure;
                           scaler.scale_and_scale_exposure(pixel_index, scale, scale_exposure);

                           const float4 shadow_catcher = shadow_catcher_calc_pixel(
                               scale, scale_exposure, in_combined, in_catcher, in_matte);

                           pixel[0] = shadow_catcher.x;
                           pixel[1] = shadow_catcher.y;
                           pixel[2] = shadow_catcher.z;
                           pixel[3] = shadow_catcher.w;
                         });
}

void PassAccessorCPU::get_pass_shadow_catcher_matte_with_shadow(
    const RenderBuffers *render_buffers, float *pixels) const
{
  const int pass_combined = render_buffers->params.get_pass_offset(PASS_COMBINED);
  const int pass_shadow_catcher = render_buffers->params.get_pass_offset(PASS_SHADOW_CATCHER);

  DCHECK_NE(pass_combined, PASS_UNUSED);
  DCHECK_NE(pass_shadow_catcher, PASS_UNUSED);

  const Scaler scaler(render_buffers, pass_access_info_.type, num_samples_, exposure_);

  run_get_pass_processor(render_buffers,
                         pass_combined,
                         pass_shadow_catcher,
                         pixels,
                         [&scaler](const int pixel_index,
                                   const float *in_matte,
                                   const float *in_combined,
                                   const float *in_catcher,
                                   float *pixel) {
                           float scale, scale_exposure;
                           scaler.scale_and_scale_exposure(pixel_index, scale, scale_exposure);

                           const float4 matte = shadow_catcher_calc_matte_with_shadow(
                               scale, scale_exposure, in_combined, in_catcher, in_matte);

                           pixel[0] = matte.x;
                           pixel[1] = matte.y;
                           pixel[2] = matte.z;
                           pixel[3] = matte.w;
                         });
}

void PassAccessorCPU::get_pass_float4(const RenderBuffers *render_buffers, float *pixels) const
{
  const Scaler scaler(render_buffers, pass_access_info_.type, num_samples_, exposure_);

  run_get_pass_processor(
      render_buffers, pixels, [scaler](const int pixel_index, const float *in, float *pixel) {
        float scale, scale_exposure;
        scaler.scale_and_scale_exposure(pixel_index, scale, scale_exposure);

        /* Note that 3rd channel contains transparency = 1 - alpha at this point. */
        const float3 color = make_float3(in[0], in[1], in[2]) * scale_exposure;
        const float transparency = in[3] * scale;

        pixel[0] = color.x;
        pixel[1] = color.y;
        pixel[2] = color.z;

        /* Clamp since alpha might end up outside of 0..1 due to Russian roulette. */
        pixel[3] = saturate(1.0f - transparency);
      });
}

CCL_NAMESPACE_END
