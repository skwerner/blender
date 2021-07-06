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

#include "integrator/denoiser_oidn.h"

#include <array>

#include "device/device.h"
#include "integrator/pass_accessor_cpu.h"
#include "render/buffers.h"
#include "util/util_array.h"
#include "util/util_logging.h"
#include "util/util_openimagedenoise.h"

#include "kernel/device/cpu/compat.h"
#include "kernel/device/cpu/kernel.h"

CCL_NAMESPACE_BEGIN

thread_mutex OIDNDenoiser::mutex_;

class OIDNDenoiser::State {
 public:
  State()
  {
  }

  ~State()
  {
  }

#ifdef WITH_OPENIMAGEDENOISE
  oidn::DeviceRef oidn_device;
  oidn::FilterRef oidn_filter;

  bool use_pass_albedo = false;
  bool use_pass_normal = false;
#endif
};

OIDNDenoiser::OIDNDenoiser(Device *path_trace_device, const DenoiseParams &params)
    : Denoiser(path_trace_device, params), state_(make_unique<State>())
{
  DCHECK_EQ(params.type, DENOISER_OPENIMAGEDENOISE);

  DCHECK(openimagedenoise_supported()) << "OpenImageDenoiser is not supported on this platform.";
}

OIDNDenoiser::~OIDNDenoiser()
{
  /* NOTE: Keep the destructor here so that it has access to the State which is an incomplete as
   * per the OIDN denoiser header. */
}

bool OIDNDenoiser::load_kernels(Progress *progress)
{
  if (!Denoiser::load_kernels(progress)) {
    return false;
  }

  /* Make sure all lazily-initializable resources are initialized and are ready for use by the
   * denoising process. */

#ifdef WITH_OPENIMAGEDENOISE
  if (state_->oidn_filter) {
    if (params_.use_pass_albedo != state_->use_pass_albedo ||
        params_.use_pass_normal != state_->use_pass_normal) {
      state_->oidn_device = nullptr;
      state_->oidn_filter = nullptr;
    }
  }

  if (!state_->oidn_device) {
    state_->oidn_device = oidn::newDevice();
    state_->oidn_device.commit();
  }

  if (!state_->oidn_filter) {
    state_->oidn_filter = state_->oidn_device.newFilter("RT");
    state_->oidn_filter.set("hdr", true);
    state_->oidn_filter.set("srgb", false);
  }

  state_->use_pass_albedo = params_.use_pass_albedo;
  state_->use_pass_normal = params_.use_pass_normal;

  return true;
#else
  return false;
#endif
}

#ifdef WITH_OPENIMAGEDENOISE
class OIDNPass {
 public:
  OIDNPass() = default;

  OIDNPass(const BufferParams &buffer_params,
           const char *name,
           PassType type,
           PassMode mode = PassMode::NOISY)
      : name(name), type(type), mode(mode)
  {
    offset = buffer_params.get_pass_offset(type, mode);
    need_scale = (type == PASS_DENOISING_ALBEDO || type == PASS_DENOISING_NORMAL);

    const PassInfo pass_info = Pass::get_info(type);
    use_compositing = pass_info.use_compositing;
  }

  /* Name of an image which will be passed to the OIDN library.
   * Should be one of the following: color, albedo, normal, output.
   * The albedo and normal images are optional. */
  const char *name = "";

  PassType type = PASS_NONE;
  PassMode mode = PassMode::NOISY;

  /* Offset of beginning of this pass in the render buffers. */
  int offset = -1;

  /* Denotes whether the data is to be scaled down with the number of passes.
   * Is required for albedo and normal passes. The color pass OIDN will perform auto-exposure, so
   * scaling is not needed for the color pass unless adaptive sampling is used.
   *
   * NOTE: Do not scale the outout pass, as that requires to be a pointer in the original buffer.
   * All the scaling on the output needed for integration with adaptive sampling will happen
   * outside of generic pass handling. */
  bool need_scale = false;

  bool use_compositing = false;

  /* For the scaled passes, the data which holds values of scaled pixels. */
  array<float> scaled_buffer;

  /* For the in-place usable passes denotes whether the underlying data has been scaled. */
  bool is_scaled = false;
};

class OIDNDenoiseContext {
 public:
  OIDNDenoiseContext(const DenoiseParams &denoise_params,
                     const BufferParams &buffer_params,
                     RenderBuffers *render_buffers,
                     oidn::FilterRef *oidn_filter,
                     const int num_samples,
                     const bool allow_inplace_modification)
      : denoise_params_(denoise_params),
        buffer_params_(buffer_params),
        render_buffers_(render_buffers),
        oidn_filter_(oidn_filter),
        num_samples_(num_samples),
        allow_inplace_modification_(allow_inplace_modification),
        pass_sample_count_(buffer_params_.get_pass_offset(PASS_SAMPLE_COUNT))
  {
    if (denoise_params_.use_pass_albedo) {
      oidn_albedo_pass_ = OIDNPass(buffer_params_, "albedo", PASS_DENOISING_ALBEDO);
      /* NOTE: The albedo pass is always ensured to be set from the denoise() call, since it is
       * possible that some passes will not use the real values. */
    }

    if (denoise_params_.use_pass_normal) {
      oidn_normal_pass_ = OIDNPass(buffer_params_, "normal", PASS_DENOISING_NORMAL);
      set_pass(oidn_normal_pass_);
    }
  }

  void denoise(const PassType pass_type)
  {
    /* Add input color image. */
    OIDNPass oidn_color_pass(buffer_params_, "color", pass_type);
    if (oidn_color_pass.offset == PASS_UNUSED) {
      return;
    }
    set_pass(oidn_color_pass);

    if (denoise_params_.use_pass_albedo) {
      const PassInfo pass_info = Pass::get_info(pass_type);
      if (!pass_info.use_denoising_albedo) {
        set_fake_albedo_pass();
      }
      else {
        set_pass(oidn_albedo_pass_);
      }
    }

    /* Add output pass. */
    OIDNPass oidn_output_pass(buffer_params_, "output", pass_type, PassMode::DENOISED);
    if (oidn_color_pass.offset == PASS_UNUSED) {
      LOG(DFATAL) << "Missing denoised pass " << pass_type_as_string(pass_type);
      return;
    }
    set_pass_referenced(oidn_output_pass);

    /* Execute filter. */
    oidn_filter_->commit();
    oidn_filter_->execute();

    postprocess_output(oidn_color_pass, oidn_output_pass);
  }

 protected:
  /* Set OIDN image to reference pixels from the given render buffer pass.
   * No transform to the pixels is done, no additional memory is used. */
  void set_pass_referenced(const OIDNPass &oidn_pass)
  {
    const int64_t x = buffer_params_.full_x;
    const int64_t y = buffer_params_.full_y;
    const int64_t width = buffer_params_.width;
    const int64_t height = buffer_params_.height;
    const int64_t offset = buffer_params_.offset;
    const int64_t stride = buffer_params_.stride;
    const int64_t pass_stride = buffer_params_.pass_stride;

    const int64_t pixel_index = offset + x + y * stride;
    const int64_t buffer_offset = pixel_index * pass_stride;

    float *buffer_data = render_buffers_->buffer.data();

    oidn_filter_->setImage(oidn_pass.name,
                           buffer_data + buffer_offset + oidn_pass.offset,
                           oidn::Format::Float3,
                           width,
                           height,
                           0,
                           pass_stride * sizeof(float),
                           stride * pass_stride * sizeof(float));
  }

  void read_pass_pixels(OIDNPass &oidn_pass)
  {
    const int64_t width = buffer_params_.width;
    const int64_t height = buffer_params_.height;

    array<float> &scaled_buffer = oidn_pass.scaled_buffer;
    scaled_buffer.resize(width * height * 3);

    PassAccessor::PassAccessInfo pass_access_info;
    pass_access_info.type = oidn_pass.type;
    pass_access_info.mode = oidn_pass.mode;
    pass_access_info.offset = oidn_pass.offset;

    /* Denoiser operates on passes which are used to calculate the approximation, and is never used
     * on the approximation. The latter is not even possible because OIDN does not support
     * denoising of semi-transparent pixels. */
    pass_access_info.use_approximate_shadow_catcher = false;
    pass_access_info.show_active_pixels = false;

    /* OIDN will perform an auto-exposure, so it is not required to know exact exposure configured
     * by users. What is important is to use same exposure for read and write access of the pass
     * pixels. */
    const PassAccessorCPU pass_accessor(pass_access_info, 1.0f, num_samples_);
    const PassAccessor::Destination destination(scaled_buffer.data(), 3);

    pass_accessor.get_render_tile_pixels(render_buffers_, buffer_params_, destination);
  }

  void set_pass_scaled(OIDNPass &oidn_pass)
  {
    if (oidn_pass.scaled_buffer.empty()) {
      read_pass_pixels(oidn_pass);
    }

    const int64_t width = buffer_params_.width;
    const int64_t height = buffer_params_.height;

    oidn_filter_->setImage(oidn_pass.name,
                           oidn_pass.scaled_buffer.data(),
                           oidn::Format::Float3,
                           width,
                           height,
                           0,
                           0,
                           0);
  }

  void set_pass(OIDNPass &oidn_pass)
  {
    if (oidn_pass.use_compositing) {
      /* TODO(sergey): Avoid extra memory for compositing passes. */
      set_pass_scaled(oidn_pass);
      return;
    }

    /* When adaptive sampling is involved scaling is always needed.
     * If the avoid scaling if there is only one sample, to save up time (so we dont divide buffer
     * by 1). */
    if (pass_sample_count_ == PASS_UNUSED && (!oidn_pass.need_scale || num_samples_ == 1)) {
      set_pass_referenced(oidn_pass);
      return;
    }

    if (allow_inplace_modification_) {
      set_pass_referenced(oidn_pass);
      scale_pass_if_needed(oidn_pass);
      return;
    }

    set_pass_scaled(oidn_pass);
  }

  void set_fake_albedo_pass()
  {
    const int64_t width = buffer_params_.width;
    const int64_t height = buffer_params_.height;

    /* TODO(sergey): Is there a way to avoid allocation of an entire frame of const values? */

    if (fake_albedo_pixels_.empty()) {
      const int64_t num_pixels = width * height * 3;
      fake_albedo_pixels_.resize(num_pixels);
      for (int i = 0; i < num_pixels; ++i) {
        fake_albedo_pixels_[i] = 0.5f;
      }
    }

    oidn_filter_->setImage(
        "albedo", fake_albedo_pixels_.data(), oidn::Format::Float3, width, height, 0, 0, 0);
  }

  /* Scale output pass to match adaptive sampling per-pixel scale, as well as bring alpha channel
   * back. */
  void postprocess_output(const OIDNPass &oidn_input_pass, const OIDNPass &oidn_output_pass)
  {
    const int64_t x = buffer_params_.full_x;
    const int64_t y = buffer_params_.full_y;
    const int64_t width = buffer_params_.width;
    const int64_t height = buffer_params_.height;
    const int64_t offset = buffer_params_.offset;
    const int64_t stride = buffer_params_.stride;
    const int64_t pass_stride = buffer_params_.pass_stride;
    const int64_t row_stride = stride * pass_stride;

    const int64_t pixel_offset = offset + x + y * stride;
    const int64_t buffer_offset = (pixel_offset * pass_stride);

    float *buffer_data = render_buffers_->buffer.data();

    const bool has_pass_sample_count = (pass_sample_count_ != PASS_UNUSED);
    const bool need_scale = has_pass_sample_count || oidn_input_pass.use_compositing;

    for (int y = 0; y < height; ++y) {
      float *buffer_row = buffer_data + buffer_offset + y * row_stride;
      for (int x = 0; x < width; ++x) {
        float *buffer_pixel = buffer_row + x * pass_stride;
        float *denoised_pixel = buffer_pixel + oidn_output_pass.offset;

        if (need_scale) {
          const float pixel_scale = has_pass_sample_count ?
                                        __float_as_uint(buffer_pixel[pass_sample_count_]) :
                                        num_samples_;

          denoised_pixel[0] = denoised_pixel[0] * pixel_scale;
          denoised_pixel[1] = denoised_pixel[1] * pixel_scale;
          denoised_pixel[2] = denoised_pixel[2] * pixel_scale;
        }

        /* Currently compositing passes are either 3-component (derived by dividing light passes)
         * or do not have transparency (shadow catcher). Implicitly rely on this logic, as it
         * simplifies logic and avoids extra memory allocation. */
        if (!oidn_input_pass.use_compositing) {
          const float *noisy_pixel = buffer_pixel + oidn_input_pass.offset;
          denoised_pixel[3] = noisy_pixel[3];
        }
        else {
          /* Assigning to zero since this is a default alpha value for 3-component passes, and it
           * is an opaque pixel for 4 component passes. */
          denoised_pixel[3] = 0;
        }
      }
    }
  }

  void scale_pass_if_needed(OIDNPass &oidn_pass)
  {
    if (!oidn_pass.need_scale) {
      return;
    }
    if (oidn_pass.is_scaled) {
      return;
    }
    oidn_pass.is_scaled = true;

    const int64_t x = buffer_params_.full_x;
    const int64_t y = buffer_params_.full_y;
    const int64_t width = buffer_params_.width;
    const int64_t height = buffer_params_.height;
    const int64_t offset = buffer_params_.offset;
    const int64_t stride = buffer_params_.stride;
    const int64_t pass_stride = buffer_params_.pass_stride;
    const int64_t row_stride = stride * pass_stride;

    const int64_t pixel_offset = offset + x + y * stride;
    const int64_t buffer_offset = (pixel_offset * pass_stride);

    float *buffer_data = render_buffers_->buffer.data();

    const bool has_pass_sample_count = (pass_sample_count_ != PASS_UNUSED);

    for (int y = 0; y < height; ++y) {
      float *buffer_row = buffer_data + buffer_offset + y * row_stride;
      for (int x = 0; x < width; ++x) {
        float *buffer_pixel = buffer_row + x * pass_stride;
        float *pass_pixel = buffer_pixel + oidn_pass.offset;

        const float pixel_scale = 1.0f / (has_pass_sample_count ?
                                              __float_as_uint(buffer_pixel[pass_sample_count_]) :
                                              num_samples_);

        pass_pixel[0] = pass_pixel[0] * pixel_scale;
        pass_pixel[1] = pass_pixel[1] * pixel_scale;
        pass_pixel[2] = pass_pixel[2] * pixel_scale;
      }
    }
  }

  const DenoiseParams &denoise_params_;
  const BufferParams &buffer_params_;
  RenderBuffers *render_buffers_;
  oidn::FilterRef *oidn_filter_;
  int num_samples_;
  bool allow_inplace_modification_;
  int pass_sample_count_;

  /* Optional albedo and normal passes, reused by denoising of different pass types. */
  OIDNPass oidn_albedo_pass_;
  OIDNPass oidn_normal_pass_;

  array<float> fake_albedo_pixels_;
};
#endif

void OIDNDenoiser::denoise_buffer(const BufferParams &buffer_params,
                                  RenderBuffers *render_buffers,
                                  const int num_samples,
                                  bool allow_inplace_modification)
{
  thread_scoped_lock lock(mutex_);

  /* Copy pixels from compute device to CPU (no-op for CPU device). */
  render_buffers->buffer.copy_from_device();

#ifdef WITH_OPENIMAGEDENOISE
  oidn::FilterRef *oidn_filter = &state_->oidn_filter;

  OIDNDenoiseContext context(params_,
                             buffer_params,
                             render_buffers,
                             oidn_filter,
                             num_samples,
                             allow_inplace_modification);
  context.denoise(PASS_COMBINED);
  context.denoise(PASS_SHADOW_CATCHER);
  context.denoise(PASS_SHADOW_CATCHER_MATTE);
#endif

  /* TODO: It may be possible to avoid this copy, but we have to ensure that when other code copies
   * data from the device it doesn't overwrite the denoiser buffers. */
  render_buffers->buffer.copy_to_device();
}

uint OIDNDenoiser::get_device_type_mask() const
{
  return DEVICE_MASK_CPU;
}

CCL_NAMESPACE_END
