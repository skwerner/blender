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

OIDNDenoiser::OIDNDenoiser(Device *device, const DenoiseParams &params)
    : Denoiser(device, params), state_(make_unique<State>())
{
  DCHECK_EQ(params.type, DENOISER_OPENIMAGEDENOISE);

  DCHECK(openimagedenoise_supported()) << "OpenImageDenoiser is not supported on this platform.";
}

OIDNDenoiser::~OIDNDenoiser()
{
  /* NOTE: Keep the destructor here so that it has access to the State which is an incomplete as
   * per the OIDN denoiser header. */
}

void OIDNDenoiser::load_kernels(Progress * /*progress*/)
{
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

  /* For the scaled passes, the data which holds values of scaled pixels. */
  array<float> scaled_buffer;
};

class OIDNDeenoiseContext {
 public:
  OIDNDeenoiseContext(const DenoiseParams &denoise_params,
                      const BufferParams &buffer_params,
                      RenderBuffers *render_buffers,
                      oidn::FilterRef *oidn_filter,
                      const int num_samples)
      : denoise_params_(denoise_params),
        buffer_params_(buffer_params),
        render_buffers_(render_buffers),
        oidn_filter_(oidn_filter),
        num_samples_(num_samples),
        pass_sample_count_(buffer_params_.get_pass_offset(PASS_SAMPLE_COUNT))
  {
  }

  void denoise()
  {
    /* Add input images.
     *
     * NOTE: Store passes for the entire duration od denoising because OIDN denoiser might
     * reference pixels from the pass buffer. */

    OIDNPass oidn_color_pass(buffer_params_, "color", PASS_COMBINED);
    OIDNPass oidn_albedo_pass;
    OIDNPass oidn_normal_pass;

    set_pass(oidn_color_pass);

    if (denoise_params_.use_pass_albedo) {
      oidn_albedo_pass = OIDNPass(buffer_params_, "albedo", PASS_DENOISING_ALBEDO);
      set_pass(oidn_albedo_pass);
    }

    if (denoise_params_.use_pass_normal) {
      oidn_normal_pass = OIDNPass(buffer_params_, "normal", PASS_DENOISING_NORMAL);
      set_pass(oidn_normal_pass);
    }

    /* Add output pass. */
    OIDNPass oidn_output_pass(buffer_params_, "output", PASS_COMBINED, PassMode::DENOISED);
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

  void set_pass_scaled(OIDNPass &oidn_pass)
  {
    const int64_t width = buffer_params_.width;
    const int64_t height = buffer_params_.height;

    array<float> &scaled_buffer = oidn_pass.scaled_buffer;
    scaled_buffer.resize(width * height * 3);

    PassAccessor::PassAccessInfo pass_access_info;
    pass_access_info.type = oidn_pass.type;
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

    oidn_filter_->setImage(
        oidn_pass.name, scaled_buffer.data(), oidn::Format::Float3, width, height, 0, 0, 0);
  }

  void set_pass(OIDNPass &oidn_pass)
  {
    if (!oidn_pass.need_scale || (num_samples_ == 1 && pass_sample_count_ == PASS_UNUSED)) {
      set_pass_referenced(oidn_pass);
      return;
    }

    set_pass_scaled(oidn_pass);
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

    for (int y = 0; y < height; ++y) {
      float *buffer_row = buffer_data + buffer_offset + y * row_stride;
      for (int x = 0; x < width; ++x) {
        float *buffer_pixel = buffer_row + x * pass_stride;
        float *noisy_pixel = buffer_pixel + oidn_input_pass.offset;
        float *denoised_pixel = buffer_pixel + oidn_output_pass.offset;

        if (pass_sample_count_ != PASS_UNUSED) {
          const float pixel_scale = __float_as_uint(buffer_pixel[pass_sample_count_]);

          denoised_pixel[0] = denoised_pixel[0] * pixel_scale;
          denoised_pixel[1] = denoised_pixel[1] * pixel_scale;
          denoised_pixel[2] = denoised_pixel[2] * pixel_scale;
        }

        denoised_pixel[3] = noisy_pixel[3];
      }
    }
  }

  const DenoiseParams &denoise_params_;
  const BufferParams &buffer_params_;
  RenderBuffers *render_buffers_;
  oidn::FilterRef *oidn_filter_;
  int num_samples_;
  int pass_sample_count_;
};
#endif

void OIDNDenoiser::denoise_buffer(const BufferParams &buffer_params,
                                  RenderBuffers *render_buffers,
                                  const int num_samples)
{
  thread_scoped_lock lock(mutex_);

  /* Copy pixels from compute device to CPU (no-op for CPU device). */
  render_buffers->buffer.copy_from_device();

  initialize();

#ifdef WITH_OPENIMAGEDENOISE
  oidn::FilterRef *oidn_filter = &state_->oidn_filter;

  OIDNDeenoiseContext context(params_, buffer_params, render_buffers, oidn_filter, num_samples);
  context.denoise();
#endif

  /* TODO: It may be possible to avoid this copy, but we have to ensure that when other code copies
   * data from the device it doesn't overwrite the denoiser buffers. */
  render_buffers->buffer.copy_to_device();
}

DeviceInfo OIDNDenoiser::get_denoiser_device_info() const
{
  /* OpenImageDenoiser runs on CPU. Access the CPU device information with some safety fallbacks
   * for possible variations of Cycles integration. */

  vector<DeviceInfo> cpu_devices = Device::available_devices(DEVICE_MASK_CPU);

  if (cpu_devices.empty()) {
    LOG(ERROR) << "No CPU devices reported.";

    DeviceInfo dummy_info;
    dummy_info.type = DEVICE_NONE;
    return dummy_info;
  }

  if (cpu_devices.size() > 1) {
    DeviceInfo device_info;
    device_info.type = DEVICE_MULTI;
    device_info.multi_devices = cpu_devices;
    return device_info;
  }

  return cpu_devices[0];
}

void OIDNDenoiser::initialize()
{
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
#endif
}

CCL_NAMESPACE_END
