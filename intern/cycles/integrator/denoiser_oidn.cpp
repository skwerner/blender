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
struct OIDNPass {
  /* Name of an image which will be passed to the OIDN library.
   * Should be one of the following: color, albedo, normal, output.
   * The albedo and normal images are optional. */
  const char *name;

  /* Offset of beginning of this pass in the render buffers. */
  const int offset;

  /* Denotes whether the data is to be scaled down with the number of passes.
   * Is required for albedo and normal passes. The color pass OIDN will perform auto-exposure, so
   * scaling is not needed for the color pass unless adaptive sampling is used.
   *
   * NOTE: Do not scale the outout pass, as that requires to be a pointer in the original buffer.
   * All the scaling on the output needed for integration with adaptive sampling will happen
   * outside of generic pass handling. */
  const bool need_scale;

  /* Whether or not send this pass to the OIDN. */
  const bool use;

  /* For the scaled passes, the data which holds values of scaled pixels. */
  array<float> scaled_buffer;
};

static void oidn_add_pass_if_needed(oidn::FilterRef *oidn_filter,
                                    OIDNPass &oidn_pass,
                                    RenderBuffers *render_buffers,
                                    const BufferParams &buffer_params,
                                    const float scale)
{
  if (!oidn_pass.use) {
    return;
  }

  const int64_t x = buffer_params.full_x;
  const int64_t y = buffer_params.full_y;
  const int64_t width = buffer_params.width;
  const int64_t height = buffer_params.height;
  const int64_t offset = buffer_params.offset;
  const int64_t stride = buffer_params.stride;
  const int64_t pass_stride = buffer_params.pass_stride;

  const int64_t pixel_offset = offset + x + y * stride;
  const int64_t buffer_offset = (pixel_offset * pass_stride);
  const int64_t pixel_stride = pass_stride;
  const int64_t row_stride = stride * pixel_stride;

  const int pass_sample_count = buffer_params.get_pass_offset(PASS_SAMPLE_COUNT);

  float *buffer_data = reinterpret_cast<float *>(render_buffers->buffer.host_pointer);

  if (!oidn_pass.need_scale || (scale == 1.0f && pass_sample_count == PASS_UNUSED)) {
    oidn_filter->setImage(oidn_pass.name,
                          buffer_data + buffer_offset + oidn_pass.offset,
                          oidn::Format::Float3,
                          width,
                          height,
                          0,
                          pixel_stride * sizeof(float),
                          row_stride * sizeof(float));
    return;
  }

  array<float> &scaled_buffer = oidn_pass.scaled_buffer;
  scaled_buffer.resize(width * height * 3);

  for (int y = 0; y < height; ++y) {
    const float *buffer_row = buffer_data + buffer_offset + y * row_stride;
    float *scaled_row = scaled_buffer.data() + y * width * 3;

    for (int x = 0; x < width; ++x) {
      const float *buffer_pixel = buffer_row + x * pixel_stride;
      const float *pass_pixel = buffer_pixel + oidn_pass.offset;

      float pixel_scale = scale;
      if (pass_sample_count != PASS_UNUSED) {
        pixel_scale = 1.0f / __float_as_uint(buffer_pixel[pass_sample_count]);
      }

      scaled_row[x * 3 + 0] = pass_pixel[0] * pixel_scale;
      scaled_row[x * 3 + 1] = pass_pixel[1] * pixel_scale;
      scaled_row[x * 3 + 2] = pass_pixel[2] * pixel_scale;
    }
  }

  oidn_filter->setImage(
      oidn_pass.name, scaled_buffer.data(), oidn::Format::Float3, width, height, 0, 0, 0);
}

static void oidn_scale_combined_pass_after_denoise(const BufferParams &buffer_params,
                                                   RenderBuffers *render_buffers)
{
  const int pass_sample_count = buffer_params.get_pass_offset(PASS_SAMPLE_COUNT);
  if (pass_sample_count == PASS_UNUSED) {
    return;
  }

  const int64_t x = buffer_params.full_x;
  const int64_t y = buffer_params.full_y;
  const int64_t width = buffer_params.width;
  const int64_t height = buffer_params.height;
  const int64_t offset = buffer_params.offset;
  const int64_t stride = buffer_params.stride;
  const int64_t pass_stride = buffer_params.pass_stride;
  const int64_t pixel_stride = pass_stride;
  const int64_t row_stride = stride * pixel_stride;

  const int64_t pixel_offset = offset + x + y * stride;
  const int64_t buffer_offset = (pixel_offset * pass_stride);

  float *buffer_data = reinterpret_cast<float *>(render_buffers->buffer.host_pointer);

  for (int y = 0; y < height; ++y) {
    float *buffer_row = buffer_data + buffer_offset + y * row_stride;
    for (int x = 0; x < width; ++x) {
      float *buffer_pixel = buffer_row + x * pixel_stride;
      const float pixel_scale = __float_as_uint(buffer_pixel[pass_sample_count]);

      buffer_pixel[0] = buffer_pixel[0] * pixel_scale;
      buffer_pixel[1] = buffer_pixel[1] * pixel_scale;
      buffer_pixel[2] = buffer_pixel[2] * pixel_scale;
    }
  }
}
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
  const bool have_sample_count_pass = (buffer_params.get_pass_offset(PASS_SAMPLE_COUNT) !=
                                       PASS_UNUSED);

  oidn::FilterRef *oidn_filter = &state_->oidn_filter;

  std::array<OIDNPass, 4> oidn_passes = {{
      {"color", buffer_params.get_pass_offset(PASS_DENOISING_COLOR), have_sample_count_pass, true},
      {"albedo",
       buffer_params.get_pass_offset(PASS_DENOISING_ALBEDO),
       true,
       params_.use_pass_albedo},
      {"normal",
       buffer_params.get_pass_offset(PASS_DENOISING_NORMAL),
       true,
       params_.use_pass_normal},
      {"output", 0, false, true},
  }};

  const float scale = 1.0f / num_samples;

  for (OIDNPass &oidn_pass : oidn_passes) {
    oidn_add_pass_if_needed(oidn_filter, oidn_pass, render_buffers, buffer_params, scale);
  }

  /* Execute filter. */
  oidn_filter->commit();
  oidn_filter->execute();

  oidn_scale_combined_pass_after_denoise(buffer_params, render_buffers);
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
  if (!state_->oidn_device) {
    state_->oidn_device = oidn::newDevice();
    state_->oidn_device.commit();
  }

  if (!state_->oidn_filter) {
    state_->oidn_filter = state_->oidn_device.newFilter("RT");
    state_->oidn_filter.set("hdr", true);
    state_->oidn_filter.set("srgb", false);
  }
#endif
}

CCL_NAMESPACE_END
