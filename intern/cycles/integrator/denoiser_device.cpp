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

#include "integrator/denoiser_device.h"

#include "device/device.h"
#include "device/device_denoise.h"
#include "device/device_memory.h"
#include "device/device_queue.h"
#include "render/buffers.h"
#include "util/util_logging.h"
#include "util/util_progress.h"

CCL_NAMESPACE_BEGIN

DeviceDenoiser::DeviceDenoiser(Device *device, const DenoiseParams &params)
    : Denoiser(device, params)
{
}

DeviceDenoiser::~DeviceDenoiser()
{
  /* Explicit implementation, to allow forward declaration of Device in the header. */
}

void DeviceDenoiser::load_kernels(Progress *progress)
{
  get_denoiser_device(progress);
}

void DeviceDenoiser::denoise_buffer(const BufferParams &buffer_params,
                                    RenderBuffers *render_buffers,
                                    const int num_samples)
{
  Device *denoiser_device = get_denoiser_device(nullptr);

  if (!denoiser_device) {
    device_->set_error("No device available to denoise on");
    return;
  }

  VLOG(3) << "Will denoise on " << denoiser_device->info.description << " ("
          << denoiser_device->info.id << ")";

  denoise_buffer_on_device(denoiser_device, buffer_params, render_buffers, num_samples);
}

/* Check whether given device is single (not a MultiDevice) and supports requested denoiser. */
static bool is_single_supported_device(Device *device, DenoiserType type)
{
  if (device->info.type == DEVICE_MULTI) {
    /* Assume multi-device is never created with a single sub-device.
     * If one requests such configuration it should be checked on the session level. */
    return false;
  }

  if (!device->info.multi_devices.empty()) {
    /* Some configurations will use multi_devices, but keep the type of an individual device.
     * This does simplify checks for homogenous setups, but here we really need a single device. */
    return false;
  }

  /* Check the denoiser type is supported. */
  return (device->info.denoisers & type);
}

/* Find best suitable device to perform denoiser on. Will iterate over possible sub-devices of
 * multi-device.
 *
 * If there is no device available which supports given denoiser type nullptr is returned. */
static Device *find_best_device(Device *device, DenoiserType type)
{
  Device *best_device = nullptr;

  device->foreach_device([&](Device *sub_device) {
    if ((sub_device->info.denoisers & type) == 0) {
      return;
    }
    if (!best_device) {
      best_device = sub_device;
    }
    else {
      /* TODO(sergey): Choose fastest device from available ones. Taking into account performance
       * of the device and data transfer cost. */
    }
  });

  return best_device;
}

Device *DeviceDenoiser::get_denoiser_device(Progress *progress)
{
  /* The best device has been found already, avoid sequential lookups. */
  if (denoiser_device_ || device_creation_attempted_) {
    return denoiser_device_;
  }

  /* Simple case: rendering happens on a single device which also supports denoiser. */
  if (is_single_supported_device(device_, params_.type)) {
    denoiser_device_ = device_;
    return device_;
  }

  /* Find best device from the ones which are already used for rendering. */
  denoiser_device_ = find_best_device(device_, params_.type);
  if (denoiser_device_) {
    return denoiser_device_;
  }

  if (progress) {
    progress->set_status("Loading denoising kernels (may take a few minutes the first time)");
  }

  denoiser_device_ = create_denoiser_device();

  return denoiser_device_;
}

Device *DeviceDenoiser::create_denoiser_device()
{
  device_creation_attempted_ = true;

  const uint device_type_mask = get_device_type_mask();
  const vector<DeviceInfo> device_infos = Device::available_devices(device_type_mask);
  if (device_infos.empty()) {
    return nullptr;
  }

  /* TODO(sergey): Use one of the already configured devices, so that OptiX denoising can happen on
   * a physical CUDA device which is already used for rendering. */

  /* TODO(sergey): Choose fastest device for denoising. */

  const DeviceInfo denoiser_device_info = device_infos.front();

  local_denoiser_device_.reset(
      Device::create(denoiser_device_info, device_->stats, device_->profiler));

  if (!local_denoiser_device_) {
    return nullptr;
  }

  if (local_denoiser_device_->have_error()) {
    return nullptr;
  }

  /* Only need denoising feature, everything else is unused. */
  DeviceRequestedFeatures denoising_features;
  denoising_features.use_denoising = true;
  denoising_features.use_path_tracing = false;
  if (!local_denoiser_device_->load_kernels(denoising_features)) {
    return nullptr;
  }

  return local_denoiser_device_.get();
}

void DeviceDenoiser::denoise_buffer_on_device(Device *device,
                                              const BufferParams &buffer_params,
                                              RenderBuffers *render_buffers,
                                              const int num_samples)
{
  DeviceDenoiseTask task;
  task.params = params_;
  task.num_samples = num_samples;
  task.buffer_params = buffer_params;

  RenderBuffers local_render_buffers(device);
  bool local_buffer_used = false;

  if (device == device_) {
    /* The device can access an existing buffer pointer. */
    local_buffer_used = false;
    task.render_buffers = render_buffers;
  }
  else {
    DeviceQueue *queue = device->get_denoise_queue();

    /* Create buffer which is available by the device used by denoiser. */

    /* TODO(sergey): Optimize data transfers. For example, only copy denoising related passes,
     * ignoring other light ad data passes. */

    local_buffer_used = true;

    render_buffers->copy_from_device();

    local_render_buffers.reset(buffer_params);

    /* NOTE: The local buffer is allocated for an exact size of the effective render size, while
     * the input render buffer is allcoated for the lowest resolution divider possible. So it is
     * important to only copy actually needed part of the input buffer. */
    memcpy(local_render_buffers.buffer.data(),
           render_buffers->buffer.data(),
           sizeof(float) * local_render_buffers.buffer.size());

    queue->copy_to_device(local_render_buffers.buffer);

    task.render_buffers = &local_render_buffers;
  }

  device->denoise_buffer(task);

  if (local_buffer_used) {
    /* TODO(sergey): Only copy denoised passes. */
    local_render_buffers.copy_from_device();
    memcpy(render_buffers->buffer.data(),
           local_render_buffers.buffer.data(),
           sizeof(float) * local_render_buffers.buffer.size());
    render_buffers->copy_to_device();
  }
}

DeviceInfo DeviceDenoiser::get_denoiser_device_info() const
{
  if (!denoiser_device_) {
    DeviceInfo device_info;
    device_info.type = DEVICE_NONE;
    return device_info;
  }

  return denoiser_device_->info;
}

CCL_NAMESPACE_END
