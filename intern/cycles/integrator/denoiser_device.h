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

#include "integrator/denoiser.h"
#include "util/util_unique_ptr.h"

CCL_NAMESPACE_BEGIN

/* Denoiser which uses device-specific denoising implementation, such as OptiX denoiser which are
 * implemented as a part of a driver of specific device.
 *
 * This implementation makes sure the to-be-denoised buffer is available on the denoising device
 * and invoke denoising kernel via device API. */
class DeviceDenoiser : public Denoiser {
 public:
  DeviceDenoiser(Device *device, const DenoiseParams &params);
  ~DeviceDenoiser();

  virtual void load_kernels(Progress *progress) override;

  virtual void denoise_buffer(const BufferParams &buffer_params,
                              RenderBuffers *render_buffers,
                              const int num_samples) override;

  virtual DeviceInfo get_denoiser_device_info() const override;

 protected:
  /* Get device on which denoising is to happen.
   * Will either use one of the devices used for rendering, or create a dedicated device if needed.
   */
  Device *get_denoiser_device(Progress *progress);

  /* Create denoiser device which is owned by this denoiser.
   * Used in the cases when none of the devices used for rendering supports requetsed denoiser
   * type. */
  Device *create_denoiser_device();

  /* Get device type mask which is used to filter available devices when new device needs to be
   * created. */
  virtual uint get_device_type_mask() const = 0;

  void denoise_buffer_on_device(Device *device,
                                const BufferParams &buffer_params,
                                RenderBuffers *render_buffers,
                                const int num_samples);

  /* Cached pointer to the device on which denoising will happen.
   * Used to avoid lookup of a device for every denoising request. */
  Device *denoiser_device_ = nullptr;

  /* Denoiser device which was created to perform denoising in the case the none of the rendering
   * devices are capable of denoising. */
  unique_ptr<Device> local_denoiser_device_;
  bool device_creation_attempted_ = false;
};

CCL_NAMESPACE_END
