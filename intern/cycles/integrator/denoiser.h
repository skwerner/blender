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

/* TODO(sergey): The integrator folder might not be the best. Is easy to move files around if the
 * better place is figured out. */

#include "device/device.h"
#include "device/device_denoise.h"
#include "util/util_unique_ptr.h"

CCL_NAMESPACE_BEGIN

class BufferParams;
class Device;
class RenderBuffers;
class Progress;

/* Implementation of a specific denoising algorithm.
 *
 * This class takes care of breaking down denosiing algorithm into a series of device calls or to
 * calls of an external API to denoise given input.
 *
 * TODO(sergey): Are we better with device or a queue here? */
class Denoiser {
 public:
  virtual ~Denoiser() = default;

  void set_params(const DenoiseParams &params);
  const DenoiseParams &get_params() const;

  /* Create denoiser for the given device.
   * Notes:
   * - The denoiser must be configured. This means that `params.use` must be true.
   *   This is checked in debug builds.
   * - The device might be MultiDevice. */
  static unique_ptr<Denoiser> create(Device *device, const DenoiseParams &params);

  /* Create devices and load kernels needed for denoising.
   * The progress is used to communicate state when kenrels actually needs to be loaded.
   *
   * NOTE: The `progress` is an optional argument, can be nullptr. */
  virtual void load_kernels(Progress *progress) = 0;

  /* Denoise the entire buffer.
   *
   * Buffer parameters denotes an effective parameters used during rendering. It could be
   * a lower resolution render into a bigger allocated buffer, which is used in viewport during
   * navigation and non-unit pixel size. Use that instead of render_buffers->params.
   *
   * The buffer might be copming from a "foreign" device from what this denoise is created for.
   * This means that in general case the denoiser will make sure the input data is available on
   * the denoiser device, perform denoising, and put data back to the device where the buffer
   * came from.
   *
   * The `num_samples` corresponds to the number of samples in the render buffers. It is used
   * to scale buffers down to the "final" value in algorithms which don't do automatic exposure,
   * or which needs "final" value for data passes. */
  virtual void denoise_buffer(const BufferParams &buffer_params,
                              RenderBuffers *render_buffers,
                              const int num_samples) = 0;

  /* Get access to the device information which is used to perform actual denoising.
   * Note that this device:
   *
   * - Can be different from the device used during denoiser creation. This happens, for example,
   *   when using OptiX denoiser and rendering on CPU.
   *
   * - The denoising device is lazily initialized, so if no denoising was perfoemed yet it is
   *   possible that device info of type DEVICE_NONE will be returned.
   *
   * - No threading safety is ensured in this call. This means, that it is up to caller to ensure
   *   that there is no threadingconflict between denoising task lazily initializing the device and
   *   access to this device happen. */
  virtual DeviceInfo get_denoiser_device_info() const = 0;

 protected:
  Denoiser(Device *device, const DenoiseParams &params);

  Device *device_;
  DenoiseParams params_;
};

CCL_NAMESPACE_END
