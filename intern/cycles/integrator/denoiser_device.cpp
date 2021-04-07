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
#include "render/buffers.h"
#include "util/util_logging.h"

CCL_NAMESPACE_BEGIN

DeviceDenoiser::DeviceDenoiser(Device *device, const DenoiseParams &params)
    : Denoiser(device, params)
{
}

void DeviceDenoiser::denoise_buffer(const DenoiserBufferParams &buffer_params,
                                    RenderBuffers *render_buffers,
                                    const int num_samples)
{
  denoise_buffer_on_device(device_, buffer_params, render_buffers, num_samples);
}

void DeviceDenoiser::denoise_buffer_on_device(Device *device,
                                              const DenoiserBufferParams &buffer_params,
                                              RenderBuffers *render_buffers,
                                              const int num_samples)
{
  DeviceDenoiseTask task;

  task.x = buffer_params.x;
  task.y = buffer_params.y;
  task.width = buffer_params.width;
  task.height = buffer_params.height;

  task.offset = buffer_params.offset;
  task.stride = buffer_params.stride;

  task.pass_stride = buffer_params.pass_stride;
  task.pass_denoising_offset = buffer_params.pass_denoising_offset;

  task.buffer = render_buffers->buffer.device_pointer;

  task.num_samples = num_samples;

  task.params = params_;

  device->denoise_buffer(task);
}

CCL_NAMESPACE_END
