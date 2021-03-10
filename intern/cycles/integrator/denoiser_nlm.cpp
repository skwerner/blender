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

#include "integrator/denoiser_nlm.h"

#include "device/device.h"
#include "render/buffers.h"
#include "util/util_logging.h"

CCL_NAMESPACE_BEGIN

NLMDenoiser::NLMDenoiser(Device *device, const DenoiseParams &params) : Denoiser(device, params)
{
  DCHECK_NE(device->info.type, DEVICE_MULTI);
  DCHECK_EQ(params.type, DENOISER_NLM);
}

void NLMDenoiser::denoise_buffer(const DenoiserBufferParams &buffer_params,
                                 RenderBuffers *render_buffers,
                                 const int num_samples)
{
#if 1
  (void)buffer_params;
  (void)render_buffers;
  (void)num_samples;
  LOG(ERROR) << "NLM denoiser needs support from the kernel.";
#else
  DeviceTask task(DeviceTask::DENOISE_BUFFER);

  const BufferParams &buffer_params = render_buffers->params;

  task.x = buffer_params.full_x;
  task.y = buffer_params.full_y;
  task.w = buffer_params.width;
  task.h = buffer_params.height;
  task.buffer = render_buffers->buffer.device_pointer;
  task.buffers = render_buffers;

  /* TODO(sergey): Check whether sample is needed. */
  task.sample = 1;
  task.num_samples = 1;

  /* TODO(sergey): Use pre-calculated/known offset and stride. No need to re-calculate it on every
   * denoiser invocation. */
  render_buffers->params.get_offset_stride(task.offset, task.stride);

  device_->task_add(task);
  device_->task_wait();
#endif
}

CCL_NAMESPACE_END
