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

#include "device/device.h"

#include "integrator/path_trace_work.h"
#include "integrator/path_trace_work_cpu.h"
#include "integrator/path_trace_work_gpu.h"

CCL_NAMESPACE_BEGIN

unique_ptr<PathTraceWork> PathTraceWork::create(Device *device,
                                                DeviceScene *device_scene,
                                                RenderBuffers *buffers,
                                                bool *cancel_requested_flag)
{
  if (device->info.type == DEVICE_CPU) {
    return make_unique<PathTraceWorkCPU>(device, device_scene, buffers, cancel_requested_flag);
  }

  return make_unique<PathTraceWorkGPU>(device, device_scene, buffers, cancel_requested_flag);
}

PathTraceWork::PathTraceWork(Device *device,
                             DeviceScene *device_scene,
                             RenderBuffers *buffers,
                             bool *cancel_requested_flag)
    : device_(device),
      device_scene_(device_scene),
      buffers_(buffers),
      effective_buffer_params_(buffers_->params),
      cancel_requested_flag_(cancel_requested_flag)
{
}

PathTraceWork::~PathTraceWork()
{
}

void PathTraceWork::set_effective_buffer_params(const BufferParams &effective_buffer_params)
{
  effective_buffer_params_ = effective_buffer_params;
}

CCL_NAMESPACE_END
