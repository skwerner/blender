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
#include "render/buffers.h"
#include "render/scene.h"

#include "kernel/kernel_types.h"

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

PassAccessor::PassAccessInfo PathTraceWork::get_display_pass_access_info(PassMode pass_mode) const
{
  const KernelFilm &kfilm = device_scene_->data.film;

  PassAccessor::PassAccessInfo pass_access_info;
  pass_access_info.type = static_cast<PassType>(kfilm.display_pass_type);

  if (pass_mode == PassMode::DENOISED && kfilm.display_pass_denoised_offset != PASS_UNUSED) {
    pass_access_info.offset = kfilm.display_pass_denoised_offset;
  }
  else {
    pass_access_info.offset = kfilm.display_pass_offset;
  }

  pass_access_info.use_approximate_shadow_catcher = kfilm.use_approximate_shadow_catcher;
  pass_access_info.show_active_pixels = kfilm.show_active_pixels;

  return pass_access_info;
}

CCL_NAMESPACE_END
