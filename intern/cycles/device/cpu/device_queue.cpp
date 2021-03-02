/*
 * Copyright 2011-2013 Blender Foundation
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

#include "device/cpu/device_queue.h"

#include "device/cpu/device_cpu_impl.h"
#include "device/device.h"
#include "render/buffers.h"
#include "util/util_logging.h"

#include "kernel/integrator/integrator_path_state.h"

CCL_NAMESPACE_BEGIN

CPUDeviceQueue::CPUDeviceQueue(CPUDevice *device) : DeviceQueue(device)
{
}

void CPUDeviceQueue::init_execution()
{
  CPUDevice *cpu_device = get_cpu_device();

  /* Load information about textures from data stored in CPUDevice to data available to kernels
   * via KernelGlobals. */
  const bool texture_info_changed = cpu_device->load_texture_info();

  /* It is possible that kernel_data changes without texture info change. Such changes needs to
   * lead to re-initialization of the local copy. Currently it is not very clear how to detect
   * such changes, so always copy globals to the local copy.
   *
   * TODO(sergey): Is not too bad, is same as how it used to work in older versions, but is
   * something what would be nice to see performance impact of, and change if needed. */
  const bool is_data_changed = true;

  if (need_copy_kernel_globals_ || texture_info_changed || is_data_changed) {
    kernel_globals_ = CPUKernelThreadGlobals(cpu_device->kernel_globals, cpu_device->osl_memory());
  }

  need_copy_kernel_globals_ = false;
}

CPUIntegratorQueue::CPUIntegratorQueue(CPUDevice *device, RenderBuffers *render_buffers)
    : CPUDeviceQueue(device), render_buffers_(render_buffers)
{
}

static KernelWorkTile init_kernel_work_tile(RenderBuffers *render_buffers,
                                            const DeviceWorkTile &work_tile)
{
  KernelWorkTile kernel_work_tile;

  kernel_work_tile.x = work_tile.x;
  kernel_work_tile.y = work_tile.y;
  kernel_work_tile.w = work_tile.width;
  kernel_work_tile.h = work_tile.height;

  kernel_work_tile.start_sample = work_tile.sample;
  kernel_work_tile.num_samples = 1;

  kernel_work_tile.offset = work_tile.offset;
  kernel_work_tile.stride = work_tile.stride;

  kernel_work_tile.buffer = render_buffers->buffer.data();

  return kernel_work_tile;
}

void CPUIntegratorQueue::enqueue(DeviceKernel kernel)
{
  CPUDevice *cpu_device = get_cpu_device();
  const CPUKernels &kernels = cpu_device->kernels;

  switch (kernel) {
    case DeviceKernel::INTEGRATOR_INIT_FROM_CAMERA: {
      KernelWorkTile kernel_work_tile = init_kernel_work_tile(render_buffers_, work_tile_);
      return kernels.integrator_init_from_camera(
          &kernel_globals_, &integrator_state_, &kernel_work_tile);
    }
    case DeviceKernel::INTEGRATOR_INTERSECT_CLOSEST:
      return kernels.integrator_intersect_closest(&kernel_globals_, &integrator_state_);
    case DeviceKernel::INTEGRATOR_INTERSECT_SHADOW:
      return kernels.integrator_intersect_shadow(&kernel_globals_, &integrator_state_);
    case DeviceKernel::INTEGRATOR_INTERSECT_SUBSURFACE:
      return kernels.integrator_intersect_subsurface(&kernel_globals_, &integrator_state_);
    case DeviceKernel::INTEGRATOR_SHADE_BACKGROUND:
      return kernels.integrator_shade_background(
          &kernel_globals_, &integrator_state_, render_buffers_->buffer.data());
    case DeviceKernel::INTEGRATOR_SHADE_SHADOW:
      return kernels.integrator_shade_shadow(
          &kernel_globals_, &integrator_state_, render_buffers_->buffer.data());
    case DeviceKernel::INTEGRATOR_SHADE_SURFACE:
      return kernels.integrator_shade_surface(
          &kernel_globals_, &integrator_state_, render_buffers_->buffer.data());
    case DeviceKernel::INTEGRATOR_SHADE_VOLUME:
      return kernels.integrator_shade_volume(
          &kernel_globals_, &integrator_state_, render_buffers_->buffer.data());
    case DeviceKernel::INTEGRATOR_NUM_ACTIVE_PATHS:
    case DeviceKernel::NUM_KERNELS:
      break;
  }

  LOG(FATAL) << "Unhandled kernel " << kernel << ", should never happen.";
}

void CPUIntegratorQueue::set_work_tile(const DeviceWorkTile &work_tile)
{
  work_tile_ = work_tile;
}

bool CPUIntegratorQueue::has_work_remaining()
{
  const IntegratorState *state = &integrator_state_;
  return !(INTEGRATOR_PATH_IS_TERMINATED && INTEGRATOR_SHADOW_PATH_IS_TERMINATED);
}

int CPUIntegratorQueue::get_max_num_path_states()
{
  return 1;
}

CCL_NAMESPACE_END
