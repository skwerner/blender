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

#include "device/cpu/queue.h"
#include "device/cpu/device_impl.h"

#include "device/device.h"

#include "util/util_logging.h"

#include "kernel/integrator/integrator_path_state.h"

CCL_NAMESPACE_BEGIN

CPUDeviceQueue::CPUDeviceQueue(CPUDevice *device)
    : DeviceQueue(device), kernels_(get_cpu_device()->kernels)
{
}

void CPUDeviceQueue::init_execution()
{
  /* Cache per-thread kernel globals. */
  CPUDevice *cpu_device = get_cpu_device();
  cpu_device->get_cpu_kernel_thread_globals(kernel_thread_globals_);
}

bool CPUDeviceQueue::enqueue(DeviceKernel kernel, const int /* work_size */, void * /* args */[])
{
  /* TODO: does it make sense to implement this for debugging? */
  switch (kernel) {
    case DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA:
#if 0
      IntegratorState *state = *(IntegratorState **)args[0];
      const int *path_index_array = *(int **)args[2];
      const KernelWorkTile *tile = *(KernelWorkTile **)args[3];
      const int path_index_offset = *(int *)args[4];

      tbb::parallel_for(0, work_size, [&](int work_index) {
        const int thread_index = tbb::this_task_arena::current_thread_index();
        DCHECK_GE(thread_index, 0);
        DCHECK_LE(thread_index, kernel_thread_globals_.size());
        KernelGlobals &kernel_globals = kernel_thread_globals_[thread_index];

        const int path_index = (path_index_array) ? path_index_array[work_index] :
                                                    path_index_offset + work_index;

        /* TODO
        uint x, y, sample;
        get_work_pixel(tile, work_index, &x, &y, &sample); */

        kernels.integrator_init_from_camera(&kernel_globals, &state[work_index], tile);
      }
#endif
      break;
    case DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST:
    case DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW:
    case DEVICE_KERNEL_INTEGRATOR_INTERSECT_SUBSURFACE:
    case DEVICE_KERNEL_INTEGRATOR_SHADE_BACKGROUND:
    case DEVICE_KERNEL_INTEGRATOR_SHADE_LIGHT:
    case DEVICE_KERNEL_INTEGRATOR_SHADE_SHADOW:
    case DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE:
    case DEVICE_KERNEL_INTEGRATOR_SHADE_VOLUME:
    case DEVICE_KERNEL_INTEGRATOR_MEGAKERNEL:
    case DEVICE_KERNEL_INTEGRATOR_QUEUED_PATHS_ARRAY:
    case DEVICE_KERNEL_INTEGRATOR_QUEUED_SHADOW_PATHS_ARRAY:
    case DEVICE_KERNEL_INTEGRATOR_TERMINATED_PATHS_ARRAY:
    case DEVICE_KERNEL_NUM:
      break;
  }

  LOG(FATAL) << "Unhandled kernel " << kernel << ", should never happen.";
  return false;
}

bool CPUDeviceQueue::synchronize()
{
  return true;
}

CCL_NAMESPACE_END
