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
  LOG(FATAL) << "Unhandled kernel " << kernel << ", should never happen.";
  return false;
}

bool CPUDeviceQueue::synchronize()
{
  return true;
}

CCL_NAMESPACE_END
