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

#include "device/cpu/device_cpu_impl.h"
#include "device/cpu/kernel.h"
#include "device/cpu/kernel_thread_globals.h"
#include "device/device_queue.h"

// clang-format off
#include "kernel/kernel.h"
#include "kernel/kernel_compat_cpu.h"
#include "kernel/kernel_types.h"
// clang-format on

CCL_NAMESPACE_BEGIN

class CPUDevice;
class RenderBuffers;

/* Base implementation of all CPU queues. Takes care of kernel function pointers and global data
 * localization. */
class CPUDeviceQueue : public DeviceQueue {
 public:
  CPUDeviceQueue(CPUDevice *device);

  virtual void init_execution() override;

  virtual bool enqueue(DeviceKernel kernel, const int work_size, void *args[]) override;

  virtual bool synchronize() override;

  inline CPUDevice *get_cpu_device() const
  {
    return static_cast<CPUDevice *>(device);
  }

 protected:
  /* CPU kernels. */
  const CPUKernels &kernels_;

  /* Copy of kernel globals which is suitable for concurrent access from multiple threads.
   *
   * More specifically, the `kernel_globals_` is local to each threads and nobody else is
   * accessing it, but some "localization" is required to decouple from kernel globals stored
   * on the device level. */
  vector<CPUKernelThreadGlobals> kernel_thread_globals_;
};

CCL_NAMESPACE_END
