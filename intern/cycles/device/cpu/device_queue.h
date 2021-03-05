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

  inline CPUDevice *get_cpu_device() const
  {
    return static_cast<CPUDevice *>(device);
  }

 protected:
  /* Copy of kernel globals which is suitable for concurrent access from multiple queues.
   *
   * More specifically, the `kernel_globals_` is local to this queue and nobody else is
   * accessing it, but some "localization" is required to decouple from kernel globals stored
   * on the device level. */
  CPUKernelThreadGlobals kernel_globals_;

  /* Optimization flag to avoid re-initialization of the copy of the kernel globals. */
  bool need_copy_kernel_globals_ = true;
};

class CPUIntegratorQueue : public CPUDeviceQueue {
 public:
  CPUIntegratorQueue(CPUDevice *device, RenderBuffers *render_buffers);

  virtual void enqueue(DeviceKernel kernel) override;

  virtual void enqueue_work_tiles(DeviceKernel kernel,
                                  const KernelWorkTile work_tiles[],
                                  const int num_work_tiles) override;

  virtual int get_num_active_paths() override;

  virtual int get_max_num_paths() override;

 protected:
  RenderBuffers *render_buffers_;

  /* TODO(sergey): Make integrator state somehow more explicit and more dependent on the number
   * of threads, or number of splits in the kernels.
   * For the quick debug keep it at 1, but it really needs to be changed soon. */
  IntegratorState integrator_state_;
};

CCL_NAMESPACE_END
