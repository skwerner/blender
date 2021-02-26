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

#include "device/device_kernel.h"
#include "device/device_memory.h"
#include "device/device_queue.h"

#include "kernel/integrator/integrator_state.h"
#include "kernel/kernel_types.h"

CCL_NAMESPACE_BEGIN

class CUDADevice;
class CUDADeviceKernels;
class RenderBuffers;

/* Base class for CUDA queues. */
class CUDADeviceQueue : public DeviceQueue {
 public:
  CUDADeviceQueue(CUDADevice *device);

  virtual void init_execution() override;

 protected:
  CUDADevice *cuda_device_;
};

/* Path tracing integrator CUDA queue. */
class CUDAIntegratorQueue : public CUDADeviceQueue {
 public:
  CUDAIntegratorQueue(CUDADevice *device, RenderBuffers *render_buffers);

  virtual void enqueue(DeviceKernel kernel) override;

  virtual void set_work_tile(const DeviceWorkTile &work_tile) override;

  virtual bool has_work_remaining() override;

  virtual int get_max_num_path_states() override;

 protected:
  RenderBuffers *render_buffers_;

  device_only_memory<IntegratorState> integrator_state_;
  device_vector<KernelWorkTile> work_tile_;
};

CCL_NAMESPACE_END
