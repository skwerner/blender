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

#ifdef WITH_CUDA

#  include "device/device_kernel.h"
#  include "device/device_memory.h"
#  include "device/device_queue.h"

#  include "kernel/integrator/integrator_path_state.h"
#  include "kernel/integrator/integrator_state.h"
#  include "kernel/kernel_types.h"

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

  virtual void enqueue_work_tiles(DeviceKernel kernel,
                                  const KernelWorkTile work_tiles[],
                                  const int num_work_tiles) override;

  virtual int get_num_active_paths() override;

  virtual int get_max_num_paths() override;

 protected:
  void compute_queued_paths(DeviceKernel kernel, int queued_kernel);

  RenderBuffers *render_buffers_;

  /* Integrate state for paths. */
  device_only_memory<IntegratorState> integrator_state_;
  /* Keep track of number of queued kernels. */
  device_vector<IntegratorPathQueue> integrator_path_queue_;

  /* Temporary buffer to get an array of queued path for a particular kernel. */
  device_vector<int> queued_paths_;
  device_vector<int> num_queued_paths_;

  /* Temporary buffer for passing work tiles to kernel. */
  device_vector<KernelWorkTile> work_tiles_;

  /* Maximum path index, effective number of paths used may be smaller than
   * the size of the integrator_state_ buffer so can avoid iterating over the
   * full buffer. */
  int max_active_path_index_;
};

CCL_NAMESPACE_END

#endif /* WITH_CUDA */
