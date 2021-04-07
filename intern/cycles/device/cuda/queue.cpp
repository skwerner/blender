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

#ifdef WITH_CUDA

#  include "device/cuda/queue.h"
#  include "device/cuda/device_impl.h"
#  include "device/cuda/kernel.h"

#  include "util/util_time.h"

CCL_NAMESPACE_BEGIN

/* CUDADeviceQueue */

CUDADeviceQueue::CUDADeviceQueue(CUDADevice *device)
    : DeviceQueue(device), cuda_device_(device), last_sync_time_(0.0)
{
}

void CUDADeviceQueue::init_execution()
{
  /* Synchronize all textures and memory copies before executing task. */
  CUDAContextScope scope(cuda_device_);
  cuda_device_->load_texture_info();
  cuda_device_assert(cuda_device_, cuCtxSynchronize());

  last_sync_time_ = time_dt();
}

bool CUDADeviceQueue::enqueue(DeviceKernel kernel, const int work_size, void *args[])
{
  if (cuda_device_->have_error()) {
    return false;
  }

  VLOG(3) << "CUDA queue launch " << device_kernel_as_string(kernel) << ", work_size "
          << work_size;

  const CUDAContextScope scope(cuda_device_);
  const CUDADeviceKernel &cuda_kernel = cuda_device_->kernels.get(kernel);

  /* Compute kernel launch parameters. */
  const int num_threads_per_block = cuda_kernel.num_threads_per_block;
  const int num_blocks = divide_up(work_size, num_threads_per_block);

  int shared_mem_bytes = 0;

  switch (kernel) {
    case DEVICE_KERNEL_INTEGRATOR_QUEUED_PATHS_ARRAY:
    case DEVICE_KERNEL_INTEGRATOR_QUEUED_SHADOW_PATHS_ARRAY:
    case DEVICE_KERNEL_INTEGRATOR_TERMINATED_PATHS_ARRAY:
    case DEVICE_KERNEL_INTEGRATOR_SORTED_PATHS_ARRAY:
      /* See parall_active_index.h for why this amount of shared memory is needed. */
      shared_mem_bytes = (num_threads_per_block + 1) * sizeof(int);
      break;
    case DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA:
    case DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST:
    case DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW:
    case DEVICE_KERNEL_INTEGRATOR_INTERSECT_SUBSURFACE:
    case DEVICE_KERNEL_INTEGRATOR_SHADE_BACKGROUND:
    case DEVICE_KERNEL_INTEGRATOR_SHADE_LIGHT:
    case DEVICE_KERNEL_INTEGRATOR_SHADE_SHADOW:
    case DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE:
    case DEVICE_KERNEL_INTEGRATOR_SHADE_VOLUME:
    case DEVICE_KERNEL_INTEGRATOR_MEGAKERNEL:
    case DEVICE_KERNEL_SHADER_EVAL_DISPLACE:
    case DEVICE_KERNEL_SHADER_EVAL_BACKGROUND:
    case DEVICE_KERNEL_CONVERT_TO_HALF_FLOAT:
    case DEVICE_KERNEL_ADAPTIVE_SAMPLING_CONVERGENCE_CHECK:
    case DEVICE_KERNEL_ADAPTIVE_SAMPLING_CONVERGENCE_FILTER_X:
    case DEVICE_KERNEL_ADAPTIVE_SAMPLING_CONVERGENCE_FILTER_Y:
    case DEVICE_KERNEL_FILTER_CONVERT_TO_RGB:
    case DEVICE_KERNEL_FILTER_CONVERT_FROM_RGB:
    case DEVICE_KERNEL_PREFIX_SUM:
    case DEVICE_KERNEL_NUM:
      break;
  }

  /* Launch kernel. */
  cuda_device_assert(cuda_device_,
                     cuLaunchKernel(cuda_kernel.function,
                                    num_blocks,
                                    1,
                                    1,
                                    num_threads_per_block,
                                    1,
                                    1,
                                    shared_mem_bytes,
                                    0,
                                    args,
                                    0));

  return !(cuda_device_->have_error());
}

bool CUDADeviceQueue::synchronize()
{
  if (cuda_device_->have_error()) {
    return false;
  }

  const CUDAContextScope scope(cuda_device_);
  cuda_device_assert(cuda_device_, cuCtxSynchronize());

  double new_time = time_dt();
  VLOG(3) << "CUDA queue synchronize, elapsed " << new_time - last_sync_time_ << "s";
  last_sync_time_ = new_time;

  return !(cuda_device_->have_error());
}

CCL_NAMESPACE_END

#endif /* WITH_CUDA */
