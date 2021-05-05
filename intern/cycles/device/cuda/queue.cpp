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

CCL_NAMESPACE_BEGIN

/* CUDADeviceQueue */

CUDADeviceQueue::CUDADeviceQueue(CUDADevice *device)
    : DeviceQueue(device), cuda_device_(device), cuda_stream_(nullptr)
{
  const CUDAContextScope scope(cuda_device_);
  cuda_device_assert(cuda_device_, cuStreamCreate(&cuda_stream_, CU_STREAM_NON_BLOCKING));
}

CUDADeviceQueue::~CUDADeviceQueue()
{
  const CUDAContextScope scope(cuda_device_);
  cuStreamDestroy(cuda_stream_);
}

int CUDADeviceQueue::num_concurrent_states(const size_t /*state_size*/) const
{
  /* TODO: compute automatically. */
  /* TODO: must have at least num_threads_per_block. */
  return 1048576;
}

int CUDADeviceQueue::num_concurrent_busy_states()
{
  const int max_num_threads = cuda_device_->get_num_multiprocessors() *
                              cuda_device_->get_max_num_threads_per_multiprocessor();

  if (max_num_threads == 0) {
    return 65536;
  }

  return 4 * max_num_threads;
}

void CUDADeviceQueue::init_execution()
{
  /* Synchronize all textures and memory copies before executing task. */
  CUDAContextScope scope(cuda_device_);
  cuda_device_->load_texture_info();
  cuda_device_assert(cuda_device_, cuCtxSynchronize());

  debug_init_execution();
}

bool CUDADeviceQueue::kernel_available(DeviceKernel kernel) const
{
  return cuda_device_->kernels.available(kernel);
}

bool CUDADeviceQueue::enqueue(DeviceKernel kernel, const int work_size, void *args[])
{
  if (cuda_device_->have_error()) {
    return false;
  }

  debug_enqueue(kernel, work_size);

  const CUDAContextScope scope(cuda_device_);
  const CUDADeviceKernel &cuda_kernel = cuda_device_->kernels.get(kernel);

  /* Compute kernel launch parameters. */
  const int num_threads_per_block = cuda_kernel.num_threads_per_block;
  const int num_blocks = divide_up(work_size, num_threads_per_block);

  int shared_mem_bytes = 0;

  switch (kernel) {
    case DEVICE_KERNEL_INTEGRATOR_QUEUED_PATHS_ARRAY:
    case DEVICE_KERNEL_INTEGRATOR_QUEUED_SHADOW_PATHS_ARRAY:
    case DEVICE_KERNEL_INTEGRATOR_ACTIVE_PATHS_ARRAY:
    case DEVICE_KERNEL_INTEGRATOR_TERMINATED_PATHS_ARRAY:
    case DEVICE_KERNEL_INTEGRATOR_SORTED_PATHS_ARRAY:
    case DEVICE_KERNEL_INTEGRATOR_COMPACT_PATHS_ARRAY:
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
    case DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE_RAYTRACE:
    case DEVICE_KERNEL_INTEGRATOR_SHADE_VOLUME:
    case DEVICE_KERNEL_INTEGRATOR_MEGAKERNEL:
    case DEVICE_KERNEL_INTEGRATOR_COMPACT_STATES:
    case DEVICE_KERNEL_INTEGRATOR_RESET:
    case DEVICE_KERNEL_SHADER_EVAL_DISPLACE:
    case DEVICE_KERNEL_SHADER_EVAL_BACKGROUND:
    case DEVICE_KERNEL_FILM_CONVERT_DEPTH_HALF_RGBA:
    case DEVICE_KERNEL_FILM_CONVERT_MIST_HALF_RGBA:
    case DEVICE_KERNEL_FILM_CONVERT_SAMPLE_COUNT_HALF_RGBA:
    case DEVICE_KERNEL_FILM_CONVERT_FLOAT_HALF_RGBA:
    case DEVICE_KERNEL_FILM_CONVERT_SHADOW3_HALF_RGBA:
    case DEVICE_KERNEL_FILM_CONVERT_DIVIDE_EVEN_COLOR_HALF_RGBA:
    case DEVICE_KERNEL_FILM_CONVERT_FLOAT3_HALF_RGBA:
    case DEVICE_KERNEL_FILM_CONVERT_SHADOW4_HALF_RGBA:
    case DEVICE_KERNEL_FILM_CONVERT_MOTION_HALF_RGBA:
    case DEVICE_KERNEL_FILM_CONVERT_CRYPTOMATTE_HALF_RGBA:
    case DEVICE_KERNEL_FILM_CONVERT_DENOISING_COLOR_HALF_RGBA:
    case DEVICE_KERNEL_FILM_CONVERT_SHADOW_CATCHER_HALF_RGBA:
    case DEVICE_KERNEL_FILM_CONVERT_SHADOW_CATCHER_MATTE_WITH_SHADOW_HALF_RGBA:
    case DEVICE_KERNEL_FILM_CONVERT_FLOAT4_HALF_RGBA:
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
                                    cuda_stream_,
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
  cuda_device_assert(cuda_device_, cuStreamSynchronize(cuda_stream_));
  debug_synchronize();

  return !(cuda_device_->have_error());
}

void CUDADeviceQueue::zero_to_device(device_memory &mem)
{
  assert(mem.type != MEM_GLOBAL && mem.type != MEM_TEXTURE);

  if (mem.memory_size() == 0) {
    return;
  }

  /* Allocate on demand. */
  if (mem.device_pointer == 0) {
    cuda_device_->mem_alloc(mem);
  }

  /* Zero memory on device. */
  assert(mem.device_pointer != 0);

  const CUDAContextScope scope(cuda_device_);
  cuda_device_assert(
      cuda_device_,
      cuMemsetD8Async((CUdeviceptr)mem.device_pointer, 0, mem.memory_size(), cuda_stream_));
}

void CUDADeviceQueue::copy_to_device(device_memory &mem)
{
  assert(mem.type != MEM_GLOBAL && mem.type != MEM_TEXTURE);

  if (mem.memory_size() == 0) {
    return;
  }

  /* Allocate on demand. */
  if (mem.device_pointer == 0) {
    cuda_device_->mem_alloc(mem);
  }

  assert(mem.device_pointer != 0);
  assert(mem.host_pointer != nullptr);

  /* Copy memory to device. */
  const CUDAContextScope scope(cuda_device_);
  cuda_device_assert(
      cuda_device_,
      cuMemcpyHtoDAsync(
          (CUdeviceptr)mem.device_pointer, mem.host_pointer, mem.memory_size(), cuda_stream_));
}

void CUDADeviceQueue::copy_from_device(device_memory &mem)
{
  assert(mem.type != MEM_GLOBAL && mem.type != MEM_TEXTURE);

  if (mem.memory_size() == 0) {
    return;
  }

  assert(mem.device_pointer != 0);
  assert(mem.host_pointer != nullptr);

  /* Copy memory from device. */
  const CUDAContextScope scope(cuda_device_);
  cuda_device_assert(
      cuda_device_,
      cuMemcpyDtoHAsync(
          mem.host_pointer, (CUdeviceptr)mem.device_pointer, mem.memory_size(), cuda_stream_));
}

CCL_NAMESPACE_END

#endif /* WITH_CUDA */
