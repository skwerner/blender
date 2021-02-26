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

#include "device/cuda/queue.h"
#include "device/cuda/device_cuda_impl.h"
#include "device/cuda/kernel.h"

#include "render/buffers.h"

CCL_NAMESPACE_BEGIN

/* CUDADeviceQueue */

CUDADeviceQueue::CUDADeviceQueue(CUDADevice *device) : DeviceQueue(device), cuda_device_(device)
{
}

void CUDADeviceQueue::init_execution()
{
  /* Synchronize all textures and memory copies before executing task. */
  CUDAContextScope scope(cuda_device_);
  cuda_device_->load_texture_info();
  cuda_device_assert(cuda_device_, cuCtxSynchronize());
}

/* CUDAIntegratorQueue */

CUDAIntegratorQueue::CUDAIntegratorQueue(CUDADevice *device, RenderBuffers *render_buffers)
    : CUDADeviceQueue(device),
      render_buffers_(render_buffers),
      integrator_state_(device, "integrator_state"),
      work_tile_(device, "work_tile", MEM_READ_WRITE)
{
  integrator_state_.alloc_to_device(get_max_num_path_states());
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

  /* TODO(sergey): Avoid temporary variable by making sign match between device and kernel. */
  int offset, stride;
  render_buffers->params.get_offset_stride(offset, stride);

  kernel_work_tile.offset = offset;
  kernel_work_tile.stride = stride;

  kernel_work_tile.buffer = (float *)(CUdeviceptr)render_buffers->buffer.device_pointer;

  return kernel_work_tile;
}

void CUDAIntegratorQueue::enqueue(DeviceKernel kernel)
{
  const CUDAContextScope scope(cuda_device_);
  const CUDADeviceKernel &cuda_kernel = cuda_device_->kernels.get(kernel);

  /* Get work size parameters for kernel execution. */
  const KernelWorkTile *wtile = work_tile_.data();
  const int num_threads_per_block = cuda_kernel.num_threads_per_block;
  const int total_work_size = wtile->w * wtile->h * wtile->num_samples;
  const int num_blocks = divide_up(total_work_size, num_threads_per_block);

  assert(total_work_size < get_max_num_path_states());

  switch (kernel) {
    case DeviceKernel::INTEGRATOR_INIT_FROM_CAMERA: {
      /* Generate camera ray kernel with work tile. */
      CUdeviceptr d_integrator_state = (CUdeviceptr)integrator_state_.device_pointer;
      CUdeviceptr d_work_tile = (CUdeviceptr)work_tile_.device_pointer;
      int total_work_size = 1;
      void *args[] = {&d_integrator_state, &d_work_tile, &total_work_size};

      cuda_device_assert(
          cuda_device_,
          cuLaunchKernel(
              cuda_kernel.function, num_blocks, 1, 1, num_threads_per_block, 1, 1, 0, 0, args, 0));
      break;
    }
    case DeviceKernel::INTEGRATOR_INTERSECT_CLOSEST:
    case DeviceKernel::INTEGRATOR_INTERSECT_SHADOW:
    case DeviceKernel::INTEGRATOR_INTERSECT_SUBSURFACE: {
      /* Ray intersection kernels with integrator state. */
      CUdeviceptr d_integrator_state = (CUdeviceptr)integrator_state_.device_pointer;
      void *args[] = {&d_integrator_state};

      cuda_device_assert(
          cuda_device_,
          cuLaunchKernel(
              cuda_kernel.function, num_blocks, 1, 1, num_threads_per_block, 1, 1, 0, 0, args, 0));
      break;
    }
    case DeviceKernel::INTEGRATOR_SHADE_BACKGROUND:
    case DeviceKernel::INTEGRATOR_SHADE_SHADOW:
    case DeviceKernel::INTEGRATOR_SHADE_SURFACE:
    case DeviceKernel::INTEGRATOR_SHADE_VOLUME: {
      /* Shading kernels with integrator state and render buffer. */
      CUdeviceptr d_integrator_state = (CUdeviceptr)integrator_state_.device_pointer;
      CUdeviceptr d_render_buffer = (CUdeviceptr)render_buffers_->buffer.device_pointer;
      void *args[] = {&d_integrator_state, &d_render_buffer};

      cuda_device_assert(
          cuda_device_,
          cuLaunchKernel(
              cuda_kernel.function, num_blocks, 1, 1, num_threads_per_block, 1, 1, 0, 0, args, 0));
      break;
    }
    case DeviceKernel::NUM_KERNELS: {
      break;
    }
  }
}

void CUDAIntegratorQueue::set_work_tile(const DeviceWorkTile &work_tile)
{
  work_tile_.alloc(1);
  *(work_tile_.data()) = init_kernel_work_tile(render_buffers_, work_tile);
  work_tile_.copy_to_device();
}

bool CUDAIntegratorQueue::has_work_remaining()
{
  const CUDAContextScope scope(cuda_device_);
  cuda_device_assert(cuda_device_, cuCtxSynchronize());
  return false; /* TODO */
}

int CUDAIntegratorQueue::get_max_num_path_states()
{
  return 131072; /* TODO */
}

CCL_NAMESPACE_END
