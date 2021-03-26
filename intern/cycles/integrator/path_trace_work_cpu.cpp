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

#include "integrator/path_trace_work_cpu.h"

#include "device/cpu/kernel.h"
#include "device/device.h"

#include "render/buffers.h"
#include "render/gpu_display.h"

#include "util/util_logging.h"
#include "util/util_tbb.h"

CCL_NAMESPACE_BEGIN

PathTraceWorkCPU::PathTraceWorkCPU(Device *render_device,
                                   RenderBuffers *buffers,
                                   bool *cancel_requested_flag)
    : PathTraceWork(render_device, buffers, cancel_requested_flag),
      kernels_(*(render_device->get_cpu_kernels())),
      render_buffers_(buffers)
{
  DCHECK_EQ(render_device->info.type, DEVICE_CPU);
}

void PathTraceWorkCPU::init_execution()
{
  /* Cache per-thread kernel globals. */
  render_device_->get_cpu_kernel_thread_globals(kernel_thread_globals_);
}

void PathTraceWorkCPU::render_samples(int start_sample, int samples_num)
{
  const int64_t image_width = effective_buffer_params_.width;
  const int64_t image_height = effective_buffer_params_.height;
  const int64_t total_pixels_num = image_width * image_height;

  int offset, stride;
  effective_buffer_params_.get_offset_stride(offset, stride);

  /* TODO: limit this to number of threads of CPU device, it may be smaller than
   * the system number of threads when we reduce the number of CPU threads in
   * CPU + GPU rendering to dedicate some cores to handling the GPU device. */
  tbb::task_arena local_arena(render_device_->info.cpu_threads);
  local_arena.execute([&]() {
    tbb::parallel_for(int64_t(0), total_pixels_num, [&](int64_t work_index) {
      if (is_cancel_requested()) {
        return;
      }

      const int y = work_index / image_width;
      const int x = work_index - y * image_width;

      KernelWorkTile work_tile;
      work_tile.x = effective_buffer_params_.full_x + x;
      work_tile.y = effective_buffer_params_.full_y + y;
      work_tile.w = 1;
      work_tile.h = 1;
      work_tile.start_sample = start_sample;
      work_tile.num_samples = 1;
      work_tile.offset = offset;
      work_tile.stride = stride;
      work_tile.buffer = render_buffers_->buffer.data();

      const int thread_index = tbb::this_task_arena::current_thread_index();
      DCHECK_GE(thread_index, 0);
      DCHECK_LE(thread_index, kernel_thread_globals_.size());

      render_samples_full_pipeline(kernel_thread_globals_[thread_index], work_tile, samples_num);
    });
  });
}

void PathTraceWorkCPU::render_samples_full_pipeline(KernelGlobals &kernel_globals,
                                                    const KernelWorkTile &work_tile,
                                                    const int samples_num)
{
  IntegratorState integrator_state;
  IntegratorState *state = &integrator_state;

  KernelWorkTile sample_work_tile = work_tile;
  float *render_buffer = render_buffers_->buffer.data();

  for (int sample = 0; sample < samples_num; ++sample) {
    if (is_cancel_requested()) {
      break;
    }

    kernels_.integrator_init_from_camera(&kernel_globals, state, &sample_work_tile);
    kernels_.integrator_megakernel(&kernel_globals, state, render_buffer);

    ++sample_work_tile.start_sample;
  }
}

void PathTraceWorkCPU::copy_to_gpu_display(GPUDisplay *gpu_display, float sample_scale)
{
  const int full_x = effective_buffer_params_.full_x;
  const int full_y = effective_buffer_params_.full_y;
  const int width = effective_buffer_params_.width;
  const int height = effective_buffer_params_.height;

  half4 *rgba_half = gpu_display->map_texture_buffer();
  if (!rgba_half) {
    /* TODO(sergey): Look into using copy_to_gpu_display() if mapping failed. Might be needed for
     * some implementations of GPUDisplay which can not map memory? */
    return;
  }

  /* NOTE: This call is supposed to happen outside of any path tracing, so can pick any of the
   * pre-configured kernel globals. */
  KernelGlobals *kernel_globals = &kernel_thread_globals_[0];

  int offset, stride;
  effective_buffer_params_.get_offset_stride(offset, stride);

  tbb::parallel_for(0, height, [&](int y) {
    for (int x = 0; x < width; ++x) {
      kernels_.convert_to_half_float(kernel_globals,
                                     reinterpret_cast<uchar4 *>(rgba_half),
                                     reinterpret_cast<float *>(buffers_->buffer.device_pointer),
                                     sample_scale,
                                     full_x + x,
                                     full_y + y,
                                     offset,
                                     stride);
    }
  });

  gpu_display->unmap_texture_buffer();
}

CCL_NAMESPACE_END
