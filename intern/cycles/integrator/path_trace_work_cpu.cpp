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
#include "render/scene.h"

#include "util/util_atomic.h"
#include "util/util_logging.h"
#include "util/util_tbb.h"

CCL_NAMESPACE_BEGIN

/* Create TBB arena for execution of path tracing and rendering tasks. */
static inline tbb::task_arena local_tbb_arena_create(const Device *device)
{
  /* TODO: limit this to number of threads of CPU device, it may be smaller than
   * the system number of threads when we reduce the number of CPU threads in
   * CPU + GPU rendering to dedicate some cores to handling the GPU device. */
  return tbb::task_arena(device->info.cpu_threads);
}

/* Get CPUKernelThreadGlobals for the current thread. */
static inline CPUKernelThreadGlobals *kernel_thread_globals_get(
    vector<CPUKernelThreadGlobals> &kernel_thread_globals)
{
  const int thread_index = tbb::this_task_arena::current_thread_index();
  DCHECK_GE(thread_index, 0);
  DCHECK_LE(thread_index, kernel_thread_globals.size());

  return &kernel_thread_globals[thread_index];
}

PathTraceWorkCPU::PathTraceWorkCPU(Device *device,
                                   DeviceScene *device_scene,
                                   RenderBuffers *buffers,
                                   bool *cancel_requested_flag)
    : PathTraceWork(device, device_scene, buffers, cancel_requested_flag),
      kernels_(*(device->get_cpu_kernels())),
      render_buffers_(buffers)
{
  DCHECK_EQ(device->info.type, DEVICE_CPU);
}

void PathTraceWorkCPU::init_execution()
{
  /* Cache per-thread kernel globals. */
  device_->get_cpu_kernel_thread_globals(kernel_thread_globals_);
}

void PathTraceWorkCPU::render_samples(int start_sample, int samples_num)
{
  const int64_t image_width = effective_buffer_params_.width;
  const int64_t image_height = effective_buffer_params_.height;
  const int64_t total_pixels_num = image_width * image_height;

  tbb::task_arena local_arena = local_tbb_arena_create(device_);
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
      work_tile.offset = effective_buffer_params_.offset;
      work_tile.stride = effective_buffer_params_.stride;

      CPUKernelThreadGlobals *kernel_globals = kernel_thread_globals_get(kernel_thread_globals_);

      render_samples_full_pipeline(kernel_globals, work_tile, samples_num);
    });
  });
}

void PathTraceWorkCPU::render_samples_full_pipeline(KernelGlobals *kernel_globals,
                                                    const KernelWorkTile &work_tile,
                                                    const int samples_num)
{
  const bool has_shadow_catcher = device_scene_->data.integrator.has_shadow_catcher;

  IntegratorState integrator_states[2];

  IntegratorState *state = &integrator_states[0];
  IntegratorState *shadow_catcher_state = &integrator_states[1];

  KernelWorkTile sample_work_tile = work_tile;
  float *render_buffer = render_buffers_->buffer.data();

  for (int sample = 0; sample < samples_num; ++sample) {
    if (is_cancel_requested()) {
      break;
    }

    if (!kernels_.integrator_init_from_camera(
            kernel_globals, state, &sample_work_tile, render_buffer)) {
      break;
    }

    kernels_.integrator_megakernel(kernel_globals, state, render_buffer);

    if (has_shadow_catcher) {
      kernels_.integrator_megakernel(kernel_globals, shadow_catcher_state, render_buffer);
    }

    ++sample_work_tile.start_sample;
  }
}

void PathTraceWorkCPU::copy_to_gpu_display(GPUDisplay *gpu_display, float sample_scale)
{
  const int full_x = effective_buffer_params_.full_x;
  const int full_y = effective_buffer_params_.full_y;
  const int width = effective_buffer_params_.width;
  const int height = effective_buffer_params_.height;
  const int offset = effective_buffer_params_.offset;
  const int stride = effective_buffer_params_.stride;

  half4 *rgba_half = gpu_display->map_texture_buffer();
  if (!rgba_half) {
    /* TODO(sergey): Look into using copy_to_gpu_display() if mapping failed. Might be needed for
     * some implementations of GPUDisplay which can not map memory? */
    return;
  }

  tbb::task_arena local_arena = local_tbb_arena_create(device_);
  local_arena.execute([&]() {
    tbb::parallel_for(0, height, [&](int y) {
      CPUKernelThreadGlobals *kernel_globals = kernel_thread_globals_get(kernel_thread_globals_);
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
  });

  gpu_display->unmap_texture_buffer();
}

int PathTraceWorkCPU::adaptive_sampling_converge_filter_count_active(float threshold, bool reset)
{
  const int full_x = effective_buffer_params_.full_x;
  const int full_y = effective_buffer_params_.full_y;
  const int width = effective_buffer_params_.width;
  const int height = effective_buffer_params_.height;
  const int offset = effective_buffer_params_.offset;
  const int stride = effective_buffer_params_.stride;

  float *render_buffer = render_buffers_->buffer.data();

  uint num_active_pixels = 0;

  tbb::task_arena local_arena = local_tbb_arena_create(device_);

  /* Check convergency and do x-filter in a single `parallel_for`, to reduce threading overhead. */
  local_arena.execute([&]() {
    tbb::parallel_for(full_y, full_y + height, [&](int y) {
      CPUKernelThreadGlobals *kernel_globals = &kernel_thread_globals_[0];

      bool row_converged = true;
      uint num_row_pixels_active = 0;
      for (int x = 0; x < width; ++x) {
        if (!kernels_.adaptive_sampling_convergence_check(
                kernel_globals, render_buffer, full_x + x, y, threshold, reset, offset, stride)) {
          ++num_row_pixels_active;
          row_converged = false;
        }
      }

      atomic_fetch_and_add_uint32(&num_active_pixels, num_row_pixels_active);

      if (!row_converged) {
        kernels_.adaptive_sampling_filter_x(
            kernel_globals, render_buffer, y, full_x, width, offset, stride);
      }
    });
  });

  if (num_active_pixels) {
    local_arena.execute([&]() {
      tbb::parallel_for(full_x, full_x + width, [&](int x) {
        CPUKernelThreadGlobals *kernel_globals = &kernel_thread_globals_[0];
        kernels_.adaptive_sampling_filter_y(
            kernel_globals, render_buffer, x, full_y, height, offset, stride);
      });
    });
  }

  return num_active_pixels;
}

CCL_NAMESPACE_END
