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

#include "integrator/path_trace_work_pixel.h"

#include "device/cpu/kernel.h"
#include "device/device.h"

#include "render/buffers.h"

#include "util/util_logging.h"
#include "util/util_tbb.h"

CCL_NAMESPACE_BEGIN

PathTraceWorkPixel::PathTraceWorkPixel(Device *render_device,
                                       RenderBuffers *buffers,
                                       bool *cancel_requested_flag)
    : PathTraceWork(render_device, buffers, cancel_requested_flag),
      kernels_(*(render_device->get_cpu_kernels())),
      render_buffers_(buffers)
{
  DCHECK_EQ(render_device->info.type, DEVICE_CPU);
}

void PathTraceWorkPixel::init_execution()
{
  /* Cache per-thread kernel globals. */
  const KernelGlobals &kernel_globals = *(render_device_->get_cpu_kernel_globals());
  void *osl_memory = render_device_->get_cpu_osl_memory();

  kernel_thread_globals_.clear();
  for (int i = 0; i < render_device_->info.cpu_threads; i++) {
    kernel_thread_globals_.emplace_back(CPUKernelThreadGlobals(kernel_globals, osl_memory));
  }
}

void PathTraceWorkPixel::render_samples(const BufferParams &scaled_render_buffer_params,
                                        int start_sample,
                                        int samples_num)
{
  const int64_t image_width = scaled_render_buffer_params.width;
  const int64_t image_height = scaled_render_buffer_params.height;
  const int64_t total_pixels_num = image_width * image_height;

  int offset, stride;
  scaled_render_buffer_params.get_offset_stride(offset, stride);

  /* TODO: limit this to number of threads of CPU device, it may be smaller than
   * the system number of threads when we reduce the number of CPU threads in
   * CPU + GPU rendering to dedicate some cores to handling the GPU device. */
  tbb::parallel_for(int64_t(0), total_pixels_num, [&](int64_t work_index) {
    if (is_cancel_requested()) {
      return;
    }

    const int y = work_index / image_width;
    const int x = work_index - y * image_width;

    KernelWorkTile work_tile;
    work_tile.x = scaled_render_buffer_params.full_x + x;
    work_tile.y = scaled_render_buffer_params.full_y + y;
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
}

void PathTraceWorkPixel::render_samples_full_pipeline(KernelGlobals &kernel_globals,
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

#if 0
    /* NOTE: The order of queuing is based on the following ideas:
     *  - It is possible that some rays will hit background, and and of them will need volume
     *    attenuation. So first do intersect which allows to see which rays hit background,
     * then do volume kernel which might enqueue background work items. After that the
     * background kernel will handle work items coming from both intersection and volume
     * kernels.
     *
     *  - Subsurface kernel might enqueue additional shadow work items, so make it so shadow
     *    intersection kernel is scheduled after work items are scheduled from both surface and
     *    subsurface kernels. */
    while (!INTEGRATOR_PATH_IS_TERMINATED)
    {
      kernels_.integrator_intersect_closest(&kernel_globals, state);
      kernels_.integrator_shade_volume(&kernel_globals, state, render_buffer);
      kernels_.integrator_shade_background(&kernel_globals, state, render_buffer);
      kernels_.integrator_shade_light(&kernel_globals, state, render_buffer);
      kernels_.integrator_shade_surface(&kernel_globals, state, render_buffer);
      kernels_.integrator_intersect_subsurface(&kernel_globals, state);

      while (!INTEGRATOR_SHADOW_PATH_IS_TERMINATED) {
        kernels_.integrator_intersect_shadow(&kernel_globals, state);
        kernels_.integrator_shade_shadow(&kernel_globals, state, render_buffer);
      }
    }
#else
    kernels_.integrator_megakernel(&kernel_globals, state, render_buffer);
#endif

    ++sample_work_tile.start_sample;
  }
}

CCL_NAMESPACE_END
