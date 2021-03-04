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

#include "device/device.h"
#include "render/buffers.h"
#include "util/util_logging.h"
#include "util/util_tbb.h"

CCL_NAMESPACE_BEGIN

PathTraceWorkPixel::PathTraceWorkPixel(Device *render_device, RenderBuffers *buffers)
    : PathTraceWork(render_device, buffers)
{
  DCHECK_EQ(render_device->info.type, DEVICE_CPU);

  const int num_queues = render_device->get_concurrent_integrator_queues_num();
  for (int i = 0; i < num_queues; ++i) {
    integrator_queues_.emplace_back(render_device->queue_create_integrator(buffers_));
  }
}

void PathTraceWorkPixel::init_execution()
{
  for (auto &&queue : integrator_queues_) {
    queue->init_execution();
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

  tbb::parallel_for(int64_t(0), total_pixels_num, [&](int64_t work_index) {
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

    const int thread_index = tbb::this_task_arena::current_thread_index();
    DCHECK_GE(thread_index, 0);
    DCHECK_LE(thread_index, integrator_queues_.size());

    render_samples_full_pipeline(integrator_queues_[thread_index].get(), work_tile, samples_num);
  });
}

void PathTraceWorkPixel::render_samples_full_pipeline(DeviceQueue *queue,
                                                      const KernelWorkTile &work_tile,
                                                      const int samples_num)
{
  KernelWorkTile sample_work_tile = work_tile;

  for (int sample = 0; sample < samples_num; ++sample) {
    queue->set_work_tile(sample_work_tile);

    queue->enqueue(DeviceKernel::INTEGRATOR_INIT_FROM_CAMERA);

    do {
      /* NOTE: The order of queuing is based on the following ideas:
       *  - It is possible that some rays will hit background, and and of them will need volume
       *    attenuation. So first do intersect which allows to see which rays hit background, then
       *    do volume kernel which might enqueue background work items. After that the background
       *    kernel will handle work items coming from both intersection and volume kernels.
       *
       *  - Subsurface kernel might enqueue additional shadow work items, so make it so shadow
       *    intersection kernel is scheduled after work items are scheduled from both surface and
       *    subsurface kernels. */

      /* TODO(sergey): For the final implementation can do something smarter, like re-generating
       * camera rays if the wavefront becomes too small but there are still a lot of samples to be
       * calculated. */

      queue->enqueue(DeviceKernel::INTEGRATOR_INTERSECT_CLOSEST);

      queue->enqueue(DeviceKernel::INTEGRATOR_SHADE_VOLUME);
      queue->enqueue(DeviceKernel::INTEGRATOR_SHADE_BACKGROUND);

      queue->enqueue(DeviceKernel::INTEGRATOR_SHADE_SURFACE);
      queue->enqueue(DeviceKernel::INTEGRATOR_INTERSECT_SUBSURFACE);

      queue->enqueue(DeviceKernel::INTEGRATOR_INTERSECT_SHADOW);
      queue->enqueue(DeviceKernel::INTEGRATOR_SHADE_SHADOW);
    } while (queue->has_work_remaining());

    ++sample_work_tile.start_sample;
  }
}

CCL_NAMESPACE_END
