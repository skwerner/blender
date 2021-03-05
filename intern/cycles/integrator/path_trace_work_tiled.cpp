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

#include "integrator/path_trace_work_tiled.h"

#include "device/device.h"
#include "render/buffers.h"
#include "util/util_logging.h"
#include "util/util_tbb.h"
#include "util/util_time.h"

#include "kernel/kernel_types.h"

CCL_NAMESPACE_BEGIN

PathTraceWorkTiled::PathTraceWorkTiled(Device *render_device,
                                       RenderBuffers *buffers,
                                       bool *cancel_requested_flag)
    : PathTraceWork(render_device, buffers, cancel_requested_flag)
{
  const int num_queues = render_device->get_concurrent_integrator_queues_num();

  DCHECK_GT(num_queues, 0);

  for (int i = 0; i < num_queues; ++i) {
    integrator_queues_.emplace_back(render_device->queue_create_integrator(buffers_));
  }

  /* NOTE: Expect that all queues have the same number of path states. */
  work_scheduler_.set_max_num_path_states(integrator_queues_[0]->get_max_num_paths());
}

void PathTraceWorkTiled::init_execution()
{
  for (auto &&queue : integrator_queues_) {
    queue->init_execution();
  }
}

void PathTraceWorkTiled::render_samples(const BufferParams &scaled_render_buffer_params,
                                        int start_sample,
                                        int samples_num)
{
  work_scheduler_.reset(scaled_render_buffer_params, start_sample, samples_num);

  tbb::parallel_for_each(integrator_queues_, [&](unique_ptr<DeviceQueue> &queue) {
    render_samples_full_pipeline(queue.get());
  });
}

void PathTraceWorkTiled::render_samples_full_pipeline(DeviceQueue *queue)
{
  const float megakernel_threshold = 0.1f;

  while (true) {
    if (is_cancel_requested()) {
      break;
    }

    vector<KernelWorkTile> work_tiles;

    /* Get work tiles until we reach the max number of paths we can render. */
    const int max_num_paths = queue->get_max_num_paths();
    int num_paths = 0;
    while (num_paths < max_num_paths) {
      KernelWorkTile work_tile;
      if (work_scheduler_.get_work(&work_tile, max_num_paths - num_paths)) {
        work_tiles.push_back(work_tile);
        num_paths += work_tile.w * work_tile.h * work_tile.num_samples;
      }
      else {
        break;
      }
    }

    /* If we couldn't get any more tiles, we're done. */
    if (work_tiles.size() == 0) {
      break;
    }

    /* Initialize paths from work tiles. */
    queue->enqueue_work_tiles(
        DeviceKernel::INTEGRATOR_INIT_FROM_CAMERA, work_tiles.data(), work_tiles.size());

    while (true) {
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

      const int num_active = queue->get_num_active_paths();

      if (num_active == 0) {
        break;
      }
      else if (num_active < megakernel_threshold * max_num_paths) {
        /* TODO: limit number of iterations to keep GPU responsive? */
        queue->enqueue(DeviceKernel::INTEGRATOR_MEGAKERNEL);
        break;
      }
    }
  }
}

CCL_NAMESPACE_END
