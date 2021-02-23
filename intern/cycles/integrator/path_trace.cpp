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

#include "integrator/path_trace.h"

#include "device/device.h"
#include "device/device_queue.h"
#include "util/util_logging.h"
#include "util/util_time.h"

CCL_NAMESPACE_BEGIN

PathTrace::PathTrace(Device *device, const BufferParams &buffer_params)
    : device_(device), buffer_params_(buffer_params)
{
  DCHECK_NE(device_, nullptr);
}

void PathTrace::render_samples(int samples_num)
{
  /* NOTE: Currently assume this is a single device.
   *
   * TODO(sergey): Support `DeviceMulti`, by creating per-device thread which will call the
   * `render_samples_on_device` for every device, allowing the kernel graph to track state of
   * deviced independently between each other.
   *
   * TODO(sergey): For the nulti-device case need to split `buffer_params_` so that every device
   * operates on its own dedicated area. */
  render_samples_on_device(device_, buffer_params_, samples_num);
}

void PathTrace::render_samples_on_device(Device *device,
                                         const BufferParams &buffer_params,
                                         int samples_num)
{
  update_reset_status();

  /* Allocate render buffers for the current device. */
  RenderBuffers render_buffers(device);
  render_buffers.reset(buffer_params);

  /* Execution queue for the kernels from the path tracing graph. */
  unique_ptr<DeviceQueue> queue = device->queue_create_integrator(&render_buffers);
  DCHECK(queue);

  /* TODO(sergey): Replace with proper WorkTile scheduling. */
  for (int sample = 0; sample < samples_num; ++sample) {
    for (int y = 0; y < buffer_params.full_height; ++y) {
      for (int x = 0; x < buffer_params.full_width; ++x) {
        render_work_on_queue(queue.get(), x, y, sample, 1);
      }
      /* TODO(sergey): Properly handle render cancel request. Perhaps needs to happen here rather
       * than in the graph evaluation. */
    }

    /* TODO(sergey): Take user cancel into account. */
    update_if_needed(&render_buffers, sample + 1);
  }

  /* TODO(sergey): Take adaptive stopping and user cancel into account. Both of these actions will
   * affect how the buffer is to be scaled. */
  write(&render_buffers, samples_num);
}

void PathTrace::render_work_on_queue(
    DeviceQueue *queue, int x, int y, int start_sample, int samples_num)
{
  /* TODO(sergey): Determine possible queue/wavefront size. For the ideal performance the wavefront
   * should be as big as it is needed to have all device threads occupied. In practice, its size
   * might need to be smaller if the state does not fit into the memory. */

  /* Number of samples which happens in parallel for the work group on the device. */
  const int parallel_samples_num = 1;

  /* Number of samples which were path traces for the big tile. */
  int traced_samples_num = 0;

  while (traced_samples_num < samples_num) {
    DeviceWorkTile work_tile;
    work_tile.x = x;
    work_tile.y = y;
    work_tile.width = 1;
    work_tile.height = 1;
    work_tile.sample = start_sample + traced_samples_num;
    queue->set_work_tile(work_tile);

    if (is_cancel_requested()) {
      /* TODO(sergey): Either wait fore the current wavefront to be fully finished, or discard its
       * unfinished changes to the render buffer.
       * Feels like the current wavefront is the only thing we care about, since for the rest we
       * can do proper scaling using similar technique as for adaptive sampling. */
      break;
    }

    queue->enqueue(DeviceKernel::GENERATE_CAMERA_RAYS);

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

      queue->enqueue(DeviceKernel::INTERSECT_CLOSEST);

      queue->enqueue(DeviceKernel::VOLUME);
      queue->enqueue(DeviceKernel::BACKGROUND);

      queue->enqueue(DeviceKernel::SURFACE);
      queue->enqueue(DeviceKernel::SUBSURFACE);

      queue->enqueue(DeviceKernel::INTERSECT_SHADOW);
      queue->enqueue(DeviceKernel::SHADOW);
    } while (queue->has_work_remaining());

    traced_samples_num += parallel_samples_num;
  }
}

bool PathTrace::is_cancel_requested()
{
  if (!get_cancel_cb) {
    return false;
  }
  return get_cancel_cb();
}

void PathTrace::update_reset_status()
{
  update_status.has_update = false;
}

void PathTrace::update_if_needed(RenderBuffers *render_buffers, int sample)
{
  if (!update_cb) {
    return;
  }

  const double current_time = time_dt();

  /* Always perform the first update, so that users see first pixels as soon as possible.
   * After that only perform updates every now and then. */
  if (update_status.has_update) {
    /* TODO(sergey): Use steady clock. */
    if (current_time - update_status.last_update_time < update_interval_in_seconds) {
      return;
    }
  }

  update_cb(render_buffers, sample);

  update_status.has_update = true;
  update_status.last_update_time = current_time;
}

void PathTrace::write(RenderBuffers *render_buffers, int sample)
{
  if (!write_cb) {
    return;
  }

  write_cb(render_buffers, sample);
}

CCL_NAMESPACE_END
