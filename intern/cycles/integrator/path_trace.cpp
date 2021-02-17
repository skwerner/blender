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
  /* Allocate render buffers for the current device. */
  RenderBuffers render_buffers(device);
  render_buffers.reset(buffer_params);

  /* Execution queue for the kernels from the path tracing graph. */
  unique_ptr<DeviceQueue> queue = device->queue_create_integrator(&render_buffers);
  DCHECK(queue);

  /* TODO(sergey): Determine possible queue/wavefront size. For the ideal performance the wavefront
   * should be as big as it is needed to have all device threads occupied. In practice, its size
   * might need to be smaller if the state does not fit into the memory. */

  /* Number of samples which happens in parallel for the work group on the device. */
  const int parallel_samples_num = 1;

  /* Number of samples which were path traces for the big tile. */
  int traced_samples_num = 0;

  while (traced_samples_num < samples_num) {
    if (is_cancel_requested()) {
      /* TODO(sergey): Either wait fore the current wavefront to be fully finished, or discard its
       * unfinished changes to the render buffer.
       * Feels like the current wavefront is the only thing we care about, since for the rest we
       * can do proper scaling using similar technique as for adaptive sampling. */
      break;
    }

    queue->enqueue(DeviceKernel::GENERATE_CAMERA_RAYS);

    bool have_alive_paths = false;
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

      /* TODO(sergey): perform actual check on number of "alive" paths in the wavefront.  */
      have_alive_paths = false;
    } while (have_alive_paths);

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

CCL_NAMESPACE_END
