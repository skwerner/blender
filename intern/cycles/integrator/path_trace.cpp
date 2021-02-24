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
#include "util/util_logging.h"
#include "util/util_tbb.h"
#include "util/util_time.h"

CCL_NAMESPACE_BEGIN

void PathTrace::RenderStatus::reset()
{
  rendered_samples_num = 0;
}

void PathTrace::UpdateStatus::reset()
{
  has_update = false;
}

PathTrace::PathTrace(Device *device, const BufferParams &full_buffer_params)
    : device_(device), pass_stride_(full_buffer_params.get_passes_size())
{
  DCHECK_NE(device_, nullptr);

  render_status.full_render_buffers = make_unique<RenderBuffers>(device);
  render_status.full_render_buffers->reset(full_buffer_params);

  /* Create path tracing contexts in advance, so that they can be reused by incremental sampling
   * as much as possible. */
  /* TODO(sergey): Support devices which can have multiple queues running in parallel. This would
   * be, for example, a CPU device which might want to have asynchronous queues per CPU thread. */
  device->foreach_device([&](Device *render_device) {
    /* For tests one can add `for (int i = 0; i < 64; ++i)` prior to the statement below and have
     * multi-threaded rendering on CPU. */

    path_trace_contexts_.push_back(make_unique<PathTraceContext>(render_device));
  });

  /* TODO(sergey): Communicate some scheduling block size to the work scheduler based on every
   * device's get_max_num_path_states(). This is a bit tricky because CPU and GPU device will
   * be opposites of each other: CPU wavefront is super tiny, and GPU wavefront is gigantic.
   * How to find an ideal scheduling for such a mixture?  */
}

void PathTrace::render_samples(int samples_num)
{
  render_status.reset();
  update_status.reset();

  /* TODO(sergey): Dp something smarter, like:
   * - Render first sample and update the interface, so user sees first pixels as soon as possible.
   * - Render in a bigger chunks of samples for the performance reasons. */

  for (int sample = 0; sample < samples_num; ++sample) {
    /* TODO(sergey): Take adaptive stopping and user cancel into account. Both of these actions
     * will affect how the buffer is to be scaled. */

    render_samples_full_pipeline(1);
    update_if_needed();

    if (is_cancel_requested()) {
      break;
    }
  }

  /* TODO(sergey): Need to write to the whole buffer, after all devices sampled the frame to the
   * given number of samples. */
  write();
}

/* XXX: Part of an experiment and transitional code to support multi-device.
 * Copies `src` (which can be thought as a smaller tile inside of `dst`) into `dst`.
 */
static void accumulate_buffer(RenderBuffers *dst, RenderBuffers *src, const int pass_stride)
{
  int dst_offset, dst_stride;
  dst->params.get_offset_stride(dst_offset, dst_stride);

  int src_offset, src_stride;
  src->params.get_offset_stride(src_offset, src_stride);

  for (int src_y = 0; src_y < src->params.height; ++src_y) {
    const int src_x = 0;
    const int dst_x = src->params.full_x + src_x;
    const int dst_y = src->params.full_y + src_y;

    const int src_pixel_offset = (src_y * src_stride + src_x) * pass_stride;
    const int dst_pixel_offset = (dst_y * dst_stride + dst_x) * pass_stride;

    for (int i = 0; i < pass_stride; ++i) {
      *(dst->buffer.data() + dst_pixel_offset + i) += *(src->buffer.data() + src_pixel_offset + i);
    }
  }
}

void PathTrace::render_samples_full_pipeline(int samples_num)
{
  /* Reset work scheduler, so that it is ready to give work tiles for the new samples range. */
  const BufferParams &full_buffer_params = render_status.full_render_buffers->params;
  work_scheduler_.reset(full_buffer_params.full_width,
                        full_buffer_params.full_height,
                        render_status.rendered_samples_num,
                        samples_num);

  tbb::parallel_for_each(path_trace_contexts_,
                         [&](unique_ptr<PathTraceContext> &path_trace_context) {
                           render_samples_full_pipeline(path_trace_context.get());
                         });

  render_status.rendered_samples_num += samples_num;
}

void PathTrace::render_samples_full_pipeline(PathTraceContext *path_trace_context)
{
  DeviceWorkTile work_tile;
  while (work_scheduler_.get_work(&work_tile)) {
    render_samples_full_pipeline(path_trace_context, work_tile);

    /* XXX: This is annoying and feels wrong. But not sure yet what is the proper way to have
     * device-side buffer for path tracing and a big-tile buffer. */
    accumulate_buffer(render_status.full_render_buffers.get(),
                      &path_trace_context->render_buffers,
                      pass_stride_);
  }
}

void PathTrace::render_samples_full_pipeline(PathTraceContext *path_trace_context,
                                             const DeviceWorkTile &work_tile)
{
  /* TODO(sergey): This is rather expensive since it involves a temporary copy of passes and
   * re-calculation of full passes size. In theory we can only do it once per PathTraceContext
   * lifetime based on the number of path states in the device (which does not change during
   * rendering) and only move the tile's position here.  */
  BufferParams buffer_params = render_status.full_render_buffers->params;
  buffer_params.width = work_tile.width;
  buffer_params.height = work_tile.height;
  buffer_params.full_x = work_tile.x;
  buffer_params.full_y = work_tile.y;
  path_trace_context->render_buffers.reset(buffer_params);

  DeviceQueue *queue = path_trace_context->queue.get();

  queue->set_work_tile(work_tile);

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
}

bool PathTrace::is_cancel_requested()
{
  if (!get_cancel_cb) {
    return false;
  }
  return get_cancel_cb();
}

void PathTrace::update_if_needed()
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

  update_cb(render_status.full_render_buffers.get(), render_status.rendered_samples_num);

  update_status.has_update = true;
  update_status.last_update_time = current_time;
}

void PathTrace::write()
{
  if (!write_cb) {
    return;
  }

  write_cb(render_status.full_render_buffers.get(), render_status.rendered_samples_num);
}

CCL_NAMESPACE_END
