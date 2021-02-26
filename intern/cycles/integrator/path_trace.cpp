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
#include "util/util_algorithm.h"
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

PathTrace::PathTrace(Device *device) : device_(device)
{
  DCHECK_NE(device_, nullptr);

  full_render_buffers_ = make_unique<RenderBuffers>(device);

  /* Create integrator queues in advance, so that they can be reused by incremental sampling
   * as much as possible. */
  device->foreach_device([&](Device *render_device) {
    const int num_queues = render_device->get_concurrent_integrator_queues_num();

    for (int i = 0; i < num_queues; ++i) {
      integrator_queues_.emplace_back(
          render_device->queue_create_integrator(full_render_buffers_.get()));
    }
  });

  /* TODO(sergey): Communicate some scheduling block size to the work scheduler based on every
   * device's get_max_num_path_states(). This is a bit tricky because CPU and GPU device will
   * be opposites of each other: CPU wavefront is super tiny, and GPU wavefront is gigantic.
   * How to find an ideal scheduling for such a mixture?  */
}

void PathTrace::reset(const BufferParams &full_buffer_params)
{
  full_render_buffers_->reset(full_buffer_params);

  scaled_render_buffer_params_ = full_buffer_params;
  update_scaled_render_buffers_resolution();
}

void PathTrace::clear_render_buffers()
{
  full_render_buffers_->zero();
}

void PathTrace::set_resolution_divider(int resolution_divider)
{
  resolution_divider_ = resolution_divider;
  update_scaled_render_buffers_resolution();
}

void PathTrace::set_start_sample(int start_sample_num)
{
  start_sample_num_ = start_sample_num;
}

void PathTrace::render_samples(int samples_num)
{
  render_init_execution();

  render_status_.reset();
  update_status_.reset();

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

void PathTrace::render_init_execution()
{
  for (auto &&queue : integrator_queues_) {
    queue->init_execution();
  }
}

void PathTrace::render_samples_full_pipeline(int samples_num)
{
  /* Reset work scheduler, so that it is ready to give work tiles for the new samples range. */
  work_scheduler_.reset(scaled_render_buffer_params_,
                        start_sample_num_ + render_status_.rendered_samples_num,
                        samples_num);

  tbb::parallel_for_each(integrator_queues_, [&](unique_ptr<DeviceQueue> &queue) {
    render_samples_full_pipeline(queue.get());
  });

  render_status_.rendered_samples_num += samples_num;
}

void PathTrace::render_samples_full_pipeline(DeviceQueue *queue)
{
  queue->init_execution();

  DeviceWorkTile work_tile;
  while (work_scheduler_.get_work(&work_tile)) {
    render_samples_full_pipeline(queue, work_tile);
  }
}

void PathTrace::render_samples_full_pipeline(DeviceQueue *queue, const DeviceWorkTile &work_tile)
{
  queue->set_work_tile(work_tile);

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
}

void PathTrace::copy_to_display_buffer(DisplayBuffer *display_buffer)
{
  DeviceTask task(DeviceTask::FILM_CONVERT);

  task.x = scaled_render_buffer_params_.full_x;
  task.y = scaled_render_buffer_params_.full_y;
  task.w = scaled_render_buffer_params_.width;
  task.h = scaled_render_buffer_params_.height;
  task.rgba_byte = display_buffer->rgba_byte.device_pointer;
  task.rgba_half = display_buffer->rgba_half.device_pointer;
  task.buffer = full_render_buffers_->buffer.device_pointer;

  /* NOTE: The device assumes the sample is the 0-based index of the last samples sample. */
  task.sample = start_sample_num_ + render_status_.rendered_samples_num - 1;

  scaled_render_buffer_params_.get_offset_stride(task.offset, task.stride);

  if (task.w > 0 && task.h > 0) {
    device_->task_add(task);
    device_->task_wait();

    /* Set display to new size. */
    display_buffer->draw_set(task.w, task.h);
  }
}

void PathTrace::update_scaled_render_buffers_resolution()
{
  const BufferParams &orig_params = full_render_buffers_->params;

  scaled_render_buffer_params_.width = max(1, orig_params.width / resolution_divider_);
  scaled_render_buffer_params_.height = max(1, orig_params.height / resolution_divider_);
  scaled_render_buffer_params_.full_x = orig_params.full_x / resolution_divider_;
  scaled_render_buffer_params_.full_y = orig_params.full_y / resolution_divider_;
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
  if (update_status_.has_update) {
    /* TODO(sergey): Use steady clock. */
    if (current_time - update_status_.last_update_time < update_interval_in_seconds) {
      return;
    }
  }

  update_cb(full_render_buffers_.get(), render_status_.rendered_samples_num);

  update_status_.has_update = true;
  update_status_.last_update_time = current_time;
}

void PathTrace::write()
{
  if (!write_cb) {
    return;
  }

  write_cb(full_render_buffers_.get(), render_status_.rendered_samples_num);
}

CCL_NAMESPACE_END
