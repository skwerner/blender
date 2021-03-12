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
#include "util/util_progress.h"
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

  /* TODO(sergey): Need to create render buffer for every individual device, so that they can write
   * directly to it. */
  full_render_buffers_ = make_unique<RenderBuffers>(device);

  /* Create path tracing work in advance, so that it can be reused by incremental sampling as much
   * as possible. */
  device->foreach_device([&](Device *render_device) {
    path_trace_works_.emplace_back(PathTraceWork::create(
        render_device, full_render_buffers_.get(), &render_cancel_.is_requested));
  });

  /* TODO(sergey): Communicate some scheduling block size to the work scheduler based on every
   * device's get_max_num_paths(). This is a bit tricky because CPU and GPU device will
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

void PathTrace::set_progress(Progress *progress)
{
  progress_ = progress;
}

void PathTrace::render_samples(int samples_num)
{
  /* Indicate that rendering has started and that it can be requested to cancel. */
  {
    thread_scoped_lock lock(render_cancel_.mutex);
    render_cancel_.is_rendering = true;
    render_cancel_.is_requested = false;
  }

  render_init_execution();

  render_status_.reset();
  update_status_.reset();

  double total_sampling_time = 0;

  total_sampling_time += render_samples_full_pipeline(1);

  /* Update as soon as possible, so that artists immediately see first pixels. */
  buffer_update_if_needed();
  progress_update_if_needed();

  while (render_status_.rendered_samples_num < samples_num) {
    /* TODO(sergey): Take adaptive stopping and user cancel into account. Both of these actions
     * will affect how the buffer is to be scaled. */

    const double time_per_sample_average = total_sampling_time /
                                           render_status_.rendered_samples_num;

    /* TODO(sergey): Take update_interval_in_seconds into account. */

    const int samples_in_second_num = max(int(1.0 / time_per_sample_average), 1);
    const int samples_to_render_num = min(samples_in_second_num,
                                          samples_num - render_status_.rendered_samples_num);

    total_sampling_time += render_samples_full_pipeline(samples_to_render_num);

    buffer_update_if_needed();
    progress_update_if_needed();

    if (is_cancel_requested()) {
      break;
    }
  }

  buffer_write();

  /* Indicate that rendering has finished, making it so thread which requested `cancel()` can carry
   * on. */
  {
    thread_scoped_lock lock(render_cancel_.mutex);
    render_cancel_.is_rendering = false;
    render_cancel_.condition.notify_one();
  }
}

void PathTrace::render_init_execution()
{
  for (auto &&path_trace_work : path_trace_works_) {
    path_trace_work->init_execution();
  }
}

double PathTrace::render_samples_full_pipeline(int samples_num)
{
  const double start_render_time = time_dt();

  const int start_sample = start_sample_num_ + render_status_.rendered_samples_num;

  tbb::parallel_for_each(path_trace_works_, [&](unique_ptr<PathTraceWork> &path_trace_work) {
    path_trace_work->render_samples(scaled_render_buffer_params_, start_sample, samples_num);
  });

  render_status_.rendered_samples_num += samples_num;

  return time_dt() - start_render_time;
}

void PathTrace::set_denoiser_params(const DenoiseParams &params)
{
  if (!params.use) {
    denoiser_.reset();
    return;
  }

  denoiser_ = Denoiser::create(device_, params);
}

void PathTrace::denoise()
{
  if (!denoiser_) {
    /* Denoiser was not configured, so nothing to do here. */
    return;
  }

  if (is_cancel_requested()) {
    return;
  }

  /* Indicate that rendering has started and that it can be requested to cancel.
   *
   * TODO(sergey): De-duplicate with render_samples(). */
  {
    thread_scoped_lock lock(render_cancel_.mutex);
    render_cancel_.is_rendering = true;
    render_cancel_.is_requested = false;
  }

  const DenoiserBufferParams buffer_params(scaled_render_buffer_params_);
  denoiser_->denoise_buffer(
      buffer_params, full_render_buffers_.get(), get_num_samples_in_buffer());

  buffer_write();

  /* Indicate that rendering has finished, making it so thread which requested `cancel()` can carry
   * on.
   *
   * TODO(sergey): De-duplicate with render_samples(). */
  {
    thread_scoped_lock lock(render_cancel_.mutex);
    render_cancel_.is_rendering = false;
    render_cancel_.condition.notify_one();
  }
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
  task.sample = get_num_samples_in_buffer() - 1;

  scaled_render_buffer_params_.get_offset_stride(task.offset, task.stride);

  if (task.w > 0 && task.h > 0) {
    device_->task_add(task);
    device_->task_wait();

    /* Set display to new size. */
    display_buffer->draw_set(task.w, task.h);
  }
}

void PathTrace::cancel()
{
  thread_scoped_lock lock(render_cancel_.mutex);

  render_cancel_.is_requested = true;

  while (render_cancel_.is_rendering) {
    render_cancel_.condition.wait(lock);
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

int PathTrace::get_num_samples_in_buffer()
{
  return start_sample_num_ + render_status_.rendered_samples_num;
}

bool PathTrace::is_cancel_requested()
{
  if (render_cancel_.is_requested) {
    return true;
  }

  if (progress_ != nullptr) {
    if (progress_->get_cancel()) {
      return true;
    }
  }

  return false;
}

void PathTrace::buffer_update_if_needed()
{
  if (!buffer_update_cb) {
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

  buffer_update_cb(full_render_buffers_.get(), render_status_.rendered_samples_num);

  update_status_.has_update = true;
  update_status_.last_update_time = current_time;
}

void PathTrace::buffer_write()
{
  if (!buffer_write_cb) {
    return;
  }

  buffer_write_cb(full_render_buffers_.get(), render_status_.rendered_samples_num);
}

void PathTrace::progress_update_if_needed()
{
  if (progress_ != nullptr) {
    progress_->add_samples(0, get_num_samples_in_buffer());
  }

  if (progress_update_cb) {
    progress_update_cb();
  }
}

CCL_NAMESPACE_END
