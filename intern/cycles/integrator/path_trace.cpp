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
#include "render/gpu_display.h"
#include "util/util_algorithm.h"
#include "util/util_logging.h"
#include "util/util_progress.h"
#include "util/util_tbb.h"
#include "util/util_time.h"

CCL_NAMESPACE_BEGIN

PathTrace::PathTrace(Device *device, bool background)
    : device_(device), render_scheduler_(background)
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

bool PathTrace::ready_to_reset()
{
  /* The logic here is optimized for the best feedback in the viewport, which implies having a GPU
   * display. Of there is no such display, the logic here will break. */
  DCHECK(gpu_display_);

  /* The logic here tries to provide behavior which feels the most interactive feel to artists.
   * General idea is to be able to reset as quickly as possible, while still providing interactive
   * feel.
   *
   * If the render result was ever drawn after previous reset, consider that reset is now possible.
   * This way camera navigation gives the quickest feedback of rendered pixels, regardless of
   * whether CPU or GPU drawing pipeline is used.
   *
   * Consider reset happening after redraw "slow" enough to not clog anything. This is a bit
   * arbitrary, but seems to work very well with viewport navigation in Blender. */

  if (did_draw_after_reset_) {
    return true;
  }

  return false;
}

void PathTrace::reset(const BufferParams &full_buffer_params)
{
  if (full_render_buffers_->params.modified(full_buffer_params)) {
    full_render_buffers_->reset(full_buffer_params);
  }

  if (gpu_display_) {
    gpu_display_->reset(full_buffer_params);
  }

  render_scheduler_.reset();

  scaled_render_buffer_params_ = full_buffer_params;
  update_scaled_render_buffers_resolution();

  did_draw_after_reset_ = false;
}

void PathTrace::clear_render_buffers()
{
  full_render_buffers_->zero();
}

void PathTrace::set_resolution_divider(int resolution_divider)
{
  /* Changing resolution divider invalidates all the rendered samples. This doesn't currently fit
   * under the reset() usage, so inform the scheduler explicitly here.
   *
   * In the future the divider will be taken care of by the scheduler, so this will not be needed
   * anymore. */
  if (resolution_divider_ != resolution_divider) {
    render_scheduler_.reset();
  }

  resolution_divider_ = resolution_divider;
  update_scaled_render_buffers_resolution();
}

void PathTrace::set_start_sample(int start_sample_num)
{
  start_sample_num_ = start_sample_num;
}

void PathTrace::set_total_samples(int num_samples)
{
  render_scheduler_.set_total_samples(num_samples);
}

void PathTrace::set_progress(Progress *progress)
{
  progress_ = progress;
}

void PathTrace::render_samples(int num_samples)
{
  /* Indicate that rendering has started and that it can be requested to cancel. */
  {
    thread_scoped_lock lock(render_cancel_.mutex);
    render_cancel_.is_rendering = true;
    render_cancel_.is_requested = false;
  }

  render_init_execution();

  render_scheduler_.add_samples_to_render(num_samples);

  while (!is_cancel_requested()) {
    RenderWork render_work;
    if (!render_scheduler_.get_render_work(render_work)) {
      break;
    }

    render_work_full_pipeline(render_work);
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

void PathTrace::render_work_full_pipeline(const RenderWork &render_work)
{
  path_trace_work(render_work);
  if (is_cancel_requested()) {
    return;
  }

  denoise_work(render_work);
  if (is_cancel_requested()) {
    return;
  }

  copy_to_gpu_display_work(render_work);

  buffer_update_if_needed();
  progress_update_if_needed();
}

void PathTrace::path_trace_work(const RenderWork &render_work)
{
  if (!render_work.path_trace.num_samples) {
    return;
  }

  VLOG(3) << "Will path trace " << render_work.path_trace.num_samples << " samples.";

  const double start_time = time_dt();

  const int start_sample = start_sample_num_ + render_work.path_trace.start_sample;

  tbb::parallel_for_each(path_trace_works_, [&](unique_ptr<PathTraceWork> &path_trace_work) {
    path_trace_work->render_samples(start_sample, render_work.path_trace.num_samples);
  });

  render_scheduler_.report_path_trace_time(render_work, time_dt() - start_time);
}

void PathTrace::set_denoiser_params(const DenoiseParams &params)
{
  render_scheduler_.set_denoiser_params(params);

  if (!params.use) {
    denoiser_.reset();
    return;
  }

  if (!denoiser_ || denoiser_->get_params().modified(params)) {
    denoiser_ = Denoiser::create(device_, params);
  }
}

void PathTrace::denoise_work(const RenderWork &render_work)
{
  if (!render_work.denoise) {
    return;
  }

  if (!denoiser_) {
    /* Denoiser was not configured, so nothing to do here. */
    return;
  }

  VLOG(3) << "Perform denoising work.";

  const double start_time = time_dt();

  const DenoiserBufferParams buffer_params(scaled_render_buffer_params_);
  denoiser_->denoise_buffer(
      buffer_params, full_render_buffers_.get(), get_num_samples_in_buffer());

  render_scheduler_.report_denoise_time(render_work, time_dt() - start_time);
}

void PathTrace::set_gpu_display(unique_ptr<GPUDisplay> gpu_display)
{
  gpu_display_ = move(gpu_display);
}

void PathTrace::draw()
{
  DCHECK(gpu_display_);

  did_draw_after_reset_ |= gpu_display_->draw();
}

void PathTrace::copy_to_gpu_display_work(const RenderWork &render_work)
{
  if (!render_work.copy_to_gpu_display) {
    return;
  }

  if (!gpu_display_) {
    return;
  }

  VLOG(3) << "Perform copy to GPUDisplay work.";

  const int width = scaled_render_buffer_params_.width;
  const int height = scaled_render_buffer_params_.height;
  if (width == 0 || height == 0) {
    return;
  }

  const float sample_scale = 1.0f / get_num_samples_in_buffer();

  if (!gpu_display_->update_begin(width, height)) {
    LOG(ERROR) << "Error beginning GPUDisplay update.";
    return;
  }

  /* TODO(sergey): In theory we would want to update parts of the buffer from multiple threads.
   * However, there could be some complications related on how texture buffer is mapped. Depending
   * on an implementation of GPUDisplay it might not be possible to map GPUBuffer in a way that the
   * PathTraceWork expects it in a threaded environment. */
  for (auto &&path_trace_work : path_trace_works_) {
    path_trace_work->copy_to_gpu_display(gpu_display_.get(), sample_scale);
  }

  gpu_display_->update_end();
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

  /* TODO(sergey): Perform slicing of the render buffers for every work. */
  for (auto &&path_trace_work : path_trace_works_) {
    path_trace_work->set_effective_buffer_params(scaled_render_buffer_params_);
  }
}

int PathTrace::get_num_samples_in_buffer()
{
  return render_scheduler_.get_num_rendered_samples();
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

  const int num_samples_rendered = get_num_samples_in_buffer();
  buffer_update_cb(full_render_buffers_.get(), num_samples_rendered);
}

void PathTrace::buffer_write()
{
  if (!buffer_write_cb) {
    return;
  }

  const int num_samples_rendered = get_num_samples_in_buffer();
  buffer_write_cb(full_render_buffers_.get(), num_samples_rendered);
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
