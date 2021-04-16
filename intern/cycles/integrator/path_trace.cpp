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
#include "integrator/render_scheduler.h"
#include "render/gpu_display.h"
#include "render/pass_accessor.h"
#include "util/util_algorithm.h"
#include "util/util_logging.h"
#include "util/util_progress.h"
#include "util/util_tbb.h"
#include "util/util_time.h"

CCL_NAMESPACE_BEGIN

PathTrace::PathTrace(Device *device, DeviceScene *device_scene, RenderScheduler &render_scheduler)
    : device_(device), render_scheduler_(render_scheduler)
{
  DCHECK_NE(device_, nullptr);

  /* Create path tracing work in advance, so that it can be reused by incremental sampling as much
   * as possible. */
  device_->foreach_device([&](Device *path_trace_device) {
    if (!path_trace_works_.empty()) {
      if (path_trace_works_.size() == 1) {
        LOG(ERROR)
            << "Multi-devices are not yet fully implemented, will render on a single device.";
      }
      return;
    }

    /* TODO(sergey): Need to create render buffer for every individual device, so that they can
     * write directly to it. */
    full_render_buffers_ = make_unique<RenderBuffers>(path_trace_device);

    path_trace_works_.emplace_back(PathTraceWork::create(path_trace_device,
                                                         device_scene,
                                                         full_render_buffers_.get(),
                                                         &render_cancel_.is_requested));
  });
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

    /* Indicate the work state that resolution divider is out of date. */
    render_state_.resolution_divider = 0;
  }

  if (gpu_display_) {
    gpu_display_->reset(full_buffer_params);
  }

  did_draw_after_reset_ = false;
}

void PathTrace::set_progress(Progress *progress)
{
  progress_ = progress;
}

void PathTrace::render(const RenderWork &render_work)
{
  /* Indicate that rendering has started and that it can be requested to cancel. */
  {
    thread_scoped_lock lock(render_cancel_.mutex);
    if (render_cancel_.is_requested) {
      return;
    }
    render_cancel_.is_rendering = true;
  }

  render_pipeline(render_work);

  /* Indicate that rendering has finished, making it so thread which requested `cancel()` can carry
   * on. */
  {
    thread_scoped_lock lock(render_cancel_.mutex);
    render_cancel_.is_rendering = false;
    render_cancel_.condition.notify_one();
  }
}

void PathTrace::render_pipeline(RenderWork render_work)
{
  /* For the interactive viewport clear the render buffer on first sample, so that changes in
   * resolution and camera and things ike that get explicitly zeroed. */
  if (!render_scheduler_.is_background() &&
      render_work.path_trace.start_sample == render_scheduler_.get_start_sample()) {
    full_render_buffers_->zero();
  }

  render_init_kernel_execution();
  render_update_resolution_divider(render_work.resolution_divider);

  path_trace(render_work);
  if (is_cancel_requested()) {
    return;
  }

  denoise(render_work);
  if (is_cancel_requested()) {
    return;
  }

  update_display(render_work);

  progress_update_if_needed();

  if (render_scheduler_.done()) {
    buffer_write();
  }
}

void PathTrace::render_init_kernel_execution()
{
  for (auto &&path_trace_work : path_trace_works_) {
    path_trace_work->init_execution();
  }
}

void PathTrace::path_trace(RenderWork &render_work)
{
  if (!render_work.path_trace.num_samples) {
    return;
  }

  VLOG(3) << "Will path trace " << render_work.path_trace.num_samples
          << " samples at the resolution divider " << render_work.resolution_divider;

  if (render_work.path_trace.adaptive_sampling_filter) {
    VLOG(3) << "Will filter adaptive stopping buffer.";
  }

  const double start_time = time_dt();

  bool all_pixels_converged = render_work.path_trace.adaptive_sampling_filter;

  tbb::parallel_for_each(path_trace_works_, [&](unique_ptr<PathTraceWork> &path_trace_work) {
    path_trace_work->render_samples(render_work.path_trace.start_sample,
                                    render_work.path_trace.num_samples);

    if (render_work.path_trace.adaptive_sampling_filter) {
      all_pixels_converged &= path_trace_work->adaptive_sampling_converge_and_filter(
          render_work.path_trace.adaptive_sampling_threshold, false);
    }
  });

  if (all_pixels_converged) {
    VLOG(3) << "All pixels converged.";
    render_scheduler_.set_path_trace_finished(render_work);
  }

  if (!is_cancel_requested()) {
    render_scheduler_.report_path_trace_time(render_work, time_dt() - start_time);
  }
}

void PathTrace::set_denoiser_params(const DenoiseParams &params)
{
  render_scheduler_.set_denoiser_params(params);

  if (!params.use) {
    denoiser_.reset();
    return;
  }

  if (denoiser_) {
    const DenoiseParams old_denoiser_params = denoiser_->get_params();
    if (old_denoiser_params.type == params.type) {
      denoiser_->set_params(params);
      return;
    }
  }

  denoiser_ = Denoiser::create(device_, params);
}

void PathTrace::set_adaptive_sampling(const AdaptiveSampling &adaptive_sampling)
{
  render_scheduler_.set_adaptive_sampling(adaptive_sampling);
}

void PathTrace::denoise(const RenderWork &render_work)
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

  denoiser_->denoise_buffer(render_state_.scaled_render_buffer_params,
                            full_render_buffers_.get(),
                            get_num_samples_in_buffer());

  if (!is_cancel_requested()) {
    render_scheduler_.report_denoise_time(render_work, time_dt() - start_time);
  }
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

void PathTrace::update_display(const RenderWork &render_work)
{
  if (!render_work.update_display) {
    return;
  }

  if (!gpu_display_) {
    /* TODO(sergey): Ideally the offline buffers update will be done using same API than the
     * viewport GPU display. Seems to be a matter of moving pixels update API to a more abstract
     * class and using it here instead of `GPUDisplay`. */
    if (buffer_update_cb) {
      VLOG(3) << "Invoke buffer update callback.";

      const double start_time = time_dt();
      buffer_update_cb();
      render_scheduler_.report_display_update_time(render_work, time_dt() - start_time);
    }
    else {
      VLOG(3) << "Ignore display update.";
    }

    return;
  }

  VLOG(3) << "Perform copy to GPUDisplay work.";

  const int width = render_state_.scaled_render_buffer_params.width;
  const int height = render_state_.scaled_render_buffer_params.height;
  if (width == 0 || height == 0) {
    return;
  }

  const double start_time = time_dt();

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

  render_scheduler_.report_display_update_time(render_work, time_dt() - start_time);
}

void PathTrace::cancel()
{
  thread_scoped_lock lock(render_cancel_.mutex);

  render_cancel_.is_requested = true;

  while (render_cancel_.is_rendering) {
    render_cancel_.condition.wait(lock);
  }

  render_cancel_.is_requested = false;
}

void PathTrace::render_update_resolution_divider(int resolution_divider)
{
  if (render_state_.resolution_divider == resolution_divider) {
    return;
  }
  render_state_.resolution_divider = resolution_divider;

  const BufferParams &orig_params = full_render_buffers_->params;

  render_state_.scaled_render_buffer_params = orig_params;

  render_state_.scaled_render_buffer_params.width = max(1, orig_params.width / resolution_divider);
  render_state_.scaled_render_buffer_params.height = max(1,
                                                         orig_params.height / resolution_divider);
  render_state_.scaled_render_buffer_params.full_x = orig_params.full_x / resolution_divider;
  render_state_.scaled_render_buffer_params.full_y = orig_params.full_y / resolution_divider;

  render_state_.scaled_render_buffer_params.update_offset_stride();

  /* TODO(sergey): Perform slicing of the render buffers for every work. */
  for (auto &&path_trace_work : path_trace_works_) {
    path_trace_work->set_effective_buffer_params(render_state_.scaled_render_buffer_params);
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

void PathTrace::buffer_write()
{
  if (!buffer_write_cb) {
    return;
  }

  buffer_write_cb();
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

bool PathTrace::get_render_tile_pixels(PassAccessor &pass_accessor, float *pixels)
{
  if (!full_render_buffers_->copy_from_device()) {
    return false;
  }

  return pass_accessor.get_render_tile_pixels(full_render_buffers_.get(), pixels);
}

CCL_NAMESPACE_END
