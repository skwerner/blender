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

void PathTrace::load_kernels()
{
  if (denoiser_) {
    denoiser_->load_kernels(progress_);
  }
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
  /* TODO(sergey): For truly resumable render might need to avoid zero-ing. */
  if (render_work.path_trace.start_sample == render_scheduler_.get_start_sample()) {
    full_render_buffers_->zero();
  }

  render_init_kernel_execution();
  render_update_resolution_divider(render_work.resolution_divider);

  path_trace(render_work);
  if (is_cancel_requested()) {
    return;
  }

  adaptive_sample(render_work);
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

  const double start_time = time_dt();

  tbb::parallel_for_each(path_trace_works_, [&](unique_ptr<PathTraceWork> &path_trace_work) {
    path_trace_work->render_samples(render_work.path_trace.start_sample,
                                    render_work.path_trace.num_samples);
  });

  render_scheduler_.report_path_trace_time(
      render_work, time_dt() - start_time, is_cancel_requested());
}

void PathTrace::adaptive_sample(RenderWork &render_work)
{
  if (!render_work.adaptive_sampling.filter) {
    return;
  }

  bool did_reschedule_on_idle = false;

  while (true) {
    VLOG(3) << "Will filter adaptive stopping buffer, threshold "
            << render_work.adaptive_sampling.threshold;
    if (render_work.adaptive_sampling.reset) {
      VLOG(3) << "Will re-calculate convergency flag for currently converged pixels.";
    }

    const double start_time = time_dt();

    uint num_active_pixels = 0;
    tbb::parallel_for_each(path_trace_works_, [&](unique_ptr<PathTraceWork> &path_trace_work) {
      const uint num_active_pixels_in_work =
          path_trace_work->adaptive_sampling_converge_filter_count_active(
              render_work.adaptive_sampling.threshold, render_work.adaptive_sampling.reset);
      if (num_active_pixels_in_work) {
        atomic_add_and_fetch_u(&num_active_pixels, num_active_pixels_in_work);
      }
    });

    render_scheduler_.report_adaptive_filter_time(
        render_work, time_dt() - start_time, is_cancel_requested());

    if (num_active_pixels == 0) {
      VLOG(3) << "All pixels converged.";
      if (!render_scheduler_.render_work_reschedule_on_converge(render_work)) {
        break;
      }
      VLOG(3) << "Continuing with lower threshold.";
    }
    else if (did_reschedule_on_idle) {
      break;
    }
    else if (num_active_pixels < 128 * 128) {
      /* NOTE: The hardcoded value of 128^2 is more of an empirical value to keep GPU busy so that
       * there is no performance loss from the progressive noise floor feature.
       *
       * A better heuristic is possible here: for example, use maximum of 128^2 and percentage of
       * the final resolution. */
      if (!render_scheduler_.render_work_reschedule_on_idle(render_work)) {
        VLOG(3) << "Rescheduling is not possible: final threshold is reached.";
        break;
      }
      VLOG(3) << "Rescheduling lower threshold.";
      did_reschedule_on_idle = true;
    }
    else {
      break;
    }
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

/* --------------------------------------------------------------------
 * Report generation.
 */

static const char *device_type_for_description(const DeviceType type)
{
  switch (type) {
    case DEVICE_NONE:
      return "None";

    case DEVICE_CPU:
      return "CPU";
    case DEVICE_OPENCL:
      return "OpenCL";
    case DEVICE_CUDA:
      return "CUDA";
    case DEVICE_OPTIX:
      return "OptiX";
    case DEVICE_DUMMY:
      return "Dummy";
    case DEVICE_MULTI:
      return "Multi";
  }

  return "UNKNOWN";
}

/* Construct description of the device which will appear in the full report. */
/* TODO(sergey): Consider making it more reusable utility. */
static string full_device_info_description(const DeviceInfo &device_info)
{
  string full_description = device_info.description;

  full_description += " (" + string(device_type_for_description(device_info.type)) + ")";

  if (device_info.display_device) {
    full_description += " (display)";
  }

  if (device_info.type == DEVICE_CPU) {
    full_description += " (" + to_string(device_info.cpu_threads) + " threads)";
  }

  full_description += " [" + device_info.id + "]";

  return full_description;
}

/* Construct string which will contain information about devices, possibly multiple of the devices.
 *
 * In the simple case the result looks like:
 *
 *   Message: Full Device Description
 *
 * If there are multiple devices then the result looks like:
 *
 *   Message: Full First Device Description
 *            Full Second Device Description
 *
 * Note that the newlines are placed in a way so that the result can be easily concatenated to the
 * full report. */
static string device_info_list_report(const string &message, const DeviceInfo &device_info)
{
  string result = "\n" + message + ": ";
  const string pad(message.length() + 2, ' ');

  if (device_info.multi_devices.empty()) {
    result += full_device_info_description(device_info) + "\n";
    return result;
  }

  bool is_first = true;
  for (const DeviceInfo &sub_device_info : device_info.multi_devices) {
    if (!is_first) {
      result += pad;
    }

    result += full_device_info_description(sub_device_info) + "\n";

    is_first = false;
  }

  return result;
}

static string path_trace_devices_report(const vector<unique_ptr<PathTraceWork>> &path_trace_works)
{
  DeviceInfo device_info;
  device_info.type = DEVICE_MULTI;

  for (auto &&path_trace_work : path_trace_works) {
    device_info.multi_devices.push_back(path_trace_work->get_device()->info);
  }

  return device_info_list_report("Path tracing on", device_info);
}

static string denoiser_device_report(const Denoiser *denoiser)
{
  if (!denoiser) {
    return "";
  }

  if (!denoiser->get_params().use) {
    return "";
  }

  const DeviceInfo device_info = denoiser->get_denoiser_device_info();
  if (device_info.type == DEVICE_NONE) {
    return "";
  }

  return device_info_list_report("Denoising on", device_info);
}

string PathTrace::full_report() const
{
  string result = "\nFull path tracing report\n";

  result += path_trace_devices_report(path_trace_works_);
  result += denoiser_device_report(denoiser_.get());

  /* Report from the render scheduler, which includes:
   * - Render mode (interactive, offline, headless)
   * - Adaptive sampling and denoiser parameters
   * - Breakdown of timing. */
  result += render_scheduler_.full_report();

  return result;
}

CCL_NAMESPACE_END
