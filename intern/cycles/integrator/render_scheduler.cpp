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

#include "integrator/render_scheduler.h"

#include "util/util_logging.h"
#include "util/util_math.h"
#include "util/util_time.h"

CCL_NAMESPACE_BEGIN

RenderScheduler::RenderScheduler(bool background, int pixel_size)
    : background_(background), pixel_size_(pixel_size)
{
}

bool RenderScheduler::is_background() const
{
  return background_;
}

void RenderScheduler::set_denoiser_params(const DenoiseParams &params)
{
  denoiser_params_ = params;
}

void RenderScheduler::set_start_sample(int start_sample)
{
  start_sample_ = start_sample;
}

int RenderScheduler::get_start_sample() const
{
  return start_sample_;
}

void RenderScheduler::set_total_samples(int num_samples)
{
  num_total_samples_ = num_samples;
}

void RenderScheduler::reset(const BufferParams &buffer_params, int num_samples)
{
  buffer_params_ = buffer_params;

  set_total_samples(num_samples);

  /* In background mode never do lower resolution render preview, as it is not really supported
   * by the software. */
  if (background_) {
    state_.resolution_divider = 1;
  }
  else {
    state_.resolution_divider = 16;
  }

  state_.num_rendered_samples = 0;
  state_.last_gpu_display_update_time = 0.0;

  path_trace_time_.total_time = 0.0;
  path_trace_time_.num_measured_times = 0;

  denoise_time_.total_time = 0.0;
  denoise_time_.num_measured_times = 0;
}

bool RenderScheduler::done() const
{
  if (state_.resolution_divider != pixel_size_) {
    return false;
  }

  return state_.num_rendered_samples >= num_total_samples_;
}

int RenderScheduler::get_num_rendered_samples() const
{
  return state_.num_rendered_samples;
}

RenderWork RenderScheduler::get_render_work()
{
  if (done()) {
    return RenderWork();
  }

  RenderWork render_work;

  if (state_.resolution_divider != pixel_size_) {
    state_.resolution_divider = max(state_.resolution_divider / 2, pixel_size_);
    state_.num_rendered_samples = 0;
  }

  render_work.resolution_divider = state_.resolution_divider;

  render_work.path_trace.start_sample = start_sample_ + state_.num_rendered_samples;
  render_work.path_trace.num_samples = get_num_samples_to_path_trace();

  /* NOTE: Advance number of samples now, so that denoising check can see that all the samples are
   * rendered. */
  state_.num_rendered_samples += render_work.path_trace.num_samples;

  bool delayed;
  render_work.denoise = work_need_denoise(delayed);

  render_work.copy_to_gpu_display = !delayed;

  if (render_work.copy_to_gpu_display) {
    state_.last_gpu_display_update_time = time_dt();
  }

  return render_work;
}

void RenderScheduler::report_path_trace_time(const RenderWork &render_work, double time)
{
  (void)render_work;

  /* TODO(sergey): Multiply the time by the resolution divider, to give a more usabel estimate of
   * how long path tracing takes when rendering final resolution. */

  path_trace_time_.total_time += time;
  path_trace_time_.num_measured_times += render_work.path_trace.num_samples;

  VLOG(4) << "Average path tracing time: " << path_trace_time_.get_average() << " seconds.";
}

void RenderScheduler::report_denoise_time(const RenderWork &render_work, double time)
{
  (void)render_work;

  /* TODO(sergey): Multiply the time by the resolution divider, to give a more usabel estimate of
   * how long path tracing takes when rendering final resolution. */

  denoise_time_.total_time += time;
  ++denoise_time_.num_measured_times;

  VLOG(4) << "Average denoising time: " << denoise_time_.get_average() << " seconds.";
}

int RenderScheduler::get_num_samples_to_path_trace()
{
  /* Always start with a single sample. Gives more instant feedback to artists, and allows to
   * gather information for a subsequent path tracing works. */
  if (state_.num_rendered_samples == 0) {
    return 1;
  }

  /* Always render single sample when in non-final resolution. */
  if (state_.resolution_divider != 1) {
    return 1;
  }

  /* TODO(sergey): Scheduler multiple samples to the viewport as well. */
  if (!background_) {
    return 1;
  }

  const double time_per_sample_average = path_trace_time_.get_average();

  const int num_samples_in_second = max(int(1.0 / time_per_sample_average), 1);

  return min(num_samples_in_second, num_total_samples_ - state_.num_rendered_samples);
}

bool RenderScheduler::work_need_denoise(bool &delayed)
{
  delayed = false;

  if (!denoiser_params_.use) {
    /* Denoising is disabled, no need to scheduler work for it. */
    return false;
  }

  if (background_) {
    /* Background render, only denoise when rendering the last sample. */
    /* TODO(sergey): Follow similar logic to viewport, giving an overview of how final denoised
     * image looks like even for the background rendering. */
    return done();
  }

  /* Viewport render. */

  /* Immediately denoise when we reach the start sample or last sample. */
  const int num_samples_finished = state_.num_rendered_samples;
  if (num_samples_finished == denoiser_params_.start_sample ||
      num_samples_finished == num_total_samples_) {
    return true;
  }

  /* Do not denoise until the sample at which denoising should start is reached. */
  if (num_samples_finished < denoiser_params_.start_sample) {
    return false;
  }

  /* Avoid excessive denoising in viewport after reaching a certain amount of samples. */
  /* TODO(sergey): Consider making time interval and sample configurable. */
  delayed = (state_.num_rendered_samples >= 20 &&
             (time_dt() - state_.last_gpu_display_update_time) < 1.0);

  return !delayed;
}

CCL_NAMESPACE_END
