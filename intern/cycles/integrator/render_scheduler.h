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

#pragma once

#include "integrator/adaptive_sampling.h"
#include "integrator/denoiser.h" /* For DenoiseParams. */
#include "render/buffers.h"

CCL_NAMESPACE_BEGIN

class RenderWork {
 public:
  int resolution_divider = 1;

  /* Path tracing samples information. */
  struct {
    int start_sample = 0;
    int num_samples = 0;

    bool adaptive_sampling_filter = false;
  } path_trace;

  bool denoise = false;

  bool copy_to_gpu_display = false;

  /* Conversion to bool, to simplify checks about whether there is anything to be done for this
   * work. */
  inline operator bool() const
  {
    return path_trace.num_samples || path_trace.adaptive_sampling_filter || denoise ||
           copy_to_gpu_display;
  }
};

class RenderScheduler {
 public:
  RenderScheduler(bool background, int pixel_size);

  bool is_background() const;

  void set_denoiser_params(const DenoiseParams &params);
  void set_adaptive_sampling(const AdaptiveSampling &adaptive_sampling);

  /* Start sample for path tracing.
   * The scheduler will schedule work using this sample as the first one. */
  void set_start_sample(int start_sample);
  int get_start_sample() const;

  /* Number of sampels to render, starting from start sample.
   * The scheduler will schedule work in the range of
   * [start_sample, start_sample + num_samples - 1], inclusively. */
  void set_num_samples(int num_samples);
  int get_num_samples() const;

  /* Get sample up to which rendering has been done.
   * This is an absolute 0-based value.
   *
   * For example, if start sample is 10 and and 5 samples were rendered, then this call will
   * return 14.
   *
   * If there were no samples rendered, then the behavior is undefined. */
  int get_rendered_sample() const;

  /* Get number of samples rendered within the current scheduling session.
   *
   * For example, if start sample is 10 and and 5 samples were rendered, then this call will
   * return 5.
   * *
   * Note that this is based on the scheduling information. In practice this means that if someone
   * requested for work to render the scheduler considers the work done. */
  int get_num_rendered_samples() const;

  /* Reset scheduler, indicating that rendering will happen from scratch.
   * Resets current rendered state, as well as scheduling information. */
  void reset(const BufferParams &buffer_params, int num_samples);

  /* Check whether all work has been scheduled. */
  bool done() const;

  RenderWork get_render_work();

  /* Report time (in seconds) which corresponding part of work took. */
  void report_path_trace_time(const RenderWork &render_work, double time);
  void report_denoise_time(const RenderWork &render_work, double time);
  void report_display_update_time(const RenderWork &render_work, double time);

 protected:
  /* Update start resolution based on the accumulated timing information, preserving nice feeling
   * navigation feel. */
  void update_start_resolution();

  /* Calculate desired update interval in seconds based on the current timings and settings.
   * Will give an interval which provides good feeling updates during viewport navigation. */
  double guess_viewport_navigation_update_interval_in_seconds() const;

  /* Check whether denoising is active during interactive update while resolution divider is not
   * unit. */
  bool is_denoise_active_during_update() const;

  /* Heuristic which aims to give perceptually pleasant update of display interval in a way that at
   * lower samples updates happens more often, but with higher number of samples updates happens
   * less often but the device occupancy goes higher. */
  double guess_display_update_interval_in_seconds() const;

  /* Calculate number of samples which can be rendered within current desred update interval which
   * is calculated by `guess_update_interval_in_seconds()`. */
  int calculate_num_samples_per_update() const;

  /* Get start sample and the number of samples which are to be path traces in the current work. */
  int get_start_sample_to_path_trace() const;
  int get_num_samples_to_path_trace() const;

  /* Check whether current work needs denoising.
   * Denoising is not needed if the denoiser is not configured, or when denosiing is happening too
   * often.
   *
   * The delayed will be true when the denoiser is configured for use, but it was delayed for a
   * later sample, to reduce overhead. */
  bool work_need_denoise(bool &delayed);

  struct TimeAverage {
    inline void reset()
    {
      total_time = 0.0;
      num_measured_times = 0;
    }

    inline double get_average() const
    {
      return total_time / num_measured_times;
    }

    double total_time = 0.0;
    int num_measured_times = 0;
  };

  struct {
    int resolution_divider = 1;

    /* Number of rendered samples on top of the start sample. */
    int num_rendered_samples = 0;

    /* Point in time the latest GPUDisplay work has been scheduled. */
    double last_gpu_display_update_time = 0.0;
  } state_;

  TimeAverage path_trace_time_;
  TimeAverage denoise_time_;
  TimeAverage display_update_time_;

  /* Path tracing work will be scheduled for samples from within
   * [start_sample_, start_sample_ + num_samples_ - 1] range, inclusively. */
  int start_sample_ = 0;
  int num_samples_ = 0;

  /* Background (offline) rendering. */
  bool background_;

  /* Pixel size is used to force lower resolution render for final pass. Useful for retina or other
   * types of hi-dpi displays. */
  int pixel_size_ = 1;

  BufferParams buffer_params_;
  DenoiseParams denoiser_params_;
  AdaptiveSampling adaptive_sampling_;

  static constexpr const int kDefaultStartResolution = 64;
  int start_resolution_ = kDefaultStartResolution;
};

CCL_NAMESPACE_END
