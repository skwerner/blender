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

#include "integrator/denoiser.h" /* For DenoiseParams. */

CCL_NAMESPACE_BEGIN

class RenderWork {
 public:
  /* Path tracing samples information. */
  struct {
    int start_sample = 0;
    int num_samples = 0;
  } path_trace;

  bool denoise = false;

  bool copy_to_gpu_display = false;
};

class RenderScheduler {
 public:
  explicit RenderScheduler(bool background);

  void set_denoiser_params(const DenoiseParams &params);

  /* Set total number of samples which are to be rendered.
   * This is different from add_samples_to_render() in a sense that it is possible that main render
   * loop will incrementally schedule samples to be rendered until the total number of samples is
   * reached. */
  void set_total_samples(int num_samples);

  /* Reset scheduler, indicating that rendering will happen from scratch.
   * Resets current rendered state, as well as scheduling information. */
  void reset();

  /* Check whether all work has been scheduled. */
  bool done() const;

  /* Add given number of samples to be rendered.
   * Is used for progressively add samples. For examples, when in viewport rendering an artist adds
   * more samples in settings. */
  void add_samples_to_render(int num_samples);

  /* Returns false when there is no more work to be done. */
  bool get_render_work(RenderWork &render_work);

  /* Get number of samples rendered within the current scheduling session.
   * Note that this is based on the scheduling information. In practice this means that if someone
   * requested for work to render the scheduler considers the work done. */
  int get_num_rendered_samples() const;

  /* Report time (in seconds) which corresponding part of work took. */
  void report_path_trace_time(const RenderWork &render_work, double time);
  void report_denoise_time(const RenderWork &render_work, double time);

 protected:
  /* Get number of samples which are to be path traces in the current work. */
  int get_num_samples_to_path_trace();

  /* Check whether current work needs denoising.
   * Denoising is not needed if the denoiser is not configured, or when denosiing is happening too
   * often.
   *
   * The delayed will be true when the denoiser is configured for use, but it was delayed for a
   * later sample, to reduce overhead. */
  bool work_need_denoise(bool &delayed);

  struct {
    int num_rendered_samples = 0;

    /* Point in time the latest GPUDisplay work has been scheduled. */
    double last_gpu_display_update_time = 0.0;
  } state_;

  struct {
    inline double get_average()
    {
      return total_time / num_measured_times;
    }

    double total_time = 0.0;
    int num_measured_times = 0;
  } path_trace_time_;

  struct {
    inline double get_average()
    {
      return total_time / num_measured_times;
    }

    double total_time = 0.0;
    int num_measured_times = 0;
  } denoise_time_;

  /* Total number if samples to be rendered within the current render session. */
  int num_total_samples_ = 0;

  /* The number of samples to render upto from the current `PathTrace::render_samples()` call. */
  int num_samples_to_render_ = 0;

  /* Background (offline) rendering. */
  bool background_;

  DenoiseParams denoiser_params_;
};

CCL_NAMESPACE_END
