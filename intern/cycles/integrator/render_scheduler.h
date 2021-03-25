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

CCL_NAMESPACE_BEGIN

class RenderWork {
 public:
  /* Path tracing samples information. */
  struct {
    int start_sample = 0;
    int num_samples = 0;
  } path_trace;

  bool denoise = false;
};

class RenderScheduler {
 public:
  RenderScheduler();

  /* Reset scheduler, indicating that rendering will happen from scratch.
   * Resets current rendered state, as well as scheduling information. */
  void reset();

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

  /* Report time (in seconds) which path tracing part of the given work took. */
  void report_path_trace_time(const RenderWork &render_work, double time);

 protected:
  /* Get number of samples which are to be path traces in the current work. */
  int get_num_samples_to_path_trace();

  struct {
    int num_rendered_samples = 0;
  } state_;

  struct {
    double total_time = 0;
    int num_measured_samples = 0;
  } path_trace_time_;

  int num_samples_to_render_ = 0;
};

CCL_NAMESPACE_END
