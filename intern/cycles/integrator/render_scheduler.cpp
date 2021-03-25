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

#include "util/util_math.h"

CCL_NAMESPACE_BEGIN

RenderScheduler::RenderScheduler()
{
}

void RenderScheduler::reset()
{
  num_samples_to_render_ = 0;

  state_.num_rendered_samples = 0;

  path_trace_time_.total_time = 0.0;
  path_trace_time_.num_measured_samples = 0;
}

void RenderScheduler::add_samples_to_render(int num_samples)
{
  num_samples_to_render_ += num_samples;
}

int RenderScheduler::get_num_rendered_samples() const
{
  return state_.num_rendered_samples;
}

bool RenderScheduler::get_render_work(RenderWork &render_work)
{
  if (state_.num_rendered_samples >= num_samples_to_render_) {
    return false;
  }

  render_work.path_trace.start_sample = state_.num_rendered_samples;
  render_work.path_trace.num_samples = get_num_samples_to_path_trace();

  state_.num_rendered_samples += render_work.path_trace.num_samples;

  return true;
}

void RenderScheduler::report_path_trace_time(const RenderWork &render_work, double time)
{
  (void)render_work;

  /* TODO(sergey): Multiply the time by the resolution divider, to give a more usabel estimate of
   * how long path tracing takes when rendering final resolution. */

  path_trace_time_.total_time += time;
  path_trace_time_.num_measured_samples += render_work.path_trace.num_samples;
}

int RenderScheduler::get_num_samples_to_path_trace()
{
  /* Always start with a single sample. Gives more instant feedback to artists, and allows to
   * gather information for a subsequent path tracing works. */
  if (state_.num_rendered_samples == 0) {
    return 1;
  }

  const double time_per_sample_average = path_trace_time_.total_time /
                                         path_trace_time_.num_measured_samples;

  const int num_samples_in_second = max(int(1.0 / time_per_sample_average), 1);

  return min(num_samples_in_second, num_samples_to_render_ - state_.num_rendered_samples);
}

CCL_NAMESPACE_END
