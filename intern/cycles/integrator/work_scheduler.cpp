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

#include "integrator/work_scheduler.h"

#include "device/device_queue.h"
#include "util/util_atomic.h"

CCL_NAMESPACE_BEGIN

WorkScheduler::WorkScheduler()
{
}

void WorkScheduler::reset(
    int full_x, int full_y, int width, int height, int sample_start, int samples_num)
{
  full_x_ = full_x;
  full_y_ = full_y;

  width_ = width;
  height_ = height;

  sample_start_ = sample_start;
  samples_num_ = samples_num;

  reset_scheduler_state();
}

void WorkScheduler::reset_scheduler_state()
{
  total_pixels_num_ = width_ * height_;

  next_work_index_ = 0;
  total_work_size_ = width_ * height_ * samples_num_;
}

bool WorkScheduler::get_work(DeviceWorkTile *work_tile)
{
  /* TODO(sergey): Implement some smarter work scheduler which will be able to scheduler tile sizes
   * different from 1x1. Currently this is a bare minimum for CPU devices. */

  const int work_index = atomic_fetch_and_add_int32(&next_work_index_, 1);
  if (work_index >= total_work_size_) {
    return false;
  }

  const int sample = work_index / total_pixels_num_;
  const int pixel_index = work_index - sample * total_pixels_num_;
  const int y = pixel_index / width_;
  const int x = pixel_index - y * width_;

  work_tile->x = full_x_ + x;
  work_tile->y = full_y_ + y;
  work_tile->width = 1;
  work_tile->height = 1;
  work_tile->sample = sample_start_ + sample;

  return true;
}

CCL_NAMESPACE_END
