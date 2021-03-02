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
#include "integrator/tile.h"
#include "render/buffers.h"
#include "util/util_atomic.h"
#include "util/util_logging.h"

CCL_NAMESPACE_BEGIN

WorkScheduler::WorkScheduler()
{
}

void WorkScheduler::set_max_num_path_states(int max_num_path_states)
{
  max_num_path_states_ = max_num_path_states;
}

void WorkScheduler::reset(const BufferParams &buffer_params, int sample_start, int samples_num)
{
  /* Image buffer parameters. */
  image_full_offset_px_.x = buffer_params.full_x;
  image_full_offset_px_.y = buffer_params.full_y;

  image_size_px_ = make_int2(buffer_params.width, buffer_params.height);

  buffer_params.get_offset_stride(offset_, stride_);

  /* Samples parameters. */
  sample_start_ = sample_start;
  samples_num_ = samples_num;

  /* Initialize new scheduling. */
  reset_scheduler_state();
}

void WorkScheduler::reset_scheduler_state()
{
  tile_size_ = tile_calculate_best_size(image_size_px_, samples_num_, max_num_path_states_);

  num_tiles_x_ = divide_up(image_size_px_.x, tile_size_.x);
  num_tiles_y_ = divide_up(image_size_px_.y, tile_size_.y);

  total_tiles_num_ = num_tiles_x_ * num_tiles_y_;

  next_work_index_ = 0;
  total_work_size_ = total_tiles_num_ * samples_num_;
}

bool WorkScheduler::get_work(DeviceWorkTile *work_tile)
{
  DCHECK_NE(max_num_path_states_, 0);

  /* TODO(sergey): Implement some smarter work scheduler which will be able to scheduler tile sizes
   * different from 1x1. Currently this is a bare minimum for CPU devices. */

  const int work_index = atomic_fetch_and_add_int32(&next_work_index_, 1);
  if (work_index >= total_work_size_) {
    return false;
  }

  const int sample = work_index / total_tiles_num_;
  const int tile_index = work_index - sample * total_tiles_num_;
  const int tile_y = tile_index / num_tiles_x_;
  const int tile_x = tile_index - tile_y * num_tiles_x_;

  work_tile->x = image_full_offset_px_.x + tile_x * tile_size_.x;
  work_tile->y = image_full_offset_px_.y + tile_y * tile_size_.y;
  work_tile->width = tile_size_.x;
  work_tile->height = tile_size_.y;
  work_tile->sample = sample_start_ + sample;
  work_tile->offset = offset_;
  work_tile->stride = stride_;

  work_tile->width = min(work_tile->width, image_size_px_.x - work_tile->x);
  work_tile->height = min(work_tile->height, image_size_px_.y - work_tile->y);

  return true;
}

CCL_NAMESPACE_END
