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

class BufferParams;
class DeviceWorkTile;

/* Scheduler of device work tiles.
 * Takes care of feeding multiple devices running in parallel a work which needs to be done. */
class WorkScheduler {
 public:
  WorkScheduler();

  /* Scheduling will happen for pixels within a big tile denotes by its parameters. */
  void reset(const BufferParams &buffer_params, int sample_start, int samples_num);

  /* Get work for a device.
   * Returns truth if there is still work to be done and initialied the work tile to all
   * parameters of this work. If there is nothing remained to be done, returns false and the
   * work tile is kept unchanged. */
  bool get_work(DeviceWorkTile *work_tile);

 protected:
  void reset_scheduler_state();

  int full_x_ = 0;
  int full_y_ = 0;

  int width_ = 0;
  int height_ = 0;

  /* Offset and stride of the buffer within which scheduing is happenning.
   * Will be passed over to the KernelWorkTile. */
  int offset_, stride_;

  int sample_start_ = 0;
  int samples_num_ = 0;

  int total_pixels_num_ = 0;

  int next_work_index_ = 0;
  int total_work_size_ = 0;
};

CCL_NAMESPACE_END
