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

#include "kernel/integrator/integrator_path_state.h"
#include "kernel/integrator/integrator_state.h"

#include "device/device_memory.h"
#include "device/device_queue.h"

#include "integrator/path_trace_work.h"
#include "integrator/work_scheduler.h"

#include "util/util_vector.h"

CCL_NAMESPACE_BEGIN

struct KernelWorkTile;

/* Implementation of PathTraceWork which schedules work to the device in tiles which are sized
 * to match device queue's number of path states.
 * This implementation suits best devices which have a lot of integrator states, such as GPU. */
class PathTraceWorkGPU : public PathTraceWork {
 public:
  PathTraceWorkGPU(Device *render_device, RenderBuffers *buffers, bool *cancel_requested_flag);

  virtual void init_execution() override;

  virtual void render_samples(int start_sample, int samples_num) override;

 protected:
  bool enqueue_work_tiles(bool &finished);
  void enqueue_work_tiles(DeviceKernel kernel,
                          const KernelWorkTile work_tiles[],
                          const int num_work_tiles);

  bool enqueue_path_iteration();
  void enqueue_path_iteration(DeviceKernel kernel);

  void compute_queued_paths(DeviceKernel kernel, int queued_kernel);

  int get_num_active_paths();

  int get_max_num_paths();

  /* Integrator queues.
   * There are as many of queues as the concurrent queues the device supports. */
  unique_ptr<DeviceQueue> queue_;

  /* Scheduler which gives work to path tracing threads. */
  WorkScheduler work_scheduler_;

  /* Output render buffer. */
  RenderBuffers *render_buffers_;

  /* Integrate state for paths. */
  device_only_memory<IntegratorState> integrator_state_;
  /* Keep track of number of queued kernels. */
  device_vector<IntegratorPathQueue> integrator_path_queue_;

  /* Temporary buffer to get an array of queued path for a particular kernel. */
  device_vector<int> queued_paths_;
  device_vector<int> num_queued_paths_;

  /* Temporary buffer for passing work tiles to kernel. */
  device_vector<KernelWorkTile> work_tiles_;

  /* Maximum path index, effective number of paths used may be smaller than
   * the size of the integrator_state_ buffer so can avoid iterating over the
   * full buffer. */
  int max_active_path_index_;
};

CCL_NAMESPACE_END
