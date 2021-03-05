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

#include "device/device_kernel.h"

CCL_NAMESPACE_BEGIN

class Device;
class RenderBuffers;

struct KernelWorkTile;

/* Abstraction of a command queue for a device.
 * Provides API to schedule kernel execution in a specific queue with minimal possible overhead
 * from driver side.
 *
 * This class encapsulates all properties needed for commands execution. */
class DeviceQueue {
 public:
  virtual ~DeviceQueue() = default;

  /* Initialize execution of kernels on this queue.
   *
   * Will, for example, load all data required by the kernels from Device to global or path state.
   *
   * Use this method after device synchronization has finished before enqueueing any kernels. */
  virtual void init_execution() = 0;

  /* Enqueue kernel execution. */
  virtual void enqueue(DeviceKernel kernel) = 0;

  /* Enqueue kernel with work tiles, to initialize paths.
   *
   * Work tiles indicate the subset of the image buffer the paths will be initialized for.
   * Multiple tiles may be scheduled at once for more flexible division of the buffer.
   *
   * TODO(sergey): See in the future if it's a concept usable for all queues, or whether it is
   * specific to render queue. */
  virtual void enqueue_work_tiles(DeviceKernel kernel,
                                  const KernelWorkTile work_tiles[],
                                  const int num_work_tiles) = 0;

  /* Get number of active paths in the queue. */
  virtual int get_num_active_paths() = 0;

  /* Get maximum number of paths which can be held by this queue.
   *
   * The number of path is determined by factors like number of threads on the device,
   * amount of free memory on the device and so on.
   *
   * This value is used by an external world to effectively implement scheduling on a
   * (multi)device. */
  virtual int get_max_num_paths() = 0;

  /* Device this queue has been created for. */
  Device *device;

 protected:
  /* Hide construction so that allocation via `Device` API is enforced. */
  explicit DeviceQueue(Device *device);
};

CCL_NAMESPACE_END
