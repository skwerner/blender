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

class Device;

/* PathTrace class takes care of kernel graph and scheduling on a (multi)device. It takes care of
 * all the common steps of path tracing which are not device-specific. The list of tasks includes
 * but is not limited to:
 *  - Kernel graph.
 *  - Scheduling logic.
 *  - Queues management.
 *  - Adaptive stopping. */
class PathTrace {
 public:
  /* TODO(sergey): Need to provide the following information:
   *  - (Device)Scene.
   *  - Render tile (could be `ccl::Tile`, but to avoid bad level call it needs to be moved to this
   *    module).
   *  - Render buffer. */
  explicit PathTrace(Device *device);

  /* Request render of the given number of tiles.
   *
   * TODO(sergey): Decide and document whether it is a blocking or asynchronous call. */
  void render_samples(int samples_num);

 protected:
  /* Render given number of samples on the given device.
   *
   * This is a blocking call and it is yp to caller to call it from thread if asynchronous
   * execution is needed. */
  void render_samples_on_device(Device *device, int samples_num);

  /* Pointer to a device which is configured to be used for path tracing. If multiple devices are
   * configured this is a `MultiDevice`. */
  Device *device_ = nullptr;
};

CCL_NAMESPACE_END
