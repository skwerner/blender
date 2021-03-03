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

#include "util/util_unique_ptr.h"

CCL_NAMESPACE_BEGIN

class BufferParams;
class Device;
class RenderBuffers;

class PathTraceWork {
 public:
  // Create path trace work which fits best the device.
  static unique_ptr<PathTraceWork> create(Device *render_device, RenderBuffers *buffers);

  virtual ~PathTraceWork();

  /* Initialize execution of kernels.
   * Will ensure that all device queues are initialized for execution.
   *
   * This method is to be called after any change in the scene. It is not needed to call it prior
   * to an every call of the `render_samples()`. */
  virtual void init_execution() = 0;

  /* Render given number of samples as a synchronous blocking call.
   * The samples are added to the render buffer associated with this work. */
  virtual void render_samples(const BufferParams &scaled_render_buffer_params,
                              int start_sample,
                              int samples_num) = 0;

 protected:
  PathTraceWork(Device *render_device, RenderBuffers *buffers);

  // Render device which will be used for path tracing.
  // Note that it is an actual render device (and never is a multi-device).
  Device *render_device_;

  /* Render buffers where sampling is being accumulated into.
   * It also defines possible subset of a big tile in the case of multi-device rendering. */
  RenderBuffers *buffers_;
};

CCL_NAMESPACE_END
