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

#include "render/buffers.h"
#include "util/util_types.h"
#include "util/util_unique_ptr.h"

CCL_NAMESPACE_BEGIN

class BufferParams;
class Device;
class DeviceScene;
class GPUDisplay;
class RenderBuffers;

class PathTraceWork {
 public:
  /* Create path trace work which fits best the device.
   *
   * The cancel request flag is used for a cheap check whether cancel is to berformed as soon as
   * possible. This could be, for rexample, request to cancel rendering on camera navigation in
   * viewport. */
  static unique_ptr<PathTraceWork> create(Device *device,
                                          DeviceScene *device_scene,
                                          RenderBuffers *buffers,
                                          bool *cancel_requested_flag);

  virtual ~PathTraceWork();

  /* Set effective parameters within the render buffers.
   *
   * TODO(sergey): Currently is used as a part of an update for resolution divider changes. Might
   * need to become more generic once/if we want to support "re-slicing" of the full render buffer
   * according to the device performance. */
  void set_effective_buffer_params(const BufferParams &effective_buffer_params);

  /* Initialize execution of kernels.
   * Will ensure that all device queues are initialized for execution.
   *
   * This method is to be called after any change in the scene. It is not needed to call it prior
   * to an every call of the `render_samples()`. */
  virtual void init_execution() = 0;

  /* Render given number of samples as a synchronous blocking call.
   * The samples are added to the render buffer associated with this work. */
  virtual void render_samples(int start_sample, int samples_num) = 0;

  /* Copy render result from this work to the corresponding place of the GPU display. */
  virtual void copy_to_gpu_display(GPUDisplay *gpu_display, float sample_scale) = 0;

  /* Perform convergence test on the render buffer, and filter the convergence mask.
   * Returns number of active pixels (the ones which did not converge yet). */
  virtual int adaptive_sampling_converge_filter_count_active(float threshold, bool reset) = 0;

  /* Cheap-ish request to see whether rendering is requested and is to be stopped as soon as
   * possible, without waiting for any samples to be finished. */
  inline bool is_cancel_requested() const
  {
    /* NOTE: Rely on the fact that on x86 CPU reading scalar can happen without atomic even in
     * threaded environment. */
    return *cancel_requested_flag_;
  }

  /* Access to the device which is used to path trace this work on. */
  Device *get_device() const
  {
    return device_;
  }

 protected:
  PathTraceWork(Device *device,
                DeviceScene *device_scene,
                RenderBuffers *buffers,
                bool *cancel_requested_flag);

  /* Device which will be used for path tracing.
   * Note that it is an actual render device (and never is a multi-device). */
  Device *device_;

  /* Device side scene storage, that may be used for integrator logic. */
  DeviceScene *device_scene_;

  /* Render buffers where sampling is being accumulated into.
   * It also defines possible subset of a big tile in the case of multi-device rendering. */
  RenderBuffers *buffers_;

  /* Effective parameters of the render buffer.
   * Might be different from buffers_->params when there is a resolution divider involved. */
  BufferParams effective_buffer_params_;

  bool *cancel_requested_flag_ = nullptr;
};

CCL_NAMESPACE_END
