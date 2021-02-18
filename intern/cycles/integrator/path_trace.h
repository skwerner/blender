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
#include "util/util_function.h"

CCL_NAMESPACE_BEGIN

class Device;
class RenderBuffers;

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
  /* `buffer_params` denotes parameters of the entire big tile which is to be path traced. */
  PathTrace(Device *device, const BufferParams &buffer_params);

  /* Request render of the given number of tiles.
   *
   * TODO(sergey): Decide and document whether it is a blocking or asynchronous call. */
  void render_samples(int samples_num);

  /* Callback which is used top check whether user requested to cancel rendering.
   * If this callback is not assigned the path tracing procfess can not be cancelled and it will be
   * finished when it fully sampled all requested samples. */
  function<bool(void)> get_cancel_cb;

  /* Callback which communicates an updates state of the render buffer.
   * Is called during path tracing to communicate work-in-progress state of the final buffer.
   *
   * The samples indicates how many samples the buffer contains. */
  function<void(RenderBuffers *render_buffers, int sample)> update_cb;

  /* The update callback will never be run more often that this interval, avoiding overhead of
   * data communication on a simple renders.  */
  double update_interval_in_seconds = 1.0;

  /* Callback which communicates final rendered buffer. Is called after pathtracing is done.
   *
   * The samples indicates how many samples the buffer contains. */
  function<void(RenderBuffers *render_buffers, int sample)> write_cb;

 protected:
  /* Render given number of samples on the given device.
   *
   * Buffer params denotes parameters of a buffer which is to be rendered on this device. In the
   * case of multi-device rendering this will be a smaller portion of the `buffer_params_`.
   *
   * This is a blocking call and it is yp to caller to call it from thread if asynchronous
   * execution is needed. */
  void render_samples_on_device(Device *device,
                                const BufferParams &buffer_params,
                                int samples_num);

  /* Check whether user requested to cancel rendering, so that path tracing is to be finished as
   * soon as possible. */
  bool is_cancel_requested();

  /* Used before path tracing begins, so that all updates can happen as user expects them. */
  void update_reset_status();

  /* Run an update callback if needed.
   * This call which check whether an update callback is configured, and do other optimization
   * checks. For example, the update will not be communicated if update happens too often, so that
   * the overhead of update does not degrade rendering performance.
   *
   * The samples indicates how many samples the buffer contains. */
  /* TODO(sergey): Ideally the render buffers will be passed by the const reference. */
  void update_if_needed(RenderBuffers *render_buffers, int sample);

  /* Write the render buffer via the write callback.
   *
   * The samples indicates how many samples the buffer contains. */
  /* TODO(sergey): Ideally the render buffers will be passed by the const reference. */
  void write(RenderBuffers *render_buffers, int sample);

  /* Pointer to a device which is configured to be used for path tracing. If multiple devices are
   * configured this is a `MultiDevice`. */
  Device *device_ = nullptr;

  /* Parameters of buffers which corresponds to the big tile. */
  /* TODO(sergey): Consider addressing naming a bit, to make it explicit that this is a big
   * buffer. Alternatively, can consider introducing BigTile as entity concept. */
  BufferParams buffer_params_;

  /* Status for the update reporting.
   * Is used to avoid updates being sent too often. */
  struct {
    /* Denotes whether update callback was ever called during the current path tracing process. */
    bool has_update;
    /* Timestamp of when the update callback was last call (only valid if `has_update` is true.) */
    double last_update_time;
  } update_status;
};

CCL_NAMESPACE_END
