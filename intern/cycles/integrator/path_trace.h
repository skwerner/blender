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

#include "integrator/denoiser.h"
#include "integrator/path_trace_work.h"
#include "render/buffers.h"
#include "util/util_function.h"
#include "util/util_thread.h"
#include "util/util_unique_ptr.h"
#include "util/util_vector.h"

CCL_NAMESPACE_BEGIN

class Device;
class DeviceQueue;
class RenderBuffers;
class Progress;

/* TODO(sergey): See if it still will be needed for the final implementation. */
class GPUDisplay;

/* PathTrace class takes care of kernel graph and scheduling on a (multi)device. It takes care of
 * all the common steps of path tracing which are not device-specific. The list of tasks includes
 * but is not limited to:
 *  - Kernel graph.
 *  - Scheduling logic.
 *  - Queues management.
 *  - Adaptive stopping. */
class PathTrace {
 public:
  explicit PathTrace(Device *device);

  /* `full_buffer_params` denotes parameters of the entire big tile which is to be path traced.
   *
   * TODO(sergey): Streamline terminology. Maybe it should be `big_tile_buffer_params`? */
  void reset(const BufferParams &full_buffer_params);

  /* Clear current render buffer.
   * Used during interactive viewport rendering rendering to force refresh accumulated result on
   * resolution change. */
  void clear_render_buffers();

  /* Configure the path tracer to perform lower resolution rendering into the full frame buffer. */
  void set_resolution_divider(int resolution_divider);

  /* Is used to either offset sampling when adding more samples to an existing render, or to
   * offset sample for a viewport render.
   *
   * The sample is 0-based. */
  void set_start_sample(int start_sample_num);

  /* Set progress tracker.
   * Used to communicate details about the progress to the outer world, check whether rendering is
   * to be canceled.
   *
   * The path tracer writes to this object, and then at a convenient moment runs
   * progress_update_cb() callback. */
  void set_progress(Progress *progress);

  /* Request render of the given number of samples.
   * Will add [start_sample_num, start_sample_num + samples_num) samples to the render buffer.
   *
   * NOTE: This is a blocking cal. Meaning, it will not return until given number of samples are
   * rendered (or until rendering is requested to be cancelled). */
  void render_samples(int samples_num);

  /* TODO(sergey): Decide whether denoiser is really a part of path tracer. Currently it is
   * convenient to have it here because then its easy to access render buffer. But the downside is
   * that this adds too much of entities which can live separately with some clear API. */

  /* Set denoiser parameters.
   * Use this to configure the denoiser before rendering any samples. */
  void set_denoiser_params(const DenoiseParams &params);

  /* Denoise current state of the big tile. */
  void denoise();

  /* TODO(sergey): This is a quick implementation for tests. */
  void copy_to_gpu_display(GPUDisplay *gpu_display);

  /* Cancel rendering process as soon as possible, without waiting for full tile to be sampled.
   * Used in cases like reset of render session.
   *
   * This is a blockign call, which returns as soon as there is no running `render_samples()` call.
   */
  void cancel();

  /* Callback which communicates an updates state of the render buffer.
   * Is called during path tracing to communicate work-in-progress state of the final buffer.
   *
   * The samples indicates how many samples the buffer contains. */
  function<void(RenderBuffers *render_buffers, int sample)> buffer_update_cb;

  /* The update callback will never be run more often that this interval, avoiding overhead of
   * data communication on a simple renders.  */
  double update_interval_in_seconds = 1.0;

  /* Callback which communicates final rendered buffer. Is called after pathtracing is done.
   *
   * The samples indicates how many samples the buffer contains. */
  function<void(RenderBuffers *render_buffers, int sample)> buffer_write_cb;

  /* Callback which is called to report current rendering progress.
   *
   * It is supposed to be cheaper than buffer update/write, hence can be called more often.
   * Additionally, it might be called form the middle of wavefront (meaning, it is not guaranteed
   * that the buffer is "uniformly" sampled at the moment of this callback). */
  function<void(void)> progress_update_cb;

 protected:
  /* Update resolution stored in the `scaled_render_buffer_params_`.
   * Used to bring the scaled parameters up to date on either full render buffers change, or on
   * resolution divider change. */
  void update_scaled_render_buffers_resolution();

  /* Initialize kernel execution on all integrator queues. */
  void render_init_execution();

  /* Run full render pipeline on all devices to add the given number of samples to the render
   * result.
   *
   * There are no update callbacks or cancellation checks are done form here, for the performance
   * reasons.
   *
   * This call advances number of samples stored in the render status.
   *
   * Returns time in seconds which it took to render. */
  double render_samples_full_pipeline(int samples_num);

  /* Get number of samples in the current state of the render buffers. */
  int get_num_samples_in_buffer();

  /* Check whether user requested to cancel rendering, so that path tracing is to be finished as
   * soon as possible. */
  bool is_cancel_requested();

  /* Run a buffer update callback if needed.
   *
   * This call which check whether an update callback is configured, and do other optimization
   * checks. For example, the update will not be communicated if update happens too often, so that
   * the overhead of update does not degrade rendering performance. */
  void buffer_update_if_needed();

  /* Write the big tile render buffer via the write callback. */
  void buffer_write();

  /* Run the progress_update_cb callback if it is needed. */
  void progress_update_if_needed();

  /* Pointer to a device which is configured to be used for path tracing. If multiple devices are
   * configured this is a `MultiDevice`. */
  Device *device_ = nullptr;

  /* Per-compute device descriptors of work which is responsible for path tracing on its configured
   * device. */
  vector<unique_ptr<PathTraceWork>> path_trace_works_;

  /* Render buffer which corresponds to the big tile.
   * It is used to accumulate work from all rendering devices, and to communicate render result
   * to the render session.
   *
   * TODO(sergey): This is actually a subject for reconsideration when multi-device support will be
   * added. */
  unique_ptr<RenderBuffers> full_render_buffers_;

  /* Denoiser which takes care of denoising the big tile. */
  unique_ptr<Denoiser> denoiser_;

  /* Number of a start sample, in the 0 based notation. */
  /* TODO(sergey): Consider moving insode of RenderState. */
  int start_sample_num_ = 0;

  /* Divider of the resolution for faster previews.
   *
   * Allows to re-use same render buffer, but have less pixels rendered into in it. The way to
   * think of render buffer in this case is as an over-allocated array: the resolution divider
   * affects both resolution and stride as visible by the integrator kernels. */
  int resolution_divider_ = 1;

  /* Parameters of render buffers which corresponds to full render buffers divided by the
   * resolution divider. */
  BufferParams scaled_render_buffer_params_;

  /* Global path tracing status. */
  /* TODO(sergey): state vs. status. */
  struct RenderStatus {
    /* Reset status before new render begins. */
    void reset();

    /* Number of samples in the render buffer. */
    int rendered_samples_num;
  };
  RenderStatus render_status_;

  /* Status for the update reporting.
   * Is used to avoid updates being sent too often. */
  struct UpdateStatus {
    /* Used before path tracing begins, so that all updates can happen as user expects them. */
    void reset();

    /* Denotes whether update callback was ever called during the current path tracing process. */
    bool has_update;
    /* Timestamp of when the update callback was last call (only valid if `has_update` is true.) */
    double last_update_time;
  };
  UpdateStatus update_status_;

  /* Progress object which is used to communicate sample progress. */
  Progress *progress_;

  /* Fields required for canceling render on demand, as quickly as possible. */
  struct {
    /* Indicates whether there is an on-going `render_samples()` call. */
    bool is_rendering = false;

    /* Indicates whether rendering is requested to be canceled by `cancel()`. */
    bool is_requested = false;

    /* Synchronization between thread which does `render_samples()` and thread which does
     * `cancel()`. */
    thread_mutex mutex;
    thread_condition_variable condition;
  } render_cancel_;
};

CCL_NAMESPACE_END
