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

class AdaptiveSampling;
class Device;
class DeviceScene;
class RenderBuffers;
class RenderScheduler;
class RenderWork;
class PassAccessor;
class Progress;
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
  /* Render scheduler is used to report timing information and access things like start/finish
   * sample. */
  PathTrace(Device *device, DeviceScene *device_scene, RenderScheduler &render_scheduler);

  /* Create devices and load kernels which are created on-demand (for example, denoising devices).
   * The progress is reported to the currently configure progress object (via `set_progress`). */
  void load_kernels();

  /* Check whether now it is a good time to reset rendering.
   * Used to avoid very often resets in the viewport, giving it a chance to draw intermediate
   * render result. */
  bool ready_to_reset();

  /* `full_buffer_params` denotes parameters of the entire big tile which is to be path traced.
   *
   * TODO(sergey): Streamline terminology. Maybe it should be `big_tile_buffer_params`? */
  void reset(const BufferParams &full_buffer_params);

  /* Set progress tracker.
   * Used to communicate details about the progress to the outer world, check whether rendering is
   * to be canceled.
   *
   * The path tracer writes to this object, and then at a convenient moment runs
   * progress_update_cb() callback. */
  void set_progress(Progress *progress);

  /* NOTE: This is a blocking call. Meaning, it will not return until given number of samples are
   * rendered (or until rendering is requested to be cancelled). */
  void render(const RenderWork &render_work);

  /* TODO(sergey): Decide whether denoiser is really a part of path tracer. Currently it is
   * convenient to have it here because then its easy to access render buffer. But the downside is
   * that this adds too much of entities which can live separately with some clear API. */

  /* Set denoiser parameters.
   * Use this to configure the denoiser before rendering any samples. */
  void set_denoiser_params(const DenoiseParams &params);

  /* Set parameters used for adaptive sampling.
   * Use this to configure the adaptive sampler before rendering any samples. */
  void set_adaptive_sampling(const AdaptiveSampling &adaptive_sampling);

  /* Set GPU display which takes care of drawing the render result. */
  void set_gpu_display(unique_ptr<GPUDisplay> gpu_display);

  /* Perform drawing of the current state of the GPUDisplay. */
  void draw();

  /* Cancel rendering process as soon as possible, without waiting for full tile to be sampled.
   * Used in cases like reset of render session.
   *
   * This is a blockign call, which returns as soon as there is no running `render_samples()` call.
   */
  void cancel();

  /* Get pass data of the entire big tile.
   * This call puts pass render result from all devices into the final pixels storage.
   *
   * Returns false if any of the accessor's `get_render_tile_pixels()` returned false. */
  bool get_render_tile_pixels(PassAccessor &pass_accessor, float *pixels);

  /* Generate full multi-line report of the rendering process, including rendering parameters,
   * times, and so on. */
  string full_report() const;

  /* Callback which communicates an updates state of the render buffer.
   * Is called during path tracing to communicate work-in-progress state of the final buffer.
   *
   * The samples indicates how many samples the buffer contains. */
  function<void(void)> buffer_update_cb;

  /* Callback which communicates final rendered buffer. Is called after pathtracing is done.
   *
   * The samples indicates how many samples the buffer contains. */
  function<void(void)> buffer_write_cb;

  /* Callback which is called to report current rendering progress.
   *
   * It is supposed to be cheaper than buffer update/write, hence can be called more often.
   * Additionally, it might be called form the middle of wavefront (meaning, it is not guaranteed
   * that the buffer is "uniformly" sampled at the moment of this callback). */
  function<void(void)> progress_update_cb;

 protected:
  /* Actual implementation of the rendering pipeline.
   * Calls steps in order, checking for the cancel to be requested inbetween.
   *
   * Is separate from `render()` to simplify dealing with the early outputs and keeping
   * `render_cancel_` in the consistent state. */
  void render_pipeline(RenderWork render_work);

  /* Initialize kernel execution on all integrator queues. */
  void render_init_kernel_execution();

  /* Update the render state to possibly changed resolution divider. */
  void render_update_resolution_divider(int resolution_divider);

  /* Perform various steps of the render work.
   *
   * Note that some steps might modify the work, forcing some steps to happen within this iteration
   * of rendering. */
  void path_trace(RenderWork &render_work);
  void adaptive_sample(RenderWork &render_work);
  void denoise(const RenderWork &render_work);
  void update_display(const RenderWork &render_work);

  /* Get number of samples in the current state of the render buffers. */
  int get_num_samples_in_buffer();

  /* Check whether user requested to cancel rendering, so that path tracing is to be finished as
   * soon as possible. */
  bool is_cancel_requested();

  /* Write the big tile render buffer via the write callback. */
  void buffer_write();

  /* Run the progress_update_cb callback if it is needed. */
  void progress_update_if_needed();

  /* Pointer to a device which is configured to be used for path tracing. If multiple devices are
   * configured this is a `MultiDevice`. */
  Device *device_ = nullptr;

  RenderScheduler &render_scheduler_;

  unique_ptr<GPUDisplay> gpu_display_;

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

  /* State which is common for all the steps of the render work.
   * Is brought up to date in the `render()` call and is accessed from all the steps involved into
   * rendering the work. */
  struct {
    /* Divider of the resolution for faster previews.
     *
     * Allows to re-use same render buffer, but have less pixels rendered into in it. The way to
     * think of render buffer in this case is as an over-allocated array: the resolution divider
     * affects both resolution and stride as visible by the integrator kernels. */
    int resolution_divider = 0;

    /* Parameters of render buffers which corresponds to full render buffers divided by the
     * resolution divider. */
    BufferParams scaled_render_buffer_params;
  } render_state_;

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

  /* Indicates whether a render result was drawn after latest session reset.
   * Used by `ready_to_reset()` to implement logic which feels the most interactive. */
  bool did_draw_after_reset_ = true;
};

CCL_NAMESPACE_END
