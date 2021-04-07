/*
 * Copyright 2011-2013 Blender Foundation
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

#ifndef __SESSION_H__
#define __SESSION_H__

#include "device/device.h"
#include "integrator/render_scheduler.h"
#include "render/buffers.h"
#include "render/shader.h"
#include "render/stats.h"
#include "render/tile.h"

#include "util/util_progress.h"
#include "util/util_stats.h"
#include "util/util_thread.h"
#include "util/util_unique_ptr.h"
#include "util/util_vector.h"

CCL_NAMESPACE_BEGIN

class BufferParams;
class Device;
class DeviceScene;
class DeviceRequestedFeatures;
class PathTrace;
class Progress;
class GPUDisplay;
class RenderBuffers;
class Scene;

/* Session Parameters */

class SessionParams {
 public:
  DeviceInfo device;

  bool headless;
  bool background;

  bool experimental;
  int samples;
  int denoising_start_sample;
  int pixel_size;
  int threads;
  bool adaptive_sampling;

  bool use_profiling;

  DenoiseParams denoising;

  ShadingSystem shadingsystem;

  function<bool(const uchar *pixels, int width, int height, int channels)> write_render_cb;

  SessionParams()
  {
    headless = false;
    background = false;

    experimental = false;
    samples = 1024;
    denoising_start_sample = 0;
    pixel_size = 1;
    threads = 0;
    adaptive_sampling = false;

    use_profiling = false;

    shadingsystem = SHADINGSYSTEM_SVM;
  }

  bool modified(const SessionParams &params)
  {
    /* Modified means we have to recreate the session, any parameter changes
     * that can be handled by an existing Session are omitted. */
    return !(device == params.device && headless == params.headless &&
             background == params.background && experimental == params.experimental &&
             pixel_size == params.pixel_size && threads == params.threads &&
             adaptive_sampling == params.adaptive_sampling &&
             use_profiling == params.use_profiling && shadingsystem == params.shadingsystem &&
             denoising.type == params.denoising.type &&
             (denoising.use == params.denoising.use || (device.denoisers & denoising.type)));
  }
};

/* Session
 *
 * This is the class that contains the session thread, running the render
 * control loop and dispatching tasks. */

class Session {
 public:
  Device *device;
  Scene *scene;
  Progress progress;
  SessionParams params;
  BufferParams buffer_params;
  TileManager tile_manager;
  Stats stats;
  Profiler profiler;

  function<void(RenderTile &)> write_render_tile_cb;
  function<void(RenderTile &, bool)> update_render_tile_cb;
  function<void(RenderTile &)> read_bake_tile_cb;

  explicit Session(const SessionParams &params);
  ~Session();

  void start();
  void cancel();
  void draw();
  void wait();

  bool ready_to_reset();
  void reset(BufferParams &params, int samples);

  void set_pause(bool pause);

  void set_samples(int samples);

  void set_denoising(const DenoiseParams &denoising);
  void set_denoising_start_sample(int sample);

  void set_gpu_display(unique_ptr<GPUDisplay> gpu_display);

  void device_free();

  /* Returns the rendering progress or 0 if no progress can be determined
   * (for example, when rendering with unlimited samples). */
  float get_progress();

  void collect_statistics(RenderStats *stats);

 protected:
  struct DelayedReset {
    thread_mutex mutex;
    bool do_reset;
    BufferParams params;
    int samples;
  } delayed_reset;

  void run();

  /* Update for the new iteration of the main loop in run implementation (run_cpu and run_gpu).
   *
   * Will take care of the following things:
   *  - Delayed reset
   *  - Scene update
   *  - Tile manager advance
   *  - Render scheduler work request
   *
   * The updates are done in a proper order with proper locking around them, which guarantees
   * that the device side of scene and render buffers are always in a consistent state.
   *
   * Returns render work which is to be rendered next. */
  RenderWork run_update_for_next_iteration();

  /* Wait for rendering to be unpaused, or for new tiles for render to arrive.
   * Returns true if new main render loop iteration is required after this function call.
   *
   * The `render_work` is the work which was scheduled by the render scheduler right before
   * checking the pause. */
  bool run_wait_for_work(const RenderWork &render_work);

  void run_main_render_loop();

  bool update_scene(int width, int height);

  void update_status_time(bool show_pause = false, bool show_done = false);

  void reset_(BufferParams &params, int samples);

  thread *session_thread;

  bool pause;
  thread_condition_variable pause_cond;
  thread_mutex pause_mutex;
  thread_mutex tile_mutex;
  thread_mutex buffers_mutex;

  /* Render scheduler is used to get work to be rendered with the current big tile. */
  RenderScheduler render_scheduler_;

  /* Path tracer object.
   *
   * Is a single full-frame path tracer for interactive viewport rendering.
   * A path tracer for the current big-tile for an offline rendering. */
  unique_ptr<PathTrace> path_trace_;
};

CCL_NAMESPACE_END

#endif /* __SESSION_H__ */
