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
class DisplayBuffer;
class PathTrace;
class Progress;
class GPUDisplay;
class RenderBuffers;
class Scene;

/* Session Parameters */

class SessionParams {
 public:
  DeviceInfo device;
  bool background;

  /* TODO(sergey): Everything is progressive nowadays, consider removing this. */
  bool progressive;

  bool experimental;
  int samples;
  int start_resolution;
  int denoising_start_sample;
  int pixel_size;
  int threads;
  bool adaptive_sampling;

  bool use_profiling;

  DenoiseParams denoising;

  double cancel_timeout;
  double reset_timeout;
  double text_timeout;
  double progressive_update_timeout;

  ShadingSystem shadingsystem;

  function<bool(const uchar *pixels, int width, int height, int channels)> write_render_cb;

  SessionParams()
  {
    background = false;

    progressive = false;
    experimental = false;
    samples = 1024;
    start_resolution = INT_MAX;
    denoising_start_sample = 0;
    pixel_size = 1;
    threads = 0;
    adaptive_sampling = false;

    use_profiling = false;

    cancel_timeout = 0.1;
    reset_timeout = 0.1;
    text_timeout = 1.0;
    progressive_update_timeout = 1.0;

    shadingsystem = SHADINGSYSTEM_SVM;
  }

  bool modified(const SessionParams &params)
  {
    /* Modified means we have to recreate the session, any parameter changes
     * that can be handled by an existing Session are omitted. */
    return !(device == params.device && background == params.background &&
             progressive == params.progressive && experimental == params.experimental &&
             start_resolution == params.start_resolution && pixel_size == params.pixel_size &&
             threads == params.threads && adaptive_sampling == params.adaptive_sampling &&
             use_profiling == params.use_profiling && cancel_timeout == params.cancel_timeout &&
             reset_timeout == params.reset_timeout && text_timeout == params.text_timeout &&
             progressive_update_timeout == params.progressive_update_timeout &&
             shadingsystem == params.shadingsystem && denoising.type == params.denoising.type &&
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
  unique_ptr<GPUDisplay> gpu_display;
  Progress progress;
  SessionParams params;
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

  bool update_scene();

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
   *
   * The updates are done in a proper order with proper locking around them, which guarantees
   * that the device side of scene and render buffers are always in a consistent state.
   *
   * Returns true if there are tiles to be rendered. */
  bool run_update_for_next_iteration();

  /* Wait for rendering to be unpaused, or for new tiles for render to arrive.
   * Returns true if new main render loop iteration is required after this function call.
   *
   * The `no_tiles` argument should be calculated at the state before advancing the tile manager.
   * in practice it means that this is an opposite of what `run_update_for_next_iteration()`
   * returns. */
  bool run_wait_for_work(bool no_tiles);

  void run_main_render_loop();

  void update_status_time(bool show_pause = false, bool show_done = false);

  void render(bool use_denoise);

  void copy_to_display_buffer();

  void reset_(BufferParams &params, int samples);

  /* TODO(sergey): Once the threading synchronization betwee synchronization and render threads is
   * properly implemented there will be no need in this. */
  void set_denoising_no_check(const DenoiseParams &denoising);

  bool render_need_denoise(bool &delayed);

#if 0
  bool steal_tile(RenderTile &tile, Device *tile_device, thread_scoped_lock &tile_lock);
  bool get_tile_stolen();
  bool acquire_tile(RenderTile &tile, Device *tile_device, uint tile_types);
  void update_tile_sample(RenderTile &tile);
  void release_tile(RenderTile &tile, const bool need_denoise);

  void map_neighbor_tiles(RenderTileNeighbors &neighbors, Device *tile_device);
  void unmap_neighbor_tiles(RenderTileNeighbors &neighbors, Device *tile_device);

  bool device_use_gl;
#endif

  thread *session_thread;

  bool pause;
  thread_condition_variable pause_cond;
  thread_mutex pause_mutex;
  thread_mutex tile_mutex;
  thread_mutex buffers_mutex;
#if 0
  thread_condition_variable denoising_cond;
  thread_condition_variable tile_steal_cond;
#endif

  double reset_time;
  double last_display_time;

#if 0
  RenderTile stolen_tile;
  typedef enum {
    NOT_STEALING,     /* There currently is no tile stealing in progress. */
    WAITING_FOR_TILE, /* A device is waiting for another device to release a tile. */
    RELEASING_TILE,   /* A device has releasing a stealable tile. */
    GOT_TILE /* A device has released a stealable tile, which is now stored in stolen_tile. */
  } TileStealingState;
  std::atomic<TileStealingState> tile_stealing_state;
  int stealable_tiles;
#endif

  /* Path tracer object.
   *
   * Is a single full-frame path tracer for interactive viewport rendering.
   * A path tracer for the current big-tile for an offline rendering. */
  unique_ptr<PathTrace> path_trace_;

  /* Indicates whether a render result was drawn after latest session reset.
   * Used by `ready_to_reset()` to implement logic which feels the most interactive. */
  bool did_draw_after_reset_ = false;
};

CCL_NAMESPACE_END

#endif /* __SESSION_H__ */
