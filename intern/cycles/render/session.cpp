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

#include <limits.h>
#include <string.h>

#include "device/device.h"
#include "integrator/path_trace.h"
#include "render/bake.h"
#include "render/buffers.h"
#include "render/camera.h"
#include "render/gpu_display.h"
#include "render/graph.h"
#include "render/integrator.h"
#include "render/light.h"
#include "render/mesh.h"
#include "render/object.h"
#include "render/scene.h"
#include "render/session.h"

#include "util/util_foreach.h"
#include "util/util_function.h"
#include "util/util_logging.h"
#include "util/util_math.h"
#include "util/util_opengl.h"
#include "util/util_task.h"
#include "util/util_time.h"

CCL_NAMESPACE_BEGIN

#if 0
/* Note about  preserve_tile_device option for tile manager:
 * progressive refine and viewport rendering does requires tiles to
 * always be allocated for the same device
 */
#endif
Session::Session(const SessionParams &params_)
    : params(params_),
      tile_manager(params.progressive,
                   params.samples,
#if 0
                    make_int2(64, 64),
#endif
                   params.start_resolution,
#if 0
                    params.background == false,
                    params.background,
                    TILE_BOTTOM_TO_TOP,
                    max(params.device.multi_devices.size(), 1),
#endif
                   params.pixel_size),
      stats(),
      profiler()
{
  TaskScheduler::init(params.threads);

  session_thread = NULL;
  scene = NULL;

  delayed_reset.do_reset = false;
  delayed_reset.samples = 0;

  pause = false;

  /* Create CPU/GPU devices. */
  /* Special trick to have current PathTracer happy: replace multi-device which has single
   * sub-device with the single device of that type. This is required because currently PathTraces
   * makes some assumptions that the device is a single device. */
  /* TODO(sergey): Remove this once multi-device is fully supported by the PathTracer. */
  if (params.device.multi_devices.empty()) {
    device = Device::create(params.device, stats, profiler, params.background);
  }
  else if (params.device.multi_devices.size() == 1) {
    device = Device::create(params.device.multi_devices[0], stats, profiler, params.background);
  }
  else {
    LOG(ERROR) << "Multi-devices are not yet fully implemented.";
    device = Device::create(params.device, stats, profiler, params.background);
  }

  if (!device->error_message().empty()) {
    progress.set_error(device->error_message());
    return;
  }

  /* Configure path tracer. */
  path_trace_ = make_unique<PathTrace>(device);
  path_trace_->set_progress(&progress);
  path_trace_->buffer_update_cb = [&](RenderBuffers *render_buffers, int sample) {
    /* TODO(sergey): Needs proper implementation, with all the update callbacks and status ported
     * the the new progressive+adaptive nature of rendering. */

    if (!update_render_tile_cb) {
      return;
    }

    RenderTile render_tile;
    render_tile.x = render_buffers->params.full_x;
    render_tile.y = render_buffers->params.full_y;
    render_tile.w = render_buffers->params.width;
    render_tile.h = render_buffers->params.height;
    render_tile.sample = sample;
    render_tile.buffers = render_buffers;
    render_tile.task = RenderTile::PATH_TRACE;

    update_render_tile_cb(render_tile, false);
  };
  path_trace_->buffer_write_cb = [&](RenderBuffers *render_buffers, int sample) {
    /* NOTE: Almost a duplicate of `update_cb`, but is done for a testing purposes, neither of
     * this callbacks are final. */

    /* TODO(sergey): Needs proper implementation, with all the update callbacks and status ported
     * the the new progressive+adaptive nature of rendering. */

    if (!write_render_tile_cb) {
      return;
    }

    RenderTile render_tile;
    render_tile.x = render_buffers->params.full_x;
    render_tile.y = render_buffers->params.full_y;
    render_tile.w = render_buffers->params.width;
    render_tile.h = render_buffers->params.height;
    render_tile.sample = sample;
    render_tile.buffers = render_buffers;
    render_tile.task = RenderTile::PATH_TRACE;

    write_render_tile_cb(render_tile);
  };
  path_trace_->progress_update_cb = [&]() { update_status_time(); };

  /* Validate denoising parameters. */
  set_denoising_no_check(params.denoising);
}

Session::~Session()
{
  cancel();

  /* TODO(sergey): Bring the passes in viewport back.
   * It is unclear why there is such an exception needed though. */
#if 0
  if (buffers && params.write_render_cb) {
    /* Copy to display buffer and write out image if requested */
    delete display;

    display = new DisplayBuffer(device, false);
    display->reset(buffers->params);
    copy_to_display_buffer(params.samples);

    int w = display->draw_width;
    int h = display->draw_height;
    uchar4 *pixels = display->rgba_byte.copy_from_device(0, w, h);
    params.write_render_cb((uchar *)pixels, w, h, 4);
  }
#endif

  /* Make sure path tracer is destroyed before the deviec. This is needed because destruction might
   * need to access device for device memory free. */
  /* TODO(sergey): Convert device to be unique_ptr, and rely on C++ to destruct objects in the
   * pre-defined order. */
  path_trace_.reset();

#if 0
  /* clean up */
  tile_manager.device_free();
#endif

  delete scene;
  delete device;

  TaskScheduler::exit();
}

void Session::start()
{
  if (!session_thread) {
    session_thread = new thread(function_bind(&Session::run, this));
  }
}

void Session::cancel()
{
  if (session_thread) {
    /* wait for session thread to end */
    progress.set_cancel("Exiting");

    {
      thread_scoped_lock pause_lock(pause_mutex);
      pause = false;
    }
    pause_cond.notify_all();

    wait();
  }
}

bool Session::ready_to_reset()
{
  /* The logic here is optimized for the best feedback in the viewport, which implies having a GPU
   * display. Of there is no such display, the logic here will break. */
  DCHECK(gpu_display);

  /* The logic here tries to provide behavior which feels the most interactive feel to artists.
   * General idea is to be able to reset as quickly as possible, while still providing interactive
   * feel.
   *
   * If the render result was ever drawn after previous reset, consider that reset is now possible.
   * This way camera navigation gives the quickest feedback of rendered pixels, regardless of
   * whether CPU or GPU drawing pipeline is used.
   *
   * Consider reset happening after redraw "slow" enough to not clog anything. This is a bit
   * arbitrary, but seems to work very well with viewport navigation in Blender. */

  if (did_draw_after_reset_) {
    return true;
  }

  return false;
}

#if 0
bool Session::steal_tile(RenderTile &rtile, Device *tile_device, thread_scoped_lock &tile_lock)
{
  /* Devices that can get their tiles stolen don't steal tiles themselves.
   * Additionally, if there are no stealable tiles in flight, give up here. */
  if (tile_device->info.type == DEVICE_CPU || stealable_tiles == 0) {
    return false;
  }

  /* Wait until no other thread is trying to steal a tile. */
  while (tile_stealing_state != NOT_STEALING && stealable_tiles > 0) {
    /* Someone else is currently trying to get a tile.
     * Wait on the condition variable and try later. */
    tile_steal_cond.wait(tile_lock);
  }
  /* If another thread stole the last stealable tile in the meantime, give up. */
  if (stealable_tiles == 0) {
    return false;
  }

  /* There are stealable tiles in flight, so signal that one should be released. */
  tile_stealing_state = WAITING_FOR_TILE;

  /* Wait until a device notices the signal and releases its tile. */
  while (tile_stealing_state != GOT_TILE && stealable_tiles > 0) {
    tile_steal_cond.wait(tile_lock);
  }
  /* If the last stealable tile finished on its own, give up. */
  if (tile_stealing_state != GOT_TILE) {
    tile_stealing_state = NOT_STEALING;
    return false;
  }

  /* Successfully stole a tile, now move it to the new device. */
  rtile = stolen_tile;
  rtile.buffers->buffer.move_device(tile_device);
  rtile.buffer = rtile.buffers->buffer.device_pointer;
  rtile.stealing_state = RenderTile::NO_STEALING;
  rtile.num_samples -= (rtile.sample - rtile.start_sample);
  rtile.start_sample = rtile.sample;

  tile_stealing_state = NOT_STEALING;

  /* Poke any threads which might be waiting for NOT_STEALING above. */
  tile_steal_cond.notify_one();

  return true;
}

bool Session::get_tile_stolen()
{
  /* If tile_stealing_state is WAITING_FOR_TILE, atomically set it to RELEASING_TILE
   * and return true. */
  TileStealingState expected = WAITING_FOR_TILE;
  return tile_stealing_state.compare_exchange_weak(expected, RELEASING_TILE);
}

bool Session::acquire_tile(RenderTile &rtile, Device *tile_device, uint tile_types)
{
  if (progress.get_cancel()) {
    return false;
  }

  thread_scoped_lock tile_lock(tile_mutex);

  /* get next tile from manager */
  Tile *tile;
  int device_num = device->device_number(tile_device);

  while (!tile_manager.next_tile(tile, device_num, tile_types)) {
    /* Can only steal tiles on devices that support rendering
     * This is because denoising tiles cannot be stolen (see below)
     */
    if ((tile_types & (RenderTile::PATH_TRACE | RenderTile::BAKE)) &&
        steal_tile(rtile, tile_device, tile_lock)) {
      return true;
    }

    /* Wait for denoising tiles to become available */
    if ((tile_types & RenderTile::DENOISE) && !progress.get_cancel() && tile_manager.has_tiles()) {
      denoising_cond.wait(tile_lock);
      continue;
    }

    return false;
  }

  /* fill render tile */
  rtile.x = tile_manager.state.buffer.full_x + tile->x;
  rtile.y = tile_manager.state.buffer.full_y + tile->y;
  rtile.w = tile->w;
  rtile.h = tile->h;
  rtile.start_sample = tile_manager.state.sample;
  rtile.num_samples = tile_manager.state.num_samples;
  rtile.resolution = tile_manager.state.resolution_divider;
  rtile.tile_index = tile->index;
  rtile.stealing_state = RenderTile::NO_STEALING;

  if (tile->state == Tile::DENOISE) {
    rtile.task = RenderTile::DENOISE;
  }
  else {
    if (tile_device->info.type == DEVICE_CPU) {
      stealable_tiles++;
      rtile.stealing_state = RenderTile::CAN_BE_STOLEN;
    }

    if (read_bake_tile_cb) {
      rtile.task = RenderTile::BAKE;
    }
    else {
      rtile.task = RenderTile::PATH_TRACE;
    }
  }

  tile_lock.unlock();

  /* in case of a permanent buffer, return it, otherwise we will allocate
   * a new temporary buffer */
  if (buffers) {
    tile_manager.state.buffer.get_offset_stride(rtile.offset, rtile.stride);

    rtile.buffer = buffers->buffer.device_pointer;
    rtile.buffers = buffers;

    device->map_tile(tile_device, rtile);

    /* Reset copy state, since buffer contents change after the tile was acquired */
    buffers->map_neighbor_copied = false;

    /* This hack ensures that the copy in 'MultiDevice::map_neighbor_tiles' accounts
     * for the buffer resolution divider. */
    buffers->buffer.data_width = (buffers->params.width * buffers->params.get_passes_size()) /
                                 tile_manager.state.resolution_divider;
    buffers->buffer.data_height = buffers->params.height / tile_manager.state.resolution_divider;

    return true;
  }

  if (tile->buffers == NULL) {
    /* fill buffer parameters */
    BufferParams buffer_params = tile_manager.params;
    buffer_params.full_x = rtile.x;
    buffer_params.full_y = rtile.y;
    buffer_params.width = rtile.w;
    buffer_params.height = rtile.h;

    /* allocate buffers */
    tile->buffers = new RenderBuffers(tile_device);
    tile->buffers->reset(buffer_params);
  }
  else if (tile->buffers->buffer.device != tile_device) {
    /* Move buffer to current tile device again in case it was stolen before.
     * Not needed for denoising since that already handles mapping of tiles and
     * neighbors to its own device. */
    if (rtile.task != RenderTile::DENOISE) {
      tile->buffers->buffer.move_device(tile_device);
    }
  }

  tile->buffers->map_neighbor_copied = false;

  tile->buffers->params.get_offset_stride(rtile.offset, rtile.stride);

  rtile.buffer = tile->buffers->buffer.device_pointer;
  rtile.buffers = tile->buffers;
  rtile.sample = tile_manager.state.sample;

  if (read_bake_tile_cb) {
    /* This will read any passes needed as input for baking. */
    if (tile_manager.state.sample == tile_manager.range_start_sample) {
      {
        thread_scoped_lock tile_lock(tile_mutex);
        read_bake_tile_cb(rtile);
      }
      rtile.buffers->buffer.copy_to_device();
    }
  }
  else {
    /* This will tag tile as IN PROGRESS in blender-side render pipeline,
     * which is needed to highlight currently rendering tile before first
     * sample was processed for it. */
    update_tile_sample(rtile);
  }

  return true;
}

void Session::update_tile_sample(RenderTile &rtile)
{
  thread_scoped_lock tile_lock(tile_mutex);

  if (update_render_tile_cb) {
    update_render_tile_cb(rtile, true);
  }

  update_status_time();
}

void Session::release_tile(RenderTile &rtile, const bool need_denoise)
{
  thread_scoped_lock tile_lock(tile_mutex);

  if (rtile.stealing_state != RenderTile::NO_STEALING) {
    stealable_tiles--;
    if (rtile.stealing_state == RenderTile::WAS_STOLEN) {
      /* If the tile is being stolen, don't release it here - the new device will pick up where
       * the old one left off. */

      assert(tile_stealing_state == RELEASING_TILE);
      assert(rtile.sample < rtile.start_sample + rtile.num_samples);

      tile_stealing_state = GOT_TILE;
      stolen_tile = rtile;
      tile_steal_cond.notify_all();
      return;
    }
    else if (stealable_tiles == 0) {
      /* If this was the last stealable tile, wake up any threads still waiting for one. */
      tile_steal_cond.notify_all();
    }
  }

  progress.add_finished_tile(rtile.task == RenderTile::DENOISE);

  bool delete_tile;

  if (tile_manager.finish_tile(rtile.tile_index, need_denoise, delete_tile)) {
    /* Finished tile pixels write. */
    if (write_render_tile_cb) {
      write_render_tile_cb(rtile);
    }

    if (delete_tile) {
      delete rtile.buffers;
      tile_manager.state.tiles[rtile.tile_index].buffers = NULL;
    }
  }
  else {
    /* In progress tile pixels update. */
    if (update_render_tile_cb) {
      update_render_tile_cb(rtile, false);
    }
  }

  update_status_time();

  /* Notify denoising thread that a tile was finished. */
  denoising_cond.notify_all();
}

void Session::map_neighbor_tiles(RenderTileNeighbors &neighbors, Device *tile_device)
{
  thread_scoped_lock tile_lock(tile_mutex);

  const int4 image_region = make_int4(
      tile_manager.state.buffer.full_x,
      tile_manager.state.buffer.full_y,
      tile_manager.state.buffer.full_x + tile_manager.state.buffer.width,
      tile_manager.state.buffer.full_y + tile_manager.state.buffer.height);

  RenderTile &center_tile = neighbors.tiles[RenderTileNeighbors::CENTER];

  if (!tile_manager.schedule_denoising) {
    /* Fix up tile slices with overlap. */
    if (tile_manager.slice_overlap != 0) {
      int y = max(center_tile.y - tile_manager.slice_overlap, image_region.y);
      center_tile.h = min(center_tile.y + center_tile.h + tile_manager.slice_overlap,
                          image_region.w) -
                      y;
      center_tile.y = y;
    }

    /* Tiles are not being denoised individually, which means the entire image is processed. */
    neighbors.set_bounds_from_center();
  }
  else {
    int center_idx = center_tile.tile_index;
    assert(tile_manager.state.tiles[center_idx].state == Tile::DENOISE);

    for (int dy = -1, i = 0; dy <= 1; dy++) {
      for (int dx = -1; dx <= 1; dx++, i++) {
        RenderTile &rtile = neighbors.tiles[i];
        int nindex = tile_manager.get_neighbor_index(center_idx, i);
        if (nindex >= 0) {
          Tile *tile = &tile_manager.state.tiles[nindex];

          rtile.x = image_region.x + tile->x;
          rtile.y = image_region.y + tile->y;
          rtile.w = tile->w;
          rtile.h = tile->h;

          if (buffers) {
            tile_manager.state.buffer.get_offset_stride(rtile.offset, rtile.stride);

            rtile.buffer = buffers->buffer.device_pointer;
            rtile.buffers = buffers;
          }
          else {
            assert(tile->buffers);
            tile->buffers->params.get_offset_stride(rtile.offset, rtile.stride);

            rtile.buffer = tile->buffers->buffer.device_pointer;
            rtile.buffers = tile->buffers;
          }
        }
        else {
          /* TODO(sergey): Tile size is not known, tiling is done internally in the PathTrace.
           * Mapping tiles will probably be not needed in the future, as the denoising will happen
           * on a big tile. Keep a dummy value so that we can have code for reference. */
          const int2 tile_size = make_int2(64, 64);

          int px = center_tile.x + dx * tile_size.x;
          int py = center_tile.y + dy * tile_size.y;

          rtile.x = clamp(px, image_region.x, image_region.z);
          rtile.y = clamp(py, image_region.y, image_region.w);
          rtile.w = rtile.h = 0;

          rtile.buffer = (device_ptr)NULL;
          rtile.buffers = NULL;
        }
      }
    }
  }

  assert(center_tile.buffers);
  device->map_neighbor_tiles(tile_device, neighbors);

  /* The denoised result is written back to the original tile. */
  neighbors.target = center_tile;
}

void Session::unmap_neighbor_tiles(RenderTileNeighbors &neighbors, Device *tile_device)
{
  thread_scoped_lock tile_lock(tile_mutex);
  device->unmap_neighbor_tiles(tile_device, neighbors);
}
#endif

void Session::run_main_render_loop()
{
  last_display_time = time_dt();

  while (!progress.get_cancel()) {
    const bool no_tiles = !run_update_for_next_iteration();
    bool need_copy_to_display_buffer = false;

    if (no_tiles) {
      if (VLOG_IS_ON(2)) {
        double total_time, render_time;
        progress.get_time(total_time, render_time);
        VLOG(2) << "Rendering in main loop is done in " << render_time << " seconds.";
      }
      if (params.background) {
        /* if no work left and in background mode, we can stop immediately. */
        progress.set_status("Finished");
        break;
      }
    }

    if (!params.background) {
      /* if in interactive mode, and we might be either paused or done for now,
       * wait for pause condition notify to wake up again */
      if (run_wait_for_work(no_tiles)) {
        continue;
      }
    }

    if (progress.get_cancel()) {
      break;
    }

    {
      /* buffers mutex is locked entirely while rendering each
       * sample, and released/reacquired on each iteration to allow
       * reset and draw in between */
      thread_scoped_lock buffers_lock(buffers_mutex);

      /* update status and timing */
      update_status_time();

      /* render */
      bool delayed_denoise = false;
      const bool need_denoise = render_need_denoise(delayed_denoise);
      render(need_denoise);

      /* update status and timing */
      update_status_time();

      if (!params.background)
        need_copy_to_display_buffer = !delayed_denoise;

      if (!device->error_message().empty())
        progress.set_error(device->error_message());
    }

    {
      thread_scoped_lock reset_lock(delayed_reset.mutex);
      thread_scoped_lock buffers_lock(buffers_mutex);

      if (delayed_reset.do_reset) {
        /* Handle reset on the next loop iteration. Here simply avoid copy of partial render buffer
         * to the display buffer. */
        /* TODO(sergey): It might be perceptually better to not cancel rendering mid-way and show
         * "delayed" render result in the viewport. */
      }
      else if (need_copy_to_display_buffer) {
        /* Only copy to display_buffer if we do not reset, we don't
         * want to show the result of an incomplete sample */
        copy_to_display_buffer();
      }

      if (!device->error_message().empty())
        progress.set_error(device->error_message());
    }

    progress.set_update();
  }
}

void Session::run()
{
  if (params.use_profiling && (params.device.type == DEVICE_CPU)) {
    profiler.start();
  }

  /* session thread loop */
  progress.set_status("Waiting for render to start");

  /* run */
  if (!progress.get_cancel()) {
    /* reset number of rendered samples */
    progress.reset_sample();

    run_main_render_loop();
  }

  profiler.stop();

  /* progress update */
  if (progress.get_cancel())
    progress.set_status(progress.get_cancel_message());
  else
    progress.set_update();
}

bool Session::run_update_for_next_iteration()
{
  thread_scoped_lock scene_lock(scene->mutex);
  thread_scoped_lock reset_lock(delayed_reset.mutex);

  if (delayed_reset.do_reset) {
    thread_scoped_lock buffers_lock(buffers_mutex);
    reset_(delayed_reset.params, delayed_reset.samples);
    delayed_reset.do_reset = false;
  }

  const bool have_tiles = tile_manager.next();

  if (have_tiles) {
    scoped_timer update_timer;
    if (update_scene()) {
      profiler.reset(scene->shaders.size(), scene->objects.size());
    }
    progress.add_skip_time(update_timer, params.background);
  }

  return have_tiles;
}

bool Session::run_wait_for_work(bool no_tiles)
{
  thread_scoped_lock pause_lock(pause_mutex);

  if (!pause && !no_tiles) {
    return false;
  }

  update_status_time(pause, no_tiles);

  while (true) {
    scoped_timer pause_timer;
    pause_cond.wait(pause_lock);
    if (pause) {
      progress.add_skip_time(pause_timer, params.background);
    }

    update_status_time(pause, no_tiles);
    progress.set_update();

    if (!pause) {
      break;
    }
  }

  return no_tiles;
}

void Session::draw()
{
  if (!gpu_display) {
    did_draw_after_reset_ = true;
    return;
  }

  did_draw_after_reset_ |= gpu_display->draw();
}

void Session::reset_(BufferParams &buffer_params, int samples)
{
  path_trace_->reset(buffer_params);

  if (gpu_display) {
    gpu_display->reset(buffer_params);
  }

  tile_manager.reset(buffer_params, samples);
#if 0
  stealable_tiles = 0;
  tile_stealing_state = NOT_STEALING;
#endif
  progress.reset_sample();

  bool show_progress = params.background || tile_manager.get_num_effective_samples() != INT_MAX;
  progress.set_total_pixel_samples(show_progress ? tile_manager.state.total_pixel_samples : 0);

  if (!params.background)
    progress.set_start_time();
  progress.set_render_start_time();

  did_draw_after_reset_ = false;
}

void Session::reset(BufferParams &buffer_params, int samples)
{
  thread_scoped_lock reset_lock(delayed_reset.mutex);
  thread_scoped_lock pause_lock(pause_mutex);

  delayed_reset.params = buffer_params;
  delayed_reset.samples = samples;
  delayed_reset.do_reset = true;
  device->task_cancel();

  path_trace_->cancel();

  pause_cond.notify_all();
}

void Session::set_samples(int samples)
{
  if (samples != params.samples) {
    params.samples = samples;
    tile_manager.set_samples(samples);

    pause_cond.notify_all();
  }
}

void Session::set_pause(bool pause_)
{
  bool notify = false;

  {
    thread_scoped_lock pause_lock(pause_mutex);

    if (pause != pause_) {
      pause = pause_;
      notify = true;
    }
  }

  if (session_thread) {
    if (notify) {
      pause_cond.notify_all();
    }
  }
  else if (pause_) {
    update_status_time(pause_);
  }
}

void Session::set_denoising(const DenoiseParams &denoising)
{
  /* If parameters did not change do an early output, avoiding any locking.
   * This ensures quick and responsive interface update during viewport rendering (interface will
   * get locked on settings change due to buffer lock during rendering). */
  if (!params.denoising.modified(denoising)) {
    return;
  }

  set_denoising_no_check(denoising);
}

void Session::set_denoising_no_check(const DenoiseParams &denoising)
{
  bool need_denoise = denoising.need_denoising_task();

  /* Lock buffers so no denoising operation is triggered while the settings are changed here. */
  /* TODO(sergey): Would be nice to have a thread synchronization which will avoid such "expensive"
   * (in terms of possible wait time) lock. */
  thread_scoped_lock buffers_lock(buffers_mutex);
  params.denoising = denoising;

  /* TODO(sergey): Finish decoupling denoiser implementation from device. */
  if (!(params.device.denoisers & denoising.type)) {
    if (need_denoise) {
      progress.set_error("Denoiser type not supported by compute device");
    }

    params.denoising.use = false;
    need_denoise = false;
  }

  /* TODO(sergey): Check which of the code is still needed. */
#if 0
  // TODO(pmours): Query the required overlap value for denoising from the device?
  tile_manager.slice_overlap = need_denoise && !params.background ? 64 : 0;

  /* Schedule per tile denoising for final renders if we are either denoising or
   * need prefiltered passes for the native denoiser. */
  tile_manager.schedule_denoising = need_denoise && params.background;
#endif

  path_trace_->set_denoiser_params(denoising);
}

void Session::set_denoising_start_sample(int sample)
{
  if (sample != params.denoising.start_sample) {
    params.denoising.start_sample = sample;

    pause_cond.notify_all();
  }
}

void Session::wait()
{
  if (session_thread) {
    session_thread->join();
    delete session_thread;
  }

  session_thread = NULL;
}

bool Session::update_scene()
{
  /* update camera if dimensions changed for progressive render. the camera
   * knows nothing about progressive or cropped rendering, it just gets the
   * image dimensions passed in */
  Camera *cam = scene->camera;
  int width = tile_manager.state.buffer.full_width;
  int height = tile_manager.state.buffer.full_height;
  int resolution = tile_manager.state.resolution_divider;

  cam->set_screen_size_and_resolution(width, height, resolution);

  /* number of samples is needed by multi jittered
   * sampling pattern and by baking */
  Integrator *integrator = scene->integrator;
  BakeManager *bake_manager = scene->bake_manager;

  if (integrator->get_sampling_pattern() != SAMPLING_PATTERN_SOBOL || bake_manager->get_baking()) {
    integrator->set_aa_samples(tile_manager.num_samples);
  }

  if (scene->update(progress)) {
    return true;
  }

  return false;
}

void Session::update_status_time(bool show_pause, bool show_done)
{
#if 0
  int progressive_sample = tile_manager.state.sample;
  int num_samples = tile_manager.get_num_effective_samples();

  int tile = progress.get_rendered_tiles();
  int num_tiles = tile_manager.state.num_tiles;

  /* update status */
  string status, substatus;

  if (!params.progressive) {
    const bool is_cpu = params.device.type == DEVICE_CPU;
    const bool rendering_finished = (tile == num_tiles);
    const bool is_last_tile = (tile + 1) == num_tiles;

    substatus = string_printf("Rendered %d/%d Tiles", tile, num_tiles);

    if (!rendering_finished && (device->show_samples() || (is_cpu && is_last_tile))) {
      /* Some devices automatically support showing the sample number:
       * - CUDADevice
       * - OpenCLDevice when using the megakernel (the split kernel renders multiple
       *   samples at the same time, so the current sample isn't really defined)
       * - CPUDevice when using one thread
       * For these devices, the current sample is always shown.
       *
       * The other option is when the last tile is currently being rendered by the CPU.
       */
      substatus += string_printf(", Sample %d/%d", progress.get_current_sample(), num_samples);
    }
    if (params.denoising.use && params.denoising.type != DENOISER_OPENIMAGEDENOISE) {
      substatus += string_printf(", Denoised %d tiles", progress.get_denoised_tiles());
    }
    else if (params.denoising.store_passes && params.denoising.type == DENOISER_NLM) {
      substatus += string_printf(", Prefiltered %d tiles", progress.get_denoised_tiles());
    }
  }
  else if (tile_manager.num_samples == Integrator::MAX_SAMPLES)
    substatus = string_printf("Path Tracing Sample %d", progressive_sample + 1);
  else
    substatus = string_printf("Path Tracing Sample %d/%d", progressive_sample + 1, num_samples);

  if (show_pause) {
    status = "Rendering Paused";
  }
  else if (show_done) {
    status = "Rendering Done";
    progress.set_end_time(); /* Save end time so that further calls to get_time are accurate. */
  }
  else {
    status = substatus;
    substatus.clear();
  }
#else
  string status, substatus;

  /* TODO(sergey): Take number of big tiles into account. */

  const int num_samples = tile_manager.get_num_effective_samples();
  substatus += string_printf("Sample %d/%d", progress.get_current_sample(), num_samples);

  if (show_pause) {
    status = "Rendering Paused";
  }
  else if (show_done) {
    status = "Rendering Done";
    progress.set_end_time(); /* Save end time so that further calls to get_time are accurate. */
  }
  else {
    status = substatus;
    substatus.clear();
  }

#endif

  progress.set_status(status, substatus);
}

bool Session::render_need_denoise(bool &delayed)
{
  delayed = false;

  /* Not supported yet for baking. */
  if (read_bake_tile_cb) {
    return false;
  }

  /* Denoising enabled? */
  if (!params.denoising.need_denoising_task()) {
    return false;
  }

  if (params.background) {
    /* Background render, only denoise when rendering the last sample. */
    return tile_manager.done();
  }

  /* Viewport render. */

  /* It can happen that denoising was already enabled, but the scene still needs an update. */
  if (scene->film->is_modified() || !scene->film->get_denoising_data_offset()) {
    return false;
  }

  /* Immediately denoise when we reach the start sample or last sample. */
  const int num_samples_finished = tile_manager.state.sample + 1;
  if (num_samples_finished == params.denoising.start_sample ||
      num_samples_finished == params.samples) {
    return true;
  }

  /* Do not denoise until the sample at which denoising should start is reached. */
  if (num_samples_finished < params.denoising.start_sample) {
    return false;
  }

  /* Avoid excessive denoising in viewport after reaching a certain amount of samples. */
  delayed = (tile_manager.state.sample >= 20 &&
             (time_dt() - last_display_time) < params.progressive_update_timeout);
  return !delayed;
}

void Session::render(bool need_denoise)
{
  if (!params.background && tile_manager.state.sample == tile_manager.range_start_sample) {
    /* Clear buffers. */
    path_trace_->clear_render_buffers();
  }

  if (tile_manager.state.buffer.width == 0 || tile_manager.state.buffer.height == 0) {
    return; /* Avoid empty launches. */
  }

#if 1

  /* Number of samples which are to be path traced. */
  const int samples_to_render_num = tile_manager.state.num_samples;

  path_trace_->set_start_sample(tile_manager.state.sample);
  path_trace_->set_resolution_divider(tile_manager.state.resolution_divider);

  /* Perform rendering. */
  path_trace_->render_samples(samples_to_render_num);

  if (need_denoise) {
    path_trace_->denoise();
  }

  /* TODO(sergey): Left for the reference. Remove after it is clear it is not needed for working on
   * the `PathTrace`. */

#else
  /* Add path trace task. */
  DeviceTask task(DeviceTask::RENDER);

  task.acquire_tile = function_bind(&Session::acquire_tile, this, _2, _1, _3);
  task.release_tile = function_bind(&Session::release_tile, this, _1, need_denoise);
  task.map_neighbor_tiles = function_bind(&Session::map_neighbor_tiles, this, _1, _2);
  task.unmap_neighbor_tiles = function_bind(&Session::unmap_neighbor_tiles, this, _1, _2);
  task.get_cancel = function_bind(&Progress::get_cancel, &this->progress);
  task.update_tile_sample = function_bind(&Session::update_tile_sample, this, _1);
  task.update_progress_sample = function_bind(&Progress::add_samples, &this->progress, _1, _2);
  task.get_tile_stolen = function_bind(&Session::get_tile_stolen, this);
  task.need_finish_queue = params.progressive_refine;
  task.integrator_branched = scene->integrator->get_method() == Integrator::BRANCHED_PATH;

  task.adaptive_sampling.use = (scene->integrator->get_sampling_pattern() ==
                                SAMPLING_PATTERN_PMJ) &&
                               scene->dscene.data.film.pass_adaptive_aux_buffer;
  task.adaptive_sampling.min_samples = scene->dscene.data.integrator.adaptive_min_samples;
  task.adaptive_sampling.adaptive_step = scene->dscene.data.integrator.adaptive_step;

  /* Acquire render tiles by default. */
  task.tile_types = RenderTile::PATH_TRACE;

  if (need_denoise) {
    task.denoising = params.denoising;

    task.pass_stride = scene->film->get_pass_stride();
    task.target_pass_stride = task.pass_stride;
    task.pass_denoising_data = scene->film->get_denoising_data_offset();
    task.pass_denoising_clean = scene->film->get_denoising_clean_offset();

    task.denoising_from_render = true;

    if (tile_manager.schedule_denoising) {
      /* Acquire denoising tiles during rendering. */
      task.tile_types |= RenderTile::DENOISE;
    }
    else {
      assert(buffers);

      /* Schedule rendering and wait for it to finish. */
      device->task_add(task);
      device->task_wait();

      /* Then run denoising on the whole image at once. */
      task.type = DeviceTask::DENOISE_BUFFER;
      task.x = tile_manager.state.buffer.full_x;
      task.y = tile_manager.state.buffer.full_y;
      task.w = tile_manager.state.buffer.width;
      task.h = tile_manager.state.buffer.height;
      task.buffer = buffers->buffer.device_pointer;
      task.sample = tile_manager.state.sample;
      task.num_samples = tile_manager.state.num_samples;
      tile_manager.state.buffer.get_offset_stride(task.offset, task.stride);
      task.buffers = buffers;
    }
  }

  device->task_add(task);
#endif
}

void Session::copy_to_display_buffer()
{
  /* TODO(sergey): Ideally this would be totally handled by the PathTrace, so that it will take
   * care of updates when they are needed (not oo often for performance, not too rare for
   * feedback). */

  path_trace_->copy_to_gpu_display(gpu_display.get());

  last_display_time = time_dt();
}

void Session::device_free()
{
  scene->device_free();

#if 0
  tile_manager.device_free();
#endif

  /* used from background render only, so no need to
   * re-create render/display buffers here
   */
}

void Session::collect_statistics(RenderStats *render_stats)
{
  scene->collect_statistics(render_stats);
  if (params.use_profiling && (params.device.type == DEVICE_CPU)) {
    render_stats->collect_profiling(scene, profiler);
  }
}

CCL_NAMESPACE_END
