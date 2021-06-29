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
#include "integrator/pass_accessor_cpu.h"
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

Session::Session(const SessionParams &params_, const SceneParams &scene_params)
    : params(params_),
      tile_manager_(make_int2(4096, 4096)),
      render_scheduler_(params.headless, params.background, params.pixel_size)
{
  TaskScheduler::init(params.threads);

  session_thread = NULL;

  delayed_reset.do_reset = false;
  delayed_reset.samples = 0;

  pause = false;

  device = Device::create(params.device, stats, profiler);

  scene = new Scene(scene_params, device);

  /* Configure path tracer. */
  path_trace_ = make_unique<PathTrace>(device, &scene->dscene, render_scheduler_);
  path_trace_->set_progress(&progress);
  path_trace_->buffer_update_cb = [&]() {
    if (!update_render_tile_cb) {
      return;
    }
    update_render_tile_cb();
  };
  path_trace_->buffer_write_cb = [&]() {
    if (!write_render_tile_cb) {
      return;
    }
    write_render_tile_cb();
  };
  path_trace_->buffer_read_cb = [&]() -> bool {
    if (!read_render_tile_cb) {
      return false;
    }
    read_render_tile_cb();
    return true;
  };
  path_trace_->progress_update_cb = [&]() { update_status_time(); };
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
  tile_manager_.device_free();
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

void Session::cancel(bool quick)
{
  if (quick && path_trace_) {
    path_trace_->cancel();
  }

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
  return path_trace_->ready_to_reset();
}

void Session::run_main_render_loop()
{
  while (!progress.get_cancel()) {
    const RenderWork render_work = run_update_for_next_iteration();

    if (!render_work) {
      if (VLOG_IS_ON(2)) {
        double total_time, render_time;
        progress.get_time(total_time, render_time);
        VLOG(2) << "Rendering in main loop is done in " << render_time << " seconds.";
        VLOG(2) << path_trace_->full_report();
      }

      if (params.background) {
        /* if no work left and in background mode, we can stop immediately. */
        progress.set_status("Finished");
        break;
      }
    }

    if (run_wait_for_work(render_work)) {
      continue;
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
      path_trace_->render(render_work);

      /* update status and timing */
      update_status_time();

      if (device->have_error()) {
        const string &error_message = device->error_message();
        progress.set_error(error_message);
        progress.set_cancel(error_message);
        break;
      }
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

RenderWork Session::run_update_for_next_iteration()
{
  RenderWork render_work;

  thread_scoped_lock scene_lock(scene->mutex);
  thread_scoped_lock reset_lock(delayed_reset.mutex);

  bool have_tiles = true;

  if (delayed_reset.do_reset) {
    thread_scoped_lock buffers_lock(buffers_mutex);
    do_delayed_reset();

    /* After reset make sure the tile manager is at the first big tile. */
    have_tiles = tile_manager_.next();
  }

  /* Update number of samples in the integrator.
   * Ideally this would need to happen once in `Session::set_samples()`, but the issue there is
   * the initial configuration when Session is created where the `set_samples()` is not used. */
  scene->integrator->set_aa_samples(params.samples);

  /* Update denoiser settings. */
  {
    const DenoiseParams denoise_params = scene->integrator->get_denoise_params();
    path_trace_->set_denoiser_params(denoise_params);
  }

  /* Update adaptive sampling. */
  {
    const AdaptiveSampling adaptive_sampling = scene->integrator->get_adaptive_sampling();
    path_trace_->set_adaptive_sampling(adaptive_sampling);
  }

  render_scheduler_.set_num_samples(params.samples);
  render_scheduler_.set_time_limit(params.time_limit);

  while (have_tiles) {
    render_work = render_scheduler_.get_render_work();
    if (render_work) {
      break;
    }

    /* TODO(sergey): Add support of the multiple big tile. */
    break;
  }

  if (render_work) {
    scoped_timer update_timer;

    const int resolution = render_work.resolution_divider;
    const int width = max(1, buffer_params_.full_width / resolution);
    const int height = max(1, buffer_params_.full_height / resolution);

    if (update_scene(width, height)) {
      profiler.reset(scene->shaders.size(), scene->objects.size());
    }
    progress.add_skip_time(update_timer, params.background);
  }

  return render_work;
}

bool Session::run_wait_for_work(const RenderWork &render_work)
{
  /* In an offline rendering there is no pause, and no tiles will mean the job is fully done. */
  if (params.background) {
    return false;
  }

  thread_scoped_lock pause_lock(pause_mutex);

  const bool no_work = !render_work;

  if (!pause && !no_work) {
    return false;
  }

  update_status_time(pause, no_work);

  while (true) {
    scoped_timer pause_timer;
    pause_cond.wait(pause_lock);
    if (pause) {
      progress.add_skip_time(pause_timer, params.background);
    }

    update_status_time(pause, no_work);
    progress.set_update();

    if (!pause) {
      break;
    }
  }

  return no_work;
}

void Session::draw()
{
  path_trace_->draw();
}

void Session::do_delayed_reset()
{
  if (!delayed_reset.do_reset) {
    return;
  }
  delayed_reset.do_reset = false;

  scene->update_passes();

  buffer_params_ = delayed_reset.params;
  buffer_params_.update_passes(scene->passes);

  render_scheduler_.reset(buffer_params_, delayed_reset.samples);
  path_trace_->reset(buffer_params_);
  tile_manager_.reset(buffer_params_);

  progress.reset_sample();

  /* TODO(sergey): Progress report needs to be worked on. */
#if 0
  bool show_progress = params.background || tile_manager_.get_num_effective_samples() != INT_MAX;
  progress.set_total_pixel_samples(show_progress ? tile_manager_.state.total_pixel_samples : 0);
#endif

  if (!params.background) {
    progress.set_start_time();
  }
  progress.set_render_start_time();
}

void Session::reset(BufferParams &buffer_params, int samples)
{

  thread_scoped_lock reset_lock(delayed_reset.mutex);
  thread_scoped_lock pause_lock(pause_mutex);

  delayed_reset.params = buffer_params;
  delayed_reset.samples = samples;
  delayed_reset.do_reset = true;

  path_trace_->cancel();

  pause_cond.notify_all();
}

void Session::set_samples(int samples)
{
  if (samples != params.samples) {
    params.samples = samples;

    pause_cond.notify_all();
  }
}

void Session::set_time_limit(double time_limit)
{
  if (time_limit != params.time_limit) {
    params.time_limit = time_limit;

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

void Session::set_gpu_display(unique_ptr<GPUDisplay> gpu_display)
{
  path_trace_->set_gpu_display(move(gpu_display));
}

void Session::wait()
{
  if (session_thread) {
    session_thread->join();
    delete session_thread;
  }

  session_thread = NULL;
}

bool Session::update_scene(int width, int height)
{
  /* update camera if dimensions changed for progressive render. the camera
   * knows nothing about progressive or cropped rendering, it just gets the
   * image dimensions passed in */
  Camera *cam = scene->camera;

  cam->set_screen_size(width, height);

  path_trace_->load_kernels();

  if (scene->update(progress)) {
    return true;
  }

  return false;
}

void Session::update_status_time(bool show_pause, bool show_done)
{
#if 0
  int progressive_sample = tile_manager_.state.sample;
  int num_samples = tile_manager_.get_num_effective_samples();

  int tile = progress.get_rendered_tiles();
  int num_tiles = tile_manager_.state.num_tiles;

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
  }
  else if (tile_manager_.num_samples == Integrator::MAX_SAMPLES)
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
  /* TODO(sergey): Take sample range into account. */

  substatus += string_printf("Sample %d/%d", progress.get_current_sample(), params.samples);

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

void Session::device_free()
{
  scene->device_free();

#if 0
  tile_manager_.device_free();
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

/* --------------------------------------------------------------------
 * Tile and tile pixels aceess.
 */

int2 Session::get_render_tile_size() const
{
  const Tile &tile = tile_manager_.get_current_tile();

  return make_int2(tile.width, tile.height);
}

int2 Session::get_render_tile_offset() const
{
  const Tile &tile = tile_manager_.get_current_tile();

  return make_int2(tile.x - tile.full_x, tile.y - tile.full_y);
}

bool Session::copy_render_tile_from_device()
{
  return path_trace_->copy_render_tile_from_device();
}

bool Session::get_render_tile_pixels(const string &pass_name, int num_components, float *pixels)
{
  const Pass *pass = Pass::find(scene->passes, pass_name);
  if (pass == nullptr) {
    return false;
  }

  const bool has_denoised_result = path_trace_->has_denoised_result();
  if (pass->mode == PassMode::DENOISED && !has_denoised_result) {
    pass = Pass::find(scene->passes, pass->type);
  }

  pass = Film::get_actual_display_pass(scene->passes, pass);

  const float exposure = scene->film->get_exposure();
  const int num_samples = render_scheduler_.get_num_rendered_samples();

  const PassAccessor::PassAccessInfo pass_access_info(*pass, *scene->film, scene->passes);
  const PassAccessorCPU pass_accessor(pass_access_info, exposure, num_samples);
  const PassAccessor::Destination destination(pixels, num_components);

  return path_trace_->get_render_tile_pixels(pass_accessor, destination);
}

bool Session::set_render_tile_pixels(const string &pass_name,
                                     int num_components,
                                     const float *pixels)
{
  /* TODO(sergey): Do we write to alias? */
  const Pass *pass = Pass::find(scene->passes, pass_name);
  if (!pass) {
    return false;
  }

  const float exposure = scene->film->get_exposure();
  const int num_samples = render_scheduler_.get_num_rendered_samples();

  const PassAccessor::PassAccessInfo pass_access_info(*pass, *scene->film, scene->passes);
  PassAccessorCPU pass_accessor(pass_access_info, exposure, num_samples);
  PassAccessor::Source source(pixels, num_components);

  return path_trace_->set_render_tile_pixels(pass_accessor, source);
}

CCL_NAMESPACE_END
