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

#include "integrator/path_trace_work_gpu.h"

#include "device/device.h"
#include "render/buffers.h"
#include "util/util_logging.h"
#include "util/util_tbb.h"
#include "util/util_time.h"

#include "kernel/kernel_types.h"

CCL_NAMESPACE_BEGIN

PathTraceWorkGPU::PathTraceWorkGPU(Device *render_device,
                                   RenderBuffers *buffers,
                                   bool *cancel_requested_flag)
    : PathTraceWork(render_device, buffers, cancel_requested_flag),
      queue_(render_device->queue_create()),
      render_buffers_(buffers),
      integrator_state_(render_device, "integrator_state"),
      integrator_path_queue_(render_device, "integrator_path_queue", MEM_READ_WRITE),
      queued_paths_(render_device, "queued_paths", MEM_READ_WRITE),
      num_queued_paths_(render_device, "num_queued_paths", MEM_READ_WRITE),
      work_tiles_(render_device, "work_tiles", MEM_READ_WRITE),
      max_active_path_index_(0)
{
  work_scheduler_.set_max_num_path_states(get_max_num_paths());

  integrator_state_.alloc_to_device(get_max_num_paths());
  integrator_state_.zero_to_device();

  integrator_path_queue_.alloc(1);
  integrator_path_queue_.zero_to_device();
  integrator_path_queue_.copy_from_device();
}

void PathTraceWorkGPU::init_execution()
{
  queue_->init_execution();
}

void PathTraceWorkGPU::render_samples(int start_sample, int samples_num)
{
  work_scheduler_.reset(effective_buffer_params_, start_sample, samples_num);

  /* TODO: set a hard limit in case of undetected kernel failures? */
  while (true) {
    /* Enqueue work from the scheduler, on start or when there are not enough
     * paths to keep the device occupied. */
    bool finished;
    if (enqueue_work_tiles(finished)) {
      if (!queue_->synchronize()) {
        break; /* Stop on error. */
      }

      /* Copy stats from the device. */
      integrator_path_queue_.copy_from_device();
    }

    /* Stop if no more work remaining. */
    if (finished) {
      break;
    }

    /* Enqueue on of the path iteration kernels. */
    if (enqueue_path_iteration()) {
      if (!queue_->synchronize()) {
        break; /* Stop on error. */
      }

      /* Copy stats from the device. */
      integrator_path_queue_.copy_from_device();
    }
  }
}

bool PathTraceWorkGPU::enqueue_path_iteration()
{
  const float megakernel_threshold = 0.1f;
  const int max_num_paths = get_max_num_paths();

  /* Find kernel to execute, with max number of queued paths. */
  IntegratorPathQueue *path_queue = integrator_path_queue_.data();

  int num_paths = 0;
  int max_num_queued = 0;
  DeviceKernel kernel = DEVICE_KERNEL_INTEGRATOR_NUM;

  for (int i = 0; i < DEVICE_KERNEL_INTEGRATOR_NUM; i++) {
    num_paths += path_queue->num_queued[i];

    if (path_queue->num_queued[i] > max_num_queued) {
      kernel = (DeviceKernel)i;
      max_num_queued = path_queue->num_queued[i];
    }
  }

  if (max_num_queued == 0) {
    return false;
  }

  /* Switch to megakernel once the number of remaining paths is low.
   * TODO: unclear if max_num_paths is the right way to measure this. */
  const bool use_megakernel = (num_paths < megakernel_threshold * max_num_paths);
  if (use_megakernel && (kernel == DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST &&
                         num_paths == path_queue->num_queued[kernel])) {
    enqueue_path_iteration(DEVICE_KERNEL_INTEGRATOR_MEGAKERNEL);
    return true;
  }

  /* Finish shadows before potentially adding more shadow rays. We can only
   * store one shadow ray in the integrator state.
   * Also finish shadow rays if we want to switch to the megakernel since
   * all paths need to be at intersect closest to execute it. */
  if (use_megakernel || kernel == DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE ||
      kernel == DEVICE_KERNEL_INTEGRATOR_SHADE_VOLUME) {
    if (path_queue->num_queued[DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW]) {
      enqueue_path_iteration(DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW);
      return true;
    }
    else if (path_queue->num_queued[DEVICE_KERNEL_INTEGRATOR_SHADE_SHADOW]) {
      enqueue_path_iteration(DEVICE_KERNEL_INTEGRATOR_SHADE_SHADOW);
      return true;
    }
  }

  /* Schedule kernel with maximum number of queued items. */
  enqueue_path_iteration(kernel);
  return true;
}

void PathTraceWorkGPU::enqueue_path_iteration(DeviceKernel kernel)
{
  void *d_integrator_state = (void *)integrator_state_.device_pointer;
  void *d_integrator_path_queue = (void *)integrator_path_queue_.device_pointer;
  void *d_path_index = (void *)NULL;

  /* Create array of path indices for which this kernel is queued to be executed. */
  int work_size = max_active_path_index_;

  DeviceKernel queue_kernel = (kernel == DEVICE_KERNEL_INTEGRATOR_MEGAKERNEL) ?
                                  DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST :
                                  kernel;
  IntegratorPathQueue *path_queue = integrator_path_queue_.data();
  const int num_queued = path_queue->num_queued[queue_kernel];

  if (num_queued < work_size) {
    work_size = num_queued;
    compute_queued_paths((kernel == DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW ||
                          kernel == DEVICE_KERNEL_INTEGRATOR_SHADE_SHADOW) ?
                             DEVICE_KERNEL_INTEGRATOR_QUEUED_SHADOW_PATHS_ARRAY :
                             DEVICE_KERNEL_INTEGRATOR_QUEUED_PATHS_ARRAY,
                         queue_kernel);
    d_path_index = (void *)queued_paths_.device_pointer;
  }

  DCHECK_LE(work_size, get_max_num_paths());

  switch (kernel) {
    case DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST:
    case DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW:
    case DEVICE_KERNEL_INTEGRATOR_INTERSECT_SUBSURFACE: {
      /* Ray intersection kernels with integrator state. */
      void *args[] = {&d_integrator_state,
                      &d_integrator_path_queue,
                      &d_path_index,
                      const_cast<int *>(&work_size)};

      queue_->enqueue(kernel, work_size, args);
      break;
    }
    case DEVICE_KERNEL_INTEGRATOR_SHADE_BACKGROUND:
    case DEVICE_KERNEL_INTEGRATOR_SHADE_LIGHT:
    case DEVICE_KERNEL_INTEGRATOR_SHADE_SHADOW:
    case DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE:
    case DEVICE_KERNEL_INTEGRATOR_SHADE_VOLUME:
    case DEVICE_KERNEL_INTEGRATOR_MEGAKERNEL: {
      /* Shading kernels with integrator state and render buffer. */
      void *d_render_buffer = (void *)render_buffers_->buffer.device_pointer;
      void *args[] = {&d_integrator_state,
                      &d_integrator_path_queue,
                      &d_path_index,
                      &d_render_buffer,
                      const_cast<int *>(&work_size)};

      queue_->enqueue(kernel, work_size, args);
      break;
    }
    case DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA:
    case DEVICE_KERNEL_INTEGRATOR_QUEUED_PATHS_ARRAY:
    case DEVICE_KERNEL_INTEGRATOR_QUEUED_SHADOW_PATHS_ARRAY:
    case DEVICE_KERNEL_INTEGRATOR_TERMINATED_PATHS_ARRAY:
    case DEVICE_KERNEL_NUM: {
      LOG(FATAL) << "Unhandled kernel " << kernel << ", should never happen.";
      break;
    }
  }
}

void PathTraceWorkGPU::compute_queued_paths(DeviceKernel kernel, int queued_kernel)
{
  /* Launch kernel to count the number of active paths. */
  /* TODO: this could be smaller for terminated paths based on amount of work we want
   * to schedule. */
  const int work_size = (kernel == DEVICE_KERNEL_INTEGRATOR_TERMINATED_PATHS_ARRAY) ?
                            get_max_num_paths() :
                            max_active_path_index_;

  if (num_queued_paths_.size() < 1) {
    num_queued_paths_.alloc(1);
  }
  if (queued_paths_.size() < work_size) {
    queued_paths_.alloc(work_size);
    queued_paths_.zero_to_device(); /* TODO: only need to allocate on device. */
  }

  /* TODO: ensure this happens as part of queue stream. */
  num_queued_paths_.zero_to_device();

  void *d_integrator_state = (void *)integrator_state_.device_pointer;
  void *d_queued_paths = (void *)queued_paths_.device_pointer;
  void *d_num_queued_paths = (void *)num_queued_paths_.device_pointer;
  void *args[] = {&d_integrator_state,
                  const_cast<int *>(&work_size),
                  &d_queued_paths,
                  &d_num_queued_paths,
                  &queued_kernel};

  queue_->enqueue(kernel, work_size, args);
}

bool PathTraceWorkGPU::enqueue_work_tiles(bool &finished)
{
  const float regenerate_threshold = 0.5f;
  const int max_num_paths = get_max_num_paths();
  int num_paths = get_num_active_paths();

  if (num_paths == 0) {
    max_active_path_index_ = 0;
  }

  /* Don't schedule more work if cancelling. */
  if (is_cancel_requested()) {
    if (num_paths == 0) {
      finished = true;
    }
    return false;
  }

  finished = false;

  vector<KernelWorkTile> work_tiles;

  /* Schedule when we're out of paths or there are too few paths to keep the
   * device occupied. */
  if (num_paths == 0 || num_paths < regenerate_threshold * max_num_paths) {
    /* Get work tiles until the maximum number of path is reached. */
    while (num_paths < max_num_paths) {
      KernelWorkTile work_tile;
      if (work_scheduler_.get_work(&work_tile, max_num_paths - num_paths)) {
        work_tiles.push_back(work_tile);
        num_paths += work_tile.w * work_tile.h * work_tile.num_samples;
      }
      else {
        break;
      }
    }

    /* If we couldn't get any more tiles, we're done. */
    if (work_tiles.size() == 0 && num_paths == 0) {
      finished = true;
      return false;
    }
  }

  /* Initialize paths from work tiles. */
  if (work_tiles.size() == 0) {
    return false;
  }

  enqueue_work_tiles(
      DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA, work_tiles.data(), work_tiles.size());
  return true;
}

void PathTraceWorkGPU::enqueue_work_tiles(DeviceKernel kernel,
                                          const KernelWorkTile work_tiles[],
                                          const int num_work_tiles)
{
  /* Copy work tiles to device. */
  if (work_tiles_.size() < num_work_tiles) {
    work_tiles_.alloc(num_work_tiles);
  }

  for (int i = 0; i < num_work_tiles; i++) {
    KernelWorkTile &work_tile = work_tiles_.data()[i];
    work_tile = work_tiles[i];
    work_tile.buffer = render_buffers_->buffer.data();
  }

  work_tiles_.copy_to_device();

  /* TODO: consider launching a single kernel with an array of work tiles.
   * Mapping global index to the right tile with different sized tiles
   * is not trivial so not done for now. */
  void *d_integrator_state = (void *)integrator_state_.device_pointer;
  void *d_integrator_path_queue = (void *)integrator_path_queue_.device_pointer;
  void *d_work_tile = (void *)work_tiles_.device_pointer;
  void *d_path_index = (void *)NULL;

  if (max_active_path_index_ != 0) {
    compute_queued_paths(DEVICE_KERNEL_INTEGRATOR_TERMINATED_PATHS_ARRAY, 0);
    d_path_index = (void *)queued_paths_.device_pointer;
  }

  int num_paths = 0;

  for (int i = 0; i < num_work_tiles; i++) {
    KernelWorkTile &work_tile = work_tiles_.data()[i];

    /* Compute kernel launch parameters. */
    const int tile_work_size = work_tile.w * work_tile.h * work_tile.num_samples;

    /* Launch kernel. */
    void *args[] = {&d_integrator_state,
                    &d_integrator_path_queue,
                    &d_path_index,
                    &d_work_tile,
                    const_cast<int *>(&tile_work_size),
                    &num_paths};

    queue_->enqueue(kernel, tile_work_size, args);

    /* Offset work tile and path index pointers for next tile. */
    num_paths += tile_work_size;
    DCHECK_LE(num_paths, get_max_num_paths());

    /* TODO: this pointer manipulation won't work for OpenCL. */
    d_work_tile = (void *)(((KernelWorkTile *)d_work_tile) + 1);
    if (d_path_index) {
      d_path_index = (void *)(((int *)d_path_index) + tile_work_size);
    }
  }

  /* TODO: this could be computed more accurately using on the last entry
   * in the queued_paths array passed to the kernel? */
  max_active_path_index_ = min(max_active_path_index_ + num_paths, get_max_num_paths());
}

int PathTraceWorkGPU::get_num_active_paths()
{
  IntegratorPathQueue *path_queue = integrator_path_queue_.data();

  int num_paths = 0;
  for (int i = 0; i < DEVICE_KERNEL_INTEGRATOR_NUM; i++) {
    num_paths += path_queue->num_queued[i];
  }

  return num_paths;
}

int PathTraceWorkGPU::get_max_num_paths()
{
  /* TODO: compute automatically. */
  /* TODO: must have at least num_threads_per_block. */
  return 1048576;
}

CCL_NAMESPACE_END
