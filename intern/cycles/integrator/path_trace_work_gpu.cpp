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
#include "render/gpu_display.h"
#include "render/scene.h"

#include "util/util_logging.h"
#include "util/util_tbb.h"
#include "util/util_time.h"

#include "kernel/kernel_types.h"

CCL_NAMESPACE_BEGIN

PathTraceWorkGPU::PathTraceWorkGPU(Device *device,
                                   DeviceScene *device_scene,
                                   RenderBuffers *buffers,
                                   bool *cancel_requested_flag)
    : PathTraceWork(device, device_scene, buffers, cancel_requested_flag),
      queue_(device->queue_create()),
      render_buffers_(buffers),
      integrator_queue_counter_(device, "integrator_queue_counter", MEM_READ_WRITE),
      integrator_sort_key_(device, "integrator_sort_key", MEM_READ_WRITE),
      integrator_sort_key_counter_(device, "integrator_sort_key_counter", MEM_READ_WRITE),
      queued_paths_(device, "queued_paths", MEM_READ_WRITE),
      num_queued_paths_(device, "num_queued_paths", MEM_READ_WRITE),
      work_tiles_(device, "work_tiles", MEM_READ_WRITE),
      gpu_display_rgba_half_(device, "display buffer half", MEM_READ_WRITE),
      max_active_path_index_(0)
{
  work_tile_scheduler_.set_max_num_path_states(get_max_num_paths());
}

void PathTraceWorkGPU::alloc_integrator_state()
{
  /* IntegrateState allocated as structure of arrays.
   *
   * Allocate a device only memory buffer before for each struct member, and then
   * write the pointers into a struct that resides in constant memory.
   *
   * This assumes the device side struct memory contains consecutive pointers for
   * each struct member, with the same 64-bit size as device_ptr.
   *
   * TODO: store float3 in separate XYZ arrays. */
  if (!integrator_state_soa_.empty()) {
    return;
  }

  vector<device_ptr> device_struct;
  const int max_num_paths = get_max_num_paths();

#define KERNEL_STRUCT_BEGIN(name) for (int array_index = 0;; array_index++) {
#define KERNEL_STRUCT_MEMBER(type, name) \
  { \
    device_only_memory<type> *array = new device_only_memory<type>(device_, \
                                                                   "integrator_state_" #name); \
    array->alloc_to_device(max_num_paths); \
    /* TODO: skip for most arrays. */ \
    array->zero_to_device(); \
    device_struct.push_back(array->device_pointer); \
    integrator_state_soa_.emplace_back(array); \
  }
#define KERNEL_STRUCT_END(name) \
  break; \
  }
#define KERNEL_STRUCT_END_ARRAY(name, array_size) \
  if (array_index == array_size - 1) { \
    break; \
  } \
  }
#include "kernel/integrator/integrator_state_template.h"
#undef KERNEL_STRUCT_BEGIN
#undef KERNEL_STRUCT_MEMBER
#undef KERNEL_STRUCT_END
#undef KERNEL_STRUCT_END_ARRAY

  /* Copy to device side struct in constant memory. */
  device_->const_copy_to(
      "__integrator_state", device_struct.data(), device_struct.size() * sizeof(device_ptr));
}

void PathTraceWorkGPU::alloc_integrator_queue()
{
  if (integrator_queue_counter_.size() == 0) {
    integrator_queue_counter_.alloc(1);
    integrator_queue_counter_.zero_to_device();
    integrator_queue_counter_.copy_from_device();

    /* Copy to device side pointer in constant memory. */
    device_->const_copy_to("__integrator_queue_counter",
                           &integrator_queue_counter_.device_pointer,
                           sizeof(device_ptr));
  }

  /* Allocate data for active path index arrays. */
  if (num_queued_paths_.size() == 0) {
    num_queued_paths_.alloc(1);
    num_queued_paths_.zero_to_device();
  }

  if (queued_paths_.size() == 0) {
    queued_paths_.alloc(get_max_num_paths());
    /* TODO: this could be skip if we had a function to just allocate on device. */
    queued_paths_.zero_to_device();
  }
}

void PathTraceWorkGPU::alloc_integrator_sorting()
{
  /* Allocate arrays for shader sorting. */
  if (integrator_sort_key_counter_.size() == 0) {
    integrator_sort_key_.alloc(get_max_num_paths());
    /* TODO: this could be skip if we had a function to just allocate on device. */
    integrator_sort_key_.zero_to_device();
    device_->const_copy_to(
        "__integrator_sort_key", &integrator_sort_key_.device_pointer, sizeof(device_ptr));
  }

  const int num_shaders = device_scene_->shaders.size();
  if (integrator_sort_key_counter_.size() < num_shaders) {
    integrator_sort_key_counter_.alloc(num_shaders);
    integrator_sort_key_counter_.zero_to_device();
    device_->const_copy_to("__integrator_sort_key_counter",
                           &integrator_sort_key_counter_.device_pointer,
                           sizeof(device_ptr));
  }
}

void PathTraceWorkGPU::init_execution()
{
  queue_->init_execution();

  alloc_integrator_state();
  alloc_integrator_queue();
  alloc_integrator_sorting();
}

void PathTraceWorkGPU::render_samples(int start_sample, int samples_num)
{
  work_tile_scheduler_.reset(effective_buffer_params_, start_sample, samples_num);

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
      integrator_queue_counter_.copy_from_device();
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
      integrator_queue_counter_.copy_from_device();
    }
  }
}

bool PathTraceWorkGPU::enqueue_path_iteration()
{
  /* Find kernel to execute, with max number of queued paths. */
  IntegratorQueueCounter *queue_counter = integrator_queue_counter_.data();

  int num_paths = 0;
  for (int i = 0; i < DEVICE_KERNEL_INTEGRATOR_NUM; i++) {
    num_paths += queue_counter->num_queued[i];
  }

  if (num_paths == 0) {
    return false;
  }

  const int max_num_paths = get_max_num_paths();
  const float megakernel_threshold = 0.02f;
  const bool use_megakernel = (num_paths < megakernel_threshold * max_num_paths);

  if (use_megakernel) {
    enqueue_path_iteration(DEVICE_KERNEL_INTEGRATOR_MEGAKERNEL);
    return true;
  }

  /* Find kernel to execute, with max number of queued paths. */
  int max_num_queued = 0;
  DeviceKernel kernel = DEVICE_KERNEL_INTEGRATOR_NUM;

  for (int i = 0; i < DEVICE_KERNEL_INTEGRATOR_NUM; i++) {
    if (queue_counter->num_queued[i] > max_num_queued) {
      kernel = (DeviceKernel)i;
      max_num_queued = queue_counter->num_queued[i];
    }
  }

  if (max_num_queued == 0) {
    return false;
  }

  /* Finish shadows before potentially adding more shadow rays. We can only
   * store one shadow ray in the integrator state. */
  if (kernel == DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE ||
      kernel == DEVICE_KERNEL_INTEGRATOR_SHADE_VOLUME) {
    if (queue_counter->num_queued[DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW]) {
      enqueue_path_iteration(DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW);
      return true;
    }
    else if (queue_counter->num_queued[DEVICE_KERNEL_INTEGRATOR_SHADE_SHADOW]) {
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
  void *d_path_index = (void *)NULL;

  /* Create array of path indices for which this kernel is queued to be executed. */
  int work_size = max_active_path_index_;

  IntegratorQueueCounter *queue_counter = integrator_queue_counter_.data();
  int num_queued = queue_counter->num_queued[kernel];

  if (kernel == DEVICE_KERNEL_INTEGRATOR_MEGAKERNEL) {
    num_queued = 0;
    for (int i = 0; i < DEVICE_KERNEL_INTEGRATOR_NUM; i++) {
      num_queued += queue_counter->num_queued[i];
    }
  }

  if (kernel == DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE) {
    /* Compute array of active paths, sorted by shader. */
    work_size = num_queued;
    d_path_index = (void *)queued_paths_.device_pointer;

    compute_sorted_queued_paths(DEVICE_KERNEL_INTEGRATOR_SORTED_PATHS_ARRAY, kernel);
  }
  else if (num_queued < work_size) {
    work_size = num_queued;
    d_path_index = (void *)queued_paths_.device_pointer;

    if (kernel == DEVICE_KERNEL_INTEGRATOR_MEGAKERNEL) {
      /* Compute array of all active paths for megakernel. */
      compute_queued_paths(DEVICE_KERNEL_INTEGRATOR_ACTIVE_PATHS_ARRAY, kernel);
      num_queued_paths_.copy_from_device();
      work_size = num_queued_paths_.data()[0];
    }
    else if (kernel == DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW ||
             kernel == DEVICE_KERNEL_INTEGRATOR_SHADE_SHADOW) {
      /* Compute array of active shadow paths for specific kernel. */
      compute_queued_paths(DEVICE_KERNEL_INTEGRATOR_QUEUED_SHADOW_PATHS_ARRAY, kernel);
    }
    else {
      /* Compute array of active paths for specific kernel. */
      compute_queued_paths(DEVICE_KERNEL_INTEGRATOR_QUEUED_PATHS_ARRAY, kernel);
    }

    /* TODO: ensure this happens as part of queue stream. */
    num_queued_paths_.zero_to_device();
  }

  DCHECK_LE(work_size, get_max_num_paths());

  switch (kernel) {
    case DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST:
    case DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW:
    case DEVICE_KERNEL_INTEGRATOR_INTERSECT_SUBSURFACE: {
      /* Ray intersection kernels with integrator state. */
      void *args[] = {&d_path_index, const_cast<int *>(&work_size)};

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
      void *args[] = {&d_path_index, &d_render_buffer, const_cast<int *>(&work_size)};

      queue_->enqueue(kernel, work_size, args);
      break;
    }
    case DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA:
    case DEVICE_KERNEL_INTEGRATOR_QUEUED_PATHS_ARRAY:
    case DEVICE_KERNEL_INTEGRATOR_QUEUED_SHADOW_PATHS_ARRAY:
    case DEVICE_KERNEL_INTEGRATOR_ACTIVE_PATHS_ARRAY:
    case DEVICE_KERNEL_INTEGRATOR_TERMINATED_PATHS_ARRAY:
    case DEVICE_KERNEL_INTEGRATOR_SORTED_PATHS_ARRAY:
    case DEVICE_KERNEL_SHADER_EVAL_DISPLACE:
    case DEVICE_KERNEL_SHADER_EVAL_BACKGROUND:
    case DEVICE_KERNEL_CONVERT_TO_HALF_FLOAT:
    case DEVICE_KERNEL_ADAPTIVE_SAMPLING_CONVERGENCE_CHECK:
    case DEVICE_KERNEL_ADAPTIVE_SAMPLING_CONVERGENCE_FILTER_X:
    case DEVICE_KERNEL_ADAPTIVE_SAMPLING_CONVERGENCE_FILTER_Y:
    case DEVICE_KERNEL_FILTER_CONVERT_TO_RGB:
    case DEVICE_KERNEL_FILTER_CONVERT_FROM_RGB:
    case DEVICE_KERNEL_PREFIX_SUM:
    case DEVICE_KERNEL_NUM: {
      LOG(FATAL) << "Unhandled kernel " << kernel << ", should never happen.";
      break;
    }
  }

  if (kernel == DEVICE_KERNEL_INTEGRATOR_MEGAKERNEL) {
    /* Megakernel ignores sorting, zero the counter for the next iteration. */
    integrator_sort_key_counter_.zero_to_device();
  }
}

void PathTraceWorkGPU::compute_sorted_queued_paths(DeviceKernel kernel, int queued_kernel)
{
  void *d_key_counter = (void *)integrator_sort_key_counter_.device_pointer;

  /* Compute prefix sum of number of active paths with each shader. */
  {
    const int work_size = 1;
    int num_shaders = integrator_sort_key_counter_.size();
    void *args[] = {&d_key_counter, &num_shaders};
    queue_->enqueue(DEVICE_KERNEL_PREFIX_SUM, work_size, args);
  }

  /* Launch kernel to fill the active paths arrays. */
  {
    /* TODO: this could be smaller for terminated paths based on amount of work we want
     * to schedule. */
    const int work_size = max_active_path_index_;

    void *d_queued_paths = (void *)queued_paths_.device_pointer;
    void *d_num_queued_paths = (void *)num_queued_paths_.device_pointer;
    void *args[] = {const_cast<int *>(&work_size),
                    &d_queued_paths,
                    &d_num_queued_paths,
                    &d_key_counter,
                    &queued_kernel};

    queue_->enqueue(kernel, work_size, args);
  }

  /* TODO: ensure this happens as part of queue stream. */
  num_queued_paths_.zero_to_device();
  integrator_sort_key_counter_.zero_to_device();
}

void PathTraceWorkGPU::compute_queued_paths(DeviceKernel kernel, int queued_kernel)
{
  /* Launch kernel to fill the active paths arrays. */
  /* TODO: this could be smaller for terminated paths based on amount of work we want
   * to schedule. */
  const int work_size = (kernel == DEVICE_KERNEL_INTEGRATOR_TERMINATED_PATHS_ARRAY) ?
                            get_max_num_paths() :
                            max_active_path_index_;

  void *d_queued_paths = (void *)queued_paths_.device_pointer;
  void *d_num_queued_paths = (void *)num_queued_paths_.device_pointer;
  void *args[] = {
      const_cast<int *>(&work_size), &d_queued_paths, &d_num_queued_paths, &queued_kernel};

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
      if (work_tile_scheduler_.get_work(&work_tile, max_num_paths - num_paths)) {
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
  }

  work_tiles_.copy_to_device();

  /* TODO: consider launching a single kernel with an array of work tiles.
   * Mapping global index to the right tile with different sized tiles
   * is not trivial so not done for now. */
  void *d_work_tile = (void *)work_tiles_.device_pointer;
  void *d_path_index = (void *)NULL;
  void *d_render_buffer = (void *)render_buffers_->buffer.device_pointer;

  if (max_active_path_index_ != 0) {
    compute_queued_paths(DEVICE_KERNEL_INTEGRATOR_TERMINATED_PATHS_ARRAY, 0);
    /* TODO: ensure this happens as part of queue stream. */
    num_queued_paths_.zero_to_device();
    d_path_index = (void *)queued_paths_.device_pointer;
  }

  int num_paths = 0;

  for (int i = 0; i < num_work_tiles; i++) {
    KernelWorkTile &work_tile = work_tiles_.data()[i];

    /* Compute kernel launch parameters. */
    const int tile_work_size = work_tile.w * work_tile.h * work_tile.num_samples;

    /* Launch kernel. */
    void *args[] = {&d_path_index,
                    &d_work_tile,
                    &d_render_buffer,
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
  /* TODO: this is wrong, does not account for duplicates with shadow! */
  IntegratorQueueCounter *queue_counter = integrator_queue_counter_.data();

  int num_paths = 0;
  for (int i = 0; i < DEVICE_KERNEL_INTEGRATOR_NUM; i++) {
    num_paths += queue_counter->num_queued[i];
  }

  return num_paths;
}

int PathTraceWorkGPU::get_max_num_paths()
{
  /* TODO: compute automatically. */
  /* TODO: must have at least num_threads_per_block. */
  return 1048576;
}

void PathTraceWorkGPU::copy_to_gpu_display(GPUDisplay *gpu_display, float sample_scale)
{
  if (!interop_use_checked_) {
    Device *device = queue_->device;
    interop_use_ = device->should_use_graphics_interop();

    if (interop_use_) {
      VLOG(2) << "Will be using graphics interop GPU display update.";
    }
    else {
      VLOG(2) << "Will be using naive GPU display update.";
    }

    interop_use_checked_ = true;
  }

  if (interop_use_) {
    if (copy_to_gpu_display_interop(gpu_display, sample_scale)) {
      return;
    }
    interop_use_ = false;
  }

  copy_to_gpu_display_naive(gpu_display, sample_scale);
}

void PathTraceWorkGPU::copy_to_gpu_display_naive(GPUDisplay *gpu_display, float sample_scale)
{
  const int width = effective_buffer_params_.width;
  const int height = effective_buffer_params_.height;
  const int final_width = render_buffers_->params.width;
  const int final_height = render_buffers_->params.height;

  /* Re-allocate display memory if needed, and make sure the device pointer is allocated.
   *
   * NOTE: allocation happens to the final resolution so that no re-allocation happens on every
   * change of the resolution divider. However, if the display becomes smaller, shrink the
   * allocated memory as well. */
  if (gpu_display_rgba_half_.data_width != final_width ||
      gpu_display_rgba_half_.data_height != final_height) {
    gpu_display_rgba_half_.alloc(width, height);
    /* TODO(sergey): There should be a way to make sure device-side memory is allocated without
     * transfering zeroes to the device. */
    gpu_display_rgba_half_.zero_to_device();
  }

  enqueue_film_convert(gpu_display_rgba_half_.device_pointer, sample_scale);
  queue_->synchronize();

  gpu_display_rgba_half_.copy_from_device();

  gpu_display->copy_pixels_to_texture(gpu_display_rgba_half_.data());
}

bool PathTraceWorkGPU::copy_to_gpu_display_interop(GPUDisplay *gpu_display, float sample_scale)
{
  Device *device = queue_->device;

  if (!device_graphics_interop_) {
    device_graphics_interop_ = device->graphics_interop_create();
  }

  const DeviceGraphicsInteropDestination graphics_interop_dst =
      gpu_display->graphics_interop_get();
  device_graphics_interop_->set_destination(graphics_interop_dst);

  const device_ptr d_rgba_half = device_graphics_interop_->map();
  if (!d_rgba_half) {
    return false;
  }

  enqueue_film_convert(d_rgba_half, sample_scale);

  device_graphics_interop_->unmap();
  queue_->synchronize();

  return true;
}

void PathTraceWorkGPU::enqueue_film_convert(device_ptr d_rgba_half, float sample_scale)
{
  const int work_size = effective_buffer_params_.width * effective_buffer_params_.height;

  void *args[] = {&d_rgba_half,
                  &render_buffers_->buffer.device_pointer,
                  const_cast<float *>(&sample_scale),
                  &effective_buffer_params_.full_x,
                  &effective_buffer_params_.full_y,
                  &effective_buffer_params_.width,
                  &effective_buffer_params_.height,
                  &effective_buffer_params_.offset,
                  &effective_buffer_params_.stride};

  queue_->enqueue(DEVICE_KERNEL_CONVERT_TO_HALF_FLOAT, work_size, args);
}

bool PathTraceWorkGPU::adaptive_sampling_converge_and_filter(int sample)
{
  enqueue_adaptive_sampling_convergence_check(sample);
  enqueue_adaptive_sampling_filter_x();
  enqueue_adaptive_sampling_filter_y();

  queue_->synchronize();

  return true;
}

void PathTraceWorkGPU::enqueue_adaptive_sampling_convergence_check(int sample)
{
  const int work_size = effective_buffer_params_.width * effective_buffer_params_.height;

  void *args[] = {&render_buffers_->buffer.device_pointer,
                  const_cast<int *>(&effective_buffer_params_.full_x),
                  const_cast<int *>(&effective_buffer_params_.full_y),
                  const_cast<int *>(&effective_buffer_params_.width),
                  const_cast<int *>(&effective_buffer_params_.height),
                  const_cast<int *>(&sample),
                  &effective_buffer_params_.offset,
                  &effective_buffer_params_.stride};

  queue_->enqueue(DEVICE_KERNEL_ADAPTIVE_SAMPLING_CONVERGENCE_CHECK, work_size, args);
}

void PathTraceWorkGPU::enqueue_adaptive_sampling_filter_x()
{
  const int work_size = effective_buffer_params_.height;

  void *args[] = {&render_buffers_->buffer.device_pointer,
                  &effective_buffer_params_.full_x,
                  &effective_buffer_params_.full_y,
                  &effective_buffer_params_.width,
                  &effective_buffer_params_.height,
                  &effective_buffer_params_.offset,
                  &effective_buffer_params_.stride};

  queue_->enqueue(DEVICE_KERNEL_ADAPTIVE_SAMPLING_CONVERGENCE_FILTER_X, work_size, args);
}

void PathTraceWorkGPU::enqueue_adaptive_sampling_filter_y()
{
  const int work_size = effective_buffer_params_.width;

  void *args[] = {&render_buffers_->buffer.device_pointer,
                  &effective_buffer_params_.full_x,
                  &effective_buffer_params_.full_y,
                  &effective_buffer_params_.width,
                  &effective_buffer_params_.height,
                  &effective_buffer_params_.offset,
                  &effective_buffer_params_.stride};

  queue_->enqueue(DEVICE_KERNEL_ADAPTIVE_SAMPLING_CONVERGENCE_FILTER_Y, work_size, args);
}

CCL_NAMESPACE_END
