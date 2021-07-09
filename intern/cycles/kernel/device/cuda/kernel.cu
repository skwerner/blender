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

/* CUDA kernel entry points */

#ifdef __CUDA_ARCH__

#  include "kernel/device/cuda/compat.h"
#  include "kernel/device/cuda/config.h"
#  include "kernel/device/cuda/globals.h"
#  include "kernel/device/cuda/image.h"
#  include "kernel/device/cuda/parallel_active_index.h"
#  include "kernel/device/cuda/parallel_prefix_sum.h"
#  include "kernel/device/cuda/parallel_sorted_index.h"

#  include "kernel/integrator/integrator_state.h"
#  include "kernel/integrator/integrator_state_flow.h"
#  include "kernel/integrator/integrator_state_util.h"

#  include "kernel/integrator/integrator_init_from_bake.h"
#  include "kernel/integrator/integrator_init_from_camera.h"
#  include "kernel/integrator/integrator_intersect_closest.h"
#  include "kernel/integrator/integrator_intersect_shadow.h"
#  include "kernel/integrator/integrator_intersect_subsurface.h"
#  include "kernel/integrator/integrator_intersect_volume_stack.h"
#  include "kernel/integrator/integrator_shade_background.h"
#  include "kernel/integrator/integrator_shade_light.h"
#  include "kernel/integrator/integrator_shade_shadow.h"
#  include "kernel/integrator/integrator_shade_surface.h"
#  include "kernel/integrator/integrator_shade_volume.h"

#  include "kernel/kernel_adaptive_sampling.h"
#  include "kernel/kernel_bake.h"
#  include "kernel/kernel_film.h"
#  include "kernel/kernel_work_stealing.h"

/* TODO: move cryptomatte post sorting to its own kernel. */
#  if 0
/* kernels */
extern "C" __global__ void CUDA_LAUNCH_BOUNDS(CUDA_KERNEL_BLOCK_NUM_THREADS, CUDA_KERNEL_MAX_REGISTERS)
    kernel_cuda_path_trace(KernelWorkTile *tile, uint work_size)
{
  int work_index = ccl_global_id(0);
  bool thread_is_active = work_index < work_size;
  uint x, y, sample;
  KernelGlobals kg;
  if(thread_is_active) {
    get_work_pixel(tile, work_index, &x, &y, &sample);

    kernel_path_trace(&kg, tile->buffer, sample, x, y, tile->offset, tile->stride);
  }

  if(kernel_data.film.cryptomatte_passes) {
    __syncthreads();
    if(thread_is_active) {
      kernel_cryptomatte_post(&kg, tile->buffer, sample, x, y, tile->offset, tile->stride);
    }
  }
}
#  endif

/* --------------------------------------------------------------------
 * Integrator.
 */

extern "C" __global__ void CUDA_LAUNCH_BOUNDS(CUDA_KERNEL_BLOCK_NUM_THREADS,
                                              CUDA_KERNEL_MAX_REGISTERS)
    kernel_cuda_integrator_reset(int num_states)
{
  const int path_index = ccl_global_id(0);

  if (path_index < num_states) {
    INTEGRATOR_STATE_WRITE(path, queued_kernel) = 0;
    INTEGRATOR_STATE_WRITE(shadow_path, queued_kernel) = 0;
  }
}

extern "C" __global__ void CUDA_LAUNCH_BOUNDS(CUDA_KERNEL_BLOCK_NUM_THREADS,
                                              CUDA_KERNEL_MAX_REGISTERS)
    kernel_cuda_integrator_init_from_camera(const int *path_index_array,
                                            KernelWorkTile *tiles,
                                            const int num_tiles,
                                            float *render_buffer,
                                            const int max_tile_work_size)
{
  const int work_index = ccl_global_id(0);

  if (work_index >= max_tile_work_size * num_tiles) {
    return;
  }

  const int tile_index = work_index / max_tile_work_size;
  const int tile_work_index = work_index - tile_index * max_tile_work_size;

  const KernelWorkTile *tile = &tiles[tile_index];

  if (tile_work_index >= tile->work_size) {
    return;
  }

  const int path_index = (path_index_array) ?
                             path_index_array[tile->path_index_offset + tile_work_index] :
                             tile->path_index_offset + tile_work_index;

  uint x, y, sample;
  get_work_pixel(tile, tile_work_index, &x, &y, &sample);

  integrator_init_from_camera(nullptr, path_index, tile, render_buffer, x, y, sample);
}

extern "C" __global__ void CUDA_LAUNCH_BOUNDS(CUDA_KERNEL_BLOCK_NUM_THREADS,
                                              CUDA_KERNEL_MAX_REGISTERS)
    kernel_cuda_integrator_init_from_bake(const int *path_index_array,
                                          KernelWorkTile *tiles,
                                          const int num_tiles,
                                          float *render_buffer,
                                          const int max_tile_work_size)
{
  const int work_index = ccl_global_id(0);

  if (work_index >= max_tile_work_size * num_tiles) {
    return;
  }

  const int tile_index = work_index / max_tile_work_size;
  const int tile_work_index = work_index - tile_index * max_tile_work_size;

  const KernelWorkTile *tile = &tiles[tile_index];

  if (tile_work_index >= tile->work_size) {
    return;
  }

  const int path_index = (path_index_array) ?
                             path_index_array[tile->path_index_offset + tile_work_index] :
                             tile->path_index_offset + tile_work_index;

  uint x, y, sample;
  get_work_pixel(tile, tile_work_index, &x, &y, &sample);

  integrator_init_from_bake(nullptr, path_index, tile, render_buffer, x, y, sample);
}

extern "C" __global__ void CUDA_LAUNCH_BOUNDS(CUDA_KERNEL_BLOCK_NUM_THREADS,
                                              CUDA_KERNEL_MAX_REGISTERS)
    kernel_cuda_integrator_intersect_closest(const int *path_index_array, const int work_size)
{
  const int global_index = ccl_global_id(0);

  if (global_index < work_size) {
    const int path_index = (path_index_array) ? path_index_array[global_index] : global_index;
    integrator_intersect_closest(NULL, path_index);
  }
}

extern "C" __global__ void CUDA_LAUNCH_BOUNDS(CUDA_KERNEL_BLOCK_NUM_THREADS,
                                              CUDA_KERNEL_MAX_REGISTERS)
    kernel_cuda_integrator_intersect_shadow(const int *path_index_array, const int work_size)
{
  const int global_index = ccl_global_id(0);

  if (global_index < work_size) {
    const int path_index = (path_index_array) ? path_index_array[global_index] : global_index;
    integrator_intersect_shadow(NULL, path_index);
  }
}

extern "C" __global__ void CUDA_LAUNCH_BOUNDS(CUDA_KERNEL_BLOCK_NUM_THREADS,
                                              CUDA_KERNEL_MAX_REGISTERS)
    kernel_cuda_integrator_intersect_subsurface(const int *path_index_array, const int work_size)
{
  const int global_index = ccl_global_id(0);

  if (global_index < work_size) {
    const int path_index = (path_index_array) ? path_index_array[global_index] : global_index;
    integrator_intersect_subsurface(NULL, path_index);
  }
}

extern "C" __global__ void CUDA_LAUNCH_BOUNDS(CUDA_KERNEL_BLOCK_NUM_THREADS,
                                              CUDA_KERNEL_MAX_REGISTERS)
    kernel_cuda_integrator_intersect_volume_stack(const int *path_index_array, const int work_size)
{
  const int global_index = ccl_global_id(0);

  if (global_index < work_size) {
    const int path_index = (path_index_array) ? path_index_array[global_index] : global_index;
    integrator_intersect_volume_stack(NULL, path_index);
  }
}

extern "C" __global__ void CUDA_LAUNCH_BOUNDS(CUDA_KERNEL_BLOCK_NUM_THREADS,
                                              CUDA_KERNEL_MAX_REGISTERS)
    kernel_cuda_integrator_shade_background(const int *path_index_array,
                                            float *render_buffer,
                                            const int work_size)
{
  const int global_index = ccl_global_id(0);

  if (global_index < work_size) {
    const int path_index = (path_index_array) ? path_index_array[global_index] : global_index;
    integrator_shade_background(NULL, path_index, render_buffer);
  }
}

extern "C" __global__ void CUDA_LAUNCH_BOUNDS(CUDA_KERNEL_BLOCK_NUM_THREADS,
                                              CUDA_KERNEL_MAX_REGISTERS)
    kernel_cuda_integrator_shade_light(const int *path_index_array,
                                       float *render_buffer,
                                       const int work_size)
{
  const int global_index = ccl_global_id(0);

  if (global_index < work_size) {
    const int path_index = (path_index_array) ? path_index_array[global_index] : global_index;
    integrator_shade_light(NULL, path_index, render_buffer);
  }
}

extern "C" __global__ void CUDA_LAUNCH_BOUNDS(CUDA_KERNEL_BLOCK_NUM_THREADS,
                                              CUDA_KERNEL_MAX_REGISTERS)
    kernel_cuda_integrator_shade_shadow(const int *path_index_array,
                                        float *render_buffer,
                                        const int work_size)
{
  const int global_index = ccl_global_id(0);

  if (global_index < work_size) {
    const int path_index = (path_index_array) ? path_index_array[global_index] : global_index;
    integrator_shade_shadow(NULL, path_index, render_buffer);
  }
}

extern "C" __global__ void CUDA_LAUNCH_BOUNDS(CUDA_KERNEL_BLOCK_NUM_THREADS,
                                              CUDA_KERNEL_MAX_REGISTERS)
    kernel_cuda_integrator_shade_surface(const int *path_index_array,
                                         float *render_buffer,
                                         const int work_size)
{
  const int global_index = ccl_global_id(0);

  if (global_index < work_size) {
    const int path_index = (path_index_array) ? path_index_array[global_index] : global_index;
    integrator_shade_surface(NULL, path_index, render_buffer);
  }
}

extern "C" __global__ void CUDA_LAUNCH_BOUNDS(CUDA_KERNEL_BLOCK_NUM_THREADS,
                                              CUDA_KERNEL_MAX_REGISTERS)
    kernel_cuda_integrator_shade_surface_raytrace(const int *path_index_array,
                                                  float *render_buffer,
                                                  const int work_size)
{
  const int global_index = ccl_global_id(0);

  if (global_index < work_size) {
    const int path_index = (path_index_array) ? path_index_array[global_index] : global_index;
    integrator_shade_surface_raytrace(NULL, path_index, render_buffer);
  }
}

extern "C" __global__ void CUDA_LAUNCH_BOUNDS(CUDA_KERNEL_BLOCK_NUM_THREADS,
                                              CUDA_KERNEL_MAX_REGISTERS)
    kernel_cuda_integrator_shade_volume(const int *path_index_array,
                                        float *render_buffer,
                                        const int work_size)
{
  const int global_index = ccl_global_id(0);

  if (global_index < work_size) {
    const int path_index = (path_index_array) ? path_index_array[global_index] : global_index;
    integrator_shade_volume(NULL, path_index, render_buffer);
  }
}

extern "C" __global__ void __launch_bounds__(CUDA_PARALLEL_ACTIVE_INDEX_DEFAULT_BLOCK_SIZE)
    kernel_cuda_integrator_queued_paths_array(int num_states,
                                              int *indices,
                                              int *num_indices,
                                              int kernel)
{
  cuda_parallel_active_index_array<CUDA_PARALLEL_ACTIVE_INDEX_DEFAULT_BLOCK_SIZE>(
      num_states, indices, num_indices, [kernel](const int path_index) {
        return (INTEGRATOR_STATE(path, queued_kernel) == kernel);
      });
}

extern "C" __global__ void __launch_bounds__(CUDA_PARALLEL_ACTIVE_INDEX_DEFAULT_BLOCK_SIZE)
    kernel_cuda_integrator_queued_shadow_paths_array(int num_states,
                                                     int *indices,
                                                     int *num_indices,
                                                     int kernel)
{
  cuda_parallel_active_index_array<CUDA_PARALLEL_ACTIVE_INDEX_DEFAULT_BLOCK_SIZE>(
      num_states, indices, num_indices, [kernel](const int path_index) {
        return (INTEGRATOR_STATE(shadow_path, queued_kernel) == kernel);
      });
}

extern "C" __global__ void __launch_bounds__(CUDA_PARALLEL_ACTIVE_INDEX_DEFAULT_BLOCK_SIZE)
    kernel_cuda_integrator_active_paths_array(int num_states, int *indices, int *num_indices)
{
  cuda_parallel_active_index_array<CUDA_PARALLEL_ACTIVE_INDEX_DEFAULT_BLOCK_SIZE>(
      num_states, indices, num_indices, [](const int path_index) {
        return (INTEGRATOR_STATE(path, queued_kernel) != 0) ||
               (INTEGRATOR_STATE(shadow_path, queued_kernel) != 0);
      });
}

extern "C" __global__ void __launch_bounds__(CUDA_PARALLEL_ACTIVE_INDEX_DEFAULT_BLOCK_SIZE)
    kernel_cuda_integrator_terminated_paths_array(int num_states,
                                                  int *indices,
                                                  int *num_indices,
                                                  int indices_offset)
{
  cuda_parallel_active_index_array<CUDA_PARALLEL_ACTIVE_INDEX_DEFAULT_BLOCK_SIZE>(
      num_states, indices + indices_offset, num_indices, [](const int path_index) {
        if (kernel_data.integrator.has_shadow_catcher) {
          /* NOTE: The kernel invocation limits number of states checked, ensuring that only
           * non-shadow-catcher states are checked here. */

          /* Only allow termination of both complementary states did finish their job. */
          if (INTEGRATOR_SHADOW_CATCHER_STATE(path, queued_kernel) != 0 ||
              INTEGRATOR_SHADOW_CATCHER_STATE(shadow_path, queued_kernel) != 0) {
            return false;
          }
        }
        return (INTEGRATOR_STATE(path, queued_kernel) == 0) &&
               (INTEGRATOR_STATE(shadow_path, queued_kernel) == 0);
      });
}

extern "C" __global__ void __launch_bounds__(CUDA_PARALLEL_SORTED_INDEX_DEFAULT_BLOCK_SIZE)
    kernel_cuda_integrator_sorted_paths_array(
        int num_states, int *indices, int *num_indices, int *key_prefix_sum, int kernel)
{
  cuda_parallel_sorted_index_array<CUDA_PARALLEL_SORTED_INDEX_DEFAULT_BLOCK_SIZE>(
      num_states, indices, num_indices, key_prefix_sum, [kernel](const int path_index) {
        return (INTEGRATOR_STATE(path, queued_kernel) == kernel) ?
                   INTEGRATOR_STATE(path, shader_sort_key) :
                   CUDA_PARALLEL_SORTED_INDEX_INACTIVE_KEY;
      });
}

extern "C" __global__ void __launch_bounds__(CUDA_PARALLEL_ACTIVE_INDEX_DEFAULT_BLOCK_SIZE)
    kernel_cuda_integrator_compact_paths_array(int num_states,
                                               int *indices,
                                               int *num_indices,
                                               int num_active_paths)
{
  cuda_parallel_active_index_array<CUDA_PARALLEL_ACTIVE_INDEX_DEFAULT_BLOCK_SIZE>(
      num_states, indices, num_indices, [num_active_paths](const int path_index) {
        return (path_index >= num_active_paths) &&
               ((INTEGRATOR_STATE(path, queued_kernel) != 0) ||
                (INTEGRATOR_STATE(shadow_path, queued_kernel) != 0));
      });
}

extern "C" __global__ void __launch_bounds__(CUDA_PARALLEL_SORTED_INDEX_DEFAULT_BLOCK_SIZE)
    kernel_cuda_integrator_compact_states(const int *active_terminated_states,
                                          const int active_states_offset,
                                          const int terminated_states_offset,
                                          const int work_size)
{
  const int global_index = ccl_global_id(0);

  if (global_index < work_size) {
    const int from_path_index = active_terminated_states[active_states_offset + global_index];
    const int to_path_index = active_terminated_states[terminated_states_offset + global_index];

    integrator_state_move(to_path_index, from_path_index);
  }
}

extern "C" __global__ void __launch_bounds__(CUDA_PARALLEL_PREFIX_SUM_DEFAULT_BLOCK_SIZE)
    kernel_cuda_prefix_sum(int *values, int num_values)
{
  cuda_parallel_prefix_sum<CUDA_PARALLEL_PREFIX_SUM_DEFAULT_BLOCK_SIZE>(values, num_values);
}

/* --------------------------------------------------------------------
 * Adaptive sampling.
 */

extern "C" __global__ void CUDA_LAUNCH_BOUNDS(CUDA_KERNEL_BLOCK_NUM_THREADS,
                                              CUDA_KERNEL_MAX_REGISTERS)
    kernel_cuda_adaptive_sampling_convergence_check(float *render_buffer,
                                                    int sx,
                                                    int sy,
                                                    int sw,
                                                    int sh,
                                                    float threshold,
                                                    bool reset,
                                                    int offset,
                                                    int stride,
                                                    uint *num_active_pixels)
{
  const int work_index = ccl_global_id(0);
  const int y = work_index / sw;
  const int x = work_index - y * sw;

  bool converged = true;

  if (x < sw && y < sh) {
    converged = kernel_adaptive_sampling_convergence_check(
        nullptr, render_buffer, sx + x, sy + y, threshold, reset, offset, stride);
  }

  /* NOTE: All threads specified in the mask must execute the intrinsic. */
  const uint num_active_pixels_mask = __ballot_sync(0xffffffff, !converged);
  if (threadIdx.x == 0) {
    atomic_fetch_and_add_uint32(num_active_pixels, __popc(num_active_pixels_mask));
  }
}

extern "C" __global__ void CUDA_LAUNCH_BOUNDS(CUDA_KERNEL_BLOCK_NUM_THREADS,
                                              CUDA_KERNEL_MAX_REGISTERS)
    kernel_cuda_adaptive_sampling_filter_x(
        float *render_buffer, int sx, int sy, int sw, int sh, int offset, int stride)
{
  const int y = ccl_global_id(0);

  if (y < sh) {
    kernel_adaptive_sampling_filter_x(NULL, render_buffer, sy + y, sx, sw, offset, stride);
  }
}

extern "C" __global__ void CUDA_LAUNCH_BOUNDS(CUDA_KERNEL_BLOCK_NUM_THREADS,
                                              CUDA_KERNEL_MAX_REGISTERS)
    kernel_cuda_adaptive_sampling_filter_y(
        float *render_buffer, int sx, int sy, int sw, int sh, int offset, int stride)
{
  const int x = ccl_global_id(0);

  if (x < sw) {
    kernel_adaptive_sampling_filter_y(NULL, render_buffer, sx + x, sy, sh, offset, stride);
  }
}

/* --------------------------------------------------------------------
 * Film.
 */

/* Common implementation for float destination. */
template<typename Processor>
ccl_device_inline void kernel_cuda_film_convert_common(const KernelFilmConvert *kfilm_convert,
                                                       float *pixels,
                                                       float *render_buffer,
                                                       int num_pixels,
                                                       int offset,
                                                       int stride,
                                                       int dst_offset,
                                                       const Processor &processor)
{
  const int render_pixel_index = ccl_global_id(0);
  if (render_pixel_index >= num_pixels) {
    return;
  }

  const uint64_t render_buffer_offset = (uint64_t)render_pixel_index * kfilm_convert->pass_stride;
  ccl_global const float *buffer = render_buffer + render_buffer_offset;
  ccl_global float *pixel = pixels +
                            (render_pixel_index + dst_offset) * kfilm_convert->pixel_stride;

  processor(kfilm_convert, buffer, pixel);
}

/* Common implementation for half4 destination and 4-channel input pass. */
template<typename Processor>
ccl_device_inline void kernel_cuda_film_convert_half_rgba_common_rgba(
    const KernelFilmConvert *kfilm_convert,
    uchar4 *rgba,
    float *render_buffer,
    int num_pixels,
    int offset,
    int stride,
    int rgba_offset,
    const Processor &processor)
{
  const int render_pixel_index = ccl_global_id(0);
  if (render_pixel_index >= num_pixels) {
    return;
  }

  const uint64_t render_buffer_offset = (uint64_t)render_pixel_index * kfilm_convert->pass_stride;
  ccl_global const float *buffer = render_buffer + render_buffer_offset;

  float pixel[4];
  processor(kfilm_convert, buffer, pixel);

  film_apply_pass_pixel_overlays_rgba(kfilm_convert, buffer, pixel);

  ccl_global half *out = (ccl_global half *)rgba + (rgba_offset + render_pixel_index) * 4;
  float4_store_half(out, make_float4(pixel[0], pixel[1], pixel[2], pixel[3]));
}

/* Common implementation for half4 destination and 3-channel input pass. */
template<typename Processor>
ccl_device_inline void kernel_cuda_film_convert_half_rgba_common_rgb(
    const KernelFilmConvert *kfilm_convert,
    uchar4 *rgba,
    float *render_buffer,
    int num_pixels,
    int offset,
    int stride,
    int rgba_offset,
    const Processor &processor)
{
  kernel_cuda_film_convert_half_rgba_common_rgba(
      kfilm_convert,
      rgba,
      render_buffer,
      num_pixels,
      offset,
      stride,
      rgba_offset,
      [&processor](const KernelFilmConvert *kfilm_convert,
                   ccl_global const float *buffer,
                   float *pixel_rgba) {
        processor(kfilm_convert, buffer, pixel_rgba);
        pixel_rgba[3] = 1.0f;
      });
}

/* Common implementation for half4 destination and single channel input pass. */
template<typename Processor>
ccl_device_inline void kernel_cuda_film_convert_half_rgba_common_value(
    const KernelFilmConvert *kfilm_convert,
    uchar4 *rgba,
    float *render_buffer,
    int num_pixels,
    int offset,
    int stride,
    int rgba_offset,
    const Processor &processor)
{
  kernel_cuda_film_convert_half_rgba_common_rgba(
      kfilm_convert,
      rgba,
      render_buffer,
      num_pixels,
      offset,
      stride,
      rgba_offset,
      [&processor](const KernelFilmConvert *kfilm_convert,
                   ccl_global const float *buffer,
                   float *pixel_rgba) {
        float value;
        processor(kfilm_convert, buffer, &value);

        pixel_rgba[0] = value;
        pixel_rgba[1] = value;
        pixel_rgba[2] = value;
        pixel_rgba[3] = 1.0f;
      });
}

#  define KERNEL_FILM_CONVERT_PROC(name) \
    extern "C" __global__ void CUDA_LAUNCH_BOUNDS(CUDA_KERNEL_BLOCK_NUM_THREADS, \
                                                  CUDA_KERNEL_MAX_REGISTERS) name

#  define KERNEL_FILM_CONVERT_DEFINE(variant, channels) \
    KERNEL_FILM_CONVERT_PROC(kernel_cuda_film_convert_##variant) \
    (const KernelFilmConvert kfilm_convert, \
     float *pixels, \
     float *render_buffer, \
     int num_pixels, \
     int offset, \
     int stride, \
     int rgba_offset) \
    { \
      kernel_cuda_film_convert_common(&kfilm_convert, \
                                      pixels, \
                                      render_buffer, \
                                      num_pixels, \
                                      offset, \
                                      stride, \
                                      rgba_offset, \
                                      film_get_pass_pixel_##variant); \
    } \
    KERNEL_FILM_CONVERT_PROC(kernel_cuda_film_convert_##variant##_half_rgba) \
    (const KernelFilmConvert kfilm_convert, \
     uchar4 *rgba, \
     float *render_buffer, \
     int num_pixels, \
     int offset, \
     int stride, \
     int rgba_offset) \
    { \
      kernel_cuda_film_convert_half_rgba_common_##channels(&kfilm_convert, \
                                                           rgba, \
                                                           render_buffer, \
                                                           num_pixels, \
                                                           offset, \
                                                           stride, \
                                                           rgba_offset, \
                                                           film_get_pass_pixel_##variant); \
    }

KERNEL_FILM_CONVERT_DEFINE(depth, value)
KERNEL_FILM_CONVERT_DEFINE(mist, value)
KERNEL_FILM_CONVERT_DEFINE(sample_count, value)
KERNEL_FILM_CONVERT_DEFINE(float, value)

KERNEL_FILM_CONVERT_DEFINE(divide_even_color, rgb)
KERNEL_FILM_CONVERT_DEFINE(float3, rgb)

KERNEL_FILM_CONVERT_DEFINE(motion, rgba)
KERNEL_FILM_CONVERT_DEFINE(cryptomatte, rgba)
KERNEL_FILM_CONVERT_DEFINE(shadow_catcher, rgba)
KERNEL_FILM_CONVERT_DEFINE(shadow_catcher_matte_with_shadow, rgba)
KERNEL_FILM_CONVERT_DEFINE(float4, rgba)

KERNEL_FILM_CONVERT_DEFINE(shadow, rgb)

#  undef KERNEL_FILM_CONVERT_DEFINE
#  undef KERNEL_FILM_CONVERT_HALF_RGBA_DEFINE
#  undef KERNEL_FILM_CONVERT_PROC

/* --------------------------------------------------------------------
 * Shader evaluaiton.
 */

/* Displacement */

extern "C" __global__ void CUDA_LAUNCH_BOUNDS(CUDA_KERNEL_BLOCK_NUM_THREADS,
                                              CUDA_KERNEL_MAX_REGISTERS)
    kernel_cuda_shader_eval_displace(KernelShaderEvalInput *input,
                                     float4 *output,
                                     const int offset,
                                     const int work_size)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < work_size) {
    kernel_displace_evaluate(NULL, input, output, offset + i);
  }
}

/* Background Shader Evaluation */

extern "C" __global__ void CUDA_LAUNCH_BOUNDS(CUDA_KERNEL_BLOCK_NUM_THREADS,
                                              CUDA_KERNEL_MAX_REGISTERS)
    kernel_cuda_shader_eval_background(KernelShaderEvalInput *input,
                                       float4 *output,
                                       const int offset,
                                       const int work_size)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < work_size) {
    kernel_background_evaluate(NULL, input, output, offset + i);
  }
}

/* --------------------------------------------------------------------
 * Denoising.
 */

extern "C" __global__ void CUDA_LAUNCH_BOUNDS(CUDA_KERNEL_BLOCK_NUM_THREADS,
                                              CUDA_KERNEL_MAX_REGISTERS)
    kernel_cuda_filter_color_preprocess(float *render_buffer,
                                        int full_x,
                                        int full_y,
                                        int width,
                                        int height,
                                        int offset,
                                        int stride,
                                        int pass_stride,
                                        int pass_denoised)
{
  const int work_index = ccl_global_id(0);
  const int y = work_index / width;
  const int x = work_index - y * width;

  if (x >= width || y >= height) {
    return;
  }

  const uint64_t render_pixel_index = offset + (x + full_x) + (y + full_y) * stride;
  float *buffer = render_buffer + render_pixel_index * pass_stride;

  float *color_out = buffer + pass_denoised;
  color_out[0] = clamp(color_out[0], 0.0f, 10000.0f);
  color_out[1] = clamp(color_out[1], 0.0f, 10000.0f);
  color_out[2] = clamp(color_out[2], 0.0f, 10000.0f);
}

extern "C" __global__ void CUDA_LAUNCH_BOUNDS(CUDA_KERNEL_BLOCK_NUM_THREADS,
                                              CUDA_KERNEL_MAX_REGISTERS)
    kernel_cuda_filter_guiding_preprocess(float *guiding_buffer,
                                          int guiding_pass_stride,
                                          int guiding_pass_albedo,
                                          int guiding_pass_normal,
                                          const float *render_buffer,
                                          int render_offset,
                                          int render_stride,
                                          int render_pass_stride,
                                          int render_pass_sample_count,
                                          int render_pass_denoising_albedo,
                                          int render_pass_denoising_normal,
                                          int full_x,
                                          int full_y,
                                          int width,
                                          int height,
                                          int num_samples)
{
  const int work_index = ccl_global_id(0);
  const int y = work_index / width;
  const int x = work_index - y * width;

  if (x >= width || y >= height) {
    return;
  }

  const uint64_t guiding_pixel_index = x + y * width;
  float *guiding_pixel = guiding_buffer + guiding_pixel_index * guiding_pass_stride;

  const uint64_t render_pixel_index = render_offset + (x + full_x) + (y + full_y) * render_stride;
  const float *buffer = render_buffer + render_pixel_index * render_pass_stride;

  float pixel_scale;
  if (render_pass_sample_count == PASS_UNUSED) {
    pixel_scale = 1.0f / num_samples;
  }
  else {
    pixel_scale = 1.0f / __float_as_uint(buffer[render_pass_sample_count]);
  }

  /* Albedo pass. */
  if (guiding_pass_albedo != PASS_UNUSED) {
    kernel_assert(render_pass_denoising_albedo != PASS_UNUSED);

    const float *aledo_in = buffer + render_pass_denoising_albedo;
    float *albedo_out = guiding_pixel + guiding_pass_albedo;

    albedo_out[0] = aledo_in[0] * pixel_scale;
    albedo_out[1] = aledo_in[1] * pixel_scale;
    albedo_out[2] = aledo_in[2] * pixel_scale;
  }

  /* Normal pass. */
  if (render_pass_denoising_normal != PASS_UNUSED) {
    kernel_assert(render_pass_denoising_normal != PASS_UNUSED);

    const float *normal_in = buffer + render_pass_denoising_normal;
    float *normal_out = guiding_pixel + guiding_pass_normal;

    normal_out[3] = normal_in[0] * pixel_scale;
    normal_out[4] = normal_in[1] * pixel_scale;
    normal_out[5] = normal_in[2] * pixel_scale;
  }
}

extern "C" __global__ void CUDA_LAUNCH_BOUNDS(CUDA_KERNEL_BLOCK_NUM_THREADS,
                                              CUDA_KERNEL_MAX_REGISTERS)
    kernel_cuda_filter_guiding_set_fake_albedo(float *guiding_buffer,
                                               int guiding_pass_stride,
                                               int guiding_pass_albedo,
                                               int width,
                                               int height)
{
  kernel_assert(guiding_pass_albedo != PASS_UNUSED);

  const int work_index = ccl_global_id(0);
  const int y = work_index / width;
  const int x = work_index - y * width;

  if (x >= width || y >= height) {
    return;
  }

  const uint64_t guiding_pixel_index = x + y * width;
  float *guiding_pixel = guiding_buffer + guiding_pixel_index * guiding_pass_stride;

  float *albedo_out = guiding_pixel + guiding_pass_albedo;

  albedo_out[0] = 0.5f;
  albedo_out[1] = 0.5f;
  albedo_out[2] = 0.5f;
}

extern "C" __global__ void CUDA_LAUNCH_BOUNDS(CUDA_KERNEL_BLOCK_NUM_THREADS,
                                              CUDA_KERNEL_MAX_REGISTERS)
    kernel_cuda_filter_color_postprocess(float *render_buffer,
                                         int full_x,
                                         int full_y,
                                         int width,
                                         int height,
                                         int offset,
                                         int stride,
                                         int pass_stride,
                                         int num_samples,
                                         int pass_noisy,
                                         int pass_denoised,
                                         int pass_sample_count,
                                         bool use_compositing)
{
  const int work_index = ccl_global_id(0);
  const int y = work_index / width;
  const int x = work_index - y * width;

  if (x >= width || y >= height) {
    return;
  }

  const uint64_t render_pixel_index = offset + (x + full_x) + (y + full_y) * stride;
  float *buffer = render_buffer + render_pixel_index * pass_stride;

  float pixel_scale;
  if (pass_sample_count == PASS_UNUSED) {
    pixel_scale = num_samples;
  }
  else {
    pixel_scale = __float_as_uint(buffer[pass_sample_count]);
  }

  float *denoised_pixel = buffer + pass_denoised;

  denoised_pixel[0] *= pixel_scale;
  denoised_pixel[1] *= pixel_scale;
  denoised_pixel[2] *= pixel_scale;

  /* Currently compositing passes are either 3-component (derived by dividing light passes)
   * or do not have transparency (shadow catcher). Implicitly rely on this logic, as it
   * simplifies logic and avoids extra memory allocation. */
  if (!use_compositing) {
    const float *noisy_pixel = buffer + pass_noisy;
    denoised_pixel[3] = noisy_pixel[3];
  }
  else {
    /* Assigning to zero since this is a default alpha value for 3-component passes, and it
     * is an opaque pixel for 4 component passes. */

    denoised_pixel[3] = 0;
  }
}

#endif
