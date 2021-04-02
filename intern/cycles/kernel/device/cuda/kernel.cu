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

#  include "kernel/integrator/integrator_path_state.h"
#  include "kernel/integrator/integrator_state.h"
#  include "kernel/integrator/integrator_state_util.h"

#  include "kernel/integrator/integrator_init_from_camera.h"
#  include "kernel/integrator/integrator_intersect_closest.h"
#  include "kernel/integrator/integrator_intersect_shadow.h"
#  include "kernel/integrator/integrator_intersect_subsurface.h"
#  include "kernel/integrator/integrator_megakernel.h"
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

/* Integrator */

extern "C" __global__ void CUDA_LAUNCH_BOUNDS(CUDA_KERNEL_BLOCK_NUM_THREADS,
                                              CUDA_KERNEL_MAX_REGISTERS)
    kernel_cuda_integrator_init_from_camera(const int *path_index_array,
                                            KernelWorkTile *tile,
                                            float *render_buffer,
                                            const int tile_work_size,
                                            const int path_index_offset)
{
  const int global_index = ccl_global_id(0);
  const int work_index = global_index;
  bool thread_is_active = work_index < tile_work_size;
  if (thread_is_active) {
    const int path_index = (path_index_array) ? path_index_array[global_index] :
                                                path_index_offset + global_index;

    uint x, y, sample;
    get_work_pixel(tile, work_index, &x, &y, &sample);
    integrator_init_from_camera(NULL, path_index, tile, render_buffer, x, y, sample);
  }
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

extern "C" __global__ void CUDA_LAUNCH_BOUNDS(CUDA_KERNEL_BLOCK_NUM_THREADS,
                                              CUDA_KERNEL_MAX_REGISTERS)
    kernel_cuda_integrator_megakernel(const int *path_index_array,
                                      float *render_buffer,
                                      const int work_size)
{
  const int global_index = ccl_global_id(0);

  if (global_index < work_size) {
    const int path_index = (path_index_array) ? path_index_array[global_index] : global_index;
    integrator_megakernel(NULL, path_index, render_buffer);
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
    kernel_cuda_integrator_terminated_paths_array(int num_states,
                                                  int *indices,
                                                  int *num_indices,
                                                  int unused_kernel)
{
  cuda_parallel_active_index_array<CUDA_PARALLEL_ACTIVE_INDEX_DEFAULT_BLOCK_SIZE>(
      num_states, indices, num_indices, [](const int path_index) {
        return (INTEGRATOR_STATE(path, queued_kernel) == 0) &&
               (INTEGRATOR_STATE(shadow_path, queued_kernel) == 0);
      });
}

/* Adaptive Sampling */

extern "C" __global__ void CUDA_LAUNCH_BOUNDS(CUDA_KERNEL_BLOCK_NUM_THREADS,
                                              CUDA_KERNEL_MAX_REGISTERS)
    kernel_cuda_adaptive_sampling_convergence_check(
        float *render_buffer, int sx, int sy, int sw, int sh, int sample, int offset, int stride)
{
  const int work_index = ccl_global_id(0);
  const int y = work_index / sw;
  const int x = work_index - y * sw;

  if (x < sw && y < sh) {
    kernel_adaptive_sampling_convergence_check(
        NULL, render_buffer, sx + x, sy + y, sample, offset, stride);
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

/* Convert to Display Buffer */

extern "C" __global__ void CUDA_LAUNCH_BOUNDS(CUDA_KERNEL_BLOCK_NUM_THREADS,
                                              CUDA_KERNEL_MAX_REGISTERS)
    kernel_cuda_convert_to_half_float(uchar4 *rgba,
                                      float *buffer,
                                      float sample_scale,
                                      int sx,
                                      int sy,
                                      int sw,
                                      int sh,
                                      int offset,
                                      int stride)
{
  const int work_index = ccl_global_id(0);
  const int y = work_index / sw;
  const int x = work_index - y * sw;

  if (x < sw && y < sh) {
    kernel_film_convert_to_half_float(
        NULL, rgba, buffer, sample_scale, sx + x, sy + y, offset, stride);
  }
}

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

/* Baking */

#  ifdef __BAKING__
extern "C" __global__ void CUDA_LAUNCH_BOUNDS(CUDA_KERNEL_BLOCK_NUM_THREADS,
                                              CUDA_KERNEL_MAX_REGISTERS)
    kernel_cuda_bake(KernelWorkTile *tile, uint work_size)
{
  /* TODO */
#    if 0
  int work_index = ccl_global_id(0);

  if (work_index < work_size) {
    uint x, y, sample;
    get_work_pixel(tile, work_index, &x, &y, &sample);

    KernelGlobals kg;
    kernel_bake_evaluate(&kg, tile->buffer, sample, x, y, tile->offset, tile->stride);
  }
#    endif
}
#  endif

#endif
