/*
 * Copyright 2021 Blender Foundation
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

CCL_NAMESPACE_BEGIN

/* Parallel sum of array input_data with size n into output_sum.
 *
 * Adapted from "Optimizing Parallel Reduction in CUDA", Mark Harris.
 *
 * This version adds multiple elements per thread sequentially.  This reduces
 * the overall cost of the algorithm while keeping the work complexity O(n) and
 * the step complexity O(log n). (Brent's Theorem optimization) */

#define CUDA_PARALLEL_SUM_DEFAULT_BLOCK_SIZE 512

template<uint blocksize, typename InputT, typename OutputT, typename ConvertOp>
__device__ void cuda_parallel_sum(
    const InputT *input_data, const uint n, OutputT *output_sum, OutputT zero, ConvertOp convert)
{
  extern __shared__ OutputT shared_data[];

  const uint tid = threadIdx.x;
  const uint gridsize = blocksize * gridDim.x;

  OutputT sum = zero;
  for (uint i = blockIdx.x * blocksize + tid; i < n; i += gridsize) {
    sum += convert(input_data[i]);
  }
  shared_data[tid] = sum;

  __syncthreads();

  if (blocksize >= 512 && tid < 256) {
    shared_data[tid] = sum = sum + shared_data[tid + 256];
  }

  __syncthreads();

  if (blocksize >= 256 && tid < 128) {
    shared_data[tid] = sum = sum + shared_data[tid + 128];
  }

  __syncthreads();

  if (blocksize >= 128 && tid < 64) {
    shared_data[tid] = sum = sum + shared_data[tid + 64];
  }

  __syncthreads();

  if (blocksize >= 64 && tid < 32) {
    shared_data[tid] = sum = sum + shared_data[tid + 32];
  }

  __syncthreads();

  if (tid < 32) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
  }

  if (tid == 0) {
    output_sum[blockIdx.x] = sum;
  }
}

CCL_NAMESPACE_END
