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

/* Device data taken from CUDA occupancy calculator.
 *
 * Terminology
 * - CUDA GPUs have multiple streaming multiprocessors
 * - Each multiprocessor executes multiple thread blocks
 * - Each thread block contains a number of threads, also known as the block size
 * - Multiprocessors have a fixed number of registers, and the amount of registers
 *   used by each threads limits the number of threads per block.
 */

/* 3.0 and 3.5 */
#if __CUDA_ARCH__ == 300 || __CUDA_ARCH__ == 350
#  define CUDA_MULTIPRESSOR_MAX_REGISTERS 65536
#  define CUDA_MULTIPROCESSOR_MAX_BLOCKS 16
#  define CUDA_BLOCK_MAX_THREADS 1024
#  define CUDA_THREAD_MAX_REGISTERS 63

/* tunable parameters */
#  define CUDA_KERNEL_BLOCK_NUM_THREADS 256
#  define CUDA_KERNEL_MAX_REGISTERS 63

/* 3.2 */
#elif __CUDA_ARCH__ == 320
#  define CUDA_MULTIPRESSOR_MAX_REGISTERS 32768
#  define CUDA_MULTIPROCESSOR_MAX_BLOCKS 16
#  define CUDA_BLOCK_MAX_THREADS 1024
#  define CUDA_THREAD_MAX_REGISTERS 63

/* tunable parameters */
#  define CUDA_KERNEL_BLOCK_NUM_THREADS 256
#  define CUDA_KERNEL_MAX_REGISTERS 63

/* 3.7 */
#elif __CUDA_ARCH__ == 370
#  define CUDA_MULTIPRESSOR_MAX_REGISTERS 65536
#  define CUDA_MULTIPROCESSOR_MAX_BLOCKS 16
#  define CUDA_BLOCK_MAX_THREADS 1024
#  define CUDA_THREAD_MAX_REGISTERS 255

/* tunable parameters */
#  define CUDA_KERNEL_BLOCK_NUM_THREADS 256
#  define CUDA_KERNEL_MAX_REGISTERS 63

/* 5.x, 6.x */
#elif __CUDA_ARCH__ <= 699
#  define CUDA_MULTIPRESSOR_MAX_REGISTERS 65536
#  define CUDA_MULTIPROCESSOR_MAX_BLOCKS 32
#  define CUDA_BLOCK_MAX_THREADS 1024
#  define CUDA_THREAD_MAX_REGISTERS 255

/* tunable parameters */
#  define CUDA_KERNEL_BLOCK_NUM_THREADS 256
/* CUDA 9.0 seems to cause slowdowns on high-end Pascal cards unless we increase the number of
 * registers */
#  if __CUDACC_VER_MAJOR__ >= 9 && __CUDA_ARCH__ >= 600
#    define CUDA_KERNEL_MAX_REGISTERS 64
#  else
#    define CUDA_KERNEL_MAX_REGISTERS 48
#  endif

/* 7.x, 8.x */
#elif __CUDA_ARCH__ <= 899
#  define CUDA_MULTIPRESSOR_MAX_REGISTERS 65536
#  define CUDA_MULTIPROCESSOR_MAX_BLOCKS 32
#  define CUDA_BLOCK_MAX_THREADS 1024
#  define CUDA_THREAD_MAX_REGISTERS 255

/* tunable parameters */
#  define CUDA_KERNEL_BLOCK_NUM_THREADS 512
#  define CUDA_KERNEL_MAX_REGISTERS 96

/* unknown architecture */
#else
#  error "Unknown or unsupported CUDA architecture, can't determine launch bounds"
#endif

/* For split kernel using all registers seems fastest for now, but this
 * is unlikely to be optimal once we resolve other bottlenecks. */

#define CUDA_KERNEL_SPLIT_MAX_REGISTERS CUDA_THREAD_MAX_REGISTERS

/* Compute number of threads per block and minimum blocks per multiprocessor
 * given the maximum number of registers per thread. */

#define CUDA_LAUNCH_BOUNDS(block_num_threads, thread_num_registers) \
  __launch_bounds__(block_num_threads, \
                    CUDA_MULTIPRESSOR_MAX_REGISTERS / (block_num_threads * thread_num_registers))

/* sanity checks */

#if CUDA_KERNEL_BLOCK_NUM_THREADS > CUDA_BLOCK_MAX_THREADS
#  error "Maximum number of threads per block exceeded"
#endif

#if CUDA_MULTIPRESSOR_MAX_REGISTERS / \
        (CUDA_KERNEL_BLOCK_NUM_THREADS * CUDA_KERNEL_MAX_REGISTERS) > \
    CUDA_MULTIPROCESSOR_MAX_BLOCKS
#  error "Maximum number of blocks per multiprocessor exceeded"
#endif

#if CUDA_KERNEL_MAX_REGISTERS > CUDA_THREAD_MAX_REGISTERS
#  error "Maximum number of registers per thread exceeded"
#endif
