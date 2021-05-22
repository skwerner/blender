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

#pragma once

#include "kernel/kernel_types.h"
#include "util/util_atomic.h"

CCL_NAMESPACE_BEGIN

/* Control Flow
 *
 * Utilities for control flow between kernels. The implementation may differ per device
 * or even be handled on the host side. To abstract such differences, experiment with
 * different implementations and for debugging, this is abstracted using macros.
 *
 * There is a main path for regular path tracing camera for path tracing. Shadows for next
 * event estimation branch off from this into their own path, that may be computed in
 * parallel while the main path continues.
 *
 * Each kernel on the main path must call one of these functions. These may not be called
 * multiple times from the same kernel.
 *
 * INTEGRATOR_PATH_INIT(next_kernel)
 * INTEGRATOR_PATH_NEXT(current_kernel, next_kernel)
 * INTEGRATOR_PATH_TERMINATE(current_kernel)
 *
 * For the shadow path similar functions are used, and again each shadow kernel must call
 * one of them, and only once.
 */

#define INTEGRATOR_PATH_IS_TERMINATED (INTEGRATOR_STATE(path, queued_kernel) == 0)
#define INTEGRATOR_SHADOW_PATH_IS_TERMINATED (INTEGRATOR_STATE(shadow_path, queued_kernel) == 0)

#ifdef __KERNEL_GPU__

#  define INTEGRATOR_PATH_INIT(next_kernel) \
    atomic_fetch_and_add_uint32(&kernel_integrator_state.queue_counter \
                                     ->num_queued[DEVICE_KERNEL_INTEGRATOR_##next_kernel], \
                                1); \
    INTEGRATOR_STATE_WRITE(path, queued_kernel) = DEVICE_KERNEL_INTEGRATOR_##next_kernel;
#  define INTEGRATOR_PATH_NEXT(current_kernel, next_kernel) \
    atomic_fetch_and_sub_uint32(&kernel_integrator_state.queue_counter \
                                     ->num_queued[DEVICE_KERNEL_INTEGRATOR_##current_kernel], \
                                1); \
    atomic_fetch_and_add_uint32(&kernel_integrator_state.queue_counter \
                                     ->num_queued[DEVICE_KERNEL_INTEGRATOR_##next_kernel], \
                                1); \
    INTEGRATOR_STATE_WRITE(path, queued_kernel) = DEVICE_KERNEL_INTEGRATOR_##next_kernel;
#  define INTEGRATOR_PATH_TERMINATE(current_kernel) \
    atomic_fetch_and_sub_uint32(&kernel_integrator_state.queue_counter \
                                     ->num_queued[DEVICE_KERNEL_INTEGRATOR_##current_kernel], \
                                1); \
    INTEGRATOR_STATE_WRITE(path, queued_kernel) = 0;

#  define INTEGRATOR_SHADOW_PATH_INIT(next_kernel) \
    atomic_fetch_and_add_uint32(&kernel_integrator_state.queue_counter \
                                     ->num_queued[DEVICE_KERNEL_INTEGRATOR_##next_kernel], \
                                1); \
    INTEGRATOR_STATE_WRITE(shadow_path, queued_kernel) = DEVICE_KERNEL_INTEGRATOR_##next_kernel;
#  define INTEGRATOR_SHADOW_PATH_NEXT(current_kernel, next_kernel) \
    atomic_fetch_and_sub_uint32(&kernel_integrator_state.queue_counter \
                                     ->num_queued[DEVICE_KERNEL_INTEGRATOR_##current_kernel], \
                                1); \
    atomic_fetch_and_add_uint32(&kernel_integrator_state.queue_counter \
                                     ->num_queued[DEVICE_KERNEL_INTEGRATOR_##next_kernel], \
                                1); \
    INTEGRATOR_STATE_WRITE(shadow_path, queued_kernel) = DEVICE_KERNEL_INTEGRATOR_##next_kernel;
#  define INTEGRATOR_SHADOW_PATH_TERMINATE(current_kernel) \
    atomic_fetch_and_sub_uint32(&kernel_integrator_state.queue_counter \
                                     ->num_queued[DEVICE_KERNEL_INTEGRATOR_##current_kernel], \
                                1); \
    INTEGRATOR_STATE_WRITE(shadow_path, queued_kernel) = 0;

#  define INTEGRATOR_PATH_SET_SORT_KEY(key) \
    { \
      const int key_ = key; \
      INTEGRATOR_STATE_WRITE(path, shader_sort_key) = key_; \
      atomic_fetch_and_add_uint32(&kernel_integrator_state.sort_key_counter[key_], 1); \
    }

#  define INTEGRATOR_SHADOW_CATCHER_PATH_INIT() \
    { \
      /* Path. */ \
      { \
        atomic_fetch_and_add_uint32( \
            &kernel_integrator_state.queue_counter \
                 ->num_queued[INTEGRATOR_SHADOW_CATCHER_STATE(path, queued_kernel)], \
            1); \
      } \
      /* Shadow path. Note that it is optional. */ \
      { \
        const int shadow_kernel = INTEGRATOR_SHADOW_CATCHER_STATE(shadow_path, queued_kernel); \
        if (shadow_kernel != 0) { \
          atomic_fetch_and_add_uint32( \
              &kernel_integrator_state.queue_counter->num_queued[shadow_kernel], 1); \
        } \
      } \
      /* Sorting. */ \
      { \
        if (INTEGRATOR_SHADOW_CATCHER_STATE(path, queued_kernel) == \
            DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE) { \
          atomic_fetch_and_add_uint32( \
              &kernel_integrator_state.sort_key_counter[INTEGRATOR_STATE(path, shader_sort_key)], \
              1); \
        } \
      } \
    }

#else

#  define INTEGRATOR_PATH_INIT(next_kernel) \
    INTEGRATOR_STATE_WRITE(path, queued_kernel) = DEVICE_KERNEL_INTEGRATOR_##next_kernel;
#  define INTEGRATOR_PATH_NEXT(current_kernel, next_kernel) \
    INTEGRATOR_STATE_WRITE(path, queued_kernel) = DEVICE_KERNEL_INTEGRATOR_##next_kernel;
#  define INTEGRATOR_PATH_TERMINATE(current_kernel) \
    INTEGRATOR_STATE_WRITE(path, queued_kernel) = 0;

#  define INTEGRATOR_SHADOW_PATH_INIT(next_kernel) \
    INTEGRATOR_STATE_WRITE(shadow_path, queued_kernel) = DEVICE_KERNEL_INTEGRATOR_##next_kernel;
#  define INTEGRATOR_SHADOW_PATH_NEXT(current_kernel, next_kernel) \
    INTEGRATOR_STATE_WRITE(shadow_path, queued_kernel) = DEVICE_KERNEL_INTEGRATOR_##next_kernel;
#  define INTEGRATOR_SHADOW_PATH_TERMINATE(current_kernel) \
    INTEGRATOR_STATE_WRITE(shadow_path, queued_kernel) = 0;

#  define INTEGRATOR_PATH_SET_SORT_KEY(key)

#  define INTEGRATOR_SHADOW_CATCHER_PATH_INIT()

#endif

CCL_NAMESPACE_END
