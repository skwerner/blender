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

/* Integrator Path State
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
 * INTEGRATOR_PATH_NEXT(next_kernel)
 * INTEGRATOR_PATH_TERMINATE
 *
 * For the shadow path these functions are used, and again each shadow kernel must call
 * one of them, and only once.
 *
 * INTEGRATOR_SHADOW_PATH_NEXT(next_kernel)
 * INTEGRATOR_SHADOW_PATH_TERMINATE
 */

#pragma once

CCL_NAMESPACE_BEGIN

/* Abstraction
 *
 * Macros for control flow on different devices. */

#define INTEGRATOR_PATH_IS_TERMINATED (INTEGRATOR_STATE(path, flag) == 0)
#define INTEGRATOR_PATH_NEXT(next_kernel)
#define INTEGRATOR_PATH_TERMINATE \
  { \
    INTEGRATOR_STATE_WRITE(path, flag) = 0; \
  }

#define INTEGRATOR_SHADOW_PATH_IS_TERMINATED (INTEGRATOR_STATE(shadow_path, flag) == 0)
#define INTEGRATOR_SHADOW_PATH_NEXT(next_kernel)
#define INTEGRATOR_SHADOW_PATH_TERMINATE \
  { \
    INTEGRATOR_STATE_WRITE(shadow_path, flag) = 0; \
  }

CCL_NAMESPACE_END
