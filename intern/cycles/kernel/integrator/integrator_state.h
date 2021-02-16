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

/* Integrator State
 *
 * This file defines the data structures that define the state of a path. Any state that is
 * preserved and passed between kernel executions is part of this.
 *
 * The size of this state must be kept as small as possible, to reduce cache misses and keep memory
 * usage under control on GPUs that may execute millions of kernels.
 *
 * Memory may be allocated and passed along in different ways depending on the device. There may
 * be a scalar layout, or AoS or SoA layout for batches. The state may be passed along as a pointer
 * to every kernel, or the pointer may exist at program scope or in constant memory. To abstract
 * these differences between devices and experiment with different layouts, macros are used.
 *
 * INTEGRATOR_STATE_ARGS: prepend to argument definitions for every function that accesses
 * path state.
 * INTEGRATOR_STATE_CONST_ARGS: same as INTEGRATOR_STATE_ARGS, when state is read-only
 * INTEGRATOR_STATE_PASS: use to pass along state to other functions access it.
 *
 * INTEGRATOR_STATE(x, y): read nested struct member x.y of IntegratorState
 * INTEGRATOR_STATE_WRITE(x, y): write to nested struct member x.y of IntegratorState
 *
 * INTEGRATOR_STATE_ARRAY(x, index, y): read x[index].y
 * INTEGRATOR_STATE_ARRAY_WRITE(x, index, y): write x[index].y
 *
 * INTEGRATOR_STATE_COPY(to_x, from_x): copy contents of one nested struct to another
 *
 * NOTE: if we end up with a device that passes no arguments, the leading comma will be a problem.
 * Can solve it with more macros if we encouter it, but rather ugly so postpone for now.
 */

#pragma once

CCL_NAMESPACE_BEGIN

/* Constants
 *
 * TODO: these could be made dynamic depending on the features used in the scene. */

#define INTEGRATOR_VOLUME_STACK_SIZE 4

#ifdef __TRANSPARENT_SHADOWS__
#  define INTEGRATOR_SHADOW_ISECT_SIZE 4
#else
#  define INTEGRATOR_SHADOW_ISECT_SIZE 1
#endif

/* Scalar Struct Definitions
 *
 * Used for single path processing on CPU. */

/* Path tracer state. */
typedef struct IntegratorPathState {
  /* Pixel x, y coordinate in current (big) tile. */
  uint16_t x, y;
  /* Current sample number. */
  uint16_t sample;
  /* Current ray depth. */
  uint16_t depth;

  /* Random number generator seed. */
  uint32_t rng;

  /* enum PathRayFlag */
  uint32_t flag;

  /* Multiple importance sampling. */
  float ray_pdf;

  /* Throughput. */
  float3 throughput;
} IntegratorPathState;

/* Ray parameters for scene intersection. */
typedef struct IntegratorRayState {
  float3 P;
  float3 D;
  float t;
  float time;

  /* TODO: compact differentials. */
} IntegratorRayState;

/* Result from scene intersection. */
typedef struct IntegratorIntersectionState {
  float t, u, v;
  int prim;
  int object;
  int type;

  /* TODO: exclude for GPU. */
  float3 Ng;
} IntegratorIntersectionState;

/* Volume stack to identify which volumes the path is inside of. */
#ifdef __VOLUME__
typedef struct IntegratorVolumeStack {
  int object;
  int shader;
} IntegratorVolumeStack;
#endif

/* Subsurface closure state for subsurface kernel. */
#ifdef __SUBSURFACE__
typedef struct IntegratorSubsurfaceState {
  /* TODO: actual BSSRDF closure parameters */
  float3 albedo;
} IntegratorSubsurfaceState;
#endif

/* Combined state for path. */
typedef struct IntegratorState {
  /* Basic Path Tracing */
  IntegratorRayState ray;
  IntegratorIntersectionState isect;
  IntegratorPathState path;

  /* Volume Rendering */
#ifdef __VOLUME__
  IntegratorVolumeStack volume_stack[INTEGRATOR_VOLUME_STACK_SIZE];
#endif

  /* Subsurface Scattering */
#ifdef __SUBSURFACE__
  IntegratorSubsurfaceState subsurface;
#endif

  /* Shadows / Next Event Estimation */
  IntegratorRayState shadow_ray;
  IntegratorIntersectionState shadow_isect[INTEGRATOR_SHADOW_ISECT_SIZE];

  /* Transparent Shadows */
#ifdef __TRANSPARENT_SHADOWS__
  IntegratorPathState shadow_path;
#  ifdef __VOLUME__
  IntegratorVolumeStack shadow_volume_stack[INTEGRATOR_VOLUME_STACK_SIZE];
#  endif
#endif
} IntegratorState;

/* Abstraction
 *
 * Macros to access data structures on different devices. */

#ifdef __KERNEL_CPU__

/* Scalar access on CPU. */

#  define INTEGRATOR_STATE_ARGS const KernelGlobals *kg, IntegratorState *state
#  define INTEGRATOR_STATE_CONST_ARGS const KernelGlobals *kg, const IntegratorState *state
#  define INTEGRATOR_STATE_PASS kg, state

#  define INTEGRATOR_STATE(nested_struct, member) \
    (((const IntegratorState *)state)->nested_struct.member)
#  define INTEGRATOR_STATE_WRITE(nested_struct, member) (state->nested_struct.member)

#  define INTEGRATOR_STATE_ARRAY(nested_struct, array_index, member) \
    (((const IntegratorState *)state)->nested_struct[array_index].member)
#  define INTEGRATOR_STATE_ARRAY_WRITE(nested_struct, array_index, member) \
    ((state)->nested_struct[array_index].member)

#  define INTEGRATOR_STATE_COPY(to_nested_struct, from_nested_struct) \
    memcpy( \
        &state->to_nested_struct, &state->from_nested_struct, sizeof(state->from_nested_struct));

#else /* __KERNEL__CPU__ */

#  error "Not implemented yet"

#endif /* __KERNEL__CPU__ */

CCL_NAMESPACE_END
