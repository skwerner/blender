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
 * INTEGRATOR_STATE_IS_NULL: test if any integrator state is available, for shader evaluation
 * INTEGRATOR_STATE_PASS_NULL: use to pass empty state to other functions.
 *
 * NOTE: if we end up with a device that passes no arguments, the leading comma will be a problem.
 * Can solve it with more macros if we encouter it, but rather ugly so postpone for now.
 */

#include "util/util_types.h"

#pragma once

CCL_NAMESPACE_BEGIN

/* Constants
 *
 * TODO: these could be made dynamic depending on the features used in the scene. */

#define INTEGRATOR_VOLUME_STACK_SIZE 4
#define INTEGRATOR_SHADOW_ISECT_SIZE 4

/* Scalar Struct Definitions
 *
 * Used for single path processing on CPU. */

/* Path tracer state. */
typedef struct IntegratorPathState {
  /* Index of a pixel within the device render buffer where this path will write its result.
   * To get an actual offset within the buffer the value needs to be multiplied by the
   * `kernel_data.film.pass_stride`.
   *
   * The multiplication is delayed for later, so that state can use 32bit integer. */
  uint32_t render_pixel_index;

  /* Current sample number. */
  uint16_t sample;
  /* Current ray bounce depth. */
  uint16_t bounce;
  /* Current transparent ray bounce depth. */
  uint16_t transparent_bounce;

  /* DeviceKernel bit indicating queued kernels.
   * TODO: reduce size? */
  uint32_t queued_kernel;

  /* Random number generator seed. */
  uint32_t rng_hash;
  /* Random number dimension offset. */
  uint32_t rng_offset;

  /* enum PathRayFlag */
  uint32_t flag;

  /* Multiple importance sampling. */
  float ray_pdf;
  /* Filter glossy. */
  float min_ray_pdf;

  /* Throughput. */
  float3 throughput;

  /* Denoising. */
  float3 denoising_feature_throughput;
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
typedef struct IntegratorVolumeStack {
  int object;
  int shader;
} IntegratorVolumeStack;

/* Subsurface closure state for subsurface kernel.
 * TODO: overlap storage with something else? */
typedef struct IntegratorSubsurfaceState {
  float3 albedo;
  float3 radius;
  float roughness;
} IntegratorSubsurfaceState;

typedef struct IntegratorShadowLight {
  /* TODO: can we write this somewhere with the additional memory usage? */
  float3 L;
  /* TODO: use a bit somewhere */
  uint8_t is_light;
} IntegratorShadowLight;

/* Combined state for path. */
typedef struct IntegratorState {
  /* Basic Path Tracing */
  IntegratorRayState ray;
  IntegratorIntersectionState isect;
  IntegratorPathState path;

  /* Volume Rendering */
  IntegratorVolumeStack volume_stack[INTEGRATOR_VOLUME_STACK_SIZE];

  /* Subsurface Scattering */
  IntegratorSubsurfaceState subsurface;

  /* Shadows / Next Event Estimation */
  IntegratorRayState shadow_ray;
  IntegratorIntersectionState shadow_isect[INTEGRATOR_SHADOW_ISECT_SIZE];
  IntegratorShadowLight shadow_light;

  /* Transparent Shadows */
  IntegratorPathState shadow_path;
  IntegratorVolumeStack shadow_volume_stack[INTEGRATOR_VOLUME_STACK_SIZE];
} IntegratorState;

/* Abstraction
 *
 * Macros to access data structures on different devices. */

#ifdef __KERNEL_CPU__

/* Scalar access on CPU. */

#  define INTEGRATOR_STATE_ARGS \
    ccl_attr_maybe_unused const KernelGlobals *ccl_restrict kg, IntegratorState *ccl_restrict state
#  define INTEGRATOR_STATE_CONST_ARGS \
    ccl_attr_maybe_unused const KernelGlobals *ccl_restrict kg, \
        const IntegratorState *ccl_restrict state
#  define INTEGRATOR_STATE_PASS kg, state

#  define INTEGRATOR_STATE_PASS_NULL kg, NULL
#  define INTEGRATOR_STATE_IS_NULL state == NULL

#  define INTEGRATOR_STATE(nested_struct, member) \
    (((const IntegratorState *)state)->nested_struct.member)
#  define INTEGRATOR_STATE_WRITE(nested_struct, member) (state->nested_struct.member)

#  define INTEGRATOR_STATE_ARRAY(nested_struct, array_index, member) \
    (((const IntegratorState *)state)->nested_struct[array_index].member)
#  define INTEGRATOR_STATE_ARRAY_WRITE(nested_struct, array_index, member) \
    ((state)->nested_struct[array_index].member)

/* NOTE: Cast to `void*` to avoid strict compiler's `-Werror=class-memaccess`. The compiler is not
 * really wrong here, as it is possible that it might be better to rely on struct's assignment
 * operator. It is possible to implement with a bit of SFINAE magic based on `is_assignable`.
 * However, things gets a bit more complicated when `INTEGRATOR_STATE_COPY` is used for an array
 * of non-trivially-copyable structs.
 *
 * TODO(sergey): Check whether using assignment operator behaves better for vectorized registers
 * and either implement it properly, or re-state this note. */
#  define INTEGRATOR_STATE_COPY(to_nested_struct, from_nested_struct) \
    memcpy((void *)&state->to_nested_struct, \
           (void *)&state->from_nested_struct, \
           sizeof(state->from_nested_struct));

#else /* __KERNEL_CPU__ */

/* Array access on GPU (TODO: SoA). */

#  define INTEGRATOR_STATE_ARGS \
    const KernelGlobals *ccl_restrict kg, ccl_global IntegratorState *ccl_restrict state, \
        ccl_global IntegratorPathQueue *ccl_restrict queue, const int path_index
#  define INTEGRATOR_STATE_CONST_ARGS \
    const KernelGlobals *ccl_restrict kg, const ccl_global IntegratorState *ccl_restrict state, \
        ccl_global IntegratorPathQueue *ccl_restrict queue, const int path_index
#  define INTEGRATOR_STATE_PASS kg, state, queue, path_index

#  define INTEGRATOR_STATE_PASS_NULL kg, NULL, NULL, -1
#  define INTEGRATOR_STATE_IS_NULL (path_index == -1)

#  define INTEGRATOR_STATE(nested_struct, member) state[path_index].nested_struct.member
#  define INTEGRATOR_STATE_WRITE(nested_struct, member) INTEGRATOR_STATE(nested_struct, member)

#  define INTEGRATOR_STATE_ARRAY(nested_struct, array_index, member) \
    state[path_index].nested_struct[array_index].member
#  define INTEGRATOR_STATE_ARRAY_WRITE(nested_struct, array_index, member) \
    INTEGRATOR_STATE_ARRAY(nested_struct, array_index, member)

/* TODO: check if memcpy emits efficient code on CUDA. */
#  define INTEGRATOR_STATE_COPY(to_nested_struct, from_nested_struct) \
    memcpy(&state[path_index].to_nested_struct, \
           &state[path_index].from_nested_struct, \
           sizeof(state[0].from_nested_struct));

#endif /* __KERNEL__CPU__ */

CCL_NAMESPACE_END
