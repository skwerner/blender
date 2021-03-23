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

#pragma once

#include "kernel_random.h"

CCL_NAMESPACE_BEGIN

ccl_device_inline void path_state_init(INTEGRATOR_STATE_ARGS,
                                       const ccl_global KernelWorkTile *ccl_restrict tile,
                                       const int sample,
                                       const int x,
                                       const int y,
                                       const uint rng_hash)
{
  const uint render_pixel_index = (uint)tile->offset + x + y * tile->stride;

  INTEGRATOR_STATE_WRITE(path, render_pixel_index) = render_pixel_index;
  INTEGRATOR_STATE_WRITE(path, sample) = sample;
  INTEGRATOR_STATE_WRITE(path, bounce) = 0;
  INTEGRATOR_STATE_WRITE(path, transparent_bounce) = 0;
  INTEGRATOR_STATE_WRITE(path, rng_hash) = rng_hash;
  INTEGRATOR_STATE_WRITE(path, rng_offset) = PRNG_BASE_NUM;
  INTEGRATOR_STATE_WRITE(path, flag) = PATH_RAY_CAMERA | PATH_RAY_MIS_SKIP |
                                       PATH_RAY_TRANSPARENT_BACKGROUND;
  INTEGRATOR_STATE_WRITE(path, ray_pdf) = 0.0f;
  INTEGRATOR_STATE_WRITE(path, min_ray_pdf) = FLT_MAX;
  INTEGRATOR_STATE_WRITE(path, throughput) = make_float3(1.0f, 1.0f, 1.0f);

  INTEGRATOR_STATE_ARRAY_WRITE(volume_stack, 0, object) = OBJECT_NONE;
  INTEGRATOR_STATE_ARRAY_WRITE(volume_stack, 0, shader) = SHADER_NONE;

  INTEGRATOR_STATE_WRITE(shadow_path, queued_kernel) = 0;

/* TODO */
#if 0
  state->diffuse_bounce = 0;
  state->glossy_bounce = 0;
  state->transmission_bounce = 0;

#  ifdef __DENOISING_FEATURES__
  if (kernel_data.film.pass_denoising_data) {
    state->flag |= PATH_RAY_STORE_SHADOW_INFO;
    state->denoising_feature_weight = 1.0f;
    state->denoising_feature_throughput = one_float3();
  }
  else {
    state->denoising_feature_weight = 0.0f;
    state->denoising_feature_throughput = zero_float3();
  }
#  endif /* __DENOISING_FEATURES__ */

#  ifdef __VOLUME__
  state->volume_bounce = 0;
  state->volume_bounds_bounce = 0;

  if (kernel_data.integrator.use_volumes) {
    /* Initialize volume stack with volume we are inside of. */
    kernel_volume_stack_init(kg, stack_sd, state, ray, state->volume_stack);
  }
  else {
    state->volume_stack[0].shader = SHADER_NONE;
  }
#  endif
#endif
}

ccl_device_inline void path_state_next(INTEGRATOR_STATE_ARGS, int label)
{
  uint32_t flag = INTEGRATOR_STATE(path, flag);

  /* ray through transparent keeps same flags from previous ray and is
   * not counted as a regular bounce, transparent has separate max */
  if (label & LABEL_TRANSPARENT) {
    uint32_t transparent_bounce = INTEGRATOR_STATE(path, transparent_bounce) + 1;

    flag |= PATH_RAY_TRANSPARENT;
    if (transparent_bounce >= kernel_data.integrator.transparent_max_bounce) {
      flag |= PATH_RAY_TERMINATE_IMMEDIATE;
    }

    if (!kernel_data.integrator.transparent_shadows)
      flag |= PATH_RAY_MIS_SKIP;

    INTEGRATOR_STATE_WRITE(path, flag) = flag;
    INTEGRATOR_STATE_WRITE(path, transparent_bounce) = transparent_bounce;
    /* Random number generator next bounce. */
    INTEGRATOR_STATE_WRITE(path, rng_offset) += PRNG_BOUNCE_NUM;
    return;
  }

  uint32_t bounce = INTEGRATOR_STATE(path, bounce) + 1;
  if (bounce >= kernel_data.integrator.max_bounce) {
    flag |= PATH_RAY_TERMINATE_AFTER_TRANSPARENT;
  }

  flag &= ~(PATH_RAY_ALL_VISIBILITY | PATH_RAY_MIS_SKIP);

#ifdef __VOLUME__
  if (label & LABEL_VOLUME_SCATTER) {
    /* volume scatter */
    flag |= PATH_RAY_VOLUME_SCATTER;
    flag &= ~PATH_RAY_TRANSPARENT_BACKGROUND;

    /* TODO */
#  if 0
    state->volume_bounce++;
    if (state->volume_bounce >= kernel_data.integrator.max_volume_bounce) {
      flag |= PATH_RAY_TERMINATE_AFTER_TRANSPARENT;
    }
#  endif
  }
  else
#endif
  {
    /* surface reflection/transmission */
    if (label & LABEL_REFLECT) {
      flag |= PATH_RAY_REFLECT;
      flag &= ~PATH_RAY_TRANSPARENT_BACKGROUND;

      /* TODO */
#if 0
      if (label & LABEL_DIFFUSE) {
        state->diffuse_bounce++;
        if (state->diffuse_bounce >= kernel_data.integrator.max_diffuse_bounce) {
          flag |= PATH_RAY_TERMINATE_AFTER_TRANSPARENT;
        }
      }
      else {
        state->glossy_bounce++;
        if (state->glossy_bounce >= kernel_data.integrator.max_glossy_bounce) {
          flag |= PATH_RAY_TERMINATE_AFTER_TRANSPARENT;
        }
      }
#endif
    }
    else {
      kernel_assert(label & LABEL_TRANSMIT);

      flag |= PATH_RAY_TRANSMIT;

      if (!(label & LABEL_TRANSMIT_TRANSPARENT)) {
        flag &= ~PATH_RAY_TRANSPARENT_BACKGROUND;
      }

      /* TODO */
#if 0
      state->transmission_bounce++;
      if (state->transmission_bounce >= kernel_data.integrator.max_transmission_bounce) {
        flag |= PATH_RAY_TERMINATE_AFTER_TRANSPARENT;
      }
#endif
    }

    /* diffuse/glossy/singular */
    if (label & LABEL_DIFFUSE) {
      flag |= PATH_RAY_DIFFUSE | PATH_RAY_DIFFUSE_ANCESTOR;
    }
    else if (label & LABEL_GLOSSY) {
      flag |= PATH_RAY_GLOSSY;
    }
    else {
      kernel_assert(label & LABEL_SINGULAR);
      flag |= PATH_RAY_GLOSSY | PATH_RAY_SINGULAR | PATH_RAY_MIS_SKIP;
    }
  }

  /* TODO */
#if 0
#  ifdef __DENOISING_FEATURES__
  if ((state->denoising_feature_weight == 0.0f) && !(flag & PATH_RAY_SHADOW_CATCHER)) {
    flag &= ~PATH_RAY_STORE_SHADOW_INFO;
  }
#  endif
#endif

  INTEGRATOR_STATE_WRITE(path, flag) = flag;
  INTEGRATOR_STATE_WRITE(path, bounce) = bounce;

  /* Random number generator next bounce. */
  INTEGRATOR_STATE_WRITE(path, rng_offset) += PRNG_BOUNCE_NUM;
}

#ifdef __VOLUME__
ccl_device_inline bool path_state_volume_next(INTEGRATOR_STATE_ARGS)
{
  /* TODO */
#  if 0
  /* For volume bounding meshes we pass through without counting transparent
   * bounces, only sanity check in case self intersection gets us stuck. */
  uint32_t volume_bounds_bounce = INTEGRATOR_STATE(path, volume_bounds_bounce) + 1;
  INTEGRATOR_STATE_WRITE(path, volume_bounds_bounce) = volume_bounds_bounce;
  if (volume_bounds_bounce > VOLUME_BOUNDS_MAX) {
    return false;
  }

  /* Random number generator next bounce. */
  if (volume_bounds_bounce > 1) {
    INTEGRATOR_STATE_WRITE(path, rng_offset) += PRNG_BOUNCE_NUM;
  }

  return true;
#  else
  return false;
#  endif
}
#endif

ccl_device_inline uint path_state_ray_visibility(INTEGRATOR_STATE_CONST_ARGS)
{
  const uint32_t flag = INTEGRATOR_STATE(path, flag);
  uint32_t visibility = flag & PATH_RAY_ALL_VISIBILITY;

  /* For visibility, diffuse/glossy are for reflection only. */
  if (visibility & PATH_RAY_TRANSMIT)
    visibility &= ~(PATH_RAY_DIFFUSE | PATH_RAY_GLOSSY);
  /* todo: this is not supported as its own ray visibility yet. */
  if (flag & PATH_RAY_VOLUME_SCATTER)
    visibility |= PATH_RAY_DIFFUSE;

  return visibility;
}

ccl_device_inline float path_state_continuation_probability(INTEGRATOR_STATE_CONST_ARGS)
{
  const uint32_t flag = INTEGRATOR_STATE(path, flag);

  if (flag & PATH_RAY_TERMINATE_IMMEDIATE) {
    /* Ray is to be terminated immediately. */
    return 0.0f;
  }
  else if (flag & PATH_RAY_TRANSPARENT) {
    const uint32_t transparent_bounce = INTEGRATOR_STATE(path, transparent_bounce);
    /* Do at least specified number of bounces without RR. */
    if (transparent_bounce <= kernel_data.integrator.transparent_min_bounce) {
      return 1.0f;
    }
#ifdef __SHADOW_TRICKS__
    /* Exception for shadow catcher not working correctly with RR. */
    else if ((flag & PATH_RAY_SHADOW_CATCHER) && (transparent_bounce <= 8)) {
      return 1.0f;
    }
#endif
  }
  else {
    const uint32_t bounce = INTEGRATOR_STATE(path, bounce);
    /* Do at least specified number of bounces without RR. */
    if (bounce <= kernel_data.integrator.min_bounce) {
      return 1.0f;
    }
#ifdef __SHADOW_TRICKS__
    /* Exception for shadow catcher not working correctly with RR. */
    else if ((flag & PATH_RAY_SHADOW_CATCHER) && (bounce <= 3)) {
      return 1.0f;
    }
#endif
  }

  /* Probabilistic termination: use sqrt() to roughly match typical view
   * transform and do path termination a bit later on average. */
  return min(sqrtf(max3(fabs(INTEGRATOR_STATE(path, throughput)))), 1.0f);
}

#if 0
/* TODO(DingTo): Find more meaningful name for this */
ccl_device_inline void path_state_modify_bounce(ccl_addr_space PathState *state, bool increase)
{
  /* Modify bounce temporarily for shader eval */
  if (increase)
    state->bounce += 1
  else
    state->bounce -= 1;
}
#endif

ccl_device_inline bool path_state_ao_bounce(INTEGRATOR_STATE_CONST_ARGS)
{
  /* TODO */
#if 0
  if (state->bounce <= kernel_data.integrator.ao_bounces) {
    return false;
  }

  int bounce = state->bounce - state->transmission_bounce - (state->glossy_bounce > 0);
  return (bounce > kernel_data.integrator.ao_bounces);
#else
  return false;
#endif
}

/* Random Number Sampling Utility Functions
 *
 * For each random number in each step of the path we must have a unique
 * dimension to avoid using the same sequence twice.
 *
 * For branches in the path we must be careful not to reuse the same number
 * in a sequence and offset accordingly.
 */

/* RNG State loaded onto stack. */
typedef struct RNGState {
  uint rng_hash;
  uint rng_offset;
  int sample;
} RNGState;

ccl_device_inline void path_state_rng_load(INTEGRATOR_STATE_CONST_ARGS, RNGState *rng_state)
{
  rng_state->rng_hash = INTEGRATOR_STATE(path, rng_hash);
  rng_state->rng_offset = INTEGRATOR_STATE(path, rng_offset);
  rng_state->sample = INTEGRATOR_STATE(path, sample);
}

ccl_device_inline void path_shadow_state_rng_load(INTEGRATOR_STATE_CONST_ARGS, RNGState *rng_state)
{
  rng_state->rng_hash = INTEGRATOR_STATE(shadow_path, rng_hash);
  rng_state->rng_offset = INTEGRATOR_STATE(shadow_path, rng_offset);
  rng_state->sample = INTEGRATOR_STATE(shadow_path, sample);
}

ccl_device_inline float path_state_rng_1D(const KernelGlobals *kg,
                                          const RNGState *rng_state,
                                          int dimension)
{
  return path_rng_1D(
      kg, rng_state->rng_hash, rng_state->sample, rng_state->rng_offset + dimension);
}

ccl_device_inline void path_state_rng_2D(
    const KernelGlobals *kg, const RNGState *rng_state, int dimension, float *fx, float *fy)
{
  path_rng_2D(
      kg, rng_state->rng_hash, rng_state->sample, rng_state->rng_offset + dimension, fx, fy);
}

ccl_device_inline float path_state_rng_1D_hash(const KernelGlobals *kg,
                                               const RNGState *rng_state,
                                               uint hash)
{
  /* Use a hash instead of dimension, this is not great but avoids adding
   * more dimensions to each bounce which reduces quality of dimensions we
   * are already using. */
  return path_rng_1D(
      kg, cmj_hash_simple(rng_state->rng_hash, hash), rng_state->sample, rng_state->rng_offset);
}

ccl_device_inline float path_branched_rng_1D(const KernelGlobals *kg,
                                             uint rng_hash,
                                             const RNGState *rng_state,
                                             int branch,
                                             int num_branches,
                                             int dimension)
{
  return path_rng_1D(
      kg, rng_hash, rng_state->sample * num_branches + branch, rng_state->rng_offset + dimension);
}

ccl_device_inline void path_branched_rng_2D(const KernelGlobals *kg,
                                            uint rng_hash,
                                            const RNGState *rng_state,
                                            int branch,
                                            int num_branches,
                                            int dimension,
                                            float *fx,
                                            float *fy)
{
  path_rng_2D(kg,
              rng_hash,
              rng_state->sample * num_branches + branch,
              rng_state->rng_offset + dimension,
              fx,
              fy);
}

/* Utility functions to get light termination value,
 * since it might not be needed in many cases.
 */
ccl_device_inline float path_state_rng_light_termination(const KernelGlobals *kg,
                                                         const RNGState *state)
{
  if (kernel_data.integrator.light_inv_rr_threshold > 0.0f) {
    return path_state_rng_1D(kg, state, PRNG_LIGHT_TERMINATE);
  }
  return 0.0f;
}

ccl_device_inline float path_branched_rng_light_termination(
    const KernelGlobals *kg, uint rng_hash, const RNGState *state, int branch, int num_branches)
{
  if (kernel_data.integrator.light_inv_rr_threshold > 0.0f) {
    return path_branched_rng_1D(kg, rng_hash, state, branch, num_branches, PRNG_LIGHT_TERMINATE);
  }
  return 0.0f;
}

CCL_NAMESPACE_END
