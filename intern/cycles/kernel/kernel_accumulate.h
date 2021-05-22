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

#include "kernel_adaptive_sampling.h"
#include "kernel_random.h"
#include "kernel_shadow_catcher.h"
#include "kernel_write_passes.h"

CCL_NAMESPACE_BEGIN

/* --------------------------------------------------------------------
 * BSDF Evaluation
 *
 * BSDF evaluation result, split between diffuse and glossy. This is used to
 * accumulate render passes separately. Note that reflection, transmission
 * and volume scattering are written to different render passes, but we assume
 * that only one of those can happen at a bounce, and so do not need to accumulate
 * them separately. */

ccl_device_inline void bsdf_eval_init(BsdfEval *eval,
                                      const bool is_diffuse,
                                      float3 value,
                                      const bool use_light_pass)
{
  eval->diffuse = zero_float3();
  eval->glossy = zero_float3();

  if (is_diffuse) {
    eval->diffuse = value;
  }
  else {
    eval->glossy = value;
  }
}

ccl_device_inline void bsdf_eval_accum(BsdfEval *eval,
                                       const bool is_diffuse,
                                       float3 value,
                                       float mis_weight)
{
  value *= mis_weight;

  if (is_diffuse) {
    eval->diffuse += value;
  }
  else {
    eval->glossy += value;
  }
}

ccl_device_inline bool bsdf_eval_is_zero(BsdfEval *eval)
{
  return is_zero(eval->diffuse) && is_zero(eval->glossy);
}

ccl_device_inline void bsdf_eval_mul(BsdfEval *eval, float value)
{
  eval->diffuse *= value;
  eval->glossy *= value;
}

ccl_device_inline void bsdf_eval_mul3(BsdfEval *eval, float3 value)
{
  eval->diffuse *= value;
  eval->glossy *= value;
}

ccl_device_inline float3 bsdf_eval_sum(const BsdfEval *eval)
{
  return eval->diffuse + eval->glossy;
}

ccl_device_inline float3 bsdf_eval_diffuse_glossy_ratio(const BsdfEval *eval)
{
  /* Ratio of diffuse and glossy to recover proportions for writing to render pass.
   * We assume reflection, transmission and volume scatter to be exclusive. */
  return safe_divide_float3_float3(eval->diffuse, eval->diffuse + eval->glossy);
}

/* --------------------------------------------------------------------
 * Clamping
 *
 * Clamping is done on a per-contribution basis so that we can write directly
 * to render buffers instead of using per-thread memory, and to avoid the
 * impact of clamping on other contributions. */

ccl_device_forceinline void kernel_accum_clamp(const KernelGlobals *kg, float3 *L, int bounce)
{
  /* Make sure all components are finite, allowing the contribution to be usable by adaptive
   * sampling convergence check, but also to make it so render result never causes issues with
   * post-processing. */
  *L = ensure_finite3(*L);

#ifdef __CLAMP_SAMPLE__
  float limit = (bounce > 0) ? kernel_data.integrator.sample_clamp_indirect :
                               kernel_data.integrator.sample_clamp_direct;
  float sum = reduce_add(fabs(*L));
  if (sum > limit) {
    *L *= limit / sum;
  }
#endif
}

/* --------------------------------------------------------------------
 * Legacy functions.
 *
 * TODO: Seems to be mainly related to shadow catcher, which is about to have new implementation.
 * Some other code is related to clamping, which is done in `kernel_accum_clamp()`. Port remaining
 * parts over, remove legacy code.
 */

#if 0
ccl_device_inline void path_radiance_accum_light(const KernelGlobals *kg,
                                                 PathRadiance *L,
                                                 ccl_addr_space PathState *state,
                                                 float3 throughput,
                                                 BsdfEval *bsdf_eval,
                                                 float3 shadow,
                                                 float shadow_fac,
                                                 bool is_lamp)
{
#  ifdef __SHADOW_TRICKS__
  if (state->flag & PATH_RAY_STORE_SHADOW_INFO) {
    float3 light = throughput * bsdf_eval->sum_no_mis;
    L->path_total += light;
    L->path_total_shaded += shadow * light;

    if (state->flag & PATH_RAY_SHADOW_CATCHER) {
      return;
    }
  }
#  endif
}

ccl_device_inline void path_radiance_accum_total_light(PathRadiance *L,
                                                       ccl_addr_space PathState *state,
                                                       float3 throughput,
                                                       const BsdfEval *bsdf_eval)
{
#  ifdef __SHADOW_TRICKS__
  if (state->flag & PATH_RAY_STORE_SHADOW_INFO) {
    L->path_total += throughput * bsdf_eval->sum_no_mis;
  }
#  else
  (void)L;
  (void)state;
  (void)throughput;
  (void)bsdf_eval;
#  endif
}

ccl_device_inline void path_radiance_accum_background(const KernelGlobals *kg,
                                                      PathRadiance *L,
                                                      ccl_addr_space PathState *state,
                                                      float3 throughput,
                                                      float3 value)
{

#  ifdef __SHADOW_TRICKS__
  if (state->flag & PATH_RAY_STORE_SHADOW_INFO) {
    L->path_total += throughput * value;
    L->path_total_shaded += throughput * value * L->shadow_transparency;

    if (state->flag & PATH_RAY_SHADOW_CATCHER) {
      return;
    }
  }
#  endif
}

ccl_device_inline void path_radiance_accum_transparent(PathRadiance *L,
                                                       ccl_addr_space PathState *state,
                                                       float3 throughput)
{
  L->transparent += average(throughput);
}

#  ifdef __SHADOW_TRICKS__
ccl_device_inline void path_radiance_accum_shadowcatcher(PathRadiance *L,
                                                         float3 throughput,
                                                         float3 background)
{
  L->shadow_throughput += average(throughput);
  L->shadow_background_color += throughput * background;
  L->has_shadow_catcher = 1;
}
#  endif

#  ifdef __SHADOW_TRICKS__
ccl_device_inline void path_radiance_sum_shadowcatcher(const KernelGlobals *kg,
                                                       PathRadiance *L,
                                                       float3 *L_sum,
                                                       float *alpha)
{
  /* Calculate current shadow of the path. */
  float path_total = average(L->path_total);
  float shadow;

  if (UNLIKELY(!isfinite_safe(path_total))) {
    kernel_assert(!"Non-finite total radiance along the path");
    shadow = 0.0f;
  }
  else if (path_total == 0.0f) {
    shadow = L->shadow_transparency;
  }
  else {
    float path_total_shaded = average(L->path_total_shaded);
    shadow = path_total_shaded / path_total;
  }

  /* Calculate final light sum and transparency for shadow catcher object. */
  if (kernel_data.background.transparent) {
    *alpha -= L->shadow_throughput * shadow;
  }
  else {
    L->shadow_background_color *= shadow;
    *L_sum += L->shadow_background_color;
  }
}
#  endif

ccl_device_inline float3 path_radiance_clamp_and_sum(const KernelGlobals *kg,
                                                     PathRadiance *L,
                                                     float *alpha)
{
  float3 L_sum;
  /* Light Passes are used */
#  ifdef __PASSES__
  float3 L_direct, L_indirect;
  if (L->use_light_pass) {
    path_radiance_sum_indirect(L);

    L_direct = L->direct_diffuse + L->direct_glossy + L->direct_transmission + L->direct_volume +
               L->emission;
    L_indirect = L->indirect_diffuse + L->indirect_glossy + L->indirect_transmission +
                 L->indirect_volume;

    if (!kernel_data.background.transparent)
      L_direct += L->background;

    L_sum = L_direct + L_indirect;
    float sum = fabsf((L_sum).x) + fabsf((L_sum).y) + fabsf((L_sum).z);

    /* Reject invalid value */
    if (!isfinite_safe(sum)) {
      kernel_assert(!"Non-finite sum in path_radiance_clamp_and_sum!");
      L_sum = zero_float3();

      L->direct_diffuse = zero_float3();
      L->direct_glossy = zero_float3();
      L->direct_transmission = zero_float3();
      L->direct_volume = zero_float3();

      L->indirect_diffuse = zero_float3();
      L->indirect_glossy = zero_float3();
      L->indirect_transmission = zero_float3();
      L->indirect_volume = zero_float3();

      L->emission = zero_float3();
    }
  }

  /* No Light Passes */
  else
#  endif
  {
    L_sum = L->emission;

    /* Reject invalid value */
    float sum = fabsf((L_sum).x) + fabsf((L_sum).y) + fabsf((L_sum).z);
    if (!isfinite_safe(sum)) {
      kernel_assert(!"Non-finite final sum in path_radiance_clamp_and_sum!");
      L_sum = zero_float3();
    }
  }

  /* Compute alpha. */
  *alpha = 1.0f - L->transparent;

  /* Add shadow catcher contributions. */
#  ifdef __SHADOW_TRICKS__
  if (L->has_shadow_catcher) {
    path_radiance_sum_shadowcatcher(kg, L, &L_sum, alpha);
  }
#  endif /* __SHADOW_TRICKS__ */

  return L_sum;
}
#endif

/* --------------------------------------------------------------------
 * Pass accumulation utilities.
 */

/* Get pointer to pixel in render buffer. */
ccl_device_forceinline ccl_global float *kernel_accum_pixel_render_buffer(
    INTEGRATOR_STATE_CONST_ARGS, ccl_global float *ccl_restrict render_buffer)
{
  const uint32_t render_pixel_index = INTEGRATOR_STATE(path, render_pixel_index);
  const uint64_t render_buffer_offset = (uint64_t)render_pixel_index *
                                        kernel_data.film.pass_stride;
  return render_buffer + render_buffer_offset;
}

/* --------------------------------------------------------------------
 * Adaptive sampling.
 */

ccl_device_inline int kernel_accum_sample(INTEGRATOR_STATE_CONST_ARGS,
                                          ccl_global float *ccl_restrict render_buffer,
                                          int sample)
{
  if (kernel_data.film.pass_sample_count == PASS_UNUSED) {
    return sample;
  }

  ccl_global float *buffer = kernel_accum_pixel_render_buffer(INTEGRATOR_STATE_PASS,
                                                              render_buffer);

  return atomic_fetch_and_add_uint32((uint *)(buffer) + kernel_data.film.pass_sample_count, 1);
}

ccl_device void kernel_accum_adaptive_buffer(INTEGRATOR_STATE_CONST_ARGS,
                                             const float3 contribution,
                                             ccl_global float *ccl_restrict buffer)
{
  /* Adaptive Sampling. Fill the additional buffer with the odd samples and calculate our stopping
   * criteria. This is the heuristic from "A hierarchical automatic stopping condition for Monte
   * Carlo global illumination" except that here it is applied per pixel and not in hierarchical
   * tiles. */

  if (kernel_data.film.pass_adaptive_aux_buffer == PASS_UNUSED) {
    return;
  }

  const int sample = INTEGRATOR_STATE(path, sample);
  if (sample_is_even(kernel_data.integrator.sampling_pattern, sample)) {
    kernel_write_pass_float4(
        buffer + kernel_data.film.pass_adaptive_aux_buffer,
        make_float4(contribution.x * 2.0f, contribution.y * 2.0f, contribution.z * 2.0f, 0.0f));
  }
}

/* --------------------------------------------------------------------
 * Shadow catcher.
 */

#ifdef __SHADOW_CATCHER__

/* Accumulate contribution to the Shadow Catcher pass.
 *
 * Returns truth if the contribution is fully handled here and is not to be added to the other
 * passes (like combined, adaptive sampling, denoising passes). */

ccl_device bool kernel_accum_shadow_catcher(INTEGRATOR_STATE_CONST_ARGS,
                                            const float3 contribution,
                                            ccl_global float *ccl_restrict buffer)
{
  /* Matte pass. */
  if (kernel_data.film.pass_shadow_catcher_matte != PASS_UNUSED) {
    if (kernel_shadow_catcher_is_matte_path(INTEGRATOR_STATE_PASS)) {
      kernel_write_pass_float3(buffer + kernel_data.film.pass_shadow_catcher_matte, contribution);
    }
  }

  /* Shadow catcher pass. */
  if (kernel_data.film.pass_shadow_catcher != PASS_UNUSED) {
    if (kernel_shadow_catcher_is_object_pass(INTEGRATOR_STATE_PASS)) {
      kernel_write_pass_float3(buffer + kernel_data.film.pass_shadow_catcher, contribution);
      return true;
    }
  }

  return false;
}

ccl_device bool kernel_accum_shadow_catcher_transparent(INTEGRATOR_STATE_CONST_ARGS,
                                                        const float3 contribution,
                                                        const float transparent,
                                                        ccl_global float *ccl_restrict buffer)
{
  /* Matte pass. */
  if (kernel_data.film.pass_shadow_catcher_matte != PASS_UNUSED) {
    if (kernel_shadow_catcher_is_matte_path(INTEGRATOR_STATE_PASS)) {
      kernel_write_pass_float4(
          buffer + kernel_data.film.pass_shadow_catcher_matte,
          make_float4(contribution.x, contribution.y, contribution.z, transparent));
    }
  }

  /* Shadow catcher pass. */
  if (kernel_data.film.pass_shadow_catcher != PASS_UNUSED) {
    if (kernel_shadow_catcher_is_object_pass(INTEGRATOR_STATE_PASS)) {
      kernel_write_pass_float4(
          buffer + kernel_data.film.pass_shadow_catcher,
          make_float4(contribution.x, contribution.y, contribution.z, transparent));
      return true;
    }
  }

  return false;
}

#endif /* __SHADOW_CATCHER__ */

/* --------------------------------------------------------------------
 * Render passes.
 */

/* Write combined pass. */
ccl_device_inline void kernel_accum_combined_pass(INTEGRATOR_STATE_CONST_ARGS,
                                                  const float3 contribution,
                                                  ccl_global float *ccl_restrict buffer)
{
#ifdef __SHADOW_CATCHER__
  if (kernel_accum_shadow_catcher(INTEGRATOR_STATE_PASS, contribution, buffer)) {
    return;
  }
#endif

  if (kernel_data.film.light_pass_flag & PASSMASK(COMBINED)) {
    kernel_write_pass_float3(buffer, contribution);
  }

#ifdef __PASSES__
  if (kernel_data.film.pass_denoising_color != PASS_UNUSED) {
    kernel_write_pass_float3_unaligned(buffer + kernel_data.film.pass_denoising_color,
                                       contribution);
  }
#endif

  kernel_accum_adaptive_buffer(INTEGRATOR_STATE_PASS, contribution, buffer);
}

/* Write combined pass with transparency. */
ccl_device_inline void kernel_accum_combined_transparent_pass(INTEGRATOR_STATE_CONST_ARGS,
                                                              const float3 contribution,
                                                              const float transparent,
                                                              ccl_global float *ccl_restrict
                                                                  buffer)
{
#ifdef __SHADOW_CATCHER__
  if (kernel_accum_shadow_catcher_transparent(
          INTEGRATOR_STATE_PASS, contribution, transparent, buffer)) {
    return;
  }
#endif

  if (kernel_data.film.light_pass_flag & PASSMASK(COMBINED)) {
    kernel_write_pass_float4(
        buffer, make_float4(contribution.x, contribution.y, contribution.z, transparent));
  }

#ifdef __PASSES__
  if (kernel_data.film.pass_denoising_color != PASS_UNUSED) {
    kernel_write_pass_float3_unaligned(buffer + kernel_data.film.pass_denoising_color,
                                       contribution);
  }
#endif

  kernel_accum_adaptive_buffer(INTEGRATOR_STATE_PASS, contribution, buffer);
}

/* Write background or emission to appropriate pass. */
ccl_device_inline void kernel_accum_emission_or_background_pass(INTEGRATOR_STATE_CONST_ARGS,
                                                                float3 contribution,
                                                                ccl_global float *ccl_restrict
                                                                    buffer,
                                                                const int pass)
{
  if (!(kernel_data.film.light_pass_flag & PASS_ANY)) {
    return;
  }

#ifdef __PASSES__
  const int path_flag = INTEGRATOR_STATE(path, flag);
  int pass_offset = PASS_UNUSED;

  if (!(path_flag & PATH_RAY_ANY_PASS)) {
    /* Directly visible, write to emission or background pass. */
    pass_offset = pass;

    /* Denoising albedo. */
#  ifdef __DENOISING_FEATURES__
    if (path_flag & PATH_RAY_DENOISING_FEATURES) {
      if (kernel_data.film.pass_denoising_albedo != PASS_UNUSED) {
        const float3 denoising_feature_throughput = INTEGRATOR_STATE(path,
                                                                     denoising_feature_throughput);
        const float3 denoising_albedo = denoising_feature_throughput * contribution;
        kernel_write_pass_float3_unaligned(buffer + kernel_data.film.pass_denoising_albedo,
                                           denoising_albedo);
      }
    }
#  endif /* __DENOISING_FEATURES__ */
  }
  else if (path_flag & PATH_RAY_REFLECT_PASS) {
    /* Indirectly visible through reflection. */
    const int glossy_pass_offset = pass_offset = (INTEGRATOR_STATE(path, bounce) == 1) ?
                                                     kernel_data.film.pass_glossy_direct :
                                                     kernel_data.film.pass_glossy_indirect;

    if (glossy_pass_offset != PASS_UNUSED) {
      /* Glossy is a subset of the throughput, reconstruct it here using the
       * diffuse-glossy ratio. */
      const float3 ratio = INTEGRATOR_STATE(path, diffuse_glossy_ratio);
      const float3 glossy_contribution = (one_float3() - ratio) * contribution;
      kernel_write_pass_float3(buffer + glossy_pass_offset, glossy_contribution);
    }

    /* Reconstruct diffuse subset of throughput. */
    pass_offset = (INTEGRATOR_STATE(path, bounce) == 1) ? kernel_data.film.pass_diffuse_direct :
                                                          kernel_data.film.pass_diffuse_indirect;
    if (pass_offset != PASS_UNUSED) {
      contribution *= INTEGRATOR_STATE(path, diffuse_glossy_ratio);
    }
  }
  else if (path_flag & PATH_RAY_TRANSMISSION_PASS) {
    /* Indirectly visible through transmission. */
    pass_offset = (INTEGRATOR_STATE(path, bounce) == 1) ?
                      kernel_data.film.pass_transmission_direct :
                      kernel_data.film.pass_transmission_indirect;
  }
  else if (path_flag & PATH_RAY_VOLUME_PASS) {
    /* Indirectly visible through volume. */
    pass_offset = (INTEGRATOR_STATE(path, bounce) == 1) ? kernel_data.film.pass_volume_direct :
                                                          kernel_data.film.pass_volume_indirect;
  }

  /* Single write call for GPU coherence. */
  if (pass_offset != PASS_UNUSED) {
    kernel_write_pass_float3(buffer + pass_offset, contribution);
  }
#endif /* __PASSES__ */
}

/* Write light contribution to render buffer. */
ccl_device_inline void kernel_accum_light(INTEGRATOR_STATE_CONST_ARGS,
                                          ccl_global float *ccl_restrict render_buffer)
{
  /* The throughput for shadow paths already contains the light shader evaluation. */
  float3 contribution = INTEGRATOR_STATE(shadow_path, throughput);
  kernel_accum_clamp(kg, &contribution, INTEGRATOR_STATE(shadow_path, bounce) - 1);

  ccl_global float *buffer = kernel_accum_pixel_render_buffer(INTEGRATOR_STATE_PASS,
                                                              render_buffer);

  kernel_accum_combined_pass(INTEGRATOR_STATE_PASS, contribution, buffer);

#ifdef __PASSES__
  if (kernel_data.film.light_pass_flag & PASS_ANY) {
    const int path_flag = INTEGRATOR_STATE(shadow_path, flag);
    int pass_offset = PASS_UNUSED;

    if (path_flag & PATH_RAY_REFLECT_PASS) {
      /* Indirectly visible through reflection. */
      const int glossy_pass_offset = pass_offset = (INTEGRATOR_STATE(shadow_path, bounce) == 0) ?
                                                       kernel_data.film.pass_glossy_direct :
                                                       kernel_data.film.pass_glossy_indirect;

      if (glossy_pass_offset != PASS_UNUSED) {
        /* Glossy is a subset of the throughput, reconstruct it here using the
         * diffuse-glossy ratio. */
        const float3 ratio = INTEGRATOR_STATE(shadow_path, diffuse_glossy_ratio);
        const float3 glossy_contribution = (one_float3() - ratio) * contribution;
        kernel_write_pass_float3(buffer + glossy_pass_offset, glossy_contribution);
      }

      /* Reconstruct diffuse subset of throughput. */
      pass_offset = (INTEGRATOR_STATE(shadow_path, bounce) == 0) ?
                        kernel_data.film.pass_diffuse_direct :
                        kernel_data.film.pass_diffuse_indirect;
      if (pass_offset != PASS_UNUSED) {
        contribution *= INTEGRATOR_STATE(shadow_path, diffuse_glossy_ratio);
      }
    }
    else if (path_flag & PATH_RAY_TRANSMISSION_PASS) {
      /* Indirectly visible through transmission. */
      pass_offset = (INTEGRATOR_STATE(shadow_path, bounce) == 0) ?
                        kernel_data.film.pass_transmission_direct :
                        kernel_data.film.pass_transmission_indirect;
    }
    else if (path_flag & PATH_RAY_VOLUME_PASS) {
      /* Indirectly visible through volume. */
      pass_offset = (INTEGRATOR_STATE(shadow_path, bounce) == 0) ?
                        kernel_data.film.pass_volume_direct :
                        kernel_data.film.pass_volume_indirect;
    }

    /* Single write call for GPU coherence. */
    if (pass_offset != PASS_UNUSED) {
      kernel_write_pass_float3(buffer + pass_offset, contribution);
    }

    /* TODO: Write shadow pass. */
#  if 0
  if (path_flag & PATH_RAY_SHADOW_FOR_LIGHT) {
    const int shadow_pass_offset = kernel_data.film.pass_shadow;
    if (shadow_pass_offset != PASS_UNUSED) {
      kernel_write_pass_float4(
          buffer + shadow_pass_offset,
          make_float4(shadow.x, shadow.y, shadow.z, kernel_data.film.pass_shadow_scale));
    }
  }
#  endif
  }
#endif
}

/* Write transparency to render buffer.
 *
 * Note that we accumulate transparency = 1 - alpha in the render buffer.
 * Otherwise we'd have to write alpha on path termination, which happens
 * in many places. */
ccl_device_inline void kernel_accum_transparent(INTEGRATOR_STATE_CONST_ARGS,
                                                const float transparent,
                                                ccl_global float *ccl_restrict render_buffer)
{
  if (kernel_data.film.light_pass_flag & PASSMASK(COMBINED)) {
    ccl_global float *buffer = kernel_accum_pixel_render_buffer(INTEGRATOR_STATE_PASS,
                                                                render_buffer);
    kernel_write_pass_float(buffer + 3, transparent);
  }
}

/* Write background contribution to render buffer.
 *
 * Includes transparency, matching kernel_accum_transparent. */
ccl_device_inline void kernel_accum_background(INTEGRATOR_STATE_CONST_ARGS,
                                               const float3 L,
                                               const float transparent,
                                               ccl_global float *ccl_restrict render_buffer)
{
  float3 contribution = INTEGRATOR_STATE(path, throughput) * L;
  kernel_accum_clamp(kg, &contribution, INTEGRATOR_STATE(path, bounce) - 1);

  ccl_global float *buffer = kernel_accum_pixel_render_buffer(INTEGRATOR_STATE_PASS,
                                                              render_buffer);

  kernel_accum_combined_transparent_pass(INTEGRATOR_STATE_PASS, contribution, transparent, buffer);
  kernel_accum_emission_or_background_pass(
      INTEGRATOR_STATE_PASS, contribution, buffer, kernel_data.film.pass_background);
}

/* Write emission to render buffer. */
ccl_device_inline void kernel_accum_emission(INTEGRATOR_STATE_CONST_ARGS,
                                             const float3 L,
                                             ccl_global float *ccl_restrict render_buffer)
{
  float3 contribution = INTEGRATOR_STATE(path, throughput) * L;
  kernel_accum_clamp(kg, &contribution, INTEGRATOR_STATE(path, bounce) - 1);

  ccl_global float *buffer = kernel_accum_pixel_render_buffer(INTEGRATOR_STATE_PASS,
                                                              render_buffer);

  kernel_accum_combined_pass(INTEGRATOR_STATE_PASS, contribution, buffer);
  kernel_accum_emission_or_background_pass(
      INTEGRATOR_STATE_PASS, contribution, buffer, kernel_data.film.pass_emission);
}

CCL_NAMESPACE_END
