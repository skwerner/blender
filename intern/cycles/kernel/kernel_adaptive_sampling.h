/*
 * Copyright 2019 Blender Foundation
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

/* Check whether the pixel has converged and should not be sampled anymore. */

ccl_device_forceinline bool kernel_need_sample_pixel(INTEGRATOR_STATE_CONST_ARGS,
                                                     ccl_global float *render_buffer)
{
  if (!kernel_data.film.pass_adaptive_aux_buffer) {
    return true;
  }

  const uint32_t render_pixel_index = INTEGRATOR_STATE(path, render_pixel_index);
  const uint64_t render_buffer_offset = (uint64_t)render_pixel_index *
                                        kernel_data.film.pass_stride;
  ccl_global float *buffer = render_buffer + render_buffer_offset;

  ccl_global float4 *aux = (ccl_global float4 *)(buffer +
                                                 kernel_data.film.pass_adaptive_aux_buffer);
  return (*aux).w == 0.0f;
}

/* Determines whether to continue sampling a given pixel or if it has sufficiently converged. */

ccl_device void kernel_adaptive_sampling_convergence_check(const KernelGlobals *kg,
                                                           ccl_global float *render_buffer,
                                                           int x,
                                                           int y,
                                                           int sample,
                                                           int offset,
                                                           int stride)
{
  const int render_pixel_index = offset + x + y * stride;
  ccl_global float *buffer = render_buffer +
                             (uint64_t)render_pixel_index * kernel_data.film.pass_stride;

  /* TODO(Stefan): Is this better in linear, sRGB or something else? */

  const float4 A = *(ccl_global float4 *)(buffer + kernel_data.film.pass_adaptive_aux_buffer);
  if (A.w != 0.0f) {
    /* If the pixel was considered converged, its state will not change in this kernmel. Early
     * output before doing any math.
     *
     * TODO(sergey): On a GPU it might be better to keep thread alive for better coherency? */
    return;
  }

  const float4 I = *((ccl_global float4 *)buffer);

  /* The per pixel error as seen in section 2.1 of
   * "A hierarchical automatic stopping condition for Monte Carlo global illumination"
   * A small epsilon is added to the divisor to prevent division by zero. */
  const float error = (fabsf(I.x - A.x) + fabsf(I.y - A.y) + fabsf(I.z - A.z)) /
                      (sample * 0.0001f + sqrtf(I.x + I.y + I.z));
  if (error < kernel_data.integrator.adaptive_threshold * (float)sample) {
    /* Set the fourth component to non-zero value to indicate that this pixel has converged. */
    buffer[kernel_data.film.pass_adaptive_aux_buffer + 3] += 1.0f;
  }
}

/* Adjust the values of an adaptively sampled pixel. */

ccl_device void kernel_adaptive_post_adjust(const KernelGlobals *kg,
                                            ccl_global float *buffer,
                                            float sample_multiplier)
{
  *(ccl_global float4 *)(buffer) *= sample_multiplier;

  /* Scale the aux pass too, this is necessary for progressive rendering to work properly. */
  kernel_assert(kernel_data.film.pass_adaptive_aux_buffer);
  *(ccl_global float4 *)(buffer + kernel_data.film.pass_adaptive_aux_buffer) *= sample_multiplier;

#ifdef __PASSES__
  int flag = kernel_data.film.pass_flag;

  if (flag & PASSMASK(NORMAL))
    *(ccl_global float3 *)(buffer + kernel_data.film.pass_normal) *= sample_multiplier;

  if (flag & PASSMASK(UV))
    *(ccl_global float3 *)(buffer + kernel_data.film.pass_uv) *= sample_multiplier;

  if (flag & PASSMASK(MOTION)) {
    *(ccl_global float4 *)(buffer + kernel_data.film.pass_motion) *= sample_multiplier;
    *(ccl_global float *)(buffer + kernel_data.film.pass_motion_weight) *= sample_multiplier;
  }

  if (kernel_data.film.use_light_pass) {
    int light_flag = kernel_data.film.light_pass_flag;

    if (light_flag & PASSMASK(MIST))
      *(ccl_global float *)(buffer + kernel_data.film.pass_mist) *= sample_multiplier;

    /* Shadow pass omitted on purpose. It has its own scale parameter. */

    if (light_flag & PASSMASK(DIFFUSE_INDIRECT))
      *(ccl_global float3 *)(buffer + kernel_data.film.pass_diffuse_indirect) *= sample_multiplier;
    if (light_flag & PASSMASK(GLOSSY_INDIRECT))
      *(ccl_global float3 *)(buffer + kernel_data.film.pass_glossy_indirect) *= sample_multiplier;
    if (light_flag & PASSMASK(TRANSMISSION_INDIRECT))
      *(ccl_global float3 *)(buffer +
                             kernel_data.film.pass_transmission_indirect) *= sample_multiplier;
    if (light_flag & PASSMASK(VOLUME_INDIRECT))
      *(ccl_global float3 *)(buffer + kernel_data.film.pass_volume_indirect) *= sample_multiplier;
    if (light_flag & PASSMASK(DIFFUSE_DIRECT))
      *(ccl_global float3 *)(buffer + kernel_data.film.pass_diffuse_direct) *= sample_multiplier;
    if (light_flag & PASSMASK(GLOSSY_DIRECT))
      *(ccl_global float3 *)(buffer + kernel_data.film.pass_glossy_direct) *= sample_multiplier;
    if (light_flag & PASSMASK(TRANSMISSION_DIRECT))
      *(ccl_global float3 *)(buffer +
                             kernel_data.film.pass_transmission_direct) *= sample_multiplier;
    if (light_flag & PASSMASK(VOLUME_DIRECT))
      *(ccl_global float3 *)(buffer + kernel_data.film.pass_volume_direct) *= sample_multiplier;

    if (light_flag & PASSMASK(EMISSION))
      *(ccl_global float3 *)(buffer + kernel_data.film.pass_emission) *= sample_multiplier;
    if (light_flag & PASSMASK(BACKGROUND))
      *(ccl_global float3 *)(buffer + kernel_data.film.pass_background) *= sample_multiplier;
    if (light_flag & PASSMASK(AO))
      *(ccl_global float3 *)(buffer + kernel_data.film.pass_ao) *= sample_multiplier;

    if (light_flag & PASSMASK(DIFFUSE_COLOR))
      *(ccl_global float3 *)(buffer + kernel_data.film.pass_diffuse_color) *= sample_multiplier;
    if (light_flag & PASSMASK(GLOSSY_COLOR))
      *(ccl_global float3 *)(buffer + kernel_data.film.pass_glossy_color) *= sample_multiplier;
    if (light_flag & PASSMASK(TRANSMISSION_COLOR))
      *(ccl_global float3 *)(buffer +
                             kernel_data.film.pass_transmission_color) *= sample_multiplier;
  }
#endif

#ifdef __DENOISING_FEATURES__

#  define scale_float3_variance(buffer, offset, scale) \
    *(buffer + offset) *= scale; \
    *(buffer + offset + 1) *= scale; \
    *(buffer + offset + 2) *= scale; \
    *(buffer + offset + 3) *= scale * scale; \
    *(buffer + offset + 4) *= scale * scale; \
    *(buffer + offset + 5) *= scale * scale;

#  define scale_shadow_variance(buffer, offset, scale) \
    *(buffer + offset) *= scale; \
    *(buffer + offset + 1) *= scale; \
    *(buffer + offset + 2) *= scale * scale;

  if (kernel_data.film.pass_denoising_data) {
    scale_float3_variance(
        buffer, kernel_data.film.pass_denoising_data + DENOISING_PASS_COLOR, sample_multiplier);
    scale_float3_variance(
        buffer, kernel_data.film.pass_denoising_data + DENOISING_PASS_NORMAL, sample_multiplier);
    scale_float3_variance(
        buffer, kernel_data.film.pass_denoising_data + DENOISING_PASS_ALBEDO, sample_multiplier);
    *(buffer + kernel_data.film.pass_denoising_data + DENOISING_PASS_DEPTH) *= sample_multiplier;
    *(buffer + kernel_data.film.pass_denoising_data + DENOISING_PASS_DEPTH +
      1) *= sample_multiplier * sample_multiplier;
  }
#endif /* __DENOISING_FEATURES__ */

  /* Cryptomatte. */
  if (kernel_data.film.cryptomatte_passes) {
    int num_slots = 0;
    num_slots += (kernel_data.film.cryptomatte_passes & CRYPT_OBJECT) ? 1 : 0;
    num_slots += (kernel_data.film.cryptomatte_passes & CRYPT_MATERIAL) ? 1 : 0;
    num_slots += (kernel_data.film.cryptomatte_passes & CRYPT_ASSET) ? 1 : 0;
    num_slots = num_slots * 2 * kernel_data.film.cryptomatte_depth;
    ccl_global float2 *id_buffer = (ccl_global float2 *)(buffer +
                                                         kernel_data.film.pass_cryptomatte);
    for (int slot = 0; slot < num_slots; slot++) {
      id_buffer[slot].y *= sample_multiplier;
    }
  }

  /* AOVs. */
  for (int i = 0; i < kernel_data.film.pass_aov_value_num; i++) {
    *(buffer + kernel_data.film.pass_aov_value + i) *= sample_multiplier;
  }
  for (int i = 0; i < kernel_data.film.pass_aov_color_num; i++) {
    *((ccl_global float4 *)(buffer + kernel_data.film.pass_aov_color) + i) *= sample_multiplier;
  }
}

/* This is a simple box filter in two passes.
 * When a pixel demands more adaptive samples, let its neighboring pixels draw more samples too. */

ccl_device bool kernel_adaptive_sampling_filter_x(const KernelGlobals *kg,
                                                  ccl_global float *render_buffer,
                                                  int y,
                                                  int start_x,
                                                  int width,
                                                  int offset,
                                                  int stride)
{
  bool any = false;
  bool prev = false;
  for (int x = start_x; x < start_x + width; ++x) {
    int index = offset + x + y * stride;
    ccl_global float *buffer = render_buffer + index * kernel_data.film.pass_stride;
    ccl_global float4 *aux = (ccl_global float4 *)(buffer +
                                                   kernel_data.film.pass_adaptive_aux_buffer);
    if ((*aux).w == 0.0f) {
      any = true;
      if (x > start_x && !prev) {
        index = index - 1;
        buffer = render_buffer + index * kernel_data.film.pass_stride;
        aux = (ccl_global float4 *)(buffer + kernel_data.film.pass_adaptive_aux_buffer);
        (*aux).w = 0.0f;
      }
      prev = true;
    }
    else {
      if (prev) {
        (*aux).w = 0.0f;
      }
      prev = false;
    }
  }
  return any;
}

ccl_device bool kernel_adaptive_sampling_filter_y(const KernelGlobals *kg,
                                                  ccl_global float *render_buffer,
                                                  int x,
                                                  int start_y,
                                                  int height,
                                                  int offset,
                                                  int stride)
{
  bool prev = false;
  bool any = false;
  for (int y = start_y; y < start_y + height; ++y) {
    int index = offset + x + y * stride;
    ccl_global float *buffer = render_buffer + index * kernel_data.film.pass_stride;
    ccl_global float4 *aux = (ccl_global float4 *)(buffer +
                                                   kernel_data.film.pass_adaptive_aux_buffer);
    if ((*aux).w == 0.0f) {
      any = true;
      if (y > start_y && !prev) {
        index = index - stride;
        buffer = render_buffer + index * kernel_data.film.pass_stride;
        aux = (ccl_global float4 *)(buffer + kernel_data.film.pass_adaptive_aux_buffer);
        (*aux).w = 0.0f;
      }
      prev = true;
    }
    else {
      if (prev) {
        (*aux).w = 0.0f;
      }
      prev = false;
    }
  }
  return any;
}

CCL_NAMESPACE_END
