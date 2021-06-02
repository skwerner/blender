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

CCL_NAMESPACE_BEGIN

/* --------------------------------------------------------------------
 * Common utilities.
 */

/* The input buffer contains transparency = 1 - alpha, this converts it to
 * alpha. Also clamp since alpha might end up outside of 0..1 due to Russian
 * roulette. */
ccl_device_forceinline float film_transparency_to_alpha(float transparency)
{
  return saturate(1.0f - transparency);
}

ccl_device_inline float film_get_scale(const KernelFilmConvert *ccl_restrict kfilm_convert,
                                       ccl_global const float *ccl_restrict buffer)
{
  if (kfilm_convert->pass_sample_count == PASS_UNUSED) {
    return kfilm_convert->scale;
  }

  if (kfilm_convert->pass_use_filter) {
    const uint sample_count = *((const uint *)(buffer + kfilm_convert->pass_sample_count));
    return 1.0f / sample_count;
  }

  return 1.0f;
}

ccl_device_inline float film_get_scale_exposure(const KernelFilmConvert *ccl_restrict
                                                    kfilm_convert,
                                                ccl_global const float *ccl_restrict buffer)
{
  if (kfilm_convert->pass_sample_count == PASS_UNUSED) {
    return kfilm_convert->scale_exposure;
  }

  if (kfilm_convert->pass_use_exposure) {
    const float scale = film_get_scale(kfilm_convert, buffer);
    return scale * kfilm_convert->exposure;
  }

  return 1.0f;
}

ccl_device_inline void film_get_scale_and_scale_exposure(
    const KernelFilmConvert *ccl_restrict kfilm_convert,
    ccl_global const float *ccl_restrict buffer,
    float *ccl_restrict scale,
    float *ccl_restrict scale_exposure)
{
  if (kfilm_convert->pass_sample_count == PASS_UNUSED) {
    *scale = kfilm_convert->scale;
    *scale_exposure = kfilm_convert->scale_exposure;
    return;
  }

  if (kfilm_convert->pass_use_filter) {
    const uint sample_count = *((const uint *)(buffer + kfilm_convert->pass_sample_count));
    *scale = 1.0f / sample_count;
  }
  else {
    *scale = 1.0f;
  }

  if (kfilm_convert->pass_use_exposure) {
    *scale_exposure = *scale * kfilm_convert->exposure;
  }
  else {
    *scale_exposure = *scale;
  }
}

/* --------------------------------------------------------------------
 * Float (scalar) passes.
 */

ccl_device_inline void film_get_pass_pixel_depth(const KernelFilmConvert *ccl_restrict
                                                     kfilm_convert,
                                                 ccl_global const float *ccl_restrict buffer,
                                                 float *ccl_restrict pixel)
{
  kernel_assert(kfilm_convert->pass_offset != PASS_UNUSED);

  const float scale_exposure = film_get_scale_exposure(kfilm_convert, buffer);

  const float *in = buffer + kfilm_convert->pass_offset;
  const float f = *in;

  pixel[0] = (f == 0.0f) ? 1e10f : f * scale_exposure;
}

ccl_device_inline void film_get_pass_pixel_mist(const KernelFilmConvert *ccl_restrict
                                                    kfilm_convert,
                                                ccl_global const float *ccl_restrict buffer,
                                                float *ccl_restrict pixel)
{
  kernel_assert(kfilm_convert->pass_offset != PASS_UNUSED);

  const float scale_exposure = film_get_scale_exposure(kfilm_convert, buffer);

  const float *in = buffer + kfilm_convert->pass_offset;
  const float f = *in;

  /* Note that we accumulate 1 - mist in the kernel to avoid having to
   * track the mist values in the integrator state. */
  pixel[0] = saturate(1.0f - f * scale_exposure);
}

ccl_device_inline void film_get_pass_pixel_sample_count(
    const KernelFilmConvert *ccl_restrict kfilm_convert,
    ccl_global const float *ccl_restrict buffer,
    float *ccl_restrict pixel)
{
  /* TODO(sergey): Consider normalizing into the [0..1] range, so that it is possible to see
   * meaningful value when adaptive sampler stopped rendering image way before the maximum
   * number of samples was reached (for examples when number of samples is set to 0 in
   * viewport). */

  kernel_assert(kfilm_convert->pass_offset != PASS_UNUSED);

  const float *in = buffer + kfilm_convert->pass_offset;
  const float f = *in;

  pixel[0] = __float_as_uint(f) * kfilm_convert->scale;
}

ccl_device_inline void film_get_pass_pixel_float(const KernelFilmConvert *ccl_restrict
                                                     kfilm_convert,
                                                 ccl_global const float *ccl_restrict buffer,
                                                 float *ccl_restrict pixel)
{
  kernel_assert(kfilm_convert->pass_offset != PASS_UNUSED);

  const float scale_exposure = film_get_scale_exposure(kfilm_convert, buffer);

  const float *in = buffer + kfilm_convert->pass_offset;
  const float f = *in;

  pixel[0] = f * scale_exposure;
}

/* --------------------------------------------------------------------
 * Float 3 passes.
 */

ccl_device_inline void film_get_pass_pixel_shadow3(const KernelFilmConvert *ccl_restrict
                                                       kfilm_convert,
                                                   ccl_global const float *ccl_restrict buffer,
                                                   float *ccl_restrict pixel)
{
  const float *in = buffer + kfilm_convert->pass_offset;

  const float weight = in[3];
  const float weight_inv = (weight > 0.0f) ? 1.0f / weight : 1.0f;

  const float3 shadow = make_float3(in[0], in[1], in[2]) * weight_inv;

  pixel[0] = shadow.x;
  pixel[1] = shadow.y;
  pixel[2] = shadow.z;
}

ccl_device_inline void film_get_pass_pixel_divide_even_color(
    const KernelFilmConvert *ccl_restrict kfilm_convert,
    ccl_global const float *ccl_restrict buffer,
    float *ccl_restrict pixel)
{
  kernel_assert(kfilm_convert->pass_offset != PASS_UNUSED);
  kernel_assert(kfilm_convert->pass_divide != PASS_UNUSED);

  const float *in = buffer + kfilm_convert->pass_offset;
  const float *in_divide = buffer + kfilm_convert->pass_divide;

  const float3 f = make_float3(in[0], in[1], in[2]);
  const float3 f_divide = make_float3(in_divide[0], in_divide[1], in_divide[2]);
  const float3 f_divided = safe_divide_even_color(f * kfilm_convert->exposure, f_divide);

  pixel[0] = f_divided.x;
  pixel[1] = f_divided.y;
  pixel[2] = f_divided.z;
}

ccl_device_inline void film_get_pass_pixel_float3(const KernelFilmConvert *ccl_restrict
                                                      kfilm_convert,
                                                  ccl_global const float *ccl_restrict buffer,
                                                  float *ccl_restrict pixel)
{
  kernel_assert(kfilm_convert->pass_offset != PASS_UNUSED);

  const float scale_exposure = film_get_scale_exposure(kfilm_convert, buffer);

  const float *in = buffer + kfilm_convert->pass_offset;

  const float3 f = make_float3(in[0], in[1], in[2]) * scale_exposure;

  pixel[0] = f.x;
  pixel[1] = f.y;
  pixel[2] = f.z;
}

/* --------------------------------------------------------------------
 * Float 4 passes.
 */

ccl_device_inline void film_get_pass_pixel_shadow4(const KernelFilmConvert *ccl_restrict
                                                       kfilm_convert,
                                                   ccl_global const float *ccl_restrict buffer,
                                                   float *ccl_restrict pixel)
{
  film_get_pass_pixel_shadow3(kfilm_convert, buffer, pixel);
  pixel[0] = 1.0f;
}

ccl_device_inline void film_get_pass_pixel_motion(const KernelFilmConvert *ccl_restrict
                                                      kfilm_convert,
                                                  ccl_global const float *ccl_restrict buffer,
                                                  float *ccl_restrict pixel)
{
  kernel_assert(kfilm_convert->pass_offset != PASS_UNUSED);
  kernel_assert(kfilm_convert->pass_motion_weight != PASS_UNUSED);

  const float *in = buffer + kfilm_convert->pass_offset;
  const float *in_weight = buffer + kfilm_convert->pass_motion_weight;

  const float weight = in_weight[0];
  const float weight_inv = (weight > 0.0f) ? 1.0f / weight : 0.0f;

  const float4 motion = make_float4(in[0], in[1], in[2], in[3]) * weight_inv;

  pixel[0] = motion.x;
  pixel[1] = motion.y;
  pixel[2] = motion.z;
  pixel[3] = motion.w;
}

ccl_device_inline void film_get_pass_pixel_cryptomatte(const KernelFilmConvert *ccl_restrict
                                                           kfilm_convert,
                                                       ccl_global const float *ccl_restrict buffer,
                                                       float *ccl_restrict pixel)
{
  kernel_assert(kfilm_convert->pass_offset != PASS_UNUSED);

  const float scale = film_get_scale(kfilm_convert, buffer);

  const float *in = buffer + kfilm_convert->pass_offset;

  const float4 f = make_float4(in[0], in[1], in[2], in[3]);

  /* x and z contain integer IDs, don't rescale them.
   * y and w contain matte weights, they get scaled. */
  pixel[0] = f.x;
  pixel[1] = f.y * scale;
  pixel[2] = f.z;
  pixel[3] = f.w * scale;
}

/* Special code which converts noisy image pass from RGB to RGBA using alpha from the combined
 * pass. */
ccl_device_inline void film_get_pass_pixel_denoising_color(
    const KernelFilmConvert *ccl_restrict kfilm_convert,
    ccl_global const float *ccl_restrict buffer,
    float *ccl_restrict pixel)
{
  kernel_assert(kfilm_convert->pass_offset != PASS_UNUSED);
  kernel_assert(kfilm_convert->pass_combined != PASS_UNUSED);

  float scale, scale_exposure;
  film_get_scale_and_scale_exposure(kfilm_convert, buffer, &scale, &scale_exposure);

  const float *in = buffer + kfilm_convert->pass_offset;
  const float *in_combined = buffer + kfilm_convert->pass_combined;

  const float3 color = make_float3(in[0], in[1], in[2]) * scale_exposure;
  const float transparency = in_combined[3] * scale;

  pixel[0] = color.x;
  pixel[1] = color.y;
  pixel[2] = color.z;
  pixel[3] = film_transparency_to_alpha(transparency);
}

ccl_device_inline void film_get_pass_pixel_float4(const KernelFilmConvert *ccl_restrict
                                                      kfilm_convert,
                                                  ccl_global const float *ccl_restrict buffer,
                                                  float *ccl_restrict pixel)
{
  kernel_assert(kfilm_convert->pass_offset != PASS_UNUSED);

  float scale, scale_exposure;
  film_get_scale_and_scale_exposure(kfilm_convert, buffer, &scale, &scale_exposure);

  const float *in = buffer + kfilm_convert->pass_offset;

  /* Note that 3rd channel contains transparency = 1 - alpha at this point. */
  const float3 color = make_float3(in[0], in[1], in[2]) * scale_exposure;
  const float transparency = in[3] * scale;

  pixel[0] = color.x;
  pixel[1] = color.y;
  pixel[2] = color.z;
  pixel[3] = film_transparency_to_alpha(transparency);
}

/* --------------------------------------------------------------------
 * Shadow catcher.
 */

ccl_device_inline float4
film_calculate_shadow_catcher(const KernelFilmConvert *ccl_restrict kfilm_convert,
                              ccl_global const float *ccl_restrict buffer)
{
  /* For the shadow catcher pass we divide combined pass by the shadow catcher.
   *
   * The non-obvious trick here is that we add matte pass to the shadow catcher, so that we
   * avoid division by zero. This solves artifacts around edges of the artificial object.
   *
   * Another trick we do here is to alpha-over the pass on top of white. and ignore the alpha.
   * This way using transparent film to render artificial objects will be easy to be combined
   * with a backdrop. */

  kernel_assert(kfilm_convert->pass_offset != PASS_UNUSED);
  kernel_assert(kfilm_convert->pass_combined != PASS_UNUSED);
  kernel_assert(kfilm_convert->pass_shadow_catcher != PASS_UNUSED);
  kernel_assert(kfilm_convert->pass_shadow_catcher_matte != PASS_UNUSED);

  float scale, scale_exposure;
  film_get_scale_and_scale_exposure(kfilm_convert, buffer, &scale, &scale_exposure);

  ccl_global const float *in_combined = buffer + kfilm_convert->pass_combined;
  ccl_global const float *in_catcher = buffer + kfilm_convert->pass_shadow_catcher;
  ccl_global const float *in_matte = buffer + kfilm_convert->pass_shadow_catcher_matte;

  const float3 color_catcher = make_float3(in_catcher[0], in_catcher[1], in_catcher[2]) *
                               scale_exposure;

  const float3 color_combined = make_float3(in_combined[0], in_combined[1], in_combined[2]) *
                                scale_exposure;

  const float3 color_matte = make_float3(in_matte[0], in_matte[1], in_matte[2]) * scale_exposure;

  const float transparency = in_combined[3] * scale;
  const float alpha = saturate(1.0f - transparency);

  const float3 shadow_catcher = safe_divide_even_color(color_combined,
                                                       color_catcher + color_matte);

  /* Restore pre-multipled nature of the color, avoiding artifacts on the edges.
   * Makes sense since the division of premultiplied color's "removes" alpha from the
   * result. */
  const float3 pixel = (1.0f - alpha) * one_float3() + alpha * shadow_catcher;

  return make_float4(pixel.x, pixel.y, pixel.z, 1.0f);
}

ccl_device_inline float4 film_calculate_shadow_catcher_matte_with_shadow(
    const KernelFilmConvert *ccl_restrict kfilm_convert,
    ccl_global const float *ccl_restrict buffer)
{
  /* The approximation of the shadow is 1 - average(shadow_catcher_pass). A better approximation
   * is possible.
   *
   * The matte is alpha-overed onto the shadow (which is kind of alpha-overing shadow onto footage,
   * and then alpha-overing synthetic objects on top). */

  kernel_assert(kfilm_convert->pass_offset != PASS_UNUSED);
  kernel_assert(kfilm_convert->pass_shadow_catcher != PASS_UNUSED);
  kernel_assert(kfilm_convert->pass_shadow_catcher_matte != PASS_UNUSED);

  float scale, scale_exposure;
  film_get_scale_and_scale_exposure(kfilm_convert, buffer, &scale, &scale_exposure);

  ccl_global const float *in_matte = buffer + kfilm_convert->pass_shadow_catcher_matte;

  const float4 shadow_catcher = film_calculate_shadow_catcher(kfilm_convert, buffer);
  const float3 color_matte = make_float3(in_matte[0], in_matte[1], in_matte[2]) * scale_exposure;

  const float transparency = in_matte[3] * scale;
  const float alpha = saturate(1.0f - transparency);

  return make_float4(color_matte.x,
                     color_matte.y,
                     color_matte.z,
                     (1.0f - alpha) * (1.0f - average(float4_to_float3(shadow_catcher))) + alpha);
}

ccl_device_inline void film_get_pass_pixel_shadow_catcher(
    const KernelFilmConvert *ccl_restrict kfilm_convert,
    ccl_global const float *ccl_restrict buffer,
    float *ccl_restrict pixel)
{
  const float4 pixel_value = film_calculate_shadow_catcher(kfilm_convert, buffer);

  pixel[0] = pixel_value.x;
  pixel[1] = pixel_value.y;
  pixel[2] = pixel_value.z;
  pixel[3] = pixel_value.w;
}

ccl_device_inline void film_get_pass_pixel_shadow_catcher_matte_with_shadow(
    const KernelFilmConvert *ccl_restrict kfilm_convert,
    ccl_global const float *ccl_restrict buffer,
    float *ccl_restrict pixel)
{
  const float4 pixel_value = film_calculate_shadow_catcher_matte_with_shadow(kfilm_convert,
                                                                             buffer);

  pixel[0] = pixel_value.x;
  pixel[1] = pixel_value.y;
  pixel[2] = pixel_value.z;
  pixel[3] = pixel_value.w;
}

/* --------------------------------------------------------------------
 * Compositing and overlays.
 */

ccl_device_inline void film_apply_pass_pixel_overlays_rgba(
    const KernelFilmConvert *ccl_restrict kfilm_convert,
    ccl_global const float *ccl_restrict buffer,
    float *ccl_restrict pixel)
{
  if (kfilm_convert->show_active_pixels &&
      kfilm_convert->pass_adaptive_aux_buffer != PASS_UNUSED) {
    if (buffer[kfilm_convert->pass_adaptive_aux_buffer + 3] == 0.0f) {
      const float3 active_rgb = make_float3(1.0f, 0.0f, 0.0f);
      const float3 mix_rgb = interp(make_float3(pixel[0], pixel[1], pixel[2]), active_rgb, 0.5f);
      pixel[0] = mix_rgb.x;
      pixel[1] = mix_rgb.y;
      pixel[2] = mix_rgb.z;
    }
  }
}

/* --------------------------------------------------------------------
 * Legacy.
 */

ccl_device float4 film_get_pass_result(const KernelGlobals *kg, ccl_global float *buffer)
{
  float4 pass_result;

  const int display_pass_offset = kernel_data.film.display_pass_offset;
  const int display_pass_components = kernel_data.film.display_pass_components;

  if (display_pass_components == 4) {
    const float4 in = *(ccl_global float4 *)(buffer + display_pass_offset);
    const float transparency = (kernel_data.film.use_display_pass_alpha) ? in.w : 0.0f;

    pass_result = make_float4(in.x, in.y, in.z, transparency);

    int display_divide_pass_offset = kernel_data.film.display_divide_pass_offset;
    if (display_divide_pass_offset != -1) {
      ccl_global const float4 *divide_in = (ccl_global float4 *)(buffer +
                                                                 display_divide_pass_offset);
      const float3 divided = safe_divide_even_color(float4_to_float3(pass_result),
                                                    float4_to_float3(*divide_in));
      pass_result = make_float4(divided.x, divided.y, divided.z, pass_result.w);
    }

    if (kernel_data.film.use_display_exposure) {
      const float exposure = kernel_data.film.exposure;
      pass_result *= make_float4(exposure, exposure, exposure, 1.0f);
    }
  }
  else if (display_pass_components == 1) {
    ccl_global const float *in = (ccl_global float *)(buffer + display_pass_offset);
    if (kernel_data.film.pass_sample_count != PASS_UNUSED &&
        kernel_data.film.pass_sample_count == display_pass_offset) {
      const float value = __float_as_uint(*in);
      pass_result = make_float4(value, value, value, 0.0f);
    }
    else {
      pass_result = make_float4(*in, *in, *in, 0.0f);
    }
  }

  return pass_result;
}

ccl_device void kernel_film_convert_to_half_float(const KernelGlobals *kg,
                                                  ccl_global uchar4 *rgba,
                                                  ccl_global float *render_buffer,
                                                  float sample_scale,
                                                  int x,
                                                  int y,
                                                  int offset,
                                                  int stride)
{
  const int render_pixel_index = offset + x + y * stride;
  const uint64_t render_buffer_offset = (uint64_t)render_pixel_index *
                                        kernel_data.film.pass_stride;
  ccl_global float *buffer = render_buffer + render_buffer_offset;

  float4 rgba_in = film_get_pass_result(kg, buffer);

  /* Filter the pixel if needed. */
  if (kernel_data.film.display_divide_pass_offset == -1) {
    /* Divide by adaptive sampling count.
     * Note that the sample count pass gets divided by the overall sampls count, so that it gives
     * meaningful result (rather than becoming uniform buffer filled with 1). */
    if (kernel_data.film.pass_sample_count != PASS_UNUSED &&
        kernel_data.film.pass_sample_count != kernel_data.film.display_pass_offset) {
      sample_scale = 1.0f / __float_as_uint(buffer[kernel_data.film.pass_sample_count]);
    }
    rgba_in *= sample_scale;
  }

  /* Highlight the pixel. */
  if (kernel_data.film.show_active_pixels &&
      kernel_data.film.pass_adaptive_aux_buffer != PASS_UNUSED) {
    if (buffer[kernel_data.film.pass_adaptive_aux_buffer + 3] == 0.0f) {
      const float3 active_rgb = make_float3(1.0f, 0.0f, 0.0f);
      const float3 mix_rgb = interp(float4_to_float3(rgba_in), active_rgb, 0.5f);
      rgba_in = make_float4(mix_rgb.x, mix_rgb.y, mix_rgb.z, rgba_in.w);
    }
  }

  rgba_in.w = film_transparency_to_alpha(rgba_in.w);

  ccl_global half *out = (ccl_global half *)rgba + render_pixel_index * 4;
  float4_store_half(out, rgba_in);
}

CCL_NAMESPACE_END
