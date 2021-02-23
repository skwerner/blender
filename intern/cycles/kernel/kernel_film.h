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

ccl_device float4 film_get_pass_result(const KernelGlobals *kg,
                                       ccl_global float *buffer,
                                       int index)
{
  float4 pass_result;

  int display_pass_stride = kernel_data.film.display_pass_stride;
  int display_pass_components = kernel_data.film.display_pass_components;

  if (display_pass_components == 4) {
    float4 in = *(ccl_global float4 *)(buffer + display_pass_stride +
                                       index * kernel_data.film.pass_stride);
    float transparency = (kernel_data.film.use_display_pass_alpha) ? in.w : 0.0f;

    pass_result = make_float4(in.x, in.y, in.z, transparency);

    int display_divide_pass_stride = kernel_data.film.display_divide_pass_stride;
    if (display_divide_pass_stride != -1) {
      ccl_global float4 *divide_in = (ccl_global float4 *)(buffer + display_divide_pass_stride +
                                                           index * kernel_data.film.pass_stride);
      float3 divided = safe_divide_even_color(float4_to_float3(pass_result),
                                              float4_to_float3(*divide_in));
      pass_result = make_float4(divided.x, divided.y, divided.z, pass_result.w);
    }

    if (kernel_data.film.use_display_exposure) {
      float exposure = kernel_data.film.exposure;
      pass_result *= make_float4(exposure, exposure, exposure, 1.0f);
    }
  }
  else if (display_pass_components == 1) {
    ccl_global float *in = (ccl_global float *)(buffer + display_pass_stride +
                                                index * kernel_data.film.pass_stride);
    pass_result = make_float4(*in, *in, *in, 0.0f);
  }

  return pass_result;
}

/* The input buffer contains transparency = 1 - alpha, this converts it to
 * alpha. Also clamp since alpha might end up outside of 0..1 due to Russian
 * roulette. */
ccl_device float film_transparency_to_alpha(float transparency)
{
  return saturate(1.0f - transparency);
}

ccl_device float4 film_map(const KernelGlobals *kg, float4 rgba_in)
{
  float4 result;

  /* Conversion to SRGB. */
  result.x = color_linear_to_srgb(rgba_in.x);
  result.y = color_linear_to_srgb(rgba_in.y);
  result.z = color_linear_to_srgb(rgba_in.z);
  result.w = film_transparency_to_alpha(rgba_in.w);

  return result;
}

ccl_device uchar4 film_float_to_byte(float4 color)
{
  uchar4 result;

  /* simple float to byte conversion */
  result.x = (uchar)(saturate(color.x) * 255.0f);
  result.y = (uchar)(saturate(color.y) * 255.0f);
  result.z = (uchar)(saturate(color.z) * 255.0f);
  result.w = (uchar)(saturate(color.w) * 255.0f);

  return result;
}

ccl_device void kernel_film_convert_to_byte(const KernelGlobals *kg,
                                            ccl_global uchar4 *rgba,
                                            ccl_global float *buffer,
                                            float sample_scale,
                                            int x,
                                            int y,
                                            int offset,
                                            int stride)
{
  /* buffer offset */
  int index = offset + x + y * stride;

  float4 rgba_in = film_get_pass_result(kg, buffer, index);
  if (kernel_data.film.display_divide_pass_stride == -1) {
    rgba_in *= sample_scale;
  }

  /* map colors */
  float4 float_result = film_map(kg, rgba_in);
  uchar4 uchar_result = film_float_to_byte(float_result);

  rgba += index;
  *rgba = uchar_result;
}

ccl_device void kernel_film_convert_to_half_float(const KernelGlobals *kg,
                                                  ccl_global uchar4 *rgba,
                                                  ccl_global float *buffer,
                                                  float sample_scale,
                                                  int x,
                                                  int y,
                                                  int offset,
                                                  int stride)
{
  /* buffer offset */
  int index = offset + x + y * stride;

  float4 rgba_in = film_get_pass_result(kg, buffer, index);
  if (kernel_data.film.display_divide_pass_stride == -1) {
    rgba_in *= sample_scale;
  }

  rgba_in.w = film_transparency_to_alpha(rgba_in.w);

  ccl_global half *out = (ccl_global half *)rgba + index * 4;
  float4_store_half(out, rgba_in);
}

CCL_NAMESPACE_END
