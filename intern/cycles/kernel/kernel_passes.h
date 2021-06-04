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

#include "kernel/geom/geom.h"

#include "kernel/kernel_id_passes.h"
#include "kernel/kernel_write_passes.h"

CCL_NAMESPACE_BEGIN

/* Get pointer to pixel in render buffer. */
ccl_device_forceinline ccl_global float *kernel_pass_pixel_render_buffer(
    INTEGRATOR_STATE_CONST_ARGS, ccl_global float *ccl_restrict render_buffer)
{
  const uint32_t render_pixel_index = INTEGRATOR_STATE(path, render_pixel_index);
  const uint64_t render_buffer_offset = (uint64_t)render_pixel_index *
                                        kernel_data.film.pass_stride;
  return render_buffer + render_buffer_offset;
}

#ifdef __DENOISING_FEATURES__

ccl_device_inline void kernel_write_denoising_features(
    INTEGRATOR_STATE_ARGS, const ShaderData *sd, ccl_global float *ccl_restrict render_buffer)
{
  if (!(INTEGRATOR_STATE(path, flag) & PATH_RAY_DENOISING_FEATURES)) {
    return;
  }

  /* Skip implicitly transparent surfaces. */
  if (sd->flag & SD_HAS_ONLY_VOLUME) {
    return;
  }

  ccl_global float *buffer = kernel_pass_pixel_render_buffer(INTEGRATOR_STATE_PASS, render_buffer);

  float3 normal = zero_float3();
  float3 diffuse_albedo = zero_float3();
  float3 specular_albedo = zero_float3();
  float sum_weight = 0.0f, sum_nonspecular_weight = 0.0f;

  for (int i = 0; i < sd->num_closure; i++) {
    const ShaderClosure *sc = &sd->closure[i];

    if (!CLOSURE_IS_BSDF_OR_BSSRDF(sc->type)) {
      continue;
    }

    /* All closures contribute to the normal feature, but only diffuse-like ones to the albedo. */
    normal += sc->N * sc->sample_weight;
    sum_weight += sc->sample_weight;

    float3 closure_albedo = sc->weight;
    /* Closures that include a Fresnel term typically have weights close to 1 even though their
     * actual contribution is significantly lower.
     * To account for this, we scale their weight by the average fresnel factor (the same is also
     * done for the sample weight in the BSDF setup, so we don't need to scale that here). */
    if (CLOSURE_IS_BSDF_MICROFACET_FRESNEL(sc->type)) {
      MicrofacetBsdf *bsdf = (MicrofacetBsdf *)sc;
      closure_albedo *= bsdf->extra->fresnel_color;
    }
    else if (sc->type == CLOSURE_BSDF_PRINCIPLED_SHEEN_ID) {
      PrincipledSheenBsdf *bsdf = (PrincipledSheenBsdf *)sc;
      closure_albedo *= bsdf->avg_value;
    }
    else if (sc->type == CLOSURE_BSDF_HAIR_PRINCIPLED_ID) {
      closure_albedo *= bsdf_principled_hair_albedo(sc);
    }

    if (bsdf_get_specular_roughness_squared(sc) > sqr(0.075f)) {
      diffuse_albedo += closure_albedo;
      sum_nonspecular_weight += sc->sample_weight;
    }
    else {
      specular_albedo += closure_albedo;
    }
  }

  /* Wait for next bounce if 75% or more sample weight belongs to specular-like closures. */
  if ((sum_weight == 0.0f) || (sum_nonspecular_weight * 4.0f > sum_weight)) {
    if (sum_weight != 0.0f) {
      normal /= sum_weight;
    }

    if (kernel_data.film.pass_denoising_normal != PASS_UNUSED) {
      /* Transform normal into camera space. */
      const Transform worldtocamera = kernel_data.cam.worldtocamera;
      normal = transform_direction(&worldtocamera, normal);

      const float3 denoising_normal = ensure_finite3(normal);
      kernel_write_pass_float3_unaligned(buffer + kernel_data.film.pass_denoising_normal,
                                         denoising_normal);
    }

    if (kernel_data.film.pass_denoising_albedo != PASS_UNUSED) {
      const float3 denoising_feature_throughput = INTEGRATOR_STATE(path,
                                                                   denoising_feature_throughput);
      const float3 denoising_albedo = ensure_finite3(denoising_feature_throughput *
                                                     diffuse_albedo);
      kernel_write_pass_float3_unaligned(buffer + kernel_data.film.pass_denoising_albedo,
                                         denoising_albedo);
    }

    INTEGRATOR_STATE_WRITE(path, flag) &= ~PATH_RAY_DENOISING_FEATURES;
  }
  else {
    INTEGRATOR_STATE_WRITE(path, denoising_feature_throughput) *= specular_albedo;
  }
}
#endif /* __DENOISING_FEATURES__ */

#ifdef __SHADOW_CATCHER__

/* Write transparency to the matte pass at a bounce off the shadow catcher object (this is where
 * the path split happens). */
ccl_device_inline void kernel_write_shadow_catcher_bounce_data(
    INTEGRATOR_STATE_ARGS, const ShaderData *sd, ccl_global float *ccl_restrict render_buffer)
{
  if (kernel_data.film.pass_shadow_catcher_matte == PASS_UNUSED) {
    return;
  }

  if (!kernel_shadow_catcher_is_path_split_bounce(INTEGRATOR_STATE_PASS, sd->object_flag)) {
    return;
  }

  ccl_global float *buffer = kernel_pass_pixel_render_buffer(INTEGRATOR_STATE_PASS, render_buffer);

  /* TODO(sergey): Use contribution and transparency based on the throughput, allowing to have
   * transparent object between camera and shadow catcher. */

  kernel_write_pass_float(buffer + kernel_data.film.pass_shadow_catcher_matte + 3, 1.0f);
}

#endif /* __SHADOW_CATCHER__ */

#ifdef __KERNEL_CPU__
#  define WRITE_ID_SLOT(buffer, depth, id, matte_weight, name) \
    kernel_write_id_pass_cpu(buffer, depth * 2, id, matte_weight, kg->coverage_##name)
ccl_device_inline size_t kernel_write_id_pass_cpu(
    float *ccl_restrict buffer, size_t depth, float id, float matte_weight, CoverageMap *map)
{
  if (map) {
    (*map)[id] += matte_weight;
    return 0;
  }
#else /* __KERNEL_CPU__ */
#  define WRITE_ID_SLOT(buffer, depth, id, matte_weight, name) \
    kernel_write_id_slots_gpu(buffer, depth * 2, id, matte_weight)
ccl_device_inline size_t kernel_write_id_slots_gpu(ccl_global float *ccl_restrict buffer,
                                                   size_t depth,
                                                   float id,
                                                   float matte_weight)
{
#endif /* __KERNEL_CPU__ */
  kernel_write_id_slots(buffer, depth, id, matte_weight);
  return depth * 2;
}

ccl_device_inline void kernel_write_data_passes(INTEGRATOR_STATE_ARGS,
                                                const ShaderData *sd,
                                                ccl_global float *ccl_restrict render_buffer)
{
#ifdef __PASSES__
  const int path_flag = INTEGRATOR_STATE(path, flag);

  if (!(path_flag & PATH_RAY_CAMERA)) {
    return;
  }

  const int flag = kernel_data.film.pass_flag;

  if (!(flag & PASS_ANY)) {
    return;
  }

  ccl_global float *buffer = kernel_pass_pixel_render_buffer(INTEGRATOR_STATE_PASS, render_buffer);

  if (!(path_flag & PATH_RAY_SINGLE_PASS_DONE)) {
    if (!(sd->flag & SD_TRANSPARENT) || kernel_data.film.pass_alpha_threshold == 0.0f ||
        average(shader_bsdf_alpha(kg, sd)) >= kernel_data.film.pass_alpha_threshold) {
      if (INTEGRATOR_STATE(path, sample) == 0) {
        if (flag & PASSMASK(DEPTH)) {
          const float depth = camera_z_depth(kg, sd->P);
          kernel_write_pass_float(buffer + kernel_data.film.pass_depth, depth);
        }
        if (flag & PASSMASK(OBJECT_ID)) {
          const float id = object_pass_id(kg, sd->object);
          kernel_write_pass_float(buffer + kernel_data.film.pass_object_id, id);
        }
        if (flag & PASSMASK(MATERIAL_ID)) {
          const float id = shader_pass_id(kg, sd);
          kernel_write_pass_float(buffer + kernel_data.film.pass_material_id, id);
        }
      }

      if (flag & PASSMASK(NORMAL)) {
        const float3 normal = shader_bsdf_average_normal(kg, sd);
        kernel_write_pass_float3(buffer + kernel_data.film.pass_normal, normal);
      }
      if (flag & PASSMASK(ROUGHNESS)) {
        const float roughness = shader_bsdf_average_roughness(sd);
        kernel_write_pass_float(buffer + kernel_data.film.pass_roughness, roughness);
      }
      if (flag & PASSMASK(UV)) {
        const float3 uv = primitive_uv(kg, sd);
        kernel_write_pass_float3(buffer + kernel_data.film.pass_uv, uv);
      }
      if (flag & PASSMASK(MOTION)) {
        const float4 speed = primitive_motion_vector(kg, sd);
        kernel_write_pass_float4(buffer + kernel_data.film.pass_motion, speed);
        kernel_write_pass_float(buffer + kernel_data.film.pass_motion_weight, 1.0f);
      }

      INTEGRATOR_STATE_WRITE(path, flag) |= PATH_RAY_SINGLE_PASS_DONE;
    }
  }

  if (kernel_data.film.cryptomatte_passes) {
    const float3 throughput = INTEGRATOR_STATE(path, throughput);
    const float matte_weight = average(throughput) *
                               (1.0f - average(shader_bsdf_transparency(kg, sd)));
    if (matte_weight > 0.0f) {
      ccl_global float *cryptomatte_buffer = buffer + kernel_data.film.pass_cryptomatte;
      if (kernel_data.film.cryptomatte_passes & CRYPT_OBJECT) {
        const float id = object_cryptomatte_id(kg, sd->object);
        cryptomatte_buffer += WRITE_ID_SLOT(
            cryptomatte_buffer, kernel_data.film.cryptomatte_depth, id, matte_weight, object);
      }
      if (kernel_data.film.cryptomatte_passes & CRYPT_MATERIAL) {
        const float id = shader_cryptomatte_id(kg, sd->shader);
        cryptomatte_buffer += WRITE_ID_SLOT(
            cryptomatte_buffer, kernel_data.film.cryptomatte_depth, id, matte_weight, material);
      }
      if (kernel_data.film.cryptomatte_passes & CRYPT_ASSET) {
        const float id = object_cryptomatte_asset_id(kg, sd->object);
        cryptomatte_buffer += WRITE_ID_SLOT(
            cryptomatte_buffer, kernel_data.film.cryptomatte_depth, id, matte_weight, asset);
      }
    }
  }

  if (flag & PASSMASK(DIFFUSE_COLOR)) {
    const float3 throughput = INTEGRATOR_STATE(path, throughput);
    kernel_write_pass_float3(buffer + kernel_data.film.pass_diffuse_color,
                             shader_bsdf_diffuse(kg, sd) * throughput);
  }
  if (flag & PASSMASK(GLOSSY_COLOR)) {
    const float3 throughput = INTEGRATOR_STATE(path, throughput);
    kernel_write_pass_float3(buffer + kernel_data.film.pass_glossy_color,
                             shader_bsdf_glossy(kg, sd) * throughput);
  }
  if (flag & PASSMASK(TRANSMISSION_COLOR)) {
    const float3 throughput = INTEGRATOR_STATE(path, throughput);
    kernel_write_pass_float3(buffer + kernel_data.film.pass_transmission_color,
                             shader_bsdf_transmission(kg, sd) * throughput);
  }
  if (flag & PASSMASK(MIST)) {
    /* Bring depth into 0..1 range. */
    const float mist_start = kernel_data.film.mist_start;
    const float mist_inv_depth = kernel_data.film.mist_inv_depth;

    const float depth = camera_distance(kg, sd->P);
    float mist = saturate((depth - mist_start) * mist_inv_depth);

    /* Falloff */
    const float mist_falloff = kernel_data.film.mist_falloff;

    if (mist_falloff == 1.0f)
      ;
    else if (mist_falloff == 2.0f)
      mist = mist * mist;
    else if (mist_falloff == 0.5f)
      mist = sqrtf(mist);
    else
      mist = powf(mist, mist_falloff);

    /* Modulate by transparency */
    const float3 throughput = INTEGRATOR_STATE(path, throughput);
    const float3 alpha = shader_bsdf_alpha(kg, sd);
    const float mist_output = (1.0f - mist) * average(throughput * alpha);

    /* Note that the final value in the render buffer we want is 1 - mist_output,
     * to avoid having to tracking this in the Integrator state we do the negation
     * after rendering. */
    kernel_write_pass_float(buffer + kernel_data.film.pass_mist, mist_output);
  }
#endif
}

#if 0
ccl_device_inline void kernel_write_light_passes(const KernelGlobals *ccl_restrict kg,
                                                 ccl_global float *ccl_restrict buffer,
                                                 PathRadiance *L)
{
#  ifdef __PASSES__
  int light_flag = kernel_data.film.light_pass_flag;

  if (!kernel_data.film.use_light_pass)
    return;

  if (light_flag & PASSMASK(AO))
    kernel_write_pass_float3(buffer + kernel_data.film.pass_ao, L->ao);
#  endif
}
#endif

#if 0
ccl_device_inline void kernel_write_result(const KernelGlobals *ccl_restrict kg,
                                           ccl_global float *ccl_restrict buffer,
                                           int sample,
                                           PathRadiance *L)
{
  PROFILING_INIT(kg, PROFILING_WRITE_RESULT);
  PROFILING_OBJECT(PRIM_NONE);

  float alpha;
  float3 L_sum = path_radiance_clamp_and_sum(kg, L, &alpha);

  if (kernel_data.film.light_pass_flag & PASSMASK(COMBINED)) {
    kernel_write_pass_float4(buffer, make_float4(L_sum.x, L_sum.y, L_sum.z, alpha));
  }


  /* Adaptive Sampling. Fill the additional buffer with the odd samples and calculate our stopping
     criteria. This is the heuristic from "A hierarchical automatic stopping condition for Monte
     Carlo global illumination" except that here it is applied per pixel and not in hierarchical
     tiles. */
  if (kernel_data.film.pass_adaptive_aux_buffer != PASS_UNUSED) {
    if (sample_is_even(kernel_data.integrator.sampling_pattern, sample)) {
      kernel_write_pass_float4(buffer + kernel_data.film.pass_adaptive_aux_buffer,
                               make_float4(L_sum.x * 2.0f, L_sum.y * 2.0f, L_sum.z * 2.0f, 0.0f));
    }
#  ifdef __KERNEL_CPU__
    if ((sample > kernel_data.integrator.adaptive_min_samples) &&
        kernel_data.integrator.adaptive_stop_per_sample) {
      const int step = kernel_data.integrator.adaptive_step;

      if ((sample & (step - 1)) == (step - 1)) {
        kernel_do_adaptive_stopping(kg, buffer, sample);
      }
    }
#  endif
  }

  /* Write the sample count as negative numbers initially to mark the samples as in progress.
   * Once the tile has finished rendering, the sign gets flipped and all the pixel values
   * are scaled as if they were taken at a uniform sample count. */
  if (kernel_data.film.pass_sample_count != PASS_UNUSED) {
    /* Make sure it's a negative number. In progressive refine mode, this bit gets flipped between
     * passes. */
#  ifdef __ATOMIC_PASS_WRITE__
    atomic_fetch_and_or_uint32((ccl_global uint *)(buffer + kernel_data.film.pass_sample_count),
                               0x80000000);
#  else
    if (buffer[kernel_data.film.pass_sample_count] > 0) {
      buffer[kernel_data.film.pass_sample_count] *= -1.0f;
    }
#  endif
    kernel_write_pass_float(buffer + kernel_data.film.pass_sample_count, -1.0f);
  }
}
#endif

CCL_NAMESPACE_END
