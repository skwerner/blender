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

#include "kernel/kernel_accumulate.h"
#include "kernel/kernel_light.h"
#include "kernel/kernel_path_state.h"
#include "kernel/kernel_shader.h"

CCL_NAMESPACE_BEGIN

/* TODO: this should move to its own kernel. */
#if 0
#  ifdef __SHADOW_TRICKS__
ccl_device_inline bool integrate_surface_shadow_catcher(INTEGRATOR_STATE_CONST_ARGS)
{
  if ((sd->object_flag & SD_OBJECT_SHADOW_CATCHER)) {
    if (state->flag & PATH_RAY_TRANSPARENT_BACKGROUND) {
      state->flag |= (PATH_RAY_SHADOW_CATCHER | PATH_RAY_STORE_SHADOW_INFO);

      float3 bg = make_float3(0.0f, 0.0f, 0.0f);
      if (!kernel_data.background.transparent) {
        bg = indirect_background(kg, emission_sd, state, NULL, ray);
      }
      path_radiance_accum_shadowcatcher(L, throughput, bg);
    }
  }
  else if (state->flag & PATH_RAY_SHADOW_CATCHER) {
    /* Only update transparency after shadow catcher bounce. */
    L->shadow_transparency *= average(shader_bsdf_transparency(kg, sd));
  }
  return true;
}
#  endif /* __SHADOW_TRICKS__ */
#endif

#ifdef __HOLDOUT__
ccl_device_inline bool integrate_surface_holdout(INTEGRATOR_STATE_CONST_ARGS,
                                                 ShaderData *sd,
                                                 ccl_global float *ccl_restrict render_buffer)
{
  /* Write holdout transparency to render buffer and stop if fully holdout. */
  const uint32_t path_flag = INTEGRATOR_STATE(path, flag);

  if (((sd->flag & SD_HOLDOUT) || (sd->object_flag & SD_OBJECT_HOLDOUT_MASK)) &&
      (path_flag & PATH_RAY_TRANSPARENT_BACKGROUND)) {
    const float3 holdout_weight = shader_holdout_apply(kg, sd);
    if (kernel_data.background.transparent) {
      const float3 throughput = INTEGRATOR_STATE(path, throughput);
      const float transparent = average(holdout_weight * throughput);
      kernel_accum_transparent(INTEGRATOR_STATE_PASS, transparent, render_buffer);
    }
    if (isequal_float3(holdout_weight, make_float3(1.0f, 1.0f, 1.0f))) {
      return false;
    }
  }

  return true;
}
#endif /* __HOLDOUT__ */

#ifdef __PASSES__
ccl_device_inline void integrate_surface_passes(INTEGRATOR_STATE_CONST_ARGS,
                                                const ShaderData *sd,
                                                ccl_global float *ccl_restrict render_buffer)

{
/* TODO */
#  if 0
  kernel_write_data_passes(kg, buffer, L, sd, state, throughput);
#  endif
}
#endif /* __PASSES__ */

#ifdef __EMISSION__
ccl_device_inline void integrate_surface_emission(INTEGRATOR_STATE_CONST_ARGS,
                                                  const ShaderData *sd,
                                                  ccl_global float *ccl_restrict render_buffer)
{
  const uint32_t path_flag = INTEGRATOR_STATE(path, flag);

  /* Evaluate emissive closure. */
  float3 L = shader_emissive_eval(sd);

#  ifdef __HAIR__
  if (!(path_flag & PATH_RAY_MIS_SKIP) && (sd->flag & SD_USE_MIS) &&
      (sd->type & PRIMITIVE_ALL_TRIANGLE))
#  else
  if (!(path_flag & PATH_RAY_MIS_SKIP) && (sd->flag & SD_USE_MIS))
#  endif
  {
    const float bsdf_pdf = INTEGRATOR_STATE(path, ray_pdf);
    const float t = sd->ray_length;

    /* Multiple importance sampling, get triangle light pdf,
     * and compute weight with respect to BSDF pdf. */
    float pdf = triangle_light_pdf(kg, sd, t);
    float mis_weight = power_heuristic(bsdf_pdf, pdf);

    L *= mis_weight;
  }

  kernel_accum_emission(INTEGRATOR_STATE_PASS, L, render_buffer);
}
#endif /* __EMISSION__ */

/* Path tracing: bounce off or through surface with new direction. */
ccl_device bool integrate_surface_bounce(INTEGRATOR_STATE_ARGS,
                                         ShaderData *sd,
                                         const RNGState *rng_state)
{
  /* no BSDF? we can stop here */
  if (sd->flag & SD_BSDF) {
    /* sample BSDF */
    float bsdf_pdf;
    BsdfEval bsdf_eval ccl_optional_struct_init;
    float3 bsdf_omega_in ccl_optional_struct_init;
    differential3 bsdf_domega_in ccl_optional_struct_init;
    float bsdf_u, bsdf_v;
    path_state_rng_2D(kg, rng_state, PRNG_BSDF_U, &bsdf_u, &bsdf_v);
    int label;

    label = shader_bsdf_sample(
        kg, sd, bsdf_u, bsdf_v, &bsdf_eval, &bsdf_omega_in, &bsdf_domega_in, &bsdf_pdf);

    if (bsdf_pdf == 0.0f || bsdf_eval_is_zero(&bsdf_eval)) {
      return false;
    }

    /* Setup ray. Note that clipping works through transparent bounces. */
    INTEGRATOR_STATE_WRITE(ray, P) = ray_offset(sd->P,
                                                (label & LABEL_TRANSMIT) ? -sd->Ng : sd->Ng);
    INTEGRATOR_STATE_WRITE(ray, D) = normalize(bsdf_omega_in);
    INTEGRATOR_STATE_WRITE(ray, t) = (INTEGRATOR_STATE(path, bounce) == 0) ?
                                         INTEGRATOR_STATE(ray, t) - sd->ray_length :
                                         FLT_MAX;

/* TODO */
#if 0
#  ifdef __RAY_DIFFERENTIALS__
    ray->dP = sd->dP;
    ray->dD = bsdf_domega_in;
#  endif
#endif

    /* Update throughput. */
    float3 throughput = INTEGRATOR_STATE(path, throughput);
    /* TODO */
#if 0
    path_radiance_bsdf_bounce(kg, L_state, throughput, &bsdf_eval, bsdf_pdf, state->bounce, label);
#else
    throughput *= bsdf_eval.diffuse / bsdf_pdf;
#endif
    INTEGRATOR_STATE_WRITE(path, throughput) = throughput;

    /* Update path state */
    if (!(label & LABEL_TRANSPARENT)) {
      INTEGRATOR_STATE_WRITE(path, ray_pdf) = bsdf_pdf;
      INTEGRATOR_STATE_WRITE(path, min_ray_pdf) = fminf(bsdf_pdf,
                                                        INTEGRATOR_STATE(path, min_ray_pdf));
      /* TODO */
#if 0
#  ifdef __LAMP_MIS__
      INTEGRATOR_STATE_WTITE(path, ray_t) = 0.0f;
#  endif
#endif
    }

    path_state_next(INTEGRATOR_STATE_PASS, label);

    /* TODO */
#if 0
#  ifdef __VOLUME__
    /* enter/exit volume */
    if (label & LABEL_TRANSMIT)
      kernel_volume_stack_enter_exit(kg, sd, state->volume_stack);
#  endif
#endif
    return true;
  }
#ifdef __VOLUME__
  else if (sd->flag & SD_HAS_ONLY_VOLUME) {
    if (!path_state_volume_next(INTEGRATOR_STATE_PASS)) {
      return false;
    }

    /* Setup ray position, direction stays unchanged. */
    INTEGRATOR_STATE_WRITE(ray, P) = ray_offset(sd->P, -sd->Ng);

    /* Clipping works through transparent. */
    INTEGRATOR_STATE_WRITE(ray, t) = (INTEGRATOR_STATE(path, bounce) == 0) ?
                                         INTEGRATOR_STATE(ray, t) - sd->ray_length :
                                         FLT_MAX;

    /* TODO */
#  if 0
#    ifdef __RAY_DIFFERENTIALS__
    ray->dP = sd->dP;
#    endif
#  endif

    /* TODO */
#  if 0
    /* enter/exit volume */
    kernel_volume_stack_enter_exit(kg, sd, state->volume_stack);
#  endif
    return true;
  }
#endif
  else {
    /* no bsdf or volume? */
    return false;
  }
}

ccl_device_inline bool integrate_surface(INTEGRATOR_STATE_ARGS,
                                         ccl_global float *ccl_restrict render_buffer)

{
  if (path_state_ao_bounce(INTEGRATOR_STATE_PASS)) {
    return false;
  }

  /* Setup shader data. */
  ShaderData sd;
  shader_setup_from_ray(INTEGRATOR_STATE_PASS, &sd);

  /* Skip most work for volume bounding surface. */
#ifdef __VOLUME__
  if (sd.flag & SD_HAS_ONLY_VOLUME) {
    return false;
  }
#endif

  /* Evaluate shader. */
  shader_eval_surface(INTEGRATOR_STATE_PASS, &sd, render_buffer, INTEGRATOR_STATE(path, flag));
  shader_prepare_closures(INTEGRATOR_STATE_PASS, &sd);

#ifdef __HOLDOUT__
  /* Evaluate holdout. */
  if (!integrate_surface_holdout(INTEGRATOR_STATE_PASS, &sd, render_buffer)) {
    return false;
  }
#endif

#ifdef __PASSES__
  /* Write render passes. */
  integrate_surface_passes(INTEGRATOR_STATE_PASS, &sd, render_buffer);
#endif

#ifdef __EMISSION__
  /* Write emission. */
  if (sd.flag & SD_EMISSION) {
    integrate_surface_emission(INTEGRATOR_STATE_PASS, &sd, render_buffer);
  }
#endif

  /* Load random number state. */
  RNGState rng_state;
  path_state_rng_load(INTEGRATOR_STATE_PASS, &rng_state);

  /* Path termination. this is a strange place to put the termination, it's
   * mainly due to the mixed in MIS that we use. gives too many unneeded
   * shader evaluations, only need emission if we are going to terminate. */
  const float probability = path_state_continuation_probability(INTEGRATOR_STATE_PASS);

  if (probability == 0.0f) {
    return false;
  }
  else if (probability != 1.0f) {
    const float terminate = path_state_rng_1D(kg, &rng_state, PRNG_TERMINATE);

    if (terminate >= probability) {
      return false;
    }

    INTEGRATOR_STATE_WRITE(path, throughput) /= probability;
  }

  /* TODO */
#if 0
#  ifdef __DENOISING_FEATURES__
  kernel_update_denoising_features(kg, &sd, state, L);
#  endif

#  ifdef __AO__
  /* ambient occlusion */
  if (kernel_data.integrator.use_ambient_occlusion) {
    kernel_path_ao(kg, &sd, emission_sd, L, state, throughput, shader_bsdf_alpha(kg, &sd));
  }
#  endif /* __AO__ */

#  ifdef __EMISSION__
  /* direct lighting */
  kernel_path_surface_connect_light(kg, &sd, emission_sd, throughput, state, L);
#  endif /* __EMISSION__ */
#endif

#if 0
  /* Direct lighting. */
  const float3 throughput = INTEGRATOR_STATE(path, throughput);
  const bool direct_lighting = false;
  if (direct_lighting) {
    /* Generate shadow ray. */
    INTEGRATOR_STATE_WRITE(shadow_ray, P) = make_float3(0.0f, 0.0f, 0.0f);
    INTEGRATOR_STATE_WRITE(shadow_ray, D) = make_float3(0.0f, 0.0f, 1.0f);
    INTEGRATOR_STATE_WRITE(shadow_ray, t) = FLT_MAX;
    INTEGRATOR_STATE_WRITE(shadow_ray, time) = 0.0f;

    /* Copy entire path state. */
    INTEGRATOR_STATE_COPY(shadow_path, path);
    INTEGRATOR_STATE_COPY(shadow_volume_stack, volume_stack);

    /* Branch of shadow kernel. */
    INTEGRATOR_SHADOW_PATH_NEXT(intersect_shadow);
  }
#endif

#if 0
  /* Subsurface scattering does scattering, direct and indirect light in own kernel. */
  const bool subsurface = false;
  if (subsurface) {
    INTEGRATOR_STATE_WRITE(path, flag) |= PATH_RAY_SUBSURFACE;
    INTEGRATOR_STATE_WRITE(subsurface, albedo) = make_float3(1.0f, 1.0f, 1.0f);
    INTEGRATOR_PATH_NEXT(subsurface);
    return;
  }
#endif

  return integrate_surface_bounce(INTEGRATOR_STATE_PASS, &sd, &rng_state);
}

ccl_device void kernel_integrate_surface(INTEGRATOR_STATE_ARGS,
                                         ccl_global float *ccl_restrict render_buffer)
{
  /* Only execute if path is active and intersection was found. */
  if (INTEGRATOR_PATH_IS_TERMINATED || INTEGRATOR_STATE(isect, prim) == PRIM_NONE) {
    return;
  }

  if (integrate_surface(INTEGRATOR_STATE_PASS, render_buffer)) {
    INTEGRATOR_PATH_NEXT(intersect_closest);
  }
  else {
    INTEGRATOR_PATH_TERMINATE;
  }
}

CCL_NAMESPACE_END
