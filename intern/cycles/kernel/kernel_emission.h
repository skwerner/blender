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

#include "kernel/kernel_light.h"
#include "kernel/kernel_montecarlo.h"
#include "kernel/kernel_path_state.h"
#include "kernel/kernel_shader.h"

CCL_NAMESPACE_BEGIN

/* Evaluate shader on light. */
ccl_device_noinline_cpu float3 light_sample_shader_eval(INTEGRATOR_STATE_ARGS,
                                                        ShaderData *emission_sd,
                                                        LightSample *ls,
                                                        float time)
{
  /* setup shading at emitter */
  float3 eval = zero_float3();

  if (shader_constant_emission_eval(kg, ls->shader, &eval)) {
    if ((ls->prim != PRIM_NONE) && dot(ls->Ng, ls->D) > 0.0f) {
      ls->Ng = -ls->Ng;
    }
  }
  else {
    /* Setup shader data and call shader_eval_surface once, better
     * for GPU coherence and compile times. */
#ifdef __BACKGROUND_MIS__
    if (ls->type == LIGHT_BACKGROUND) {
      shader_setup_from_background(kg, emission_sd, ls->P, ls->D, time);
    }
    else
#endif
    {
      shader_setup_from_sample(kg,
                               emission_sd,
                               ls->P,
                               ls->Ng,
                               -ls->D,
                               ls->shader,
                               ls->object,
                               ls->prim,
                               ls->u,
                               ls->v,
                               ls->t,
                               time,
                               false,
                               ls->lamp);

      ls->Ng = emission_sd->Ng;
    }

    /* No proper path flag, we're evaluating this for all closures. that's
     * weak but we'd have to do multiple evaluations otherwise. */
    shader_eval_surface<NODE_FEATURE_MASK_SURFACE_LIGHT>(
        INTEGRATOR_STATE_PASS, emission_sd, NULL, PATH_RAY_EMISSION);

    /* Evaluate closures. */
#ifdef __BACKGROUND_MIS__
    if (ls->type == LIGHT_BACKGROUND) {
      eval = shader_background_eval(emission_sd);
    }
    else
#endif
    {
      eval = shader_emissive_eval(emission_sd);
    }
  }

  eval *= ls->eval_fac;

  if (ls->lamp != LAMP_NONE) {
    const ccl_global KernelLight *klight = &kernel_tex_fetch(__lights, ls->lamp);
    eval *= make_float3(klight->strength[0], klight->strength[1], klight->strength[2]);
  }

  return eval;
}

/* Test if light sample is from a light or emission from geometry. */
ccl_device_inline bool light_sample_is_light(const LightSample *ls)
{
  /* return if it's a lamp for shadow pass */
  return (ls->prim == PRIM_NONE && ls->type != LIGHT_BACKGROUND);
}

/* Early path termination of shadow rays. */
ccl_device_inline bool light_sample_terminate(const KernelGlobals *ccl_restrict kg,
                                              const LightSample *ls,
                                              BsdfEval *eval,
                                              const float rand_terminate)
{
  /* TODO: move in volume phase evaluation. */
#if 0
#  ifdef __PASSES__
  /* use visibility flag to skip lights */
    if (ls->shader & SHADER_EXCLUDE_SCATTER)
      eval->volume = zero_float3();
#  endif
#endif

  if (bsdf_eval_is_zero(eval)) {
    return true;
  }

  if (kernel_data.integrator.light_inv_rr_threshold > 0.0f) {
    float probability = max3(fabs(bsdf_eval_sum(eval))) *
                        kernel_data.integrator.light_inv_rr_threshold;
    if (probability < 1.0f) {
      if (rand_terminate >= probability) {
        return true;
      }
      bsdf_eval_mul(eval, 1.0f / probability);
    }
  }

  return false;
}

/* Create shadow ray towards light sample. */
ccl_device_inline void light_sample_to_shadow_ray(const ShaderData *sd,
                                                  const LightSample *ls,
                                                  Ray *ray)
{
  if (ls->shader & SHADER_CAST_SHADOW) {
    /* setup ray */
    bool transmit = (dot(sd->Ng, ls->D) < 0.0f);
    ray->P = ray_offset(sd->P, (transmit) ? -sd->Ng : sd->Ng);

    if (ls->t == FLT_MAX) {
      /* distant light */
      ray->D = ls->D;
      ray->t = ls->t;
    }
    else {
      /* other lights, avoid self-intersection */
      ray->D = ray_offset(ls->P, ls->Ng) - ray->P;
      ray->D = normalize_len(ray->D, &ray->t);
    }

    ray->dP = differential_make_compact(sd->dP);
    ray->dD = differential_zero_compact();
  }
  else {
    /* signal to not cast shadow ray */
    ray->t = 0.0f;
  }

  ray->time = sd->time;
}

/* Volume phase evaluation code - to be moved into volume code. */
#if 0
#  ifdef __VOLUME__
    float bsdf_pdf;
    shader_volume_phase_eval(kg, sd, ls->D, eval, &bsdf_pdf);
    if (ls->shader & SHADER_USE_MIS) {
      /* Multiple importance sampling. */
      float mis_weight = power_heuristic(ls->pdf, bsdf_pdf);
      light_eval *= mis_weight;
    }
  }
#  endif
#endif

CCL_NAMESPACE_END
