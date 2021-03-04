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
#include "kernel/kernel_shader.h"

CCL_NAMESPACE_BEGIN

ccl_device_noinline_cpu float3 integrator_eval_background_shader(
    INTEGRATOR_STATE_ARGS, ccl_global float *ccl_restrict render_buffer)
{
#ifdef __BACKGROUND__
  const int shader = kernel_data.background.surface_shader;
  const uint32_t path_flag = INTEGRATOR_STATE(path, flag);

  /* Use visibility flag to skip lights. */
  if (shader & SHADER_EXCLUDE_ANY) {
    if (((shader & SHADER_EXCLUDE_DIFFUSE) && (path_flag & PATH_RAY_DIFFUSE)) ||
        ((shader & SHADER_EXCLUDE_GLOSSY) && ((path_flag & (PATH_RAY_GLOSSY | PATH_RAY_REFLECT)) ==
                                              (PATH_RAY_GLOSSY | PATH_RAY_REFLECT))) ||
        ((shader & SHADER_EXCLUDE_TRANSMIT) && (path_flag & PATH_RAY_TRANSMIT)) ||
        ((shader & SHADER_EXCLUDE_CAMERA) && (path_flag & PATH_RAY_CAMERA)) ||
        ((shader & SHADER_EXCLUDE_SCATTER) && (path_flag & PATH_RAY_VOLUME_SCATTER)))
      return make_float3(0.0f, 0.0f, 0.0f);
  }

  /* Fast path for constant color shader. */
  float3 L = make_float3(0.0f, 0.0f, 0.0f);
  if (shader_constant_emission_eval(kg, shader, &L)) {
    return L;
  }

  /* Evaluate background shader. */
  {
    /* TODO: does aliasing like this break automatic SoA in CUDA?
     * Should we instead store closures separate from ShaderData? */
    ShaderDataTinyStorage emission_sd_storage;
    ShaderData *emission_sd = AS_SHADER_DATA(&emission_sd_storage);

    shader_setup_from_background(kg,
                                 emission_sd,
                                 INTEGRATOR_STATE(ray, P),
                                 INTEGRATOR_STATE(ray, D),
                                 INTEGRATOR_STATE(ray, time));
    shader_eval_surface(
        INTEGRATOR_STATE_PASS, emission_sd, render_buffer, path_flag | PATH_RAY_EMISSION);

    L = shader_background_eval(emission_sd);
  }

  /* Background MIS weights. */
#  ifdef __BACKGROUND_MIS__
  /* Check if background light exists or if we should skip pdf. */
  if (!(INTEGRATOR_STATE(path, flag) & PATH_RAY_MIS_SKIP) && kernel_data.background.use_mis) {
    const float3 ray_P = INTEGRATOR_STATE(ray, P);
    const float3 ray_D = INTEGRATOR_STATE(ray, D);
    const float ray_pdf = INTEGRATOR_STATE(path, ray_pdf);

    /* multiple importance sampling, get background light pdf for ray
     * direction, and compute weight with respect to BSDF pdf */
    const float pdf = background_light_pdf(kg, ray_P, ray_D);
    const float mis_weight = power_heuristic(ray_pdf, pdf);

    L *= mis_weight;
  }
#  endif

  return L;
#else
  return make_float3(0.8f, 0.8f, 0.8f);
#endif
}

ccl_device_inline void integrate_background(INTEGRATOR_STATE_ARGS,
                                            ccl_global float *ccl_restrict render_buffer)
{
  /* Accumulate transparency for transparent background. We can skip background
   * shader evaluation unless a background pass is used. */
  bool eval_background = true;
  float transparent = 0.0f;

  if (kernel_data.background.transparent &&
      (INTEGRATOR_STATE(path, flag) & PATH_RAY_TRANSPARENT_BACKGROUND)) {
    transparent = average(INTEGRATOR_STATE(path, throughput));

#ifdef __PASSES__
    eval_background = (kernel_data.film.light_pass_flag & PASSMASK(BACKGROUND));
#else
    eval_background = false;
#endif
  }

  /* TODO */
#if 0
  /* When using the ao bounces approximation, adjust background
   * shader intensity with ao factor. */
  if (path_state_ao_bounce(kg, state)) {
    throughput *= kernel_data.background.ao_bounces_factor;
  }
#endif

  /* Evaluate background shader. */
  const float3 L = (eval_background) ?
                       integrator_eval_background_shader(INTEGRATOR_STATE_PASS, render_buffer) :
                       make_float3(0.0f, 0.0f, 0.0f);

  /* Write to render buffer. */
  kernel_accum_background(INTEGRATOR_STATE_PASS, L, transparent, render_buffer);
}

ccl_device void integrator_shade_background(INTEGRATOR_STATE_ARGS,
                                            ccl_global float *ccl_restrict render_buffer)
{
  /* Only execute for active path and nothing hit. */
  if (INTEGRATOR_PATH_IS_TERMINATED || (INTEGRATOR_STATE(isect, prim) != PRIM_NONE)) {
    return;
  }

  integrate_background(INTEGRATOR_STATE_PASS, render_buffer);

  /* Path ends here. */
  INTEGRATOR_PATH_TERMINATE(shade_background);
}

CCL_NAMESPACE_END
