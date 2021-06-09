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

#ifdef __OSL__
#  include "kernel/osl/osl_shader.h"
#endif

// clang-format off
#include "kernel/kernel_random.h"
#include "kernel/kernel_projection.h"
#include "kernel/kernel_montecarlo.h"
#include "kernel/kernel_differential.h"
#include "kernel/kernel_camera.h"

#include "kernel/geom/geom.h"
#include "kernel/bvh/bvh.h"

#include "kernel/kernel_write_passes.h"
#include "kernel/kernel_accumulate.h"
#include "kernel/kernel_shader.h"
#include "kernel/kernel_light.h"
#include "kernel/kernel_adaptive_sampling.h"
#include "kernel/kernel_passes.h"

#if defined(__VOLUME__) || defined(__SUBSURFACE__)
#  include "kernel/kernel_volume.h"
#endif

#ifdef __SUBSURFACE__
#  include "kernel/kernel_subsurface.h"
#endif

#include "kernel/kernel_path_state.h"
#include "kernel/kernel_shadow.h"
#include "kernel/kernel_emission.h"
#include "kernel/kernel_path_common.h"
#include "kernel/kernel_path_surface.h"
#include "kernel/kernel_path_volume.h"
#include "kernel/kernel_path_subsurface.h"
// clang-format on

CCL_NAMESPACE_BEGIN

#ifdef __VOLUME__
ccl_device_forceinline VolumeIntegrateResult kernel_path_volume(const KernelGlobals *kg,
                                                                ShaderData *sd,
                                                                PathState *state,
                                                                Ray *ray,
                                                                float3 *throughput,
                                                                ccl_addr_space Intersection *isect,
                                                                bool hit,
                                                                ShaderData *emission_sd,
                                                                PathRadiance *L)
{
  PROFILING_INIT(kg, PROFILING_VOLUME);

  /* Sanitize volume stack. */
  if (!hit) {
    kernel_volume_clean_stack(kg, state->volume_stack);
  }

  if (state->volume_stack[0].shader == SHADER_NONE) {
    return VOLUME_PATH_ATTENUATED;
  }

  /* volume attenuation, emission, scatter */
  Ray volume_ray = *ray;
  volume_ray.t = (hit) ? isect->t : FLT_MAX;

  float step_size = volume_stack_step_size(kg, state->volume_stack);

#  ifdef __VOLUME_DECOUPLED__
  int sampling_method = volume_stack_sampling_method(kg, state->volume_stack);
  bool direct = (state->flag & PATH_RAY_CAMERA) != 0;
  bool decoupled = kernel_volume_use_decoupled(kg, step_size, direct, sampling_method);

  if (decoupled) {
    /* cache steps along volume for repeated sampling */
    VolumeSegment volume_segment;

    shader_setup_from_volume(kg, sd, &volume_ray);
    kernel_volume_decoupled_record(kg, state, &volume_ray, sd, &volume_segment, step_size);

    volume_segment.sampling_method = sampling_method;

    /* emission */
    if (volume_segment.closure_flag & SD_EMISSION)
      path_radiance_accum_emission(kg, L, state, *throughput, volume_segment.accum_emission);

    /* scattering */
    VolumeIntegrateResult result = VOLUME_PATH_ATTENUATED;

    if (volume_segment.closure_flag & SD_SCATTER) {
      int all = kernel_data.integrator.sample_all_lights_indirect;

      /* direct light sampling */
      kernel_branched_path_volume_connect_light(
          kg, sd, emission_sd, *throughput, state, L, all, &volume_ray, &volume_segment);

      /* indirect sample. if we use distance sampling and take just
       * one sample for direct and indirect light, we could share
       * this computation, but makes code a bit complex */
      float rphase = path_state_rng_1D(kg, state, PRNG_PHASE_CHANNEL);
      float rscatter = path_state_rng_1D(kg, state, PRNG_SCATTER_DISTANCE);

      result = kernel_volume_decoupled_scatter(
          kg, state, &volume_ray, sd, throughput, rphase, rscatter, &volume_segment, NULL, true);
    }

    /* free cached steps */
    kernel_volume_decoupled_free(kg, &volume_segment);

    if (result == VOLUME_PATH_SCATTERED) {
      if (kernel_path_volume_bounce(kg, sd, throughput, state, &L->state, ray))
        return VOLUME_PATH_SCATTERED;
      else
        return VOLUME_PATH_MISSED;
    }
    else {
      *throughput *= volume_segment.accum_transmittance;
    }
  }
  else
#  endif /* __VOLUME_DECOUPLED__ */
  {
    /* integrate along volume segment with distance sampling */
    VolumeIntegrateResult result = kernel_volume_integrate(
        kg, state, sd, &volume_ray, L, throughput, step_size);

#  ifdef __VOLUME_SCATTER__
    if (result == VOLUME_PATH_SCATTERED) {
      /* direct lighting */
      kernel_path_volume_connect_light(kg, sd, emission_sd, *throughput, state, L);

      /* indirect light bounce */
      if (kernel_path_volume_bounce(kg, sd, throughput, state, &L->state, ray))
        return VOLUME_PATH_SCATTERED;
      else
        return VOLUME_PATH_MISSED;
    }
#  endif /* __VOLUME_SCATTER__ */
  }

  return VOLUME_PATH_ATTENUATED;
}
#endif /* __VOLUME__ */

ccl_device_inline void kernel_path_ao(const KernelGlobals *kg,
                                      ShaderData *sd,
                                      ShaderData *emission_sd,
                                      PathRadiance *L,
                                      ccl_addr_space PathState *state,
                                      float3 throughput,
                                      float3 ao_alpha)
{
  PROFILING_INIT(kg, PROFILING_AO);

  /* todo: solve correlation */
  float bsdf_u, bsdf_v;

  path_state_rng_2D(kg, state, PRNG_BSDF_U, &bsdf_u, &bsdf_v);

  float ao_factor = kernel_data.background.ao_factor;
  float3 ao_N;
  float3 ao_bsdf = shader_bsdf_ao(kg, sd, ao_factor, &ao_N);
  float3 ao_D;
  float ao_pdf;

  sample_cos_hemisphere(ao_N, bsdf_u, bsdf_v, &ao_D, &ao_pdf);

  if (dot(sd->Ng, ao_D) > 0.0f && ao_pdf != 0.0f) {
    Ray light_ray;
    float3 ao_shadow;

    light_ray.P = ray_offset(sd->P, sd->Ng);
    light_ray.D = ao_D;
    light_ray.t = kernel_data.background.ao_distance;
    light_ray.time = sd->time;
    light_ray.dP = sd->dP;
    light_ray.dD = differential3_zero();

    if (!shadow_blocked(kg, sd, emission_sd, state, &light_ray, &ao_shadow)) {
      path_radiance_accum_ao(kg, L, state, throughput, ao_alpha, ao_bsdf, ao_shadow);
    }
    else {
      path_radiance_accum_total_ao(L, state, throughput, ao_bsdf);
    }
  }
}

CCL_NAMESPACE_END
