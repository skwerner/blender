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
