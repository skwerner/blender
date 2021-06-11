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

#include "kernel/integrator/integrator_intersect_closest.h"
#include "kernel/integrator/integrator_volume_stack.h"

CCL_NAMESPACE_BEGIN

#ifdef __VOLUME__

/* Events for probalistic scattering */

typedef enum VolumeIntegrateResult {
  VOLUME_PATH_SCATTERED = 0,
  VOLUME_PATH_ATTENUATED = 1,
  VOLUME_PATH_MISSED = 2
} VolumeIntegrateResult;

/* Volume shader properties
 *
 * extinction coefficient = absorption coefficient + scattering coefficient
 * sigma_t = sigma_a + sigma_s */

typedef struct VolumeShaderCoefficients {
  float3 sigma_t;
  float3 sigma_s;
  float3 emission;
} VolumeShaderCoefficients;

/* Evaluate shader to get extinction coefficient at P. */
ccl_device_inline bool shadow_volume_shader_sample(INTEGRATOR_STATE_ARGS,
                                                   ShaderData *ccl_restrict sd,
                                                   float3 *ccl_restrict extinction)
{
  shader_eval_volume(INTEGRATOR_STATE_PASS, sd, PATH_RAY_SHADOW, [=](const int i) {
    return integrator_state_read_shadow_volume_stack(INTEGRATOR_STATE_PASS, i);
  });

  if (!(sd->flag & SD_EXTINCTION)) {
    return false;
  }

  const float density = object_volume_density(kg, sd->object);
  *extinction = sd->closure_transparent_extinction * density;
  return true;
}

/* Evaluate shader to get absorption, scattering and emission at P. */
ccl_device_inline bool volume_shader_sample(INTEGRATOR_STATE_ARGS,
                                            ShaderData *ccl_restrict sd,
                                            VolumeShaderCoefficients *coeff)
{
  const int path_flag = INTEGRATOR_STATE(path, flag);
  shader_eval_volume(INTEGRATOR_STATE_PASS, sd, path_flag, [=](const int i) {
    return integrator_state_read_volume_stack(INTEGRATOR_STATE_PASS, i);
  });

  if (!(sd->flag & (SD_EXTINCTION | SD_SCATTER | SD_EMISSION))) {
    return false;
  }

  coeff->sigma_s = zero_float3();
  coeff->sigma_t = (sd->flag & SD_EXTINCTION) ? sd->closure_transparent_extinction : zero_float3();
  coeff->emission = (sd->flag & SD_EMISSION) ? sd->closure_emission_background : zero_float3();

  if (sd->flag & SD_SCATTER) {
    for (int i = 0; i < sd->num_closure; i++) {
      const ShaderClosure *sc = &sd->closure[i];

      if (CLOSURE_IS_VOLUME(sc->type))
        coeff->sigma_s += sc->weight;
    }
  }

  const float density = object_volume_density(kg, sd->object);
  coeff->sigma_s *= density;
  coeff->sigma_t *= density;
  coeff->emission *= density;

  return true;
}
#endif

ccl_device void integrator_shade_volume(INTEGRATOR_STATE_ARGS,
                                        ccl_global float *ccl_restrict render_buffer)
{
#ifdef __VOLUME__
  VolumeIntegrateResult result = VOLUME_PATH_ATTENUATED;

  /* Setup shader data. */
  Ray ray ccl_optional_struct_init;
  integrator_state_read_ray(INTEGRATOR_STATE_PASS, &ray);

  Intersection isect ccl_optional_struct_init;
  integrator_state_read_isect(INTEGRATOR_STATE_PASS, &isect);

  ShaderData sd;
  shader_setup_from_volume(kg, &sd, &ray);

  /* Clean volume stack for background rays. */
  if (isect.prim == PRIM_NONE) {
    volume_stack_clean(INTEGRATOR_STATE_PASS);
  }

  /* Evaluate shader. */
  /* TODO: implement scattering and heterogeneous media. */
  VolumeShaderCoefficients coeff ccl_optional_struct_init;
  if (volume_shader_sample(INTEGRATOR_STATE_PASS, &sd, &coeff)) {
    /* Integrate extinction over segment. */
    float3 throughput = INTEGRATOR_STATE(path, throughput);
    throughput *= exp3(-coeff.sigma_t * isect.t);
    INTEGRATOR_STATE_WRITE(path, throughput) = throughput;
  }

  if (result == VOLUME_PATH_MISSED) {
    /* End path. */
    INTEGRATOR_PATH_TERMINATE(DEVICE_KERNEL_INTEGRATOR_SHADE_VOLUME);
    return;
  }
  else if (result == VOLUME_PATH_SCATTERED) {
    /* TODO: handle path termination like intersect closest. */

    /* Queue intersect_closest kernel. */
    INTEGRATOR_PATH_NEXT(DEVICE_KERNEL_INTEGRATOR_SHADE_VOLUME,
                         DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST);
    return;
  }
  else {
    /* Continue to background, light or surface. */
    if (isect.prim == PRIM_NONE) {
      INTEGRATOR_PATH_NEXT(DEVICE_KERNEL_INTEGRATOR_SHADE_VOLUME,
                           DEVICE_KERNEL_INTEGRATOR_SHADE_BACKGROUND);
      return;
    }
    else if (isect.type & PRIMITIVE_LAMP) {
      INTEGRATOR_PATH_NEXT(DEVICE_KERNEL_INTEGRATOR_SHADE_VOLUME,
                           DEVICE_KERNEL_INTEGRATOR_SHADE_LIGHT);
      return;
    }
    else {
      /* Hit a surface, continue with surface kernel unless terminated. */
      if (integrator_intersect_shader_next_kernel<DEVICE_KERNEL_INTEGRATOR_SHADE_VOLUME>(
              INTEGRATOR_STATE_PASS, &isect)) {
        return;
      }
      else {
        INTEGRATOR_PATH_TERMINATE(DEVICE_KERNEL_INTEGRATOR_SHADE_VOLUME);
        return;
      }
    }
  }
#endif /* __VOLUME__ */
}

CCL_NAMESPACE_END
