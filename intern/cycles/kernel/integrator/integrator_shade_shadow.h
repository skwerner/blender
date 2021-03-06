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

CCL_NAMESPACE_BEGIN

#ifdef __TRANSPARENT_SHADOWS__
ccl_device_inline float3 integrate_transparent_shadow_shader_eval(INTEGRATOR_STATE_ARGS,
                                                                  const int hit)
{
  /* TODO: does aliasing like this break automatic SoA in CUDA?
   * Should we instead store closures separate from ShaderData?
   *
   * TODO: is it better to declare this outside the loop or keep it local
   * so the compiler can see there is no dependency between iterations? */
  ShaderDataTinyStorage shadow_sd_storage;
  ShaderData *shadow_sd = AS_SHADER_DATA(&shadow_sd_storage);

  /* Setup shader data at surface.
   *
   * TODO: old logic would modify ray.P and isect.t for each step, why was that
   * needed? Can we avoid it? */
  Intersection isect ccl_optional_struct_init;
  isect.prim = INTEGRATOR_STATE_ARRAY(shadow_isect, hit, prim);
  isect.object = INTEGRATOR_STATE_ARRAY(shadow_isect, hit, object);
  isect.type = INTEGRATOR_STATE_ARRAY(shadow_isect, hit, type);
  isect.u = INTEGRATOR_STATE_ARRAY(shadow_isect, hit, u);
  isect.v = INTEGRATOR_STATE_ARRAY(shadow_isect, hit, v);
  isect.t = INTEGRATOR_STATE_ARRAY(shadow_isect, hit, t);

  const float3 ray_P = INTEGRATOR_STATE(shadow_ray, P);
  const float3 ray_D = INTEGRATOR_STATE(shadow_ray, D);
  const float ray_time = INTEGRATOR_STATE(shadow_ray, time);
  shader_setup_from_ray(kg, shadow_sd, ray_P, ray_D, ray_time, &isect);

  /* Evaluate shader. */
  if (!(shadow_sd->flag & SD_HAS_ONLY_VOLUME)) {
    shader_eval_surface(INTEGRATOR_STATE_PASS, shadow_sd, NULL, PATH_RAY_SHADOW);
  }

  /* Compute transparency from closures. */
  return shader_bsdf_transparency(kg, shadow_sd);
}

ccl_device_inline bool integrate_transparent_shadow(INTEGRATOR_STATE_ARGS)
{
  /* Accumulate shadow for transparent surfaces. */
  for (int hit = 0; hit < INTEGRATOR_SHADOW_ISECT_SIZE; hit++) {
    if (INTEGRATOR_STATE_ARRAY(shadow_isect, hit, prim) == PRIM_NONE) {
      break;
    }

    const float3 shadow = integrate_transparent_shadow_shader_eval(INTEGRATOR_STATE_PASS, hit);
    const float3 throughput = INTEGRATOR_STATE(shadow_path, throughput) * shadow;
    INTEGRATOR_STATE_WRITE(shadow_path, throughput) = throughput;

    if (is_zero(throughput)) {
      return true;
    }
  }

  return false;
}
#endif /* __TRANSPARENT_SHADOWS__ */

ccl_device void integrator_shade_shadow(INTEGRATOR_STATE_ARGS,
                                        ccl_global float *ccl_restrict render_buffer)
{
  /* Only execute if anything was hit, otherwise path must have been terminated. */
  if (INTEGRATOR_SHADOW_PATH_IS_TERMINATED) {
    return;
  }

#ifdef __TRANSPARENT_SHADOWS__
  /* Evaluate transparent shadows. */
  const bool opaque = integrate_transparent_shadow(INTEGRATOR_STATE_PASS);
  if (opaque) {
    INTEGRATOR_SHADOW_PATH_TERMINATE(SHADE_SHADOW);
    return;
  }
#endif

  const bool shadow_isect_done = true;
  if (shadow_isect_done) {
    const float3 L = INTEGRATOR_STATE(shadow_light, L);
    kernel_accum_light(INTEGRATOR_STATE_PASS, L, render_buffer);

    INTEGRATOR_SHADOW_PATH_TERMINATE(SHADE_SHADOW);
    return;
  }
  else {
    /* TODO: add mechanism to detect and continue tracing if max_hits exceeded. */
    INTEGRATOR_SHADOW_PATH_NEXT(SHADE_SHADOW, INTERSECT_SHADOW);
    return;
  }
}

CCL_NAMESPACE_END
