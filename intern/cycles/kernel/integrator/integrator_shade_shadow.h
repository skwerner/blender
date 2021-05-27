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

  /* Setup shader data at surface. */
  Intersection isect ccl_optional_struct_init;
  integrator_state_read_shadow_isect(INTEGRATOR_STATE_PASS, &isect, hit);

  Ray ray ccl_optional_struct_init;
  integrator_state_read_shadow_ray(INTEGRATOR_STATE_PASS, &ray);

  shader_setup_from_ray(kg, shadow_sd, &ray, &isect);

  /* Evaluate shader. */
  if (!(shadow_sd->flag & SD_HAS_ONLY_VOLUME)) {
    shader_eval_surface<NODE_FEATURE_MASK_SURFACE_SHADOW>(
        INTEGRATOR_STATE_PASS, shadow_sd, NULL, PATH_RAY_SHADOW);
  }

  /* Compute transparency from closures. */
  return shader_bsdf_transparency(kg, shadow_sd);
}

ccl_device_inline bool integrate_transparent_shadow(INTEGRATOR_STATE_ARGS, const int num_hits)
{
  /* Accumulate shadow for transparent surfaces. */
  const int num_recorded_hits = min(num_hits, INTEGRATOR_SHADOW_ISECT_SIZE);

  for (int hit = 0; hit < num_recorded_hits; hit++) {
    const float3 shadow = integrate_transparent_shadow_shader_eval(INTEGRATOR_STATE_PASS, hit);
    const float3 throughput = INTEGRATOR_STATE(shadow_path, throughput) * shadow;
    if (is_zero(throughput)) {
      return true;
    }

    INTEGRATOR_STATE_WRITE(shadow_path, throughput) = throughput;
    INTEGRATOR_STATE_WRITE(shadow_path, transparent_bounce) += 1;

    /* Note we do not need to check max_transparent_bounce here, the number
     * of intersections is already limited and made opaque in the
     * INTERSECT_SHADOW kernel. */
  }

  if (num_hits >= INTEGRATOR_SHADOW_ISECT_SIZE) {
    /* There are more hits that we could not recorded due to memory usage,
     * adjust ray to intersect again from the last hit. */
    const float last_hit_t = INTEGRATOR_STATE_ARRAY(shadow_isect, num_recorded_hits - 1, t);
    const float3 ray_P = INTEGRATOR_STATE(shadow_ray, P);
    const float3 ray_D = INTEGRATOR_STATE(shadow_ray, D);
    INTEGRATOR_STATE_WRITE(shadow_ray, P) = ray_offset(ray_P + last_hit_t * ray_D, ray_D);
    INTEGRATOR_STATE_WRITE(shadow_ray, t) -= last_hit_t;
  }

  return false;
}
#endif /* __TRANSPARENT_SHADOWS__ */

ccl_device void integrator_shade_shadow(INTEGRATOR_STATE_ARGS,
                                        ccl_global float *ccl_restrict render_buffer)
{
  const int num_hits = INTEGRATOR_STATE(shadow_path, num_hits);

#ifdef __TRANSPARENT_SHADOWS__
  /* Evaluate transparent shadows. */
  const bool opaque = integrate_transparent_shadow(INTEGRATOR_STATE_PASS, num_hits);
  if (opaque) {
    INTEGRATOR_SHADOW_PATH_TERMINATE(DEVICE_KERNEL_INTEGRATOR_SHADE_SHADOW);
    return;
  }
#endif

  if (num_hits >= INTEGRATOR_SHADOW_ISECT_SIZE) {
    /* More intersections to find, continue shadow ray. */
    INTEGRATOR_SHADOW_PATH_NEXT(DEVICE_KERNEL_INTEGRATOR_SHADE_SHADOW,
                                DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW);
    return;
  }
  else {
    kernel_accum_light(INTEGRATOR_STATE_PASS, render_buffer);
    INTEGRATOR_SHADOW_PATH_TERMINATE(DEVICE_KERNEL_INTEGRATOR_SHADE_SHADOW);
    return;
  }
}

CCL_NAMESPACE_END
