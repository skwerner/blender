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

#include "kernel/kernel_path_state.h"
#include "kernel/kernel_projection.h"
#include "kernel/kernel_shader.h"

#include "kernel/bvh/bvh.h"

#include "kernel/closure/alloc.h"
#include "kernel/closure/bsdf_diffuse.h"
#include "kernel/closure/bsdf_principled_diffuse.h"
#include "kernel/closure/bssrdf.h"
#include "kernel/closure/volume.h"

CCL_NAMESPACE_BEGIN

#ifdef __SUBSURFACE__

/* TODO: restore or remove this.
 * If we reevaluate the shader for the normal and color, it should happen in
 * the shade_surface kernel. */

#  if 0
/* optionally do blurring of color and/or bump mapping, at the cost of a shader evaluation */
ccl_device float3 subsurface_color_pow(float3 color, float exponent)
{
  color = max(color, zero_float3());

  if (exponent == 1.0f) {
    /* nothing to do */
  }
  else if (exponent == 0.5f) {
    color.x = sqrtf(color.x);
    color.y = sqrtf(color.y);
    color.z = sqrtf(color.z);
  }
  else {
    color.x = powf(color.x, exponent);
    color.y = powf(color.y, exponent);
    color.z = powf(color.z, exponent);
  }

  return color;
}

ccl_device void subsurface_color_bump_blur(const KernelGlobals *kg,
                                           ShaderData *sd,
                                           ccl_addr_space PathState *state,
                                           float3 *eval,
                                           float3 *N)
{
  /* average color and texture blur at outgoing point */
  float texture_blur;
  float3 out_color = shader_bssrdf_sum(sd, NULL, &texture_blur);

  /* do we have bump mapping? */
  bool bump = (sd->flag & SD_HAS_BSSRDF_BUMP) != 0;

  if (bump || texture_blur > 0.0f) {
    /* average color and normal at incoming point */
    shader_eval_surface(kg, sd, state, NULL, state->flag);
    float3 in_color = shader_bssrdf_sum(sd, (bump) ? N : NULL, NULL);

    /* we simply divide out the average color and multiply with the average
     * of the other one. we could try to do this per closure but it's quite
     * tricky to match closures between shader evaluations, their number and
     * order may change, this is simpler */
    if (texture_blur > 0.0f) {
      out_color = subsurface_color_pow(out_color, texture_blur);
      in_color = subsurface_color_pow(in_color, texture_blur);

      *eval *= safe_divide_color(in_color, out_color);
    }
  }
}
#  endif

ccl_device bool subsurface_bounce(INTEGRATOR_STATE_ARGS, ShaderData *sd, const ShaderClosure *sc)
{
  /* We should never have two consecutive BSSRDF bounces, the second one should
   * be converted to a diffuse BSDF to avoid this. */
  kernel_assert(!(INTEGRATOR_STATE(path, flag) & PATH_RAY_DIFFUSE_ANCESTOR));

  /* Setup path state for intersect_subsurface kernel. */
  const Bssrdf *bssrdf = (const Bssrdf *)sc;

  /* Setup ray into surface. */
  INTEGRATOR_STATE_WRITE(ray, P) = sd->P;
  INTEGRATOR_STATE_WRITE(ray, D) = sd->N;
  INTEGRATOR_STATE_WRITE(ray, t) = FLT_MAX;
  INTEGRATOR_STATE_WRITE(ray, dP) = differential_make_compact(sd->dP);
  INTEGRATOR_STATE_WRITE(ray, dD) = differential_zero_compact();

  /* Pass along object info, reusing isect to save memory. */
  INTEGRATOR_STATE_WRITE(isect, Ng) = sd->Ng;
  INTEGRATOR_STATE_WRITE(isect, object) = sd->object;

  /* Pass BSSRDF parameters. */
  INTEGRATOR_STATE_WRITE(path, flag) |= PATH_RAY_SUBSURFACE;
  INTEGRATOR_STATE_WRITE(path, throughput) *= shader_bssrdf_sample_weight(sd, sc);
  INTEGRATOR_STATE_WRITE(path, diffuse_glossy_ratio) = one_float3();

  const float roughness = (sc->type == CLOSURE_BSSRDF_PRINCIPLED_ID ||
                           sc->type == CLOSURE_BSSRDF_PRINCIPLED_RANDOM_WALK_ID) ?
                              bssrdf->roughness :
                              FLT_MAX;
  INTEGRATOR_STATE_WRITE(subsurface, albedo) = bssrdf->albedo;
  INTEGRATOR_STATE_WRITE(subsurface, radius) = bssrdf->radius;
  INTEGRATOR_STATE_WRITE(subsurface, roughness) = roughness;

  return true;
}

ccl_device void subsurface_shader_data_setup(INTEGRATOR_STATE_ARGS, ShaderData *sd)
{
  /* Get bump mapped normal from shader evaluation at exit point. */
  float3 N = sd->N;
  if (sd->flag & SD_HAS_BSSRDF_BUMP) {
    N = shader_bssrdf_normal(sd);
  }

  /* Setup diffuse BSDF at the exit point. This replaces shader_eval_surface. */
  sd->flag &= ~SD_CLOSURE_FLAGS;
  sd->num_closure = 0;
  sd->num_closure_left = kernel_data.integrator.max_closures;

  const float3 weight = one_float3();
  const float roughness = INTEGRATOR_STATE(subsurface, roughness);

#  ifdef __PRINCIPLED__
  if (roughness != FLT_MAX) {
    PrincipledDiffuseBsdf *bsdf = (PrincipledDiffuseBsdf *)bsdf_alloc(
        sd, sizeof(PrincipledDiffuseBsdf), weight);

    if (bsdf) {
      bsdf->N = N;
      bsdf->roughness = roughness;
      sd->flag |= bsdf_principled_diffuse_setup(bsdf);

      /* replace CLOSURE_BSDF_PRINCIPLED_DIFFUSE_ID with this special ID so render passes
       * can recognize it as not being a regular Disney principled diffuse closure */
      bsdf->type = CLOSURE_BSDF_BSSRDF_PRINCIPLED_ID;
    }
  }
  else
#  endif /* __PRINCIPLED__ */
  {
    DiffuseBsdf *bsdf = (DiffuseBsdf *)bsdf_alloc(sd, sizeof(DiffuseBsdf), weight);

    if (bsdf) {
      bsdf->N = N;
      sd->flag |= bsdf_diffuse_setup(bsdf);

      /* replace CLOSURE_BSDF_DIFFUSE_ID with this special ID so render passes
       * can recognize it as not being a regular diffuse closure */
      bsdf->type = CLOSURE_BSDF_BSSRDF_ID;
    }
  }
}

/* Random walk subsurface scattering.
 *
 * "Practical and Controllable Subsurface Scattering for Production Path
 *  Tracing". Matt Jen-Yuan Chiang, Peter Kutz, Brent Burley. SIGGRAPH 2016. */

ccl_device void subsurface_random_walk_remap(const float A,
                                             const float d,
                                             float *sigma_t,
                                             float *alpha)
{
  /* Compute attenuation and scattering coefficients from albedo. */
  *alpha = 1.0f - expf(A * (-5.09406f + A * (2.61188f - A * 4.31805f)));
  const float s = 1.9f - A + 3.5f * sqr(A - 0.8f);

  *sigma_t = 1.0f / fmaxf(d * s, 1e-16f);
}

ccl_device void subsurface_random_walk_coefficients(
    const float3 albedo, const float3 radius, float3 *sigma_t, float3 *alpha, float3 *throughput)
{
  float sigma_t_x, sigma_t_y, sigma_t_z;
  float alpha_x, alpha_y, alpha_z;

  subsurface_random_walk_remap(albedo.x, radius.x, &sigma_t_x, &alpha_x);
  subsurface_random_walk_remap(albedo.y, radius.y, &sigma_t_y, &alpha_y);
  subsurface_random_walk_remap(albedo.z, radius.z, &sigma_t_z, &alpha_z);

  *sigma_t = make_float3(sigma_t_x, sigma_t_y, sigma_t_z);
  *alpha = make_float3(alpha_x, alpha_y, alpha_z);

  /* Throughput already contains closure weight at this point, which includes the
   * albedo, as well as closure mixing and Fresnel weights. Divide out the albedo
   * which will be added through scattering. */
  *throughput = safe_divide_color(*throughput, albedo);
}

/* References for Dwivedi sampling:
 *
 * [1] "A Zero-variance-based Sampling Scheme for Monte Carlo Subsurface Scattering"
 * by Jaroslav Křivánek and Eugene d'Eon (SIGGRAPH 2014)
 * https://cgg.mff.cuni.cz/~jaroslav/papers/2014-zerovar/
 *
 * [2] "Improving the Dwivedi Sampling Scheme"
 * by Johannes Meng, Johannes Hanika, and Carsten Dachsbacher (EGSR 2016)
 * https://cg.ivd.kit.edu/1951.php
 *
 * [3] "Zero-Variance Theory for Efficient Subsurface Scattering"
 * by Eugene d'Eon and Jaroslav Křivánek (SIGGRAPH 2020)
 * https://iliyan.com/publications/RenderingCourse2020
 */

ccl_device_forceinline float eval_phase_dwivedi(float v, float phase_log, float cos_theta)
{
  /* Eq. 9 from [2] using precomputed log((v + 1) / (v - 1))*/
  return 1.0f / ((v - cos_theta) * phase_log);
}

ccl_device_forceinline float sample_phase_dwivedi(float v, float phase_log, float rand)
{
  /* Based on Eq. 10 from [2]: `v - (v + 1) * pow((v - 1) / (v + 1), rand)`
   * Since we're already pre-computing `phase_log = log((v + 1) / (v - 1))` for the evaluation,
   * we can implement the power function like this. */
  return v - (v + 1) * expf(-rand * phase_log);
}

ccl_device_forceinline float diffusion_length_dwivedi(float alpha)
{
  /* Eq. 67 from [3] */
  return 1.0f / sqrtf(1.0f - powf(alpha, 2.44294f - 0.0215813f * alpha + 0.578637f / alpha));
}

ccl_device_forceinline float3 direction_from_cosine(float3 D, float cos_theta, float randv)
{
  float sin_theta = safe_sqrtf(1.0f - cos_theta * cos_theta);
  float phi = M_2PI_F * randv;
  float3 dir = make_float3(sin_theta * cosf(phi), sin_theta * sinf(phi), cos_theta);

  float3 T, B;
  make_orthonormals(D, &T, &B);
  return dir.x * T + dir.y * B + dir.z * D;
}

ccl_device_forceinline float3 subsurface_random_walk_pdf(float3 sigma_t,
                                                         float t,
                                                         bool hit,
                                                         float3 *transmittance)
{
  float3 T = volume_color_transmittance(sigma_t, t);
  if (transmittance) {
    *transmittance = T;
  }
  return hit ? T : sigma_t * T;
}

ccl_device_inline bool subsurface_random_walk(INTEGRATOR_STATE_ARGS)
{
  RNGState rng_state;
  path_state_rng_load(INTEGRATOR_STATE_PASS, &rng_state);
  float bssrdf_u, bssrdf_v;
  path_state_rng_2D(kg, &rng_state, PRNG_BSDF_U, &bssrdf_u, &bssrdf_v);

  const float3 P = INTEGRATOR_STATE(ray, P);
  const float3 N = INTEGRATOR_STATE(ray, D);
  const float ray_dP = INTEGRATOR_STATE(ray, dP);
  const float time = INTEGRATOR_STATE(ray, time);
  const float3 Ng = INTEGRATOR_STATE(isect, Ng);
  const int object = INTEGRATOR_STATE(isect, object);

  /* Sample diffuse surface scatter into the object. */
  float3 D;
  float pdf;
  sample_cos_hemisphere(-N, bssrdf_u, bssrdf_v, &D, &pdf);
  if (dot(-Ng, D) <= 0.0f) {
    return false;
  }

#  ifndef __KERNEL_OPTIX__
  /* Compute or fetch object transforms. */
  Transform ob_itfm ccl_optional_struct_init;
  Transform ob_tfm = object_fetch_transform_motion_test(kg, object, time, &ob_itfm);
#  endif

  /* Setup ray. */
  Ray ray ccl_optional_struct_init;
  ray.P = ray_offset(P, -Ng);
  ray.D = D;
  ray.t = FLT_MAX;
  ray.time = time;
  ray.dP = ray_dP;
  ray.dD = differential_zero_compact();

  /* Setup intersections.
   * TODO: make this more compact if we don't bring back disk based SSS that needs
   * multiple intersections. */
  LocalIntersection ss_isect ccl_optional_struct_init;

  /* Convert subsurface to volume coefficients.
   * The single-scattering albedo is named alpha to avoid confusion with the surface albedo. */
  const float3 albedo = INTEGRATOR_STATE(subsurface, albedo);
  const float3 radius = INTEGRATOR_STATE(subsurface, radius);

  float3 sigma_t, alpha;
  float3 throughput = INTEGRATOR_STATE_WRITE(path, throughput);
  subsurface_random_walk_coefficients(albedo, radius, &sigma_t, &alpha, &throughput);
  float3 sigma_s = sigma_t * alpha;

  /* Theoretically it should be better to use the exact alpha for the channel we're sampling at
   * each bounce, but in practice there doesn't seem to be a noticeable difference in exchange
   * for making the code significantly more complex and slower (if direction sampling depends on
   * the sampled channel, we need to compute its PDF per-channel and consider it for MIS later on).
   *
   * Since the strength of the guided sampling increases as alpha gets lower, using a value that
   * is too low results in fireflies while one that's too high just gives a bit more noise.
   * Therefore, the code here uses the highest of the three albedos to be safe. */
  const float diffusion_length = diffusion_length_dwivedi(max3(alpha));

  if (diffusion_length == 1.0f) {
    /* With specific values of alpha the length might become 1, which in asymptotic makes phase to
     * be infinite. After first bounce it will cause throughput to be 0. Do early output, avoiding
     * numerical issues and extra unneeded work. */
    return false;
  }

  /* Precompute term for phase sampling. */
  const float phase_log = logf((diffusion_length + 1) / (diffusion_length - 1));

  /* Modify state for RNGs, decorrelated from other paths. */
  rng_state.rng_hash = cmj_hash(rng_state.rng_hash + rng_state.rng_offset, 0xdeadbeef);

  /* Random walk until we hit the surface again. */
  bool hit = false;
  bool have_opposite_interface = false;
  float opposite_distance = 0.0f;

  /* Todo: Disable for alpha>0.999 or so? */
  const float guided_fraction = 0.75f;

  for (int bounce = 0; bounce < BSSRDF_MAX_BOUNCES; bounce++) {
    /* Advance random number offset. */
    rng_state.rng_offset += PRNG_BOUNCE_NUM;

    /* Sample color channel, use MIS with balance heuristic. */
    float rphase = path_state_rng_1D(kg, &rng_state, PRNG_PHASE_CHANNEL);
    float3 channel_pdf;
    int channel = volume_sample_channel(alpha, throughput, rphase, &channel_pdf);
    float sample_sigma_t = volume_channel_get(sigma_t, channel);
    float randt = path_state_rng_1D(kg, &rng_state, PRNG_SCATTER_DISTANCE);

    /* We need the result of the raycast to compute the full guided PDF, so just remember the
     * relevant terms to avoid recomputing them later. */
    float backward_fraction = 0.0f;
    float forward_pdf_factor = 0.0f;
    float forward_stretching = 1.0f;
    float backward_pdf_factor = 0.0f;
    float backward_stretching = 1.0f;

    /* For the initial ray, we already know the direction, so just do classic distance sampling. */
    if (bounce > 0) {
      /* Decide whether we should use guided or classic sampling. */
      bool guided = (path_state_rng_1D(kg, &rng_state, PRNG_LIGHT_TERMINATE) < guided_fraction);

      /* Determine if we want to sample away from the incoming interface.
       * This only happens if we found a nearby opposite interface, and the probability for it
       * depends on how close we are to it already.
       * This probability term comes from the recorded presentation of [3]. */
      bool guide_backward = false;
      if (have_opposite_interface) {
        /* Compute distance of the random walk between the tangent plane at the starting point
         * and the assumed opposite interface (the parallel plane that contains the point we
         * found in our ray query for the opposite side). */
        float x = clamp(dot(ray.P - P, -N), 0.0f, opposite_distance);
        backward_fraction = 1.0f / (1.0f + expf((opposite_distance - 2 * x) / diffusion_length));
        guide_backward = path_state_rng_1D(kg, &rng_state, PRNG_TERMINATE) < backward_fraction;
      }

      /* Sample scattering direction. */
      float scatter_u, scatter_v;
      path_state_rng_2D(kg, &rng_state, PRNG_BSDF_U, &scatter_u, &scatter_v);
      float cos_theta;
      if (guided) {
        cos_theta = sample_phase_dwivedi(diffusion_length, phase_log, scatter_u);
        /* The backwards guiding distribution is just mirrored along sd->N, so swapping the
         * sign here is enough to sample from that instead. */
        if (guide_backward) {
          cos_theta = -cos_theta;
        }
      }
      else {
        cos_theta = 2.0f * scatter_u - 1.0f;
      }
      ray.D = direction_from_cosine(N, cos_theta, scatter_v);

      /* Compute PDF factor caused by phase sampling (as the ratio of guided / classic).
       * Since phase sampling is channel-independent, we can get away with applying a factor
       * to the guided PDF, which implicitly means pulling out the classic PDF term and letting
       * it cancel with an equivalent term in the numerator of the full estimator.
       * For the backward PDF, we again reuse the same probability distribution with a sign swap.
       */
      forward_pdf_factor = 2.0f * eval_phase_dwivedi(diffusion_length, phase_log, cos_theta);
      backward_pdf_factor = 2.0f * eval_phase_dwivedi(diffusion_length, phase_log, -cos_theta);

      /* Prepare distance sampling.
       * For the backwards case, this also needs the sign swapped since now directions against
       * sd->N (and therefore with negative cos_theta) are preferred. */
      forward_stretching = (1.0f - cos_theta / diffusion_length);
      backward_stretching = (1.0f + cos_theta / diffusion_length);
      if (guided) {
        sample_sigma_t *= guide_backward ? backward_stretching : forward_stretching;
      }
    }

    /* Sample direction along ray. */
    float t = -logf(1.0f - randt) / sample_sigma_t;

    /* On the first bounce, we use the raycast to check if the opposite side is nearby.
     * If yes, we will later use backwards guided sampling in order to have a decent
     * chance of connecting to it.
     * Todo: Maybe use less than 10 times the mean free path? */
    ray.t = (bounce == 0) ? max(t, 10.0f / (min3(sigma_t))) : t;
    scene_intersect_local(kg, &ray, &ss_isect, object, NULL, 1);
    hit = (ss_isect.num_hits > 0);

    if (hit) {
#  ifdef __KERNEL_OPTIX__
      /* t is always in world space with OptiX. */
      ray.t = ss_isect.hits[0].t;
#  else
      /* Compute world space distance to surface hit. */
      float3 D = transform_direction(&ob_itfm, ray.D);
      D = normalize(D) * ss_isect.hits[0].t;
      ray.t = len(transform_direction(&ob_tfm, D));
#  endif
    }

    if (bounce == 0) {
      /* Check if we hit the opposite side. */
      if (hit) {
        have_opposite_interface = true;
        opposite_distance = dot(ray.P + ray.t * ray.D - P, -N);
      }
      /* Apart from the opposite side check, we were supposed to only trace up to distance t,
       * so check if there would have been a hit in that case. */
      hit = ray.t < t;
    }

    /* Use the distance to the exit point for the throughput update if we found one. */
    if (hit) {
      t = ray.t;
    }
    else if (bounce == 0) {
      /* Restore original position if nothing was hit after the first bounce,
       * without the ray_offset() that was added to avoid self-intersection.
       * Otherwise if that offset is relatively large compared to the scattering
       * radius, we never go back up high enough to exit the surface. */
      ray.P = P;
    }

    /* Advance to new scatter location. */
    ray.P += t * ray.D;

    float3 transmittance;
    float3 pdf = subsurface_random_walk_pdf(sigma_t, t, hit, &transmittance);
    if (bounce > 0) {
      /* Compute PDF just like we do for classic sampling, but with the stretched sigma_t. */
      float3 guided_pdf = subsurface_random_walk_pdf(forward_stretching * sigma_t, t, hit, NULL);

      if (have_opposite_interface) {
        /* First step of MIS: Depending on geometry we might have two methods for guided
         * sampling, so perform MIS between them. */
        float3 back_pdf = subsurface_random_walk_pdf(backward_stretching * sigma_t, t, hit, NULL);
        guided_pdf = mix(
            guided_pdf * forward_pdf_factor, back_pdf * backward_pdf_factor, backward_fraction);
      }
      else {
        /* Just include phase sampling factor otherwise. */
        guided_pdf *= forward_pdf_factor;
      }

      /* Now we apply the MIS balance heuristic between the classic and guided sampling. */
      pdf = mix(pdf, guided_pdf, guided_fraction);
    }

    /* Finally, we're applying MIS again to combine the three color channels.
     * Altogether, the MIS computation combines up to nine different estimators:
     * {classic, guided, backward_guided} x {r, g, b} */
    throughput *= (hit ? transmittance : sigma_s * transmittance) / dot(channel_pdf, pdf);

    if (hit) {
      /* If we hit the surface, we are done. */
      break;
    }
    else if (throughput.x < VOLUME_THROUGHPUT_EPSILON &&
             throughput.y < VOLUME_THROUGHPUT_EPSILON &&
             throughput.z < VOLUME_THROUGHPUT_EPSILON) {
      /* Avoid unnecessary work and precision issue when throughput gets really small. */
      break;
    }
  }

  kernel_assert(isfinite3_safe(throughput));

  /* Return number of hits in ss_isect. */
  if (!hit) {
    return false;
  }

  /* Pretend ray is coming from the outside towards the exit point. This ensures
   * correct front/back facing normals.
   * TODO: find a more elegant solution? */
  ray.P += ray.D * ray.t * 2.0f;
  ray.D = -ray.D;

  integrator_state_write_isect(INTEGRATOR_STATE_PASS, &ss_isect.hits[0]);
  integrator_state_write_ray(INTEGRATOR_STATE_PASS, &ray);
  INTEGRATOR_STATE_WRITE(path, throughput) = throughput;

  const int shader = intersection_get_shader(kg, &ss_isect.hits[0]);
  const int next_kernel = DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE;
  INTEGRATOR_PATH_NEXT_SORTED(DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST, next_kernel, shader);

  return true;
}

#endif /* __SUBSURFACE__ */

CCL_NAMESPACE_END
