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

/* Ignore paths that have volume throughput below this value, to avoid unnecessary work
 * and precision issues.
 * todo: this value could be tweaked or turned into a probability to avoid unnecessary
 * work in volumes and subsurface scattering. */
#define VOLUME_THROUGHPUT_EPSILON 1e-6f

ccl_device float volume_stack_step_size(const KernelGlobals *kg, ccl_addr_space VolumeStack *stack)
{
  float step_size = FLT_MAX;

  for (int i = 0; stack[i].shader != SHADER_NONE; i++) {
    int shader_flag = kernel_tex_fetch(__shaders, (stack[i].shader & SHADER_MASK)).flags;

    bool heterogeneous = false;

    if (shader_flag & SD_HETEROGENEOUS_VOLUME) {
      heterogeneous = true;
    }
    else if (shader_flag & SD_NEED_VOLUME_ATTRIBUTES) {
      /* We want to render world or objects without any volume grids
       * as homogeneous, but can only verify this at run-time since other
       * heterogeneous volume objects may be using the same shader. */
      int object = stack[i].object;
      if (object != OBJECT_NONE) {
        int object_flag = kernel_tex_fetch(__object_flag, object);
        if (object_flag & SD_OBJECT_HAS_VOLUME_ATTRIBUTES) {
          heterogeneous = true;
        }
      }
    }

    if (heterogeneous) {
      float object_step_size = object_volume_step_size(kg, stack[i].object);
      object_step_size *= kernel_data.integrator.volume_step_rate;
      step_size = fminf(object_step_size, step_size);
    }
  }

  return step_size;
}

ccl_device int volume_stack_sampling_method(const KernelGlobals *kg, VolumeStack *stack)
{
  if (kernel_data.integrator.num_all_lights == 0)
    return 0;

  int method = -1;

  for (int i = 0; stack[i].shader != SHADER_NONE; i++) {
    int shader_flag = kernel_tex_fetch(__shaders, (stack[i].shader & SHADER_MASK)).flags;

    if (shader_flag & SD_VOLUME_MIS) {
      return SD_VOLUME_MIS;
    }
    else if (shader_flag & SD_VOLUME_EQUIANGULAR) {
      if (method == 0)
        return SD_VOLUME_MIS;

      method = SD_VOLUME_EQUIANGULAR;
    }
    else {
      if (method == SD_VOLUME_EQUIANGULAR)
        return SD_VOLUME_MIS;

      method = 0;
    }
  }

  return method;
}

ccl_device_inline void kernel_volume_step_init(const KernelGlobals *kg,
                                               ccl_addr_space PathState *state,
                                               const float object_step_size,
                                               float t,
                                               float *step_size,
                                               float *step_shade_offset,
                                               float *steps_offset)
{
  const int max_steps = kernel_data.integrator.volume_max_steps;
  float step = min(object_step_size, t);

  /* compute exact steps in advance for malloc */
  if (t > max_steps * step) {
    step = t / (float)max_steps;
  }

  *step_size = step;

  /* Perform shading at this offset within a step, to integrate over
   * over the entire step segment. */
  *step_shade_offset = path_state_rng_1D_hash(kg, state, 0x1e31d8a4);

  /* Shift starting point of all segment by this random amount to avoid
   * banding artifacts from the volume bounding shape. */
  *steps_offset = path_state_rng_1D_hash(kg, state, 0x3d22c7b3);
}

/* Volume Shadows
 *
 * These functions are used to attenuate shadow rays to lights. Both absorption
 * and scattering will block light, represented by the extinction coefficient. */

/* homogeneous volume: assume shader evaluation at the starts gives
 * the extinction coefficient for the entire line segment */
ccl_device void kernel_volume_shadow_homogeneous(const KernelGlobals *kg,
                                                 ccl_addr_space PathState *state,
                                                 Ray *ray,
                                                 ShaderData *sd,
                                                 float3 *throughput)
{
  float3 sigma_t = zero_float3();

  if (volume_shader_extinction_sample(kg, sd, state, ray->P, &sigma_t))
    *throughput *= volume_color_transmittance(sigma_t, ray->t);
}

/* heterogeneous volume: integrate stepping through the volume until we
 * reach the end, get absorbed entirely, or run out of iterations */
ccl_device void kernel_volume_shadow_heterogeneous(const KernelGlobals *kg,
                                                   ccl_addr_space PathState *state,
                                                   Ray *ray,
                                                   ShaderData *sd,
                                                   float3 *throughput,
                                                   const float object_step_size)
{
  float3 tp = *throughput;

  /* Prepare for stepping.
   * For shadows we do not offset all segments, since the starting point is
   * already a random distance inside the volume. It also appears to create
   * banding artifacts for unknown reasons. */
  int max_steps = kernel_data.integrator.volume_max_steps;
  float step_size, step_shade_offset, unused;
  kernel_volume_step_init(
      kg, state, object_step_size, ray->t, &step_size, &step_shade_offset, &unused);
  const float steps_offset = 1.0f;

  /* compute extinction at the start */
  float t = 0.0f;

  float3 sum = zero_float3();

  for (int i = 0; i < max_steps; i++) {
    /* advance to new position */
    float new_t = min(ray->t, (i + steps_offset) * step_size);
    float dt = new_t - t;

    float3 new_P = ray->P + ray->D * (t + dt * step_shade_offset);
    float3 sigma_t = zero_float3();

    /* compute attenuation over segment */
    if (volume_shader_extinction_sample(kg, sd, state, new_P, &sigma_t)) {
      /* Compute expf() only for every Nth step, to save some calculations
       * because exp(a)*exp(b) = exp(a+b), also do a quick VOLUME_THROUGHPUT_EPSILON
       * check then. */
      sum += (-sigma_t * dt);
      if ((i & 0x07) == 0) { /* ToDo: Other interval? */
        tp = *throughput * exp3(sum);

        /* stop if nearly all light is blocked */
        if (tp.x < VOLUME_THROUGHPUT_EPSILON && tp.y < VOLUME_THROUGHPUT_EPSILON &&
            tp.z < VOLUME_THROUGHPUT_EPSILON)
          break;
      }
    }

    /* stop if at the end of the volume */
    t = new_t;
    if (t == ray->t) {
      /* Update throughput in case we haven't done it above */
      tp = *throughput * exp3(sum);
      break;
    }
  }

  *throughput = tp;
}

/* get the volume attenuation over line segment defined by ray, with the
 * assumption that there are no surfaces blocking light between the endpoints */
#if defined(__KERNEL_OPTIX__) && defined(__SHADER_RAYTRACE__)
ccl_device_inline void kernel_volume_shadow(const KernelGlobals *kg,
                                            ShaderData *shadow_sd,
                                            ccl_addr_space PathState *state,
                                            Ray *ray,
                                            float3 *throughput)
{
  optixDirectCall<void>(1, kg, shadow_sd, state, ray, throughput);
}
extern "C" __device__ void __direct_callable__kernel_volume_shadow(
#else
ccl_device void kernel_volume_shadow(
#endif
    const KernelGlobals *kg,
    ShaderData *shadow_sd,
    ccl_addr_space PathState *state,
    Ray *ray,
    float3 *throughput)
{
  shader_setup_from_volume(kg, shadow_sd, ray);

  float step_size = volume_stack_step_size(kg, state->volume_stack);
  if (step_size != FLT_MAX)
    kernel_volume_shadow_heterogeneous(kg, state, ray, shadow_sd, throughput, step_size);
  else
    kernel_volume_shadow_homogeneous(kg, state, ray, shadow_sd, throughput);
}

#endif /* __VOLUME__ */

/* Equi-angular sampling as in:
 * "Importance Sampling Techniques for Path Tracing in Participating Media" */

ccl_device float kernel_volume_equiangular_sample(Ray *ray, float3 light_P, float xi, float *pdf)
{
  float t = ray->t;

  float delta = dot((light_P - ray->P), ray->D);
  float D = safe_sqrtf(len_squared(light_P - ray->P) - delta * delta);
  if (UNLIKELY(D == 0.0f)) {
    *pdf = 0.0f;
    return 0.0f;
  }
  float theta_a = -atan2f(delta, D);
  float theta_b = atan2f(t - delta, D);
  float t_ = D * tanf((xi * theta_b) + (1 - xi) * theta_a);
  if (UNLIKELY(theta_b == theta_a)) {
    *pdf = 0.0f;
    return 0.0f;
  }
  *pdf = D / ((theta_b - theta_a) * (D * D + t_ * t_));

  return min(t, delta + t_); /* min is only for float precision errors */
}

ccl_device float kernel_volume_equiangular_pdf(Ray *ray, float3 light_P, float sample_t)
{
  float delta = dot((light_P - ray->P), ray->D);
  float D = safe_sqrtf(len_squared(light_P - ray->P) - delta * delta);
  if (UNLIKELY(D == 0.0f)) {
    return 0.0f;
  }

  float t = ray->t;
  float t_ = sample_t - delta;

  float theta_a = -atan2f(delta, D);
  float theta_b = atan2f(t - delta, D);
  if (UNLIKELY(theta_b == theta_a)) {
    return 0.0f;
  }

  float pdf = D / ((theta_b - theta_a) * (D * D + t_ * t_));

  return pdf;
}

/* Distance sampling */

ccl_device float kernel_volume_distance_sample(
    float max_t, float3 sigma_t, int channel, float xi, float3 *transmittance, float3 *pdf)
{
  /* xi is [0, 1[ so log(0) should never happen, division by zero is
   * avoided because sample_sigma_t > 0 when SD_SCATTER is set */
  float sample_sigma_t = volume_channel_get(sigma_t, channel);
  float3 full_transmittance = volume_color_transmittance(sigma_t, max_t);
  float sample_transmittance = volume_channel_get(full_transmittance, channel);

  float sample_t = min(max_t, -logf(1.0f - xi * (1.0f - sample_transmittance)) / sample_sigma_t);

  *transmittance = volume_color_transmittance(sigma_t, sample_t);
  *pdf = safe_divide_color(sigma_t * *transmittance, one_float3() - full_transmittance);

  /* todo: optimization: when taken together with hit/miss decision,
   * the full_transmittance cancels out drops out and xi does not
   * need to be remapped */

  return sample_t;
}

ccl_device float3 kernel_volume_distance_pdf(float max_t, float3 sigma_t, float sample_t)
{
  float3 full_transmittance = volume_color_transmittance(sigma_t, max_t);
  float3 transmittance = volume_color_transmittance(sigma_t, sample_t);

  return safe_divide_color(sigma_t * transmittance, one_float3() - full_transmittance);
}

/* Emission */

ccl_device float3 kernel_volume_emission_integrate(VolumeShaderCoefficients *coeff,
                                                   int closure_flag,
                                                   float3 transmittance,
                                                   float t)
{
  /* integral E * exp(-sigma_t * t) from 0 to t = E * (1 - exp(-sigma_t * t))/sigma_t
   * this goes to E * t as sigma_t goes to zero
   *
   * todo: we should use an epsilon to avoid precision issues near zero sigma_t */
  float3 emission = coeff->emission;

  if (closure_flag & SD_EXTINCTION) {
    float3 sigma_t = coeff->sigma_t;

    emission.x *= (sigma_t.x > 0.0f) ? (1.0f - transmittance.x) / sigma_t.x : t;
    emission.y *= (sigma_t.y > 0.0f) ? (1.0f - transmittance.y) / sigma_t.y : t;
    emission.z *= (sigma_t.z > 0.0f) ? (1.0f - transmittance.z) / sigma_t.z : t;
  }
  else
    emission *= t;

  return emission;
}

/* Volume Path */

#ifdef __VOLUME__

/* homogeneous volume: assume shader evaluation at the start gives
 * the volume shading coefficient for the entire line segment */
ccl_device VolumeIntegrateResult
kernel_volume_integrate_homogeneous(const KernelGlobals *kg,
                                    ccl_addr_space PathState *state,
                                    Ray *ray,
                                    ShaderData *sd,
                                    PathRadiance *L,
                                    ccl_addr_space float3 *throughput,
                                    bool probalistic_scatter)
{
  VolumeShaderCoefficients coeff ccl_optional_struct_init;

  if (!volume_shader_sample(kg, sd, state, ray->P, &coeff))
    return VOLUME_PATH_MISSED;

  int closure_flag = sd->flag;
  float t = ray->t;
  float3 new_tp;

#  ifdef __VOLUME_SCATTER__
  /* randomly scatter, and if we do t is shortened */
  if (closure_flag & SD_SCATTER) {
    /* Sample channel, use MIS with balance heuristic. */
    float rphase = path_state_rng_1D(kg, state, PRNG_PHASE_CHANNEL);
    float3 albedo = safe_divide_color(coeff.sigma_s, coeff.sigma_t);
    float3 channel_pdf;
    int channel = volume_sample_channel(albedo, *throughput, rphase, &channel_pdf);

    /* decide if we will hit or miss */
    bool scatter = true;
    float xi = path_state_rng_1D(kg, state, PRNG_SCATTER_DISTANCE);

    if (probalistic_scatter) {
      float sample_sigma_t = volume_channel_get(coeff.sigma_t, channel);
      float sample_transmittance = expf(-sample_sigma_t * t);

      if (1.0f - xi >= sample_transmittance) {
        scatter = true;

        /* rescale random number so we can reuse it */
        xi = 1.0f - (1.0f - xi - sample_transmittance) / (1.0f - sample_transmittance);
      }
      else
        scatter = false;
    }

    if (scatter) {
      /* scattering */
      float3 pdf;
      float3 transmittance;
      float sample_t;

      /* distance sampling */
      sample_t = kernel_volume_distance_sample(
          ray->t, coeff.sigma_t, channel, xi, &transmittance, &pdf);

      /* modify pdf for hit/miss decision */
      if (probalistic_scatter)
        pdf *= one_float3() - volume_color_transmittance(coeff.sigma_t, t);

      new_tp = *throughput * coeff.sigma_s * transmittance / dot(channel_pdf, pdf);
      t = sample_t;
    }
    else {
      /* no scattering */
      float3 transmittance = volume_color_transmittance(coeff.sigma_t, t);
      float pdf = dot(channel_pdf, transmittance);
      new_tp = *throughput * transmittance / pdf;
    }
  }
  else
#  endif
      if (closure_flag & SD_EXTINCTION) {
    /* absorption only, no sampling needed */
    float3 transmittance = volume_color_transmittance(coeff.sigma_t, t);
    new_tp = *throughput * transmittance;
  }
  else {
    new_tp = *throughput;
  }

  /* integrate emission attenuated by extinction */
  if (L && (closure_flag & SD_EMISSION)) {
    float3 transmittance = volume_color_transmittance(coeff.sigma_t, ray->t);
    float3 emission = kernel_volume_emission_integrate(
        &coeff, closure_flag, transmittance, ray->t);
    path_radiance_accum_emission(kg, L, state, *throughput, emission);
  }

  /* modify throughput */
  if (closure_flag & SD_EXTINCTION) {
    *throughput = new_tp;

    /* prepare to scatter to new direction */
    if (t < ray->t) {
      /* adjust throughput and move to new location */
      sd->P = ray->P + t * ray->D;

      return VOLUME_PATH_SCATTERED;
    }
  }

  return VOLUME_PATH_ATTENUATED;
}

/* heterogeneous volume distance sampling: integrate stepping through the
 * volume until we reach the end, get absorbed entirely, or run out of
 * iterations. this does probabilistically scatter or get transmitted through
 * for path tracing where we don't want to branch. */
ccl_device VolumeIntegrateResult
kernel_volume_integrate_heterogeneous_distance(const KernelGlobals *kg,
                                               ccl_addr_space PathState *state,
                                               Ray *ray,
                                               ShaderData *sd,
                                               PathRadiance *L,
                                               ccl_addr_space float3 *throughput,
                                               const float object_step_size)
{
  float3 tp = *throughput;

  /* Prepare for stepping.
   * Using a different step offset for the first step avoids banding artifacts. */
  int max_steps = kernel_data.integrator.volume_max_steps;
  float step_size, step_shade_offset, steps_offset;
  kernel_volume_step_init(
      kg, state, object_step_size, ray->t, &step_size, &step_shade_offset, &steps_offset);

  /* compute coefficients at the start */
  float t = 0.0f;
  float3 accum_transmittance = one_float3();

  /* pick random color channel, we use the Veach one-sample
   * model with balance heuristic for the channels */
  float xi = path_state_rng_1D(kg, state, PRNG_SCATTER_DISTANCE);
  float rphase = path_state_rng_1D(kg, state, PRNG_PHASE_CHANNEL);
  bool has_scatter = false;

  for (int i = 0; i < max_steps; i++) {
    /* advance to new position */
    float new_t = min(ray->t, (i + steps_offset) * step_size);
    float dt = new_t - t;

    float3 new_P = ray->P + ray->D * (t + dt * step_shade_offset);
    VolumeShaderCoefficients coeff ccl_optional_struct_init;

    /* compute segment */
    if (volume_shader_sample(kg, sd, state, new_P, &coeff)) {
      int closure_flag = sd->flag;
      float3 new_tp;
      float3 transmittance;
      bool scatter = false;

      /* distance sampling */
#  ifdef __VOLUME_SCATTER__
      if ((closure_flag & SD_SCATTER) || (has_scatter && (closure_flag & SD_EXTINCTION))) {
        has_scatter = true;

        /* Sample channel, use MIS with balance heuristic. */
        float3 albedo = safe_divide_color(coeff.sigma_s, coeff.sigma_t);
        float3 channel_pdf;
        int channel = volume_sample_channel(albedo, tp, rphase, &channel_pdf);

        /* compute transmittance over full step */
        transmittance = volume_color_transmittance(coeff.sigma_t, dt);

        /* decide if we will scatter or continue */
        float sample_transmittance = volume_channel_get(transmittance, channel);

        if (1.0f - xi >= sample_transmittance) {
          /* compute sampling distance */
          float sample_sigma_t = volume_channel_get(coeff.sigma_t, channel);
          float new_dt = -logf(1.0f - xi) / sample_sigma_t;
          new_t = t + new_dt;

          /* transmittance and pdf */
          float3 new_transmittance = volume_color_transmittance(coeff.sigma_t, new_dt);
          float3 pdf = coeff.sigma_t * new_transmittance;

          /* throughput */
          new_tp = tp * coeff.sigma_s * new_transmittance / dot(channel_pdf, pdf);
          scatter = true;
        }
        else {
          /* throughput */
          float pdf = dot(channel_pdf, transmittance);
          new_tp = tp * transmittance / pdf;

          /* remap xi so we can reuse it and keep thing stratified */
          xi = 1.0f - (1.0f - xi) / sample_transmittance;
        }
      }
      else
#  endif
          if (closure_flag & SD_EXTINCTION) {
        /* absorption only, no sampling needed */
        transmittance = volume_color_transmittance(coeff.sigma_t, dt);
        new_tp = tp * transmittance;
      }
      else {
        transmittance = zero_float3();
        new_tp = tp;
      }

      /* integrate emission attenuated by absorption */
      if (L && (closure_flag & SD_EMISSION)) {
        float3 emission = kernel_volume_emission_integrate(
            &coeff, closure_flag, transmittance, dt);
        path_radiance_accum_emission(kg, L, state, tp, emission);
      }

      /* modify throughput */
      if (closure_flag & SD_EXTINCTION) {
        tp = new_tp;

        /* stop if nearly all light blocked */
        if (tp.x < VOLUME_THROUGHPUT_EPSILON && tp.y < VOLUME_THROUGHPUT_EPSILON &&
            tp.z < VOLUME_THROUGHPUT_EPSILON) {
          tp = zero_float3();
          break;
        }
      }

      /* prepare to scatter to new direction */
      if (scatter) {
        /* adjust throughput and move to new location */
        sd->P = ray->P + new_t * ray->D;
        *throughput = tp;

        return VOLUME_PATH_SCATTERED;
      }
      else {
        /* accumulate transmittance */
        accum_transmittance *= transmittance;
      }
    }

    /* stop if at the end of the volume */
    t = new_t;
    if (t == ray->t)
      break;
  }

  *throughput = tp;

  return VOLUME_PATH_ATTENUATED;
}

/* get the volume attenuation and emission over line segment defined by
 * ray, with the assumption that there are no surfaces blocking light
 * between the endpoints. distance sampling is used to decide if we will
 * scatter or not. */
ccl_device_noinline_cpu VolumeIntegrateResult
kernel_volume_integrate(const KernelGlobals *kg,
                        ccl_addr_space PathState *state,
                        ShaderData *sd,
                        Ray *ray,
                        PathRadiance *L,
                        ccl_addr_space float3 *throughput,
                        float step_size)
{
  shader_setup_from_volume(kg, sd, ray);

  if (step_size != FLT_MAX)
    return kernel_volume_integrate_heterogeneous_distance(
        kg, state, ray, sd, L, throughput, step_size);
  else
    return kernel_volume_integrate_homogeneous(kg, state, ray, sd, L, throughput, true);
}

#  ifndef __SPLIT_KERNEL__
/* Decoupled Volume Sampling
 *
 * VolumeSegment is list of coefficients and transmittance stored at all steps
 * through a volume. This can then later be used for decoupled sampling as in:
 * "Importance Sampling Techniques for Path Tracing in Participating Media"
 *
 * On the GPU this is only supported (but currently not enabled)
 * for homogeneous volumes (1 step), due to
 * no support for malloc/free and too much stack usage with a fix size array. */

typedef struct VolumeStep {
  float3 sigma_s;             /* scatter coefficient */
  float3 sigma_t;             /* extinction coefficient */
  float3 accum_transmittance; /* accumulated transmittance including this step */
  float3 cdf_distance;        /* cumulative density function for distance sampling */
  float t;                    /* distance at end of this step */
  float shade_t;              /* jittered distance where shading was done in step */
  int closure_flag;           /* shader evaluation closure flags */
} VolumeStep;

typedef struct VolumeSegment {
  VolumeStep stack_step; /* stack storage for homogeneous step, to avoid malloc */
  VolumeStep *steps;     /* recorded steps */
  int numsteps;          /* number of steps */
  int closure_flag;      /* accumulated closure flags from all steps */

  float3 accum_emission;      /* accumulated emission at end of segment */
  float3 accum_transmittance; /* accumulated transmittance at end of segment */
  float3 accum_albedo;        /* accumulated average albedo over segment */

  int sampling_method; /* volume sampling method */
} VolumeSegment;

/* record volume steps to the end of the volume.
 *
 * it would be nice if we could only record up to the point that we need to scatter,
 * but the entire segment is needed to do always scattering, rather than probabilistically
 * hitting or missing the volume. if we don't know the transmittance at the end of the
 * volume we can't generate stratified distance samples up to that transmittance */
#    ifdef __VOLUME_DECOUPLED__
ccl_device void kernel_volume_decoupled_record(const KernelGlobals *kg,
                                               PathState *state,
                                               Ray *ray,
                                               ShaderData *sd,
                                               VolumeSegment *segment,
                                               const float object_step_size)
{
  /* prepare for volume stepping */
  int max_steps;
  float step_size, step_shade_offset, steps_offset;

  if (object_step_size != FLT_MAX) {
    max_steps = kernel_data.integrator.volume_max_steps;
    kernel_volume_step_init(
        kg, state, object_step_size, ray->t, &step_size, &step_shade_offset, &steps_offset);

#      ifdef __KERNEL_CPU__
    /* NOTE: For the branched path tracing it's possible to have direct
     * and indirect light integration both having volume segments allocated.
     * We detect this using index in the pre-allocated memory. Currently we
     * only support two segments allocated at a time, if more needed some
     * modifications to the const KernelGlobals will be needed.
     *
     * This gives us restrictions that decoupled record should only happen
     * in the stack manner, meaning if there's subsequent call of decoupled
     * record it'll need to free memory before its caller frees memory.
     */
    const int index = kg->decoupled_volume_steps_index;
    assert(index < sizeof(kg->decoupled_volume_steps) / sizeof(*kg->decoupled_volume_steps));
    if (kg->decoupled_volume_steps[index] == NULL) {
      kg->decoupled_volume_steps[index] = (VolumeStep *)malloc(sizeof(VolumeStep) * max_steps);
    }
    segment->steps = kg->decoupled_volume_steps[index];
    ++kg->decoupled_volume_steps_index;
#      else
    segment->steps = (VolumeStep *)malloc(sizeof(VolumeStep) * max_steps);
#      endif
  }
  else {
    max_steps = 1;
    step_size = ray->t;
    step_shade_offset = 0.0f;
    steps_offset = 1.0f;
    segment->steps = &segment->stack_step;
  }

  /* init accumulation variables */
  float3 accum_emission = zero_float3();
  float3 accum_transmittance = one_float3();
  float3 accum_albedo = zero_float3();
  float3 cdf_distance = zero_float3();
  float t = 0.0f;

  segment->numsteps = 0;
  segment->closure_flag = 0;
  bool is_last_step_empty = false;

  VolumeStep *step = segment->steps;

  for (int i = 0; i < max_steps; i++, step++) {
    /* advance to new position */
    float new_t = min(ray->t, (i + steps_offset) * step_size);
    float dt = new_t - t;

    float3 new_P = ray->P + ray->D * (t + dt * step_shade_offset);
    VolumeShaderCoefficients coeff ccl_optional_struct_init;

    /* compute segment */
    if (volume_shader_sample(kg, sd, state, new_P, &coeff)) {
      int closure_flag = sd->flag;
      float3 sigma_t = coeff.sigma_t;

      /* compute average albedo for channel sampling */
      if (closure_flag & SD_SCATTER) {
        accum_albedo += (dt / ray->t) * safe_divide_color(coeff.sigma_s, sigma_t);
      }

      /* compute accumulated transmittance */
      float3 transmittance = volume_color_transmittance(sigma_t, dt);

      /* compute emission attenuated by absorption */
      if (closure_flag & SD_EMISSION) {
        float3 emission = kernel_volume_emission_integrate(
            &coeff, closure_flag, transmittance, dt);
        accum_emission += accum_transmittance * emission;
      }

      accum_transmittance *= transmittance;

      /* compute pdf for distance sampling */
      float3 pdf_distance = dt * accum_transmittance * coeff.sigma_s;
      cdf_distance = cdf_distance + pdf_distance;

      /* write step data */
      step->sigma_t = sigma_t;
      step->sigma_s = coeff.sigma_s;
      step->closure_flag = closure_flag;

      segment->closure_flag |= closure_flag;

      is_last_step_empty = false;
      segment->numsteps++;
    }
    else {
      if (is_last_step_empty) {
        /* consecutive empty step, merge */
        step--;
      }
      else {
        /* store empty step */
        step->sigma_t = zero_float3();
        step->sigma_s = zero_float3();
        step->closure_flag = 0;

        segment->numsteps++;
        is_last_step_empty = true;
      }
    }

    step->accum_transmittance = accum_transmittance;
    step->cdf_distance = cdf_distance;
    step->t = new_t;
    step->shade_t = t + dt * step_shade_offset;

    /* stop if at the end of the volume */
    t = new_t;
    if (t == ray->t)
      break;

    /* stop if nearly all light blocked */
    if (accum_transmittance.x < VOLUME_THROUGHPUT_EPSILON &&
        accum_transmittance.y < VOLUME_THROUGHPUT_EPSILON &&
        accum_transmittance.z < VOLUME_THROUGHPUT_EPSILON)
      break;
  }

  /* store total emission and transmittance */
  segment->accum_emission = accum_emission;
  segment->accum_transmittance = accum_transmittance;
  segment->accum_albedo = accum_albedo;

  /* normalize cumulative density function for distance sampling */
  VolumeStep *last_step = segment->steps + segment->numsteps - 1;

  if (!is_zero(last_step->cdf_distance)) {
    VolumeStep *step = &segment->steps[0];
    int numsteps = segment->numsteps;
    float3 inv_cdf_distance_sum = safe_invert_color(last_step->cdf_distance);

    for (int i = 0; i < numsteps; i++, step++)
      step->cdf_distance *= inv_cdf_distance_sum;
  }
}

ccl_device void kernel_volume_decoupled_free(const KernelGlobals *kg, VolumeSegment *segment)
{
  if (segment->steps != &segment->stack_step) {
#      ifdef __KERNEL_CPU__
    /* NOTE: We only allow free last allocated segment.
     * No random order of alloc/free is supported.
     */
    assert(kg->decoupled_volume_steps_index > 0);
    assert(segment->steps == kg->decoupled_volume_steps[kg->decoupled_volume_steps_index - 1]);
    --kg->decoupled_volume_steps_index;
#      else
    free(segment->steps);
#      endif
  }
}
#    endif /* __VOLUME_DECOUPLED__ */

/* scattering for homogeneous and heterogeneous volumes, using decoupled ray
 * marching.
 *
 * function is expected to return VOLUME_PATH_SCATTERED when probalistic_scatter is false */
ccl_device VolumeIntegrateResult kernel_volume_decoupled_scatter(const KernelGlobals *kg,
                                                                 PathState *state,
                                                                 Ray *ray,
                                                                 ShaderData *sd,
                                                                 float3 *throughput,
                                                                 float rphase,
                                                                 float rscatter,
                                                                 const VolumeSegment *segment,
                                                                 const float3 *light_P,
                                                                 bool probalistic_scatter)
{
  kernel_assert(segment->closure_flag & SD_SCATTER);

  /* Sample color channel, use MIS with balance heuristic. */
  float3 channel_pdf;
  int channel = volume_sample_channel(segment->accum_albedo, *throughput, rphase, &channel_pdf);

  float xi = rscatter;

  /* probabilistic scattering decision based on transmittance */
  if (probalistic_scatter) {
    float sample_transmittance = volume_channel_get(segment->accum_transmittance, channel);

    if (1.0f - xi >= sample_transmittance) {
      /* rescale random number so we can reuse it */
      xi = 1.0f - (1.0f - xi - sample_transmittance) / (1.0f - sample_transmittance);
    }
    else {
      *throughput /= sample_transmittance;
      return VOLUME_PATH_MISSED;
    }
  }

  VolumeStep *step;
  float3 transmittance;
  float pdf, sample_t;
  float mis_weight = 1.0f;
  bool distance_sample = true;
  bool use_mis = false;

  if (segment->sampling_method && light_P) {
    if (segment->sampling_method == SD_VOLUME_MIS) {
      /* multiple importance sample: randomly pick between
       * equiangular and distance sampling strategy */
      if (xi < 0.5f) {
        xi *= 2.0f;
      }
      else {
        xi = (xi - 0.5f) * 2.0f;
        distance_sample = false;
      }

      use_mis = true;
    }
    else {
      /* only equiangular sampling */
      distance_sample = false;
    }
  }

  /* distance sampling */
  if (distance_sample) {
    /* find step in cdf */
    step = segment->steps;

    float prev_t = 0.0f;
    float3 step_pdf_distance = one_float3();

    if (segment->numsteps > 1) {
      float prev_cdf = 0.0f;
      float step_cdf = 1.0f;
      float3 prev_cdf_distance = zero_float3();

      for (int i = 0;; i++, step++) {
        /* todo: optimize using binary search */
        step_cdf = volume_channel_get(step->cdf_distance, channel);

        if (xi < step_cdf || i == segment->numsteps - 1)
          break;

        prev_cdf = step_cdf;
        prev_t = step->t;
        prev_cdf_distance = step->cdf_distance;
      }

      /* remap xi so we can reuse it */
      xi = (xi - prev_cdf) / (step_cdf - prev_cdf);

      /* pdf for picking step */
      step_pdf_distance = step->cdf_distance - prev_cdf_distance;
    }

    /* determine range in which we will sample */
    float step_t = step->t - prev_t;

    /* sample distance and compute transmittance */
    float3 distance_pdf;
    sample_t = prev_t + kernel_volume_distance_sample(
                            step_t, step->sigma_t, channel, xi, &transmittance, &distance_pdf);

    /* modify pdf for hit/miss decision */
    if (probalistic_scatter)
      distance_pdf *= one_float3() - segment->accum_transmittance;

    pdf = dot(channel_pdf, distance_pdf * step_pdf_distance);

    /* multiple importance sampling */
    if (use_mis) {
      float equi_pdf = kernel_volume_equiangular_pdf(ray, *light_P, sample_t);
      mis_weight = 2.0f * power_heuristic(pdf, equi_pdf);
    }
  }
  /* equi-angular sampling */
  else {
    /* sample distance */
    sample_t = kernel_volume_equiangular_sample(ray, *light_P, xi, &pdf);

    /* find step in which sampled distance is located */
    step = segment->steps;

    float prev_t = 0.0f;
    float3 step_pdf_distance = one_float3();

    if (segment->numsteps > 1) {
      float3 prev_cdf_distance = zero_float3();

      int numsteps = segment->numsteps;
      int high = numsteps - 1;
      int low = 0;
      int mid;

      while (low < high) {
        mid = (low + high) >> 1;

        if (sample_t < step[mid].t)
          high = mid;
        else if (sample_t >= step[mid + 1].t)
          low = mid + 1;
        else {
          /* found our interval in step[mid] .. step[mid+1] */
          prev_t = step[mid].t;
          prev_cdf_distance = step[mid].cdf_distance;
          step += mid + 1;
          break;
        }
      }

      if (low >= numsteps - 1) {
        prev_t = step[numsteps - 1].t;
        prev_cdf_distance = step[numsteps - 1].cdf_distance;
        step += numsteps - 1;
      }

      /* pdf for picking step with distance sampling */
      step_pdf_distance = step->cdf_distance - prev_cdf_distance;
    }

    /* determine range in which we will sample */
    float step_t = step->t - prev_t;
    float step_sample_t = sample_t - prev_t;

    /* compute transmittance */
    transmittance = volume_color_transmittance(step->sigma_t, step_sample_t);

    /* multiple importance sampling */
    if (use_mis) {
      float3 distance_pdf3 = kernel_volume_distance_pdf(step_t, step->sigma_t, step_sample_t);
      float distance_pdf = dot(channel_pdf, distance_pdf3 * step_pdf_distance);
      mis_weight = 2.0f * power_heuristic(pdf, distance_pdf);
    }
  }
  if (sample_t < 0.0f || pdf == 0.0f) {
    return VOLUME_PATH_MISSED;
  }

  /* compute transmittance up to this step */
  if (step != segment->steps)
    transmittance *= (step - 1)->accum_transmittance;

  /* modify throughput */
  *throughput *= step->sigma_s * transmittance * (mis_weight / pdf);

  /* evaluate shader to create closures at shading point */
  if (segment->numsteps > 1) {
    sd->P = ray->P + step->shade_t * ray->D;

    VolumeShaderCoefficients coeff;
    volume_shader_sample(kg, sd, state, sd->P, &coeff);
  }

  /* move to new position */
  sd->P = ray->P + sample_t * ray->D;

  return VOLUME_PATH_SCATTERED;
}
#  endif /* __SPLIT_KERNEL */

/* decide if we need to use decoupled or not */
ccl_device bool kernel_volume_use_decoupled(const KernelGlobals *kg,
                                            bool heterogeneous,
                                            bool direct,
                                            int sampling_method)
{
  /* decoupled ray marching for heterogeneous volumes not supported on the GPU,
   * which also means equiangular and multiple importance sampling is not
   * support for that case */
  if (!kernel_data.integrator.volume_decoupled)
    return false;

#  ifdef __KERNEL_GPU__
  if (heterogeneous)
    return false;
#  endif

  /* equiangular and multiple importance sampling only implemented for decoupled */
  if (sampling_method != 0)
    return true;

  /* for all light sampling use decoupled, reusing shader evaluations is
   * typically faster in that case */
  if (direct)
    return kernel_data.integrator.sample_all_lights_direct;
  else
    return kernel_data.integrator.sample_all_lights_indirect;
}

#endif /* __VOLUME__ */

CCL_NAMESPACE_END
