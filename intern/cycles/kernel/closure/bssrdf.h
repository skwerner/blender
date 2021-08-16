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

typedef ccl_addr_space struct Bssrdf {
  SHADER_CLOSURE_BASE;

  float3 radius;
  float3 albedo;
  float roughness;
  float channels;
} Bssrdf;

static_assert(sizeof(ShaderClosure) >= sizeof(Bssrdf), "Bssrdf is too large!");

/* Approximate Reflectance Profiles
 * http://graphics.pixar.com/library/ApproxBSSRDF/paper.pdf
 */

/* This is a bit arbitrary, just need big enough radius so it matches
 * the mean free length, but still not too big so sampling is still
 * effective. Might need some further tweaks.
 */
#define BURLEY_TRUNCATE 16.0f
#define BURLEY_TRUNCATE_CDF 0.9963790093708328f  // cdf(BURLEY_TRUNCATE)

ccl_device_inline float bssrdf_burley_fitting(float A)
{
  /* Diffuse surface transmission, equation (6). */
  return 1.9f - A + 3.5f * (A - 0.8f) * (A - 0.8f);
}

/* Scale mean free path length so it gives similar looking result
 * to Cubic and Gaussian models.
 */
ccl_device_inline float3 bssrdf_burley_compatible_mfp(float3 r)
{
  return 0.25f * M_1_PI_F * r;
}

ccl_device float bssrdf_burley_eval(const float d, float r)
{
  const float Rm = BURLEY_TRUNCATE * d;

  if (r >= Rm)
    return 0.0f;

  /* Burley reflectance profile, equation (3).
   *
   * NOTES:
   * - Surface albedo is already included into sc->weight, no need to
   *   multiply by this term here.
   * - This is normalized diffuse model, so the equation is multiplied
   *   by 2*pi, which also matches cdf().
   */
  float exp_r_3_d = expf(-r / (3.0f * d));
  float exp_r_d = exp_r_3_d * exp_r_3_d * exp_r_3_d;
  return (exp_r_d + exp_r_3_d) / (4.0f * d);
}

ccl_device float bssrdf_burley_pdf(const float d, float r)
{
  return bssrdf_burley_eval(d, r) * (1.0f / BURLEY_TRUNCATE_CDF);
}

/* Find the radius for desired CDF value.
 * Returns scaled radius, meaning the result is to be scaled up by d.
 * Since there's no closed form solution we do Newton-Raphson method to find it.
 */
ccl_device_forceinline float bssrdf_burley_root_find(float xi)
{
  const float tolerance = 1e-6f;
  const int max_iteration_count = 10;
  /* Do initial guess based on manual curve fitting, this allows us to reduce
   * number of iterations to maximum 4 across the [0..1] range. We keep maximum
   * number of iteration higher just to be sure we didn't miss root in some
   * corner case.
   */
  float r;
  if (xi <= 0.9f) {
    r = expf(xi * xi * 2.4f) - 1.0f;
  }
  else {
    /* TODO(sergey): Some nicer curve fit is possible here. */
    r = 15.0f;
  }
  /* Solve against scaled radius. */
  for (int i = 0; i < max_iteration_count; i++) {
    float exp_r_3 = expf(-r / 3.0f);
    float exp_r = exp_r_3 * exp_r_3 * exp_r_3;
    float f = 1.0f - 0.25f * exp_r - 0.75f * exp_r_3 - xi;
    float f_ = 0.25f * exp_r + 0.25f * exp_r_3;

    if (fabsf(f) < tolerance || f_ == 0.0f) {
      break;
    }

    r = r - f / f_;
    if (r < 0.0f) {
      r = 0.0f;
    }
  }
  return r;
}

ccl_device void bssrdf_burley_sample(const float d, float xi, float *r, float *h)
{
  const float Rm = BURLEY_TRUNCATE * d;
  const float r_ = bssrdf_burley_root_find(xi * BURLEY_TRUNCATE_CDF) * d;

  *r = r_;

  /* h^2 + r^2 = Rm^2 */
  *h = safe_sqrtf(Rm * Rm - r_ * r_);
}

/* None BSSRDF falloff
 *
 * Samples distributed over disk with no falloff, for reference. */

ccl_device float bssrdf_none_eval(const float radius, float r)
{
  const float Rm = radius;
  return (r < Rm) ? 1.0f : 0.0f;
}

ccl_device float bssrdf_none_pdf(const float radius, float r)
{
  /* integrate (2*pi*r)/(pi*Rm*Rm) from 0 to Rm = 1 */
  const float Rm = radius;
  const float area = (M_PI_F * Rm * Rm);

  return bssrdf_none_eval(radius, r) / area;
}

ccl_device void bssrdf_none_sample(const float radius, float xi, float *r, float *h)
{
  /* xi = integrate (2*pi*r)/(pi*Rm*Rm) = r^2/Rm^2
   * r = sqrt(xi)*Rm */
  const float Rm = radius;
  const float r_ = sqrtf(xi) * Rm;

  *r = r_;

  /* h^2 + r^2 = Rm^2 */
  *h = safe_sqrtf(Rm * Rm - r_ * r_);
}

/* Generic */

ccl_device_inline Bssrdf *bssrdf_alloc(ShaderData *sd, float3 weight)
{
  Bssrdf *bssrdf = (Bssrdf *)closure_alloc(sd, sizeof(Bssrdf), CLOSURE_NONE_ID, weight);

  if (bssrdf == NULL) {
    return NULL;
  }

  float sample_weight = fabsf(average(weight));
  bssrdf->sample_weight = sample_weight;
  return (sample_weight >= CLOSURE_WEIGHT_CUTOFF) ? bssrdf : NULL;
}

ccl_device int bssrdf_setup(ShaderData *sd, Bssrdf *bssrdf, ClosureType type)
{
  int flag = 0;
  int bssrdf_channels = 3;
  float3 diffuse_weight = make_float3(0.0f, 0.0f, 0.0f);

  /* Verify if the radii are large enough to sample without precision issues. */
  if (bssrdf->radius.x < BSSRDF_MIN_RADIUS) {
    diffuse_weight.x = bssrdf->weight.x;
    bssrdf->weight.x = 0.0f;
    bssrdf->radius.x = 0.0f;
    bssrdf_channels--;
  }
  if (bssrdf->radius.y < BSSRDF_MIN_RADIUS) {
    diffuse_weight.y = bssrdf->weight.y;
    bssrdf->weight.y = 0.0f;
    bssrdf->radius.y = 0.0f;
    bssrdf_channels--;
  }
  if (bssrdf->radius.z < BSSRDF_MIN_RADIUS) {
    diffuse_weight.z = bssrdf->weight.z;
    bssrdf->weight.z = 0.0f;
    bssrdf->radius.z = 0.0f;
    bssrdf_channels--;
  }

  if (bssrdf_channels < 3) {
    /* Add diffuse BSDF if any radius too small. */
#ifdef __PRINCIPLED__
    if (bssrdf->roughness != FLT_MAX) {
      float roughness = bssrdf->roughness;
      float3 N = bssrdf->N;

      PrincipledDiffuseBsdf *bsdf = (PrincipledDiffuseBsdf *)bsdf_alloc(
          sd, sizeof(PrincipledDiffuseBsdf), diffuse_weight);

      if (bsdf) {
        bsdf->type = CLOSURE_BSDF_BSSRDF_PRINCIPLED_ID;
        bsdf->N = N;
        bsdf->roughness = roughness;
        flag |= bsdf_principled_diffuse_setup(bsdf);
      }
    }
    else
#endif /* __PRINCIPLED__ */
    {
      DiffuseBsdf *bsdf = (DiffuseBsdf *)bsdf_alloc(sd, sizeof(DiffuseBsdf), diffuse_weight);

      if (bsdf) {
        bsdf->type = CLOSURE_BSDF_BSSRDF_ID;
        bsdf->N = bssrdf->N;
        flag |= bsdf_diffuse_setup(bsdf);
      }
    }
  }

  /* Setup BSSRDF if radius is large enough. */
  if (bssrdf_channels > 0) {
    bssrdf->type = type;
    bssrdf->channels = bssrdf_channels;
    bssrdf->sample_weight = fabsf(average(bssrdf->weight)) * bssrdf->channels;

    /* Mean free path length. */
    bssrdf->radius = bssrdf_burley_compatible_mfp(bssrdf->radius);

    flag |= SD_BSSRDF;
  }
  else {
    bssrdf->type = type;
    bssrdf->sample_weight = 0.0f;
  }

  return flag;
}

ccl_device void bssrdf_sample(const ShaderClosure *sc, float xi, float *r, float *h)
{
  const Bssrdf *bssrdf = (const Bssrdf *)sc;
  float radius;

  /* Sample color channel and reuse random number. Only a subset of channels
   * may be used if their radius was too small to handle as BSSRDF. */
  xi *= bssrdf->channels;

  if (xi < 1.0f) {
    radius = (bssrdf->radius.x > 0.0f) ? bssrdf->radius.x :
             (bssrdf->radius.y > 0.0f) ? bssrdf->radius.y :
                                         bssrdf->radius.z;
  }
  else if (xi < 2.0f) {
    xi -= 1.0f;
    radius = (bssrdf->radius.x > 0.0f && bssrdf->radius.y > 0.0f) ? bssrdf->radius.y :
                                                                    bssrdf->radius.z;
  }
  else {
    xi -= 2.0f;
    radius = bssrdf->radius.z;
  }

  /* Sample BSSRDF. */
  bssrdf_burley_sample(radius, xi, r, h);
}

ccl_device float bssrdf_channel_pdf(const Bssrdf *bssrdf, float radius, float r)
{
  if (radius == 0.0f) {
    return 0.0f;
  }
  return bssrdf_burley_pdf(radius, r);
}

ccl_device_forceinline float3 bssrdf_eval(const ShaderClosure *sc, float r)
{
  const Bssrdf *bssrdf = (const Bssrdf *)sc;

  return make_float3(bssrdf_channel_pdf(bssrdf, bssrdf->radius.x, r),
                     bssrdf_channel_pdf(bssrdf, bssrdf->radius.y, r),
                     bssrdf_channel_pdf(bssrdf, bssrdf->radius.z, r));
}

ccl_device_forceinline float bssrdf_pdf(const ShaderClosure *sc, float r)
{
  const Bssrdf *bssrdf = (const Bssrdf *)sc;
  float3 pdf = bssrdf_eval(sc, r);

  return (pdf.x + pdf.y + pdf.z) / bssrdf->channels;
}

CCL_NAMESPACE_END
