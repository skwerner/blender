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
} Bssrdf;

static_assert(sizeof(ShaderClosure) >= sizeof(Bssrdf), "Bssrdf is too large!");

/* Scale mean free path length so it gives similar looking result to older
 * to Cubic, Gaussian and Burley models.
 */
ccl_device_inline float3 bssrdf_burley_compatible_mfp(float3 r)
{
  return 0.25f * M_1_PI_F * r;
}

/* Setup */

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
    bssrdf->sample_weight = fabsf(average(bssrdf->weight)) * bssrdf_channels;

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

CCL_NAMESPACE_END
