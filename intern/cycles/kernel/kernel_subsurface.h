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

/* BSSRDF using disk based importance sampling.
 *
 * BSSRDF Importance Sampling, SIGGRAPH 2013
 * http://library.imageworks.com/pdfs/imageworks-library-BSSRDF-sampling.pdf
 */

ccl_device_inline float3
subsurface_scatter_eval(ShaderData *sd, const ShaderClosure *sc, float disk_r, float r, bool all)
{
  /* This is the Veach one-sample model with balance heuristic, some pdf
   * factors drop out when using balance heuristic weighting. For branched
   * path tracing (all) we sample all closure and don't use MIS. */
  float3 eval_sum = zero_float3();
  float pdf_sum = 0.0f;
  float sample_weight_inv = 0.0f;

  if (!all) {
    float sample_weight_sum = 0.0f;

    for (int i = 0; i < sd->num_closure; i++) {
      sc = &sd->closure[i];

      if (CLOSURE_IS_DISK_BSSRDF(sc->type)) {
        sample_weight_sum += sc->sample_weight;
      }
    }

    sample_weight_inv = 1.0f / sample_weight_sum;
  }

  for (int i = 0; i < sd->num_closure; i++) {
    sc = &sd->closure[i];

    if (CLOSURE_IS_DISK_BSSRDF(sc->type)) {
      /* we pick one BSSRDF, so adjust pdf for that */
      float sample_weight = (all) ? 1.0f : sc->sample_weight * sample_weight_inv;

      /* compute pdf */
      float3 eval = bssrdf_eval(sc, r);
      float pdf = bssrdf_pdf(sc, disk_r);

      eval_sum += sc->weight * eval;
      pdf_sum += sample_weight * pdf;
    }
  }

  return (pdf_sum > 0.0f) ? eval_sum / pdf_sum : zero_float3();
}

/* Subsurface scattering step, from a point on the surface to other
 * nearby points on the same object.
 */
ccl_device_inline int subsurface_scatter_disk(const KernelGlobals *kg,
                                              LocalIntersection *ss_isect,
                                              ShaderData *sd,
                                              const ShaderClosure *sc,
                                              uint *lcg_state,
                                              float disk_u,
                                              float disk_v,
                                              bool all)
{
  /* pick random axis in local frame and point on disk */
  float3 disk_N, disk_T, disk_B;
  float pick_pdf_N, pick_pdf_T, pick_pdf_B;

  disk_N = sd->Ng;
  make_orthonormals(disk_N, &disk_T, &disk_B);

  if (disk_v < 0.5f) {
    pick_pdf_N = 0.5f;
    pick_pdf_T = 0.25f;
    pick_pdf_B = 0.25f;
    disk_v *= 2.0f;
  }
  else if (disk_v < 0.75f) {
    float3 tmp = disk_N;
    disk_N = disk_T;
    disk_T = tmp;
    pick_pdf_N = 0.25f;
    pick_pdf_T = 0.5f;
    pick_pdf_B = 0.25f;
    disk_v = (disk_v - 0.5f) * 4.0f;
  }
  else {
    float3 tmp = disk_N;
    disk_N = disk_B;
    disk_B = tmp;
    pick_pdf_N = 0.25f;
    pick_pdf_T = 0.25f;
    pick_pdf_B = 0.5f;
    disk_v = (disk_v - 0.75f) * 4.0f;
  }

  /* sample point on disk */
  float phi = M_2PI_F * disk_v;
  float disk_height, disk_r;

  bssrdf_sample(sc, disk_u, &disk_r, &disk_height);

  float3 disk_P = (disk_r * cosf(phi)) * disk_T + (disk_r * sinf(phi)) * disk_B;

  /* create ray */
  Ray *ray = &ss_isect->ray;
  ray->P = sd->P + disk_N * disk_height + disk_P;
  ray->D = -disk_N;
  ray->t = 2.0f * disk_height;
  ray->dP = sd->dP;
  ray->dD = differential3_zero();
  ray->time = sd->time;

  /* intersect with the same object. if multiple intersections are found it
   * will use at most BSSRDF_MAX_HITS hits, a random subset of all hits */
  scene_intersect_local(kg, ray, ss_isect, sd->object, lcg_state, BSSRDF_MAX_HITS);
  int num_eval_hits = min(ss_isect->num_hits, BSSRDF_MAX_HITS);

  for (int hit = 0; hit < num_eval_hits; hit++) {
    /* Quickly retrieve P and Ng without setting up ShaderData. */
    float3 hit_P;
    if (sd->type & PRIMITIVE_TRIANGLE) {
      hit_P = triangle_refine_local(kg, sd, &ss_isect->hits[hit], ray);
    }
#ifdef __OBJECT_MOTION__
    else if (sd->type & PRIMITIVE_MOTION_TRIANGLE) {
      float3 verts[3];
      motion_triangle_vertices(kg,
                               sd->object,
                               kernel_tex_fetch(__prim_index, ss_isect->hits[hit].prim),
                               sd->time,
                               verts);
      hit_P = motion_triangle_refine_local(kg, sd, &ss_isect->hits[hit], ray, verts);
    }
#endif /* __OBJECT_MOTION__ */
    else {
      ss_isect->weight[hit] = zero_float3();
      continue;
    }

    float3 hit_Ng = ss_isect->Ng[hit];
    if (ss_isect->hits[hit].object != OBJECT_NONE) {
      object_normal_transform(kg, sd, &hit_Ng);
    }

    /* Probability densities for local frame axes. */
    float pdf_N = pick_pdf_N * fabsf(dot(disk_N, hit_Ng));
    float pdf_T = pick_pdf_T * fabsf(dot(disk_T, hit_Ng));
    float pdf_B = pick_pdf_B * fabsf(dot(disk_B, hit_Ng));

    /* Multiple importance sample between 3 axes, power heuristic
     * found to be slightly better than balance heuristic. pdf_N
     * in the MIS weight and denominator cancelled out. */
    float w = pdf_N / (sqr(pdf_N) + sqr(pdf_T) + sqr(pdf_B));
    if (ss_isect->num_hits > BSSRDF_MAX_HITS) {
      w *= ss_isect->num_hits / (float)BSSRDF_MAX_HITS;
    }

    /* Real distance to sampled point. */
    float r = len(hit_P - sd->P);

    /* Evaluate profiles. */
    float3 eval = subsurface_scatter_eval(sd, sc, disk_r, r, all) * w;

    ss_isect->weight[hit] = eval;
  }

  return num_eval_hits;
}

CCL_NAMESPACE_END
