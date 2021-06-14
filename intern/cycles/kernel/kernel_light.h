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

#include "kernel_light_background.h"

CCL_NAMESPACE_BEGIN

/* Light Sample result */

typedef struct LightSample {
  float3 P;       /* position on light, or direction for distant light */
  float3 Ng;      /* normal on light */
  float3 D;       /* direction from shading point to light */
  float t;        /* distance to light (FLT_MAX for distant light) */
  float u, v;     /* parametric coordinate on primitive */
  float pdf;      /* light sampling probability density function */
  float eval_fac; /* intensity multiplier */
  int object;     /* object id for triangle/curve lights */
  int prim;       /* primitive id for triangle/curve lights */
  int shader;     /* shader id */
  int lamp;       /* lamp id */
  LightType type; /* type of light */
} LightSample;

/* Updates the position and normal used to pick a light for direct lighting.
 *
 * The importance calculation for the light tree is different for scattering events on a surface
 * and in a volume. For volume events we need to know the point where the ray started and the point
 * it scattered. To do this we keep track of the ray direction and length. For surface events we
 * need the normal at the surface. In this case we set the ray length to -1 to mark that surface
 * importance sampling should be used.
 */
ccl_device void kernel_update_light_picking(ShaderData *sd, Ray *ray)
{
  if (ray) {
    sd->P_pick = ray->P;
    sd->V_pick = ray->D;
    sd->t_pick = ray->t;
  }
  else {
    sd->P_pick = sd->P;
    sd->V_pick = sd->N;
    sd->t_pick = -1.0f;
  }
}

/* Regular Light */

/* returns the PDF of sampling a point on this lamp */
ccl_device_inline bool lamp_light_sample(
    KernelGlobals *kg, int lamp, float randu, float randv, float3 P, LightSample *ls)
{
  const ccl_global KernelLight *klight = &kernel_tex_fetch(__lights, lamp);
  LightType type = (LightType)klight->type;
  ls->type = type;
  ls->shader = klight->shader_id;
  ls->object = PRIM_NONE;
  ls->prim = PRIM_NONE;
  ls->lamp = lamp;
  ls->u = randu;
  ls->v = randv;

  if (type == LIGHT_DISTANT) {
    /* distant light */
    float3 lightD = make_float3(klight->co[0], klight->co[1], klight->co[2]);
    float3 D = lightD;
    float radius = klight->distant.radius;
    float invarea = klight->distant.invarea;

    if (radius > 0.0f)
      D = distant_light_sample(D, radius, randu, randv);

    ls->P = D;
    ls->Ng = D;
    ls->D = -D;
    ls->t = FLT_MAX;

    float costheta = dot(lightD, D);
    ls->pdf = invarea / (costheta * costheta * costheta);
    ls->eval_fac = ls->pdf;
  }
#ifdef __BACKGROUND_MIS__
  else if (type == LIGHT_BACKGROUND) {
    /* infinite area light (e.g. light dome or env light) */
    float3 D = -background_light_sample(kg, P, randu, randv, &ls->pdf);

    ls->P = D;
    ls->Ng = D;
    ls->D = -D;
    ls->t = FLT_MAX;
    ls->eval_fac = 1.0f;
  }
#endif
  else {
    ls->P = make_float3(klight->co[0], klight->co[1], klight->co[2]);

    if (type == LIGHT_POINT || type == LIGHT_SPOT) {
      float radius = klight->spot.radius;

      if (radius > 0.0f)
        /* sphere light */
        ls->P += sphere_light_sample(P, ls->P, radius, randu, randv);

      ls->D = normalize_len(ls->P - P, &ls->t);
      ls->Ng = -ls->D;

      float invarea = klight->spot.invarea;
      ls->eval_fac = (0.25f * M_1_PI_F) * invarea;
      ls->pdf = invarea;

      if (type == LIGHT_SPOT) {
        /* spot light attenuation */
        float3 dir = make_float3(klight->spot.dir[0], klight->spot.dir[1], klight->spot.dir[2]);
        ls->eval_fac *= spot_light_attenuation(
            dir, klight->spot.spot_angle, klight->spot.spot_smooth, ls->Ng);
        if (ls->eval_fac == 0.0f) {
          return false;
        }
      }
      float2 uv = map_to_sphere(ls->Ng);
      ls->u = uv.x;
      ls->v = uv.y;

      ls->pdf *= lamp_light_pdf(kg, ls->Ng, -ls->D, ls->t);
    }
    else {
      /* area light */
      float3 axisu = make_float3(
          klight->area.axisu[0], klight->area.axisu[1], klight->area.axisu[2]);
      float3 axisv = make_float3(
          klight->area.axisv[0], klight->area.axisv[1], klight->area.axisv[2]);
      float3 Ng = make_float3(klight->area.dir[0], klight->area.dir[1], klight->area.dir[2]);
      float invarea = fabsf(klight->area.invarea);
      bool is_round = (klight->area.invarea < 0.0f);

      if (dot(ls->P - P, Ng) > 0.0f) {
        return false;
      }

      float3 inplane;

      if (is_round) {
        inplane = ellipse_sample(axisu * 0.5f, axisv * 0.5f, randu, randv);
        ls->P += inplane;
        ls->pdf = invarea;
      }
      else {
        inplane = ls->P;

        float3 sample_axisu = axisu;
        float3 sample_axisv = axisv;

        if (klight->area.tan_spread > 0.0f) {
          if (!light_spread_clamp_area_light(
                  P, Ng, &ls->P, &sample_axisu, &sample_axisv, klight->area.tan_spread)) {
            return false;
          }
        }

        ls->pdf = rect_light_sample(P, &ls->P, sample_axisu, sample_axisv, randu, randv, true);
        inplane = ls->P - inplane;
      }

      ls->u = dot(inplane, axisu) * (1.0f / dot(axisu, axisu)) + 0.5f;
      ls->v = dot(inplane, axisv) * (1.0f / dot(axisv, axisv)) + 0.5f;

      ls->Ng = Ng;
      ls->D = normalize_len(ls->P - P, &ls->t);

      ls->eval_fac = 0.25f * invarea;

      if (klight->area.tan_spread > 0.0f) {
        /* Area Light spread angle attenuation */
        ls->eval_fac *= light_spread_attenuation(
            ls->D, ls->Ng, klight->area.tan_spread, klight->area.normalize_spread);
      }

      if (is_round) {
        ls->pdf *= lamp_light_pdf(kg, Ng, -ls->D, ls->t);
      }
    }
  }

  return (ls->pdf > 0.0f);
}

ccl_device bool lamp_light_eval(
    KernelGlobals *kg, int lamp, float3 P, float3 D, float t, LightSample *ls)
{
  const ccl_global KernelLight *klight = &kernel_tex_fetch(__lights, lamp);
  LightType type = (LightType)klight->type;
  ls->type = type;
  ls->shader = klight->shader_id;
  ls->object = PRIM_NONE;
  ls->prim = PRIM_NONE;
  ls->lamp = lamp;
  /* todo: missing texture coordinates */
  ls->u = 0.0f;
  ls->v = 0.0f;

  if (!(ls->shader & SHADER_USE_MIS))
    return false;

  if (type == LIGHT_DISTANT) {
    /* distant light */
    float radius = klight->distant.radius;

    if (radius == 0.0f)
      return false;
    if (t != FLT_MAX)
      return false;

    /* a distant light is infinitely far away, but equivalent to a disk
     * shaped light exactly 1 unit away from the current shading point.
     *
     *     radius              t^2/cos(theta)
     *  <---------->           t = sqrt(1^2 + tan(theta)^2)
     *       tan(th)           area = radius*radius*pi
     *       <----->
     *        \    |           (1 + tan(theta)^2)/cos(theta)
     *         \   |           (1 + tan(acos(cos(theta)))^2)/cos(theta)
     *       t  \th| 1         simplifies to
     *           \-|           1/(cos(theta)^3)
     *            \|           magic!
     *             P
     */

    float3 lightD = make_float3(klight->co[0], klight->co[1], klight->co[2]);
    float costheta = dot(-lightD, D);
    float cosangle = klight->distant.cosangle;

    if (costheta < cosangle)
      return false;

    ls->P = -D;
    ls->Ng = -D;
    ls->D = D;
    ls->t = FLT_MAX;

    /* compute pdf */
    float invarea = klight->distant.invarea;
    ls->pdf = invarea / (costheta * costheta * costheta);
    ls->eval_fac = ls->pdf;
  }
  else if (type == LIGHT_POINT || type == LIGHT_SPOT) {
    float3 lightP = make_float3(klight->co[0], klight->co[1], klight->co[2]);

    float radius = klight->spot.radius;

    /* sphere light */
    if (radius == 0.0f)
      return false;

    if (!ray_aligned_disk_intersect(P, D, t, lightP, radius, &ls->P, &ls->t)) {
      return false;
    }

    ls->Ng = -D;
    ls->D = D;

    float invarea = klight->spot.invarea;
    ls->eval_fac = (0.25f * M_1_PI_F) * invarea;
    ls->pdf = invarea;

    if (type == LIGHT_SPOT) {
      /* spot light attenuation */
      float3 dir = make_float3(klight->spot.dir[0], klight->spot.dir[1], klight->spot.dir[2]);
      ls->eval_fac *= spot_light_attenuation(
          dir, klight->spot.spot_angle, klight->spot.spot_smooth, ls->Ng);

      if (ls->eval_fac == 0.0f)
        return false;
    }
    float2 uv = map_to_sphere(ls->Ng);
    ls->u = uv.x;
    ls->v = uv.y;

    /* compute pdf */
    if (ls->t != FLT_MAX)
      ls->pdf *= lamp_light_pdf(kg, ls->Ng, -ls->D, ls->t);
  }
  else if (type == LIGHT_AREA) {
    /* area light */
    float invarea = fabsf(klight->area.invarea);
    bool is_round = (klight->area.invarea < 0.0f);
    if (invarea == 0.0f)
      return false;

    float3 axisu = make_float3(
        klight->area.axisu[0], klight->area.axisu[1], klight->area.axisu[2]);
    float3 axisv = make_float3(
        klight->area.axisv[0], klight->area.axisv[1], klight->area.axisv[2]);
    float3 Ng = make_float3(klight->area.dir[0], klight->area.dir[1], klight->area.dir[2]);

    /* one sided */
    if (dot(D, Ng) >= 0.0f)
      return false;

    float3 light_P = make_float3(klight->co[0], klight->co[1], klight->co[2]);

    if (!ray_quad_intersect(
            P, D, 0.0f, t, light_P, axisu, axisv, Ng, &ls->P, &ls->t, &ls->u, &ls->v, is_round)) {
      return false;
    }

    ls->D = D;
    ls->Ng = Ng;
    if (is_round) {
      ls->pdf = invarea * lamp_light_pdf(kg, Ng, -D, ls->t);
    }
    else {
      float3 sample_axisu = axisu;
      float3 sample_axisv = axisv;

      if (klight->area.tan_spread > 0.0f) {
        if (!light_spread_clamp_area_light(
                P, Ng, &light_P, &sample_axisu, &sample_axisv, klight->area.tan_spread)) {
          return false;
        }
      }

      ls->pdf = rect_light_sample(P, &light_P, sample_axisu, sample_axisv, 0, 0, false);
    }
    ls->eval_fac = 0.25f * invarea;

    if (klight->area.tan_spread > 0.0f) {
      /* Area Light spread angle attenuation */
      ls->eval_fac *= light_spread_attenuation(
          ls->D, ls->Ng, klight->area.tan_spread, klight->area.normalize_spread);
      if (ls->eval_fac == 0.0f) {
        return false;
      }
    }
  }
  else {
    return false;
  }

  return true;
}

/* Triangle Light */

/* returns true if the triangle is has motion blur or an instancing transform applied */
ccl_device_inline bool triangle_world_space_vertices(
    KernelGlobals *kg, int object, int prim, float time, float3 V[3])
{
  bool has_motion = false;
  const int object_flag = kernel_tex_fetch(__object_flag, object);

  if (object_flag & SD_OBJECT_HAS_VERTEX_MOTION && time >= 0.0f) {
    motion_triangle_vertices(kg, object, prim, time, V);
    has_motion = true;
  }
  else {
    triangle_vertices(kg, prim, V);
  }

  if (!(object_flag & SD_OBJECT_TRANSFORM_APPLIED)) {
#ifdef __OBJECT_MOTION__
    float object_time = (time >= 0.0f) ? time : 0.5f;
    Transform tfm = object_fetch_transform_motion_test(kg, object, object_time, NULL);
#else
    Transform tfm = object_fetch_transform(kg, object, OBJECT_TRANSFORM);
#endif
    V[0] = transform_point(&tfm, V[0]);
    V[1] = transform_point(&tfm, V[1]);
    V[2] = transform_point(&tfm, V[2]);
    has_motion = true;
  }
  return has_motion;
}

ccl_device_inline float triangle_light_pdf_area(
    KernelGlobals *kg, const float3 Ng, const float3 I, float t, float triangle_area)
{
  float cos_pi = fabsf(dot(Ng, I));

  if (cos_pi == 0.0f)
    return 0.0f;

  return t * t / (cos_pi * triangle_area);
}

ccl_device_forceinline float triangle_light_pdf(KernelGlobals *kg, ShaderData *sd, float t)
{
  /* A naive heuristic to decide between costly solid angle sampling
   * and simple area sampling, comparing the distance to the triangle plane
   * to the length of the edges of the triangle. */

  float3 V[3];
  triangle_world_space_vertices(kg, sd->object, sd->prim, sd->time, V);

  const float3 e0 = V[1] - V[0];
  const float3 e1 = V[2] - V[0];
  const float3 e2 = V[2] - V[1];
  const float longest_edge_squared = max(len_squared(e0), max(len_squared(e1), len_squared(e2)));
  const float3 N = cross(e0, e1);
  const float distance_to_plane = fabsf(dot(N, sd->I * t)) / dot(N, N);

  if (longest_edge_squared > distance_to_plane * distance_to_plane) {
    /* sd contains the point on the light source
     * calculate Px, the point that we're shading */
    const float3 Px = sd->P + sd->I * t;
    const float3 v0_p = V[0] - Px;
    const float3 v1_p = V[1] - Px;
    const float3 v2_p = V[2] - Px;

    const float3 u01 = safe_normalize(cross(v0_p, v1_p));
    const float3 u02 = safe_normalize(cross(v0_p, v2_p));
    const float3 u12 = safe_normalize(cross(v1_p, v2_p));

    const float alpha = fast_acosf(dot(u02, u01));
    const float beta = fast_acosf(-dot(u01, u12));
    const float gamma = fast_acosf(dot(u02, u12));
    const float solid_angle = alpha + beta + gamma - M_PI_F;

    /* pdf_triangles is calculated over triangle area, but we're not sampling over its area */
    if (UNLIKELY(solid_angle == 0.0f)) {
      return 0.0f;
    }
    else {
      return 1.0f / solid_angle;
    }
  }
  else {
    const float area = 0.5f * len(N);
    return triangle_light_pdf_area(kg, sd->Ng, sd->I, t, area);
  }
}

ccl_device_forceinline void triangle_light_sample(KernelGlobals *kg,
                                                  int prim,
                                                  int object,
                                                  float randu,
                                                  float randv,
                                                  float time,
                                                  LightSample *ls,
                                                  const float3 P)
{
  /* A naive heuristic to decide between costly solid angle sampling
   * and simple area sampling, comparing the distance to the triangle plane
   * to the length of the edges of the triangle. */

  float3 V[3];
  triangle_world_space_vertices(kg, object, prim, time, V);

  const float3 e0 = V[1] - V[0];
  const float3 e1 = V[2] - V[0];
  const float3 e2 = V[2] - V[1];
  const float longest_edge_squared = max(len_squared(e0), max(len_squared(e1), len_squared(e2)));
  const float3 N0 = cross(e0, e1);
  float Nl = 0.0f;
  ls->Ng = safe_normalize_len(N0, &Nl);

  /* flip normal if necessary */
  const int object_flag = kernel_tex_fetch(__object_flag, object);
  if (object_flag & SD_OBJECT_NEGATIVE_SCALE_APPLIED) {
    ls->Ng = -ls->Ng;
  }
  ls->eval_fac = 1.0f;
  ls->shader = kernel_tex_fetch(__tri_shader, prim);
  ls->object = object;
  ls->prim = prim;
  ls->lamp = LAMP_NONE;
  ls->shader |= SHADER_USE_MIS;
  ls->type = LIGHT_TRIANGLE;

  float distance_to_plane = fabsf(dot(N0, V[0] - P) / dot(N0, N0));

  if (longest_edge_squared > distance_to_plane * distance_to_plane) {
    /* see James Arvo, "Stratified Sampling of Spherical Triangles"
     * http://www.graphics.cornell.edu/pubs/1995/Arv95c.pdf */

    /* project the triangle to the unit sphere
     * and calculate its edges and angles */
    const float3 v0_p = V[0] - P;
    const float3 v1_p = V[1] - P;
    const float3 v2_p = V[2] - P;

    const float3 u01 = safe_normalize(cross(v0_p, v1_p));
    const float3 u02 = safe_normalize(cross(v0_p, v2_p));
    const float3 u12 = safe_normalize(cross(v1_p, v2_p));

    const float3 A = safe_normalize(v0_p);
    const float3 B = safe_normalize(v1_p);
    const float3 C = safe_normalize(v2_p);

    const float cos_alpha = dot(u02, u01);
    const float cos_beta = -dot(u01, u12);
    const float cos_gamma = dot(u02, u12);

    /* calculate dihedral angles */
    const float alpha = fast_acosf(cos_alpha);
    const float beta = fast_acosf(cos_beta);
    const float gamma = fast_acosf(cos_gamma);
    /* the area of the unit spherical triangle = solid angle */
    const float solid_angle = alpha + beta + gamma - M_PI_F;

    /* precompute a few things
     * these could be re-used to take several samples
     * as they are independent of randu/randv */
    const float cos_c = dot(A, B);
    const float sin_alpha = fast_sinf(alpha);
    const float product = sin_alpha * cos_c;

    /* Select a random sub-area of the spherical triangle
     * and calculate the third vertex C_ of that new triangle */
    const float phi = randu * solid_angle - alpha;
    float s, t;
    fast_sincosf(phi, &s, &t);
    const float u = t - cos_alpha;
    const float v = s + product;

    const float3 U = safe_normalize(C - dot(C, A) * A);

    float q = 1.0f;
    const float det = ((v * s + u * t) * sin_alpha);
    if (det != 0.0f) {
      q = ((v * t - u * s) * cos_alpha - v) / det;
    }
    const float temp = max(1.0f - q * q, 0.0f);

    const float3 C_ = safe_normalize(q * A + sqrtf(temp) * U);

    /* Finally, select a random point along the edge of the new triangle
     * That point on the spherical triangle is the sampled ray direction */
    const float z = 1.0f - randv * (1.0f - dot(C_, B));
    ls->D = z * B + safe_sqrtf(1.0f - z * z) * safe_normalize(C_ - dot(C_, B) * B);

    /* calculate intersection with the planar triangle */
    if (!ray_triangle_intersect(P,
                                ls->D,
                                FLT_MAX,
#if defined(__KERNEL_SSE2__) && defined(__KERNEL_SSE__)
                                (ssef *)V,
#else
                                V[0],
                                V[1],
                                V[2],
#endif
                                &ls->u,
                                &ls->v,
                                &ls->t)) {
      ls->pdf = 0.0f;
      return;
    }

    ls->P = P + ls->D * ls->t;

    /* pdf_triangles is calculated over triangle area, but we're sampling over solid angle */
    if (UNLIKELY(solid_angle == 0.0f)) {
      ls->pdf = 0.0f;
      return;
    }
    else {
      ls->pdf = 1.0f / solid_angle;
    }
  }
  else {
    /* compute random point in triangle. From Eric Heitz's "A Low-Distortion Map Between Triangle
     * and Square" */
    float u = randu;
    float v = randv;
    if (v > u) {
      u *= 0.5f;
      v -= u;
    }
    else {
      v *= 0.5f;
      u -= v;
    }

    const float t = 1.0f - u - v;
    ls->P = u * V[0] + v * V[1] + t * V[2];
    /* compute incoming direction, distance and pdf */
    ls->D = normalize_len(ls->P - P, &ls->t);
    float area = 0.5f * Nl;
    ls->pdf = triangle_light_pdf_area(kg, ls->Ng, -ls->D, ls->t, area);

    ls->u = u;
    ls->v = v;
  }
}

/* chooses either to sample the light tree, distant or background lights by
 * sampling a CDF based on energy */
ccl_device int light_group_distribution_sample(KernelGlobals *kg, float *randu)
{
  /* This is basically std::upper_bound as used by pbrt, to find a point light or
   * triangle to emit from, proportional to area. a good improvement would be to
   * also sample proportional to power, though it's not so well defined with
   * arbitrary shaders. */
  const int num_groups = LIGHTGROUP_NUM;
  int first = 0;
  int len = num_groups + 1;
  float r = *randu;
  // todo: refactor this into its own function. It is used in several places
  while (len > 0) {
    int half_len = len >> 1;
    int middle = first + half_len;

    if (r < kernel_tex_fetch(__light_group_sample_cdf, middle)) {
      len = half_len;
    }
    else {
      first = middle + 1;
      len = len - half_len - 1;
    }
  }

  /* Clamping should not be needed but float rounding errors seem to
   * make this fail on rare occasions. */
  int index = clamp(first - 1, 0, num_groups - 1);

  /* Rescale to reuse random number. this helps the 2D samples within
   * each area light be stratified as well. */
  float distr_min = kernel_tex_fetch(__light_group_sample_cdf, index);
  float distr_max = kernel_tex_fetch(__light_group_sample_cdf, index + 1);
  *randu = (r - distr_min) / (distr_max - distr_min);

  return index;
}

/* Light Distribution */

ccl_device int light_distribution_sample(KernelGlobals *kg, float *randu)
{
  /* This is basically std::upper_bound as used by PBRT, to find a point light or
   * triangle to emit from, proportional to area. a good improvement would be to
   * also sample proportional to power, though it's not so well defined with
   * arbitrary shaders. */
  int first = 0;
  int len = kernel_data.integrator.num_distribution + 1;
  float r = *randu;

  do {
    int half_len = len >> 1;
    int middle = first + half_len;

    if (r < kernel_tex_fetch(__light_distribution, middle).totarea) {
      len = half_len;
    }
    else {
      first = middle + 1;
      len = len - half_len - 1;
    }
  } while (len > 0);

  /* Clamping should not be needed but float rounding errors seem to
   * make this fail on rare occasions. */
  int index = clamp(first - 1, 0, kernel_data.integrator.num_distribution - 1);

  /* Rescale to reuse random number. this helps the 2D samples within
   * each area light be stratified as well. */
  float distr_min = kernel_tex_fetch(__light_distribution, index).totarea;
  float distr_max = kernel_tex_fetch(__light_distribution, index + 1).totarea;
  *randu = (r - distr_min) / (distr_max - distr_min);

  return index;
}

/* Generic Light */

ccl_device_inline bool light_select_reached_max_bounces(KernelGlobals *kg, int index, int bounce)
{
  return (bounce > kernel_tex_fetch(__lights, index).max_bounces);
}

/*
 * Finds the solid angle of the smallest cone with vertex P that contains the bounding box. If P is
 * inside the bounding box light can be in any direction so use the entire sphere.
 */
ccl_device float calc_bbox_solid_angle(float3 P,
                                       float3 centroid_to_P_dir,
                                       float3 bboxMin,
                                       float3 bboxMax)
{
  if (P.x < bboxMax.x && P.y < bboxMax.y && P.z < bboxMax.z && P.x > bboxMin.x &&
      P.y > bboxMin.y && P.z > bboxMin.z) {
    /* P is inside bounding box */
    return M_PI_F;
  }
  else {
    /* Find the smallest cone that contains the bounding box by checking which bbox vertex is
     * farthest out. If we use a bounding sphere we get a too big cone. For example consider a long
     * skinny bbox oriented with P next to one of the small sides. */
    float theta_u = 0;
    float3 corners[8];
    corners[0] = bboxMin;
    corners[1] = make_float3(bboxMin.x, bboxMin.y, bboxMax.z);
    corners[2] = make_float3(bboxMin.x, bboxMax.y, bboxMin.z);
    corners[3] = make_float3(bboxMin.x, bboxMax.y, bboxMax.z);
    corners[4] = make_float3(bboxMax.x, bboxMin.y, bboxMin.z);
    corners[5] = make_float3(bboxMax.x, bboxMin.y, bboxMax.z);
    corners[6] = make_float3(bboxMax.x, bboxMax.y, bboxMin.z);
    corners[7] = bboxMax;
    for (int i = 0; i < 8; ++i) {
      float3 P_to_corner = normalize(P - corners[i]);
      const float cos_theta_u = dot(-centroid_to_P_dir, P_to_corner);
      theta_u = fmaxf(fast_acosf(cos_theta_u), theta_u);
    }
    return theta_u;
  }
}

/* calculates the importance metric for the given node and shading point P
 * If t_max is negative assume that V is the normal vector and use surface importance calculation.
 * Otherwise assume that V is the ray direction and use volume importance.
 */
ccl_device float calc_importance(KernelGlobals *kg,
                                 float3 P,
                                 float3 V,
                                 float t_max,
                                 float3 bboxMax,
                                 float3 bboxMin,
                                 float theta_o,
                                 float theta_e,
                                 float3 axis,
                                 float energy,
                                 float3 centroid)
{
  if (t_max < 0.0f) {
    /* eq. 3 */

    const float3 centroid_to_P = P - centroid;
    const float3 centroid_to_P_dir = normalize(centroid_to_P);
    const float r2 = len_squared(bboxMax - centroid);
    float d2 = len_squared(centroid_to_P);

    /* based on comment in the implementation details of the paper */
    const bool splitting = kernel_data.integrator.splitting_threshold != 0.0f;
    if (!splitting) {
      d2 = fmaxf(d2, r2 * 0.25f);
    }

    /* "theta_u captures the solid angle of the entire box" */

    float theta_u = calc_bbox_solid_angle(P, centroid_to_P_dir, bboxMin, bboxMax);

    /* cos(theta') */
    float cos_theta = dot(axis, centroid_to_P_dir);
    const float theta = fast_acosf(cos_theta);
    const float theta_prime = fmaxf(theta - theta_o - theta_u, 0.0f);
    if (theta_prime >= theta_e) {
      return 0.0f;
    }
    const float cos_theta_prime = fast_cosf(theta_prime);

    /* f_a|cos(theta'_i)| -- diffuse approximation */
    const float cos_theta_i = dot(V, -centroid_to_P_dir);
    const float theta_i = fast_acosf(cos_theta_i);
    const float theta_i_prime = fmaxf(theta_i - theta_u, 0.0f);
    const float cos_theta_i_prime = fast_cosf(theta_i_prime);
    const float abs_cos_theta_i_prime = fabsf(cos_theta_i_prime);
    /* doing something similar to bsdf_diffuse_eval_reflect() */
    /* TODO: Use theta_i or theta_i_prime here? */
    const float f_a = fmaxf(cos_theta_i_prime, 0.0f) * M_1_PI_F;

    return f_a * abs_cos_theta_i_prime * energy * cos_theta_prime / d2;
  }
  else {
    const float3 p_to_c = centroid - P;
    /* find closest point to centroid */
    const float t = fminf(t_max, fmaxf(0.0f, dot(V, p_to_c)));
    const float3 geometry_point = P + V * t;
    const float d_min = len(centroid - geometry_point);

    const float3 V0 = normalize(P - centroid);
    const float3 V1 = normalize(P + V * fminf(t_max, 1e12f) - centroid);
    const float3 O0 = V0;
    float3 O1, O2;
    make_orthonormals_tangent(O0, V1, &O1, &O2);

    const float b_max = fmaxf(dot(V0, axis), dot(V1, axis));
    const float O0_dot_a = dot(O0, axis);
    const float O1_dot_a = dot(O1, axis);
    const float cos_phi_o = O0_dot_a / sqrtf(O1_dot_a * O1_dot_a + O0_dot_a * O0_dot_a); /* Eq 5 */
    const float sin_phi_o = sqrtf(1 - cos_phi_o * cos_phi_o);
    const float3 v = O0 * cos_phi_o + O1 * sin_phi_o;

    /* Eq 6 */
    float cos_theta_min;
    if (O1_dot_a < 0 || dot(V0, V1) > cos_phi_o) {
      cos_theta_min = b_max;
    }
    else {
      cos_theta_min = dot(axis, v);
    }

    float theta_min = fast_acosf(cos_theta_min);
    float theta_u = calc_bbox_solid_angle(
        geometry_point, normalize(geometry_point - centroid), bboxMin, bboxMax);
    float theta_prime = fmaxf(theta_min - theta_o - theta_u, 0);

    if (theta_prime >= theta_e) {
      return 0;
    }

    return energy * fast_cosf(theta_prime) / d_min; /* Eq 7 */
  }
}

/* the energy, spatial and orientation bounds for the light are loaded and decoded
 * and then this information is used to calculate the importance for this light.
 * This function is used to calculate the importance for a light in a node
 * containing several lights. */
ccl_device float calc_light_importance(
    KernelGlobals *kg, float3 P, float3 V, float t_max, int node_offset, int light_offset)
{
  /* find offset into light_tree_leaf_emitters array */
  int first_emitter = kernel_tex_fetch(__leaf_to_first_emitter, node_offset / 4);
  kernel_assert(first_emitter != -1);
  int offset = first_emitter + light_offset * 3;

  /* get relevant information to be able to calculate the importance */
  const float4 data0 = kernel_tex_fetch(__light_tree_leaf_emitters, offset + 0);
  const float4 data1 = kernel_tex_fetch(__light_tree_leaf_emitters, offset + 1);
  const float4 data2 = kernel_tex_fetch(__light_tree_leaf_emitters, offset + 2);

  /* decode data for this light */
  const float3 bbox_min = make_float3(data0.x, data0.y, data0.z);
  const float3 bbox_max = make_float3(data0.w, data1.x, data1.y);
  const float theta_o = data1.z;
  const float theta_e = data1.w;
  const float3 axis = make_float3(data2.x, data2.y, data2.z);
  const float energy = data2.w;
  const float3 centroid = 0.5f * (bbox_max + bbox_min);

  return calc_importance(
      kg, P, V, t_max, bbox_max, bbox_min, theta_o, theta_e, axis, energy, centroid);
}

/* the combined energy, spatial and orientation bounds for all the lights for the
 * given node are loaded and decoded and then this information is used to
 * calculate the importance for this node. */
ccl_device float calc_node_importance(
    KernelGlobals *kg, float3 P, float3 V, float t_max, int node_offset)
{
  /* load the data for this node */
  const float4 node0 = kernel_tex_fetch(__light_tree_nodes, node_offset + 0);
  const float4 node1 = kernel_tex_fetch(__light_tree_nodes, node_offset + 1);
  const float4 node2 = kernel_tex_fetch(__light_tree_nodes, node_offset + 2);
  const float4 node3 = kernel_tex_fetch(__light_tree_nodes, node_offset + 3);

  /* decode the data so it can be used to calculate the importance */
  const float energy = node0.x;
  const float3 bbox_min = make_float3(node1.x, node1.y, node1.z);
  const float3 bbox_max = make_float3(node1.w, node2.x, node2.y);
  const float theta_o = node2.z;
  const float theta_e = node2.w;
  const float3 axis = make_float3(node3.x, node3.y, node3.z);
  const float3 centroid = 0.5f * (bbox_max + bbox_min);

  return calc_importance(
      kg, P, V, t_max, bbox_max, bbox_min, theta_o, theta_e, axis, energy, centroid);
}

/* given a node offset, this function loads and decodes the minimum amount of
 * data needed for a the given node to be able to only either identify if it is
 * a leaf node or how to find its two children
 *
 * child_o  ffset is an offset into the nodes array to this nodes right child. the
 * left child has index node_offset+4.
 * distribution_id corresponds to an offset into the distribution array for the
 * first light contained in this node. num_emitters is how many lights there are
 * in this node. */
ccl_device void update_node(
    KernelGlobals *kg, int node_offset, int *child_offset, int *distribution_id, int *num_emitters)
{
  float4 node = kernel_tex_fetch(__light_tree_nodes, node_offset);
  (*child_offset) = __float_as_int(node.y);
  (*distribution_id) = __float_as_int(node.z);
  (*num_emitters) = __float_as_int(node.w);
}

/* picks one of the distant lights and computes the probability of picking it */
ccl_device void light_distant_sample(
    KernelGlobals *kg, float3 P, float *randu, int *index, float *pdf)
{
  light_distribution_sample(kg, randu);  // rescale random number

  /* choose one of the distant lights randomly */
  int num_distant = kernel_data.integrator.num_distant_lights;
  int light = min((int)(*randu * (float)num_distant), num_distant - 1);

  /* This assumes the distant lights are next to each other in the
   * distribution array starting at distant_lights_offset. */
  int distant_lights_offset = kernel_data.integrator.distant_lights_offset;

  *index = light + distant_lights_offset;
  *pdf = kernel_data.integrator.inv_num_distant_lights;
}

/* picks the background light and sets the probability of picking it */
ccl_device void light_background_sample(
    KernelGlobals *kg, float3 P, float *randu, int *index, float *pdf)
{
  *index = kernel_tex_fetch(__lamp_to_distribution, kernel_data.integrator.background_light_index);
  *pdf = 1.0f;
}

/* picks a light from the light tree and returns its index and the probability of
 * picking this light. */
ccl_device void light_tree_sample(KernelGlobals *kg,
                                  float3 P,
                                  float3 V,
                                  float t_max,
                                  float *randu,
                                  int *index,
                                  float *pdf_factor)
{
  int sampled_index = -1;
  *pdf_factor = 1.0f;

  int offset = 0;
  int right_child_offset, distribution_id, num_emitters;
  do {

    /* read in first part of node of light tree */
    update_node(kg, offset, &right_child_offset, &distribution_id, &num_emitters);

    /* Found a leaf - Choose which light to use */
    if (right_child_offset == -1) {  // Found a leaf
      if (num_emitters == 1) {
        sampled_index = distribution_id;
      }
      else {

        /* At a leaf node containing several lights. Pick one of these
         * by creating and sampling a CDF based on the importance metric.
         *
         * The number of lights in this leaf node is not known at compile
         * time and dynamic allocation is not allowed on the GPU, so
         * some more computations have to be done instead.
         * (TODO: Could we allocate a fixed array of the same size as
         * the maximum allowed number of lights per node in the
         * construction algorithm? i.e. max_lights_in_node.)
         *
         * First, the total importance of all the lights are calculated.
         * Then, a linear loop over the lights are done where the
         * current CDF value is calculated. This loop can stop as soon
         * as the random value used to sample the CDF is less than the
         * current CDF value. The sampled light has index i-1 if i is
         * the iteration counter of the loop over the lights. This is
         * similar to for example light the old light_distribution_sample()
         * except not having an array to store the CDF in. */
        float sum = 0.0f;
        for (int i = 0; i < num_emitters; ++i) {
          sum += calc_light_importance(kg, P, V, t_max, offset, i);
        }

        if (sum == 0.0f) {
          *pdf_factor = 0.0f;
          return;
        }

        float sum_inv = 1.0f / sum;

        float cdf_L = 0.0f;
        float cdf_R = 0.0f;
        float prob = 0.0f;
        int light = 0;
        for (int i = 1; i < num_emitters + 1; ++i) {
          prob = calc_light_importance(kg, P, V, t_max, offset, i - 1) * sum_inv;
          cdf_R = cdf_L + prob;
          if (*randu < cdf_R) {
            light = i - 1;
            break;
          }
          cdf_L = cdf_R;
        }

        sampled_index = distribution_id + light;
        *pdf_factor *= prob;
        /* rescale random number */
        *randu = (*randu - cdf_L) / (cdf_R - cdf_L);
      }
      break;
    }
    else {  // Interior node, pick left or right randomly

      /* calculate probability of going down left node */
      int child_offsetL = offset + 4;
      int child_offsetR = 4 * right_child_offset;
      float I_L = calc_node_importance(kg, P, V, t_max, child_offsetL);
      float I_R = calc_node_importance(kg, P, V, t_max, child_offsetR);
      if ((I_L == 0.0f) && (I_R == 0.0f)) {
        *pdf_factor = 0.0f;
        break;
      }

      float P_L = I_L / (I_L + I_R);

      /* choose which node to go down */
      if (*randu <= P_L) {  // Going down left node
        /* rescale random number */
        *randu = *randu / P_L;

        offset = child_offsetL;
        *pdf_factor *= P_L;
      }
      else {  // Going down right node
        /* rescale random number */
        *randu = (*randu * (I_L + I_R) - I_L) / I_R;

        offset = child_offsetR;
        *pdf_factor *= 1.0f - P_L;
      }
    }
  } while (true);

  *index = sampled_index;
}

/* converts from an emissive triangle index to the corresponding
 * light distribution index. */
ccl_device int triangle_to_distribution(KernelGlobals *kg, int triangle_id, int object_id)
{
  /* binary search to find triangle_id which then gives distribution_id */
  /* equivalent to implementation of std::lower_bound */
  /* todo: of complexity log(N) now. could be made constant with a hash table? */
  /* __triangle_to_distribution is an array of uints of the format below:
   * [triangle_id0, object_id0, distribution_id0, triangle_id1,... ]
   * where e.g. [triangle_id0,object_id0] corresponds to distribution id
   * distribution_id0
   */
  int first = 0;
  int last = kernel_data.integrator.num_triangle_lights;
  int count = last - first;
  int middle, step;
  while (count > 0) {
    step = count / 2;
    middle = first + step;
    int triangle = kernel_tex_fetch(__triangle_to_distribution, middle * 3);
    if (triangle < triangle_id) {
      first = middle + 1;
      count -= step + 1;
    }
    else
      count = step;
  }

  /* If instancing then we can have several triangles with the same triangle_id
   * so loop over object_id too. */
  /* todo: do a binary search here too if many instances */
  while (true) {
    int object = kernel_tex_fetch(__triangle_to_distribution, first * 3 + 1);
    if (object == object_id)
      break;
    ++first;
  }

  kernel_assert(kernel_tex_fetch(__triangle_to_distribution, first * 3) == triangle_id);

  return kernel_tex_fetch(__triangle_to_distribution, first * 3 + 2);
}

/* Decides whether to go down both childen or only one in the tree traversal.
 * The split heuristic is based on the variance of the lighting within the node.
 * There are two types of variances that are considered: variance in energy and
 * in the distance term 1/d^2. The variance in energy is pre-computed on the
 * host but the distance term is calculated here. These variances are then
 * combined and normalized to get the final splitting heuristic. High variance
 * leads to a lower splitting heuristic which leads to more splits during the
 * traversal. */
ccl_device bool split(KernelGlobals *kg, float3 P, int node_offset)
{
  /* early exists if never/always splitting */
  const float threshold = kernel_data.integrator.splitting_threshold;
  if (threshold == 0.0f) {
    return false;
  }
  else if (threshold == 1.0f) {
    return true;
  }

  /* extract bounding box of cluster */
  const float4 node1 = kernel_tex_fetch(__light_tree_nodes, node_offset + 1);
  const float4 node2 = kernel_tex_fetch(__light_tree_nodes, node_offset + 2);
  const float3 bboxMin = make_float3(node1.x, node1.y, node1.z);
  const float3 bboxMax = make_float3(node1.w, node2.x, node2.y);

  /* if P is inside bounding sphere then split */
  const float3 centroid = 0.5f * (bboxMax + bboxMin);
  const float radius_squared = len_squared(bboxMax - centroid);
  const float dist_squared = len_squared(centroid - P);

  if (dist_squared <= radius_squared) {
    return true;
  }

  /* eq. 8 & 9 */

  /* the integral in eq. 8 requires us to know the interval the distance can
   * be in: [a,b]. This is found by considering a bounding sphere around the
   * bounding box of the node and "a" then becomes the smallest distance to
   * this sphere and "b" becomes the largest. */
  const float radius = sqrt(radius_squared);
  const float dist = sqrt(dist_squared);
  const float a = dist - radius;
  const float b = dist + radius;

  const float g_mean = 1.0f / (a * b);
  const float g_mean_squared = g_mean * g_mean;
  const float a3 = a * a * a;
  const float b3 = b * b * b;
  const float g_variance = (b3 - a3) / (3.0f * (b - a) * a3 * b3) - g_mean_squared;

  /* eq. 10 */
  const float4 node0 = kernel_tex_fetch(__light_tree_nodes, node_offset);
  const float4 node3 = kernel_tex_fetch(__light_tree_nodes, node_offset + 3);
  const float energy = node0.x;
  const float e_variance = node3.w;
  const float num_emitters = (float)__float_as_int(node0.w);
  const float num_emitters_squared = num_emitters * num_emitters;
  const float e_mean = energy / num_emitters;
  const float e_mean_squared = e_mean * e_mean;
  const float variance = (e_variance * (g_variance + g_mean_squared) +
                          e_mean_squared * g_variance) *
                         num_emitters_squared;

  /* normalize the variance heuristic to be within [0,1]. Note that high
   * variance corresponds to a low normalized variance. To give an idea of
   * how this normalization function looks like:
   *            variance: 0    1   10  100  1000  10000  100000
   * normalized variance: 1  0.8  0.7  0.5   0.4    0.3     0.2  */
  const float variance_normalized = sqrt(sqrt(1.0f / (1.0f + sqrt(variance))));

  return variance_normalized < threshold;
}

/* given a light in the form of a distribution id, this function computes the
 * the probability of picking it using the light tree. this mimics the
 * probability calculations in accum_light_tree_contribution()
 *
 * the nodes array contains all the nodes of the tree and each interior node
 * has its left child directly after it in the nodes array and the right child
 * is also after it at the second_child_offset.
 *
 * Given the structure of the nodes array we can find the path from the root
 * to the leaf node the given light belongs to as follows:
 *
 * 1. Find the offset of the leaf node the given light belongs to.
 * 2. Traverse the tree in a top-down manner where the decision to go down the
 *    left or right child is determined as follows:
 *    If the node we are looking for has a lower offset than the right child
 *    then it belongs to a node between the current node and the right child.
 *    That is, we should go down the left node. Otherwise, the right node.
 *    This is done recursively until the leaf node is found.
 * 3. Before going down the left or right child:
 *    a) If we are splitting then the probability is not affected.
 *    b) If we are not splitting then the probability is multiplied by the
 *       probability of choosing this particular child node.
 */
ccl_device float light_tree_pdf(KernelGlobals *kg,
                                float3 P,
                                float3 V,
                                float t_max,
                                int distribution_id,
                                int offset,
                                float pdf,
                                bool can_split)
{

  /* find mapping from distribution_id to node_id */
  int node_id = kernel_tex_fetch(__light_distribution_to_node, distribution_id);

  /* read in first part of node of light tree */
  int right_child_offset, first_distribution_id, num_emitters;
  update_node(kg, offset, &right_child_offset, &first_distribution_id, &num_emitters);

  while (right_child_offset != -1) {
    int child_offsetL = offset + 4;
    int child_offsetR = 4 * right_child_offset;

    /* choose whether to go down both(split) or only one of the children */
    if (can_split && split(kg, P, offset)) {
      /* go down to the child node that is an ancestor of this node_id
       * without changing the probability since we split here */

      if (node_id < child_offsetR) {
        offset = child_offsetL;
      }
      else {
        offset = child_offsetR;
      }

      return light_tree_pdf(kg, P, V, t_max, distribution_id, offset, pdf, true);
    }
    else {
      /* go down one of the child nodes */

      /* evaluate the importance of each of the child nodes */
      float I_L = calc_node_importance(kg, P, V, t_max, child_offsetL);
      float I_R = calc_node_importance(kg, P, V, t_max, child_offsetR);

      if ((I_L == 0.0f) && (I_R == 0.0f)) {
        return 0.0f;
      }

      /* calculate the probability of going down the left node */
      float P_L = I_L / (I_L + I_R);

      /* choose which node to go down */
      if (node_id < child_offsetR) {
        offset = child_offsetL;
        pdf *= P_L;
      }
      else {
        offset = child_offsetR;
        pdf *= 1.0f - P_L;
      }
      update_node(kg, offset, &right_child_offset, &first_distribution_id, &num_emitters);
    }
  }

  /* if there are several emitters in this leaf then pick one of them */
  if (num_emitters > 1) {

    /* the case of being a light inside a leaf node with several lights.
     * during sampling, a CDF is created based on importance, so here
     * the probability of sampling this light using the CDF has to be
     * computed. This is done by dividing the importance of this light
     * by the total sum of the importance of all lights in the leaf. */
    float sum = 0.0f;
    for (int i = 0; i < num_emitters; ++i) {
      sum += calc_light_importance(kg, P, V, t_max, offset, i);
    }

    if (sum == 0.0f) {
      return 0.0f;
    }

    pdf *= calc_light_importance(
               kg, P, V, t_max, offset, distribution_id - first_distribution_id) /
           sum;
  }

  return pdf;
}

/* computes the the probability of picking the given light out of all lights.
 * this mimics the probability calculations in light_distribution_sample() */
ccl_device float light_distribution_pdf(
    KernelGlobals *kg, float3 P, float3 V, float t_max, int prim_id, int object_id)
{
  /* convert from triangle/lamp to light distribution */
  int distribution_id;
  if (prim_id >= 0) {  // Triangle_id = prim_id
    distribution_id = triangle_to_distribution(kg, prim_id, object_id);
  }
  else {  // Lamp
    int lamp_id = -prim_id - 1;
    distribution_id = kernel_tex_fetch(__lamp_to_distribution, lamp_id);
  }

  kernel_assert((distribution_id >= 0) &&
                (distribution_id < kernel_data.integrator.num_distribution));

  /* compute picking pdf for this light */
  if (kernel_data.integrator.use_light_tree) {
    /* find out which group of lights to sample */
    int group;
    if (prim_id >= 0) {
      group = LIGHTGROUP_TREE;
    }
    else {
      int lamp = -prim_id - 1;
      int light_type = kernel_tex_fetch(__lights, lamp).type;
      if (light_type == LIGHT_DISTANT) {
        group = LIGHTGROUP_DISTANT;
      }
      else if (light_type == LIGHT_BACKGROUND) {
        group = LIGHTGROUP_BACKGROUND;
      }
      else {
        group = LIGHTGROUP_TREE;
      }
    }

    /* get probabilty to sample this group of lights */
    float group_prob = kernel_tex_fetch(__light_group_sample_prob, group);
    float pdf = group_prob;

    if (group == LIGHTGROUP_TREE) {
      pdf *= light_tree_pdf(kg, P, V, t_max, distribution_id, 0, 1.0f, true);
    }
    else if (group == LIGHTGROUP_DISTANT) {
      pdf *= kernel_data.integrator.inv_num_distant_lights;
    }
    else if (group == LIGHTGROUP_BACKGROUND) {
      /* there is only one background light so nothing to do here */
    }
    else {
      kernel_assert(false);
    }

    return pdf;
  }
  else {
    const ccl_global KernelLightDistribution *kdistribution = &kernel_tex_fetch(
        __light_distribution, distribution_id);
    return kdistribution->area * kernel_data.integrator.pdf_inv_totarea;
  }
}

/* picks a light and returns its index and the probability of picking it */
ccl_device void light_distribution_sample(
    KernelGlobals *kg, float3 P, float3 V, float t_max, float *randu, int *index, float *pdf)
{
  if (kernel_data.integrator.use_light_tree) {
    /* sample light type distribution */
    int group = light_group_distribution_sample(kg, randu);
    float group_prob = kernel_tex_fetch(__light_group_sample_prob, group);

    if (group == LIGHTGROUP_TREE) {
      light_tree_sample(kg, P, V, t_max, randu, index, pdf);
    }
    else if (group == LIGHTGROUP_DISTANT) {
      light_distant_sample(kg, P, randu, index, pdf);
    }
    else if (group == LIGHTGROUP_BACKGROUND) {
      light_background_sample(kg, P, randu, index, pdf);
    }
    else {
      kernel_assert(false);
    }

    *pdf *= group_prob;
  }
  else {  // Sample light distribution CDF
    *index = light_distribution_sample(kg, randu);
    const ccl_global KernelLightDistribution *kdistribution = &kernel_tex_fetch(
        __light_distribution, *index);
    *pdf = kdistribution->area * kernel_data.integrator.pdf_inv_totarea;
  }
}

/* picks a point on a given light and computes the probability of picking this point*/
ccl_device void light_point_sample(KernelGlobals *kg,
                                   int lamp,
                                   float randu,
                                   float randv,
                                   float time,
                                   float3 P,
                                   int bounce,
                                   int distribution_id,
                                   LightSample *ls)
{
  if (lamp < 0) {
    /* fetch light data and compute rest of light pdf */
    const ccl_global KernelLightDistribution *kdistribution = &kernel_tex_fetch(
        __light_distribution, distribution_id);
    int prim = kdistribution->prim;

    if (prim >= 0) {
      int object = kdistribution->mesh_light.object_id;
      int shader_flag = kdistribution->mesh_light.shader_flag;

      triangle_light_sample(kg, prim, object, randu, randv, time, ls, P);
      ls->shader |= shader_flag;
      return;
    }
    lamp = -prim - 1;
  }

  if (UNLIKELY(light_select_reached_max_bounces(kg, lamp, bounce))) {
    ls->pdf = 0.0f;
    return;
  }

  if (!lamp_light_sample(kg, lamp, randu, randv, P, ls)) {
    ls->pdf = 0.0f;
    return;
  }
}

/* picks a light and then picks a point on the light and computes the
 * probability of doing so. V and t_max are only used when using light tree for sampling. If t_max
 * is negative V should be the normal vector at P. Otherwise V should be the direction of the light
 * ray starting at P.
 */
ccl_device_noinline bool light_sample(KernelGlobals *kg,
                                      int lamp,
                                      float randu,
                                      float randv,
                                      float time,
                                      float3 P,
                                      float3 V,
                                      float t_max,
                                      int bounce,
                                      LightSample *ls)
{
  /* pick a light and compute the probability of picking this light */
  float pdf_factor = 0.0f;
  int index = -1;
  light_distribution_sample(kg, P, V, t_max, &randu, &index, &pdf_factor);

  if (pdf_factor == 0.0f) {
    return false;
  }

  /* pick a point on the light and the probability of picking this point */
  light_point_sample(kg, lamp, randu, randv, time, P, bounce, index, ls);

  /* combine pdfs */
  ls->pdf *= pdf_factor;

  return (ls->pdf > 0.0f);
}

ccl_device_inline int light_select_num_samples(KernelGlobals *kg, int index)
{
  return kernel_tex_fetch(__lights, index).samples;
}

CCL_NAMESPACE_END
