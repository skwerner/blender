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

/*
 * ShaderData, used in four steps:
 *
 * Setup from incoming ray, sampled position and background.
 * Execute for surface, volume or displacement.
 * Evaluate one or more closures.
 * Release.
 */

#pragma once

// clang-format off
#include "kernel/closure/alloc.h"
#include "kernel/closure/bsdf_util.h"
#include "kernel/closure/bsdf.h"
#include "kernel/closure/emissive.h"
// clang-format on

#include "kernel/kernel_accumulate.h"
#include "kernel/kernel_random.h"
#include "kernel/geom/geom.h"
#include "kernel/svm/svm.h"

#ifdef __OSL__
#  include "kernel/osl/osl_shader.h"
#endif

CCL_NAMESPACE_BEGIN

/* ShaderData setup from incoming ray */

#ifdef __OBJECT_MOTION__
ccl_device void shader_setup_object_transforms(const KernelGlobals *ccl_restrict kg,
                                               ShaderData *ccl_restrict sd,
                                               float time)
{
  if (sd->object_flag & SD_OBJECT_MOTION) {
    sd->ob_tfm_motion = object_fetch_transform_motion(kg, sd->object, time);
    sd->ob_itfm_motion = transform_quick_inverse(sd->ob_tfm_motion);
  }
}
#endif

/* TODO: break this up if it helps reduce register pressure to load data from
 * global memory as we write it to shaderdata. */
ccl_device_inline void shader_setup_from_ray(const KernelGlobals *ccl_restrict kg,
                                             ShaderData *ccl_restrict sd,
                                             const Ray *ccl_restrict ray,
                                             const Intersection *ccl_restrict isect)
{
  PROFILING_INIT(kg, PROFILING_SHADER_SETUP);

  /* Read intersection data into shader globals.
   *
   * TODO: this is redundant, could potentially remove some of this from
   * ShaderData but would need to ensure that it also works for shadow
   * shader evaluation. */
  sd->u = isect->u;
  sd->v = isect->v;
  sd->ray_length = isect->t;
  sd->type = isect->type;
  sd->object = (isect->object == OBJECT_NONE) ? kernel_tex_fetch(__prim_object, isect->prim) :
                                                isect->object;
  sd->object_flag = kernel_tex_fetch(__object_flag, sd->object);
  sd->prim = kernel_tex_fetch(__prim_index, isect->prim);
  sd->lamp = LAMP_NONE;
  sd->flag = 0;

  /* Read matrices and time. */
  sd->time = ray->time;

#ifdef __OBJECT_MOTION__
  shader_setup_object_transforms(kg, sd, ray->time);
#endif

  /* Read ray data into shader globals. */
  sd->I = -ray->D;

#ifdef __HAIR__
  if (sd->type & PRIMITIVE_ALL_CURVE) {
    /* curve */
    curve_shader_setup(kg, sd, ray->P, ray->D, isect->t, isect->object, isect->prim);
  }
  else
#endif
      if (sd->type & PRIMITIVE_TRIANGLE) {
    /* static triangle */
    float3 Ng = triangle_normal(kg, sd);
    sd->shader = kernel_tex_fetch(__tri_shader, sd->prim);

    /* vectors */
    sd->P = triangle_refine(kg, sd, ray->P, ray->D, isect->t, isect->object, isect->prim);
    sd->Ng = Ng;
    sd->N = Ng;

    /* smooth normal */
    if (sd->shader & SHADER_SMOOTH_NORMAL)
      sd->N = triangle_smooth_normal(kg, Ng, sd->prim, sd->u, sd->v);

#ifdef __DPDU__
    /* dPdu/dPdv */
    triangle_dPdudv(kg, sd->prim, &sd->dPdu, &sd->dPdv);
#endif
  }
  else {
    /* motion triangle */
    motion_triangle_shader_setup(
        kg, sd, ray->P, ray->D, isect->t, isect->object, isect->prim, false);
  }

  sd->flag |= kernel_tex_fetch(__shaders, (sd->shader & SHADER_MASK)).flags;

  if (isect->object != OBJECT_NONE) {
    /* instance transform */
    object_normal_transform_auto(kg, sd, &sd->N);
    object_normal_transform_auto(kg, sd, &sd->Ng);
#ifdef __DPDU__
    object_dir_transform_auto(kg, sd, &sd->dPdu);
    object_dir_transform_auto(kg, sd, &sd->dPdv);
#endif
  }

  /* backfacing test */
  bool backfacing = (dot(sd->Ng, sd->I) < 0.0f);

  if (backfacing) {
    sd->flag |= SD_BACKFACING;
    sd->Ng = -sd->Ng;
    sd->N = -sd->N;
#ifdef __DPDU__
    sd->dPdu = -sd->dPdu;
    sd->dPdv = -sd->dPdv;
#endif
  }

#ifdef __RAY_DIFFERENTIALS__
  /* differentials */
  differential_transfer_compact(&sd->dP, ray->dP, ray->D, ray->dD, sd->Ng, sd->ray_length);
  differential_incoming_compact(&sd->dI, ray->D, ray->dD);
  differential_dudv(&sd->du, &sd->dv, sd->dPdu, sd->dPdv, sd->dP, sd->Ng);
#endif
  PROFILING_SHADER(sd->shader);
  PROFILING_OBJECT(sd->object);
}

/* ShaderData setup from BSSRDF scatter */

#if 0
#  ifdef __SUBSURFACE__
#    ifndef __KERNEL_CUDA__
ccl_device
#    else
ccl_device_inline
#    endif
    void
    shader_setup_from_subsurface(const KernelGlobals *kg,
                                 ShaderData *sd,
                                 const Intersection *isect,
                                 const Ray *ray)
{
  PROFILING_INIT(kg, PROFILING_SHADER_SETUP);

  const bool backfacing = sd->flag & SD_BACKFACING;

  /* object, matrices, time, ray_length stay the same */
  sd->flag = 0;
  sd->object_flag = kernel_tex_fetch(__object_flag, sd->object);
  sd->prim = kernel_tex_fetch(__prim_index, isect->prim);
  sd->type = isect->type;

  sd->u = isect->u;
  sd->v = isect->v;

  /* fetch triangle data */
  if (sd->type == PRIMITIVE_TRIANGLE) {
    float3 Ng = triangle_normal(kg, sd);
    sd->shader = kernel_tex_fetch(__tri_shader, sd->prim);

    /* static triangle */
    sd->P = triangle_refine_local(kg, sd, isect, ray);
    sd->Ng = Ng;
    sd->N = Ng;

    if (sd->shader & SHADER_SMOOTH_NORMAL)
      sd->N = triangle_smooth_normal(kg, Ng, sd->prim, sd->u, sd->v);

#    ifdef __DPDU__
    /* dPdu/dPdv */
    triangle_dPdudv(kg, sd->prim, &sd->dPdu, &sd->dPdv);
#    endif
  }
  else {
    /* motion triangle */
    motion_triangle_shader_setup(kg, sd, isect, ray, true);
  }

  sd->flag |= kernel_tex_fetch(__shaders, (sd->shader & SHADER_MASK)).flags;

  if (isect->object != OBJECT_NONE) {
    /* instance transform */
    object_normal_transform_auto(kg, sd, &sd->N);
    object_normal_transform_auto(kg, sd, &sd->Ng);
#    ifdef __DPDU__
    object_dir_transform_auto(kg, sd, &sd->dPdu);
    object_dir_transform_auto(kg, sd, &sd->dPdv);
#    endif
  }

  /* backfacing test */
  if (backfacing) {
    sd->flag |= SD_BACKFACING;
    sd->Ng = -sd->Ng;
    sd->N = -sd->N;
#    ifdef __DPDU__
    sd->dPdu = -sd->dPdu;
    sd->dPdv = -sd->dPdv;
#    endif
  }

  /* should not get used in principle as the shading will only use a diffuse
   * BSDF, but the shader might still access it */
  sd->I = sd->N;

#    ifdef __RAY_DIFFERENTIALS__
  /* differentials */
  differential_dudv(&sd->du, &sd->dv, sd->dPdu, sd->dPdv, sd->dP, sd->Ng);
  /* don't modify dP and dI */
#    endif

  PROFILING_SHADER(sd->shader);
}
#  endif
#endif

/* ShaderData setup from position sampled on mesh */

ccl_device_inline void shader_setup_from_sample(const KernelGlobals *ccl_restrict kg,
                                                ShaderData *ccl_restrict sd,
                                                const float3 P,
                                                const float3 Ng,
                                                const float3 I,
                                                int shader,
                                                int object,
                                                int prim,
                                                float u,
                                                float v,
                                                float t,
                                                float time,
                                                bool object_space,
                                                int lamp)
{
  PROFILING_INIT(kg, PROFILING_SHADER_SETUP);

  /* vectors */
  sd->P = P;
  sd->N = Ng;
  sd->Ng = Ng;
  sd->I = I;
  sd->shader = shader;
  if (prim != PRIM_NONE)
    sd->type = PRIMITIVE_TRIANGLE;
  else if (lamp != LAMP_NONE)
    sd->type = PRIMITIVE_LAMP;
  else
    sd->type = PRIMITIVE_NONE;

  /* primitive */
  sd->object = object;
  sd->lamp = LAMP_NONE;
  /* currently no access to bvh prim index for strand sd->prim*/
  sd->prim = prim;
  sd->u = u;
  sd->v = v;
  sd->time = time;
  sd->ray_length = t;

  sd->flag = kernel_tex_fetch(__shaders, (sd->shader & SHADER_MASK)).flags;
  sd->object_flag = 0;
  if (sd->object != OBJECT_NONE) {
    sd->object_flag |= kernel_tex_fetch(__object_flag, sd->object);

#ifdef __OBJECT_MOTION__
    shader_setup_object_transforms(kg, sd, time);
#endif
  }
  else if (lamp != LAMP_NONE) {
    sd->lamp = lamp;
  }

  /* transform into world space */
  if (object_space) {
    object_position_transform_auto(kg, sd, &sd->P);
    object_normal_transform_auto(kg, sd, &sd->Ng);
    sd->N = sd->Ng;
    object_dir_transform_auto(kg, sd, &sd->I);
  }

  if (sd->type & PRIMITIVE_TRIANGLE) {
    /* smooth normal */
    if (sd->shader & SHADER_SMOOTH_NORMAL) {
      sd->N = triangle_smooth_normal(kg, Ng, sd->prim, sd->u, sd->v);

      if (!(sd->object_flag & SD_OBJECT_TRANSFORM_APPLIED)) {
        object_normal_transform_auto(kg, sd, &sd->N);
      }
    }

    /* dPdu/dPdv */
#ifdef __DPDU__
    triangle_dPdudv(kg, sd->prim, &sd->dPdu, &sd->dPdv);

    if (!(sd->object_flag & SD_OBJECT_TRANSFORM_APPLIED)) {
      object_dir_transform_auto(kg, sd, &sd->dPdu);
      object_dir_transform_auto(kg, sd, &sd->dPdv);
    }
#endif
  }
  else {
#ifdef __DPDU__
    sd->dPdu = zero_float3();
    sd->dPdv = zero_float3();
#endif
  }

  /* backfacing test */
  if (sd->prim != PRIM_NONE) {
    bool backfacing = (dot(sd->Ng, sd->I) < 0.0f);

    if (backfacing) {
      sd->flag |= SD_BACKFACING;
      sd->Ng = -sd->Ng;
      sd->N = -sd->N;
#ifdef __DPDU__
      sd->dPdu = -sd->dPdu;
      sd->dPdv = -sd->dPdv;
#endif
    }
  }

#ifdef __RAY_DIFFERENTIALS__
  /* no ray differentials here yet */
  sd->dP = differential3_zero();
  sd->dI = differential3_zero();
  sd->du = differential_zero();
  sd->dv = differential_zero();
#endif

  PROFILING_SHADER(sd->shader);
  PROFILING_OBJECT(sd->object);
}

/* ShaderData setup for displacement */

ccl_device void shader_setup_from_displace(const KernelGlobals *ccl_restrict kg,
                                           ShaderData *ccl_restrict sd,
                                           int object,
                                           int prim,
                                           float u,
                                           float v)
{
  float3 P, Ng, I = zero_float3();
  int shader;

  triangle_point_normal(kg, object, prim, u, v, &P, &Ng, &shader);

  /* force smooth shading for displacement */
  shader |= SHADER_SMOOTH_NORMAL;

  shader_setup_from_sample(
      kg,
      sd,
      P,
      Ng,
      I,
      shader,
      object,
      prim,
      u,
      v,
      0.0f,
      0.5f,
      !(kernel_tex_fetch(__object_flag, object) & SD_OBJECT_TRANSFORM_APPLIED),
      LAMP_NONE);
}

/* ShaderData setup from ray into background */

ccl_device_inline void shader_setup_from_background(const KernelGlobals *ccl_restrict kg,
                                                    ShaderData *ccl_restrict sd,
                                                    const float3 ray_P,
                                                    const float3 ray_D,
                                                    const float ray_time)
{
  PROFILING_INIT(kg, PROFILING_SHADER_SETUP);

  /* for NDC coordinates */
  sd->ray_P = ray_P;

  /* vectors */
  sd->P = ray_D;
  sd->N = -ray_D;
  sd->Ng = -ray_D;
  sd->I = -ray_D;
  sd->shader = kernel_data.background.surface_shader;
  sd->flag = kernel_tex_fetch(__shaders, (sd->shader & SHADER_MASK)).flags;
  sd->object_flag = 0;
  sd->time = ray_time;
  sd->ray_length = 0.0f;

  sd->object = OBJECT_NONE;
  sd->lamp = LAMP_NONE;
  sd->prim = PRIM_NONE;
  sd->u = 0.0f;
  sd->v = 0.0f;

#ifdef __DPDU__
  /* dPdu/dPdv */
  sd->dPdu = zero_float3();
  sd->dPdv = zero_float3();
#endif

#ifdef __RAY_DIFFERENTIALS__
  /* differentials */
  sd->dP = differential3_zero(); /* TODO: ray->dP */
  differential_incoming(&sd->dI, sd->dP);
  sd->du = differential_zero();
  sd->dv = differential_zero();
#endif

  PROFILING_SHADER(sd->shader);
  PROFILING_OBJECT(sd->object);
}

/* ShaderData setup from point inside volume */

#ifdef __VOLUME__
ccl_device_inline void shader_setup_from_volume(const KernelGlobals *ccl_restrict kg,
                                                ShaderData *ccl_restrict sd,
                                                const Ray *ccl_restrict ray)
{
  PROFILING_INIT(kg, PROFILING_SHADER_SETUP);

  /* vectors */
  sd->P = ray->P;
  sd->N = -ray->D;
  sd->Ng = -ray->D;
  sd->I = -ray->D;
  sd->shader = SHADER_NONE;
  sd->flag = 0;
  sd->object_flag = 0;
  sd->time = ray->time;
  sd->ray_length = 0.0f; /* todo: can we set this to some useful value? */

  sd->object = OBJECT_NONE; /* todo: fill this for texture coordinates */
  sd->lamp = LAMP_NONE;
  sd->prim = PRIM_NONE;
  sd->type = PRIMITIVE_NONE;

  sd->u = 0.0f;
  sd->v = 0.0f;

#  ifdef __DPDU__
  /* dPdu/dPdv */
  sd->dPdu = zero_float3();
  sd->dPdv = zero_float3();
#  endif

#  ifdef __RAY_DIFFERENTIALS__
  /* differentials */
  sd->dP = differential3_zero(); /* TODO ray->dD */
  differential_incoming(&sd->dI, sd->dP);
  sd->du = differential_zero();
  sd->dv = differential_zero();
#  endif

  /* for NDC coordinates */
  sd->ray_P = ray->P;
  sd->ray_dP = ray->dP;

  PROFILING_SHADER(sd->shader);
  PROFILING_OBJECT(sd->object);
}
#endif /* __VOLUME__ */

/* Merging */

#if defined(__VOLUME__)
ccl_device_inline void shader_merge_closures(ShaderData *sd)
{
  /* merge identical closures, better when we sample a single closure at a time */
  for (int i = 0; i < sd->num_closure; i++) {
    ShaderClosure *sci = &sd->closure[i];

    for (int j = i + 1; j < sd->num_closure; j++) {
      ShaderClosure *scj = &sd->closure[j];

      if (sci->type != scj->type)
        continue;
      if (!bsdf_merge(sci, scj))
        continue;

      sci->weight += scj->weight;
      sci->sample_weight += scj->sample_weight;

      int size = sd->num_closure - (j + 1);
      if (size > 0) {
        for (int k = 0; k < size; k++) {
          scj[k] = scj[k + 1];
        }
      }

      sd->num_closure--;
      kernel_assert(sd->num_closure >= 0);
      j--;
    }
  }
}
#endif /* __VOLUME__ */

ccl_device_inline void shader_prepare_closures(INTEGRATOR_STATE_CONST_ARGS, ShaderData *sd)
{
  /* Defensive sampling.
   *
   * We can likely also do defensive sampling at deeper bounces, particularly
   * for cases like a perfect mirror but possibly also others. This will need
   * a good heuristic. */
  if (INTEGRATOR_STATE(path, bounce) + INTEGRATOR_STATE(path, transparent_bounce) == 0 &&
      sd->num_closure > 1) {
    float sum = 0.0f;

    for (int i = 0; i < sd->num_closure; i++) {
      ShaderClosure *sc = &sd->closure[i];
      if (CLOSURE_IS_BSDF_OR_BSSRDF(sc->type)) {
        sum += sc->sample_weight;
      }
    }

    for (int i = 0; i < sd->num_closure; i++) {
      ShaderClosure *sc = &sd->closure[i];
      if (CLOSURE_IS_BSDF_OR_BSSRDF(sc->type)) {
        sc->sample_weight = max(sc->sample_weight, 0.125f * sum);
      }
    }
  }

  /* Filter glossy.
   *
   * Blurring of bsdf after bounces, for rays that have a small likelihood
   * of following this particular path (diffuse, rough glossy) */
  if (kernel_data.integrator.filter_glossy != FLT_MAX) {
    float blur_pdf = kernel_data.integrator.filter_glossy * INTEGRATOR_STATE(path, min_ray_pdf);

    if (blur_pdf < 1.0f) {
      float blur_roughness = sqrtf(1.0f - blur_pdf) * 0.5f;

      for (int i = 0; i < sd->num_closure; i++) {
        ShaderClosure *sc = &sd->closure[i];
        if (CLOSURE_IS_BSDF(sc->type)) {
          bsdf_blur(kg, sc, blur_roughness);
        }
      }
    }
  }
}

/* BSDF */

ccl_device_inline bool shader_bsdf_is_transmission(const ShaderData *sd, const float3 omega_in)
{
  /* For curves use the smooth normal, particularly for ribbons the geometric
   * normal gives too much darkening otherwise. */
  const float3 Ng = (sd->type & PRIMITIVE_ALL_CURVE) ? sd->N : sd->Ng;

  return dot(Ng, omega_in) < 0.0f;
}

ccl_device_forceinline bool _shader_bsdf_exclude(ClosureType type, uint light_shader_flags)
{
  if (!(light_shader_flags & SHADER_EXCLUDE_ANY)) {
    return false;
  }
  if (light_shader_flags & SHADER_EXCLUDE_DIFFUSE) {
    if (CLOSURE_IS_BSDF_DIFFUSE(type) || CLOSURE_IS_BSDF_BSSRDF(type)) {
      return true;
    }
  }
  if (light_shader_flags & SHADER_EXCLUDE_GLOSSY) {
    if (CLOSURE_IS_BSDF_GLOSSY(type)) {
      return true;
    }
  }
  if (light_shader_flags & SHADER_EXCLUDE_TRANSMIT) {
    if (CLOSURE_IS_BSDF_TRANSMISSION(type)) {
      return true;
    }
  }
  return false;
}

ccl_device_inline void _shader_bsdf_multi_eval(const KernelGlobals *kg,
                                               ShaderData *sd,
                                               const float3 omega_in,
                                               const bool is_transmission,
                                               float *pdf,
                                               const ShaderClosure *skip_sc,
                                               BsdfEval *result_eval,
                                               float sum_pdf,
                                               float sum_sample_weight,
                                               const uint light_shader_flags)
{
  /* this is the veach one-sample model with balance heuristic, some pdf
   * factors drop out when using balance heuristic weighting */
  for (int i = 0; i < sd->num_closure; i++) {
    const ShaderClosure *sc = &sd->closure[i];

    if (sc == skip_sc) {
      continue;
    }

    if (CLOSURE_IS_BSDF_OR_BSSRDF(sc->type)) {
      if (CLOSURE_IS_BSDF(sc->type) && !_shader_bsdf_exclude(sc->type, light_shader_flags)) {
        float bsdf_pdf = 0.0f;
        float3 eval = bsdf_eval(kg, sd, sc, omega_in, is_transmission, &bsdf_pdf);

        if (bsdf_pdf != 0.0f) {
          const bool is_diffuse = (CLOSURE_IS_BSDF_DIFFUSE(sc->type) ||
                                   CLOSURE_IS_BSDF_BSSRDF(sc->type));
          bsdf_eval_accum(result_eval, is_diffuse, eval * sc->weight, 1.0f);
          sum_pdf += bsdf_pdf * sc->sample_weight;
        }
      }

      sum_sample_weight += sc->sample_weight;
    }
  }

  *pdf = (sum_sample_weight > 0.0f) ? sum_pdf / sum_sample_weight : 0.0f;
}

#ifndef __KERNEL_CUDA__
ccl_device
#else
ccl_device_inline
#endif
    void
    shader_bsdf_eval(const KernelGlobals *kg,
                     ShaderData *sd,
                     const float3 omega_in,
                     const bool is_transmission,
                     BsdfEval *eval,
                     const float light_pdf,
                     const uint light_shader_flags)
{
  PROFILING_INIT(kg, PROFILING_CLOSURE_EVAL);

  bsdf_eval_init(eval, false, zero_float3(), kernel_data.film.use_light_pass);

  float pdf;
  _shader_bsdf_multi_eval(
      kg, sd, omega_in, is_transmission, &pdf, NULL, eval, 0.0f, 0.0f, light_shader_flags);
  if (light_shader_flags & SHADER_USE_MIS) {
    float weight = power_heuristic(light_pdf, pdf);
    bsdf_eval_mul(eval, weight);
  }
}

/* Randomly sample a BSSRDF or BSDF proportional to ShaderClosure.sample_weight. */
ccl_device_inline const ShaderClosure *shader_bsdf_bssrdf_pick(const ShaderData *ccl_restrict sd,
                                                               float *randu)
{
  int sampled = 0;

  if (sd->num_closure > 1) {
    /* Pick a BSDF or based on sample weights. */
    float sum = 0.0f;

    for (int i = 0; i < sd->num_closure; i++) {
      const ShaderClosure *sc = &sd->closure[i];

      if (CLOSURE_IS_BSDF_OR_BSSRDF(sc->type)) {
        sum += sc->sample_weight;
      }
    }

    float r = (*randu) * sum;
    float partial_sum = 0.0f;

    for (int i = 0; i < sd->num_closure; i++) {
      const ShaderClosure *sc = &sd->closure[i];

      if (CLOSURE_IS_BSDF_OR_BSSRDF(sc->type)) {
        float next_sum = partial_sum + sc->sample_weight;

        if (r < next_sum) {
          sampled = i;

          /* Rescale to reuse for direction sample, to better preserve stratification. */
          *randu = (r - partial_sum) / sc->sample_weight;
          break;
        }

        partial_sum = next_sum;
      }
    }
  }

  return &sd->closure[sampled];
}

/* Return weight for picked BSSRDF. */
ccl_device_inline float3 shader_bssrdf_sample_weight(const ShaderData *ccl_restrict sd,
                                                     const ShaderClosure *ccl_restrict bssrdf_sc)
{
  float3 weight = bssrdf_sc->weight;

  if (sd->num_closure > 1) {
    float sum = 0.0f;
    for (int i = 0; i < sd->num_closure; i++) {
      const ShaderClosure *sc = &sd->closure[i];

      if (CLOSURE_IS_BSDF_OR_BSSRDF(sc->type)) {
        sum += sc->sample_weight;
      }
    }
    weight *= sum / bssrdf_sc->sample_weight;
  }

  return weight;
}

/* Sample direction for picked BSDF, and return evaluation and pdf for all
 * BSDFs combined using MIS. */
ccl_device int shader_bsdf_sample_closure(const KernelGlobals *kg,
                                          ShaderData *sd,
                                          const ShaderClosure *sc,
                                          float randu,
                                          float randv,
                                          BsdfEval *bsdf_eval,
                                          float3 *omega_in,
                                          differential3 *domega_in,
                                          float *pdf)
{
  PROFILING_INIT(kg, PROFILING_CLOSURE_SAMPLE);

  /* BSSRDF should already have been handled elsewhere. */
  kernel_assert(CLOSURE_IS_BSDF(sc->type));

  int label;
  float3 eval = zero_float3();

  *pdf = 0.0f;
  label = bsdf_sample(kg, sd, sc, randu, randv, &eval, omega_in, domega_in, pdf);

  if (*pdf != 0.0f) {
    const bool is_diffuse = (CLOSURE_IS_BSDF_DIFFUSE(sc->type) ||
                             CLOSURE_IS_BSDF_BSSRDF(sc->type));
    bsdf_eval_init(bsdf_eval, is_diffuse, eval * sc->weight, kernel_data.film.use_light_pass);

    if (sd->num_closure > 1) {
      const bool is_transmission = shader_bsdf_is_transmission(sd, *omega_in);
      float sweight = sc->sample_weight;
      _shader_bsdf_multi_eval(
          kg, sd, *omega_in, is_transmission, pdf, sc, bsdf_eval, *pdf * sweight, sweight, 0);
    }
  }

  return label;
}

ccl_device float shader_bsdf_average_roughness(const ShaderData *sd)
{
  float roughness = 0.0f;
  float sum_weight = 0.0f;

  for (int i = 0; i < sd->num_closure; i++) {
    const ShaderClosure *sc = &sd->closure[i];

    if (CLOSURE_IS_BSDF(sc->type)) {
      /* sqrt once to undo the squaring from multiplying roughness on the
       * two axes, and once for the squared roughness convention. */
      float weight = fabsf(average(sc->weight));
      roughness += weight * sqrtf(safe_sqrtf(bsdf_get_roughness_squared(sc)));
      sum_weight += weight;
    }
  }

  return (sum_weight > 0.0f) ? roughness / sum_weight : 0.0f;
}

ccl_device float3 shader_bsdf_transparency(const KernelGlobals *kg, const ShaderData *sd)
{
  if (sd->flag & SD_HAS_ONLY_VOLUME) {
    return one_float3();
  }
  else if (sd->flag & SD_TRANSPARENT) {
    return sd->closure_transparent_extinction;
  }
  else {
    return zero_float3();
  }
}

ccl_device void shader_bsdf_disable_transparency(const KernelGlobals *kg, ShaderData *sd)
{
  if (sd->flag & SD_TRANSPARENT) {
    for (int i = 0; i < sd->num_closure; i++) {
      ShaderClosure *sc = &sd->closure[i];

      if (sc->type == CLOSURE_BSDF_TRANSPARENT_ID) {
        sc->sample_weight = 0.0f;
        sc->weight = zero_float3();
      }
    }

    sd->flag &= ~SD_TRANSPARENT;
  }
}

ccl_device float3 shader_bsdf_alpha(const KernelGlobals *kg, const ShaderData *sd)
{
  float3 alpha = one_float3() - shader_bsdf_transparency(kg, sd);

  alpha = max(alpha, zero_float3());
  alpha = min(alpha, one_float3());

  return alpha;
}

ccl_device float3 shader_bsdf_diffuse(const KernelGlobals *kg, const ShaderData *sd)
{
  float3 eval = zero_float3();

  for (int i = 0; i < sd->num_closure; i++) {
    const ShaderClosure *sc = &sd->closure[i];

    if (CLOSURE_IS_BSDF_DIFFUSE(sc->type) || CLOSURE_IS_BSSRDF(sc->type) ||
        CLOSURE_IS_BSDF_BSSRDF(sc->type))
      eval += sc->weight;
  }

  return eval;
}

ccl_device float3 shader_bsdf_glossy(const KernelGlobals *kg, const ShaderData *sd)
{
  float3 eval = zero_float3();

  for (int i = 0; i < sd->num_closure; i++) {
    const ShaderClosure *sc = &sd->closure[i];

    if (CLOSURE_IS_BSDF_GLOSSY(sc->type))
      eval += sc->weight;
  }

  return eval;
}

ccl_device float3 shader_bsdf_transmission(const KernelGlobals *kg, const ShaderData *sd)
{
  float3 eval = zero_float3();

  for (int i = 0; i < sd->num_closure; i++) {
    const ShaderClosure *sc = &sd->closure[i];

    if (CLOSURE_IS_BSDF_TRANSMISSION(sc->type))
      eval += sc->weight;
  }

  return eval;
}

ccl_device float3 shader_bsdf_average_normal(const KernelGlobals *kg, const ShaderData *sd)
{
  float3 N = zero_float3();

  for (int i = 0; i < sd->num_closure; i++) {
    const ShaderClosure *sc = &sd->closure[i];
    if (CLOSURE_IS_BSDF_OR_BSSRDF(sc->type))
      N += sc->N * fabsf(average(sc->weight));
  }

  return (is_zero(N)) ? sd->N : normalize(N);
}

#ifdef __SUBSURFACE__
ccl_device float3 shader_bssrdf_normal(const ShaderData *sd)
{
  float3 N = zero_float3();

  for (int i = 0; i < sd->num_closure; i++) {
    const ShaderClosure *sc = &sd->closure[i];

    if (CLOSURE_IS_BSSRDF(sc->type)) {
      const Bssrdf *bssrdf = (const Bssrdf *)sc;
      float avg_weight = fabsf(average(sc->weight));

      N += bssrdf->N * avg_weight;
    }
  }

  return (is_zero(N)) ? sd->N : normalize(N);
}
#endif /* __SUBSURFACE__ */

/* Constant emission optimization */

ccl_device bool shader_constant_emission_eval(const KernelGlobals *kg, int shader, float3 *eval)
{
  int shader_index = shader & SHADER_MASK;
  int shader_flag = kernel_tex_fetch(__shaders, shader_index).flags;

  if (shader_flag & SD_HAS_CONSTANT_EMISSION) {
    *eval = make_float3(kernel_tex_fetch(__shaders, shader_index).constant_emission[0],
                        kernel_tex_fetch(__shaders, shader_index).constant_emission[1],
                        kernel_tex_fetch(__shaders, shader_index).constant_emission[2]);

    return true;
  }

  return false;
}

/* Background */

ccl_device float3 shader_background_eval(const ShaderData *sd)
{
  if (sd->flag & SD_EMISSION) {
    return sd->closure_emission_background;
  }
  else {
    return zero_float3();
  }
}

/* Emission */

ccl_device float3 shader_emissive_eval(const ShaderData *sd)
{
  if (sd->flag & SD_EMISSION) {
    return emissive_simple_eval(sd->Ng, sd->I) * sd->closure_emission_background;
  }
  else {
    return zero_float3();
  }
}

/* Holdout */

ccl_device float3 shader_holdout_apply(const KernelGlobals *kg, ShaderData *sd)
{
  float3 weight = zero_float3();

  /* For objects marked as holdout, preserve transparency and remove all other
   * closures, replacing them with a holdout weight. */
  if (sd->object_flag & SD_OBJECT_HOLDOUT_MASK) {
    if ((sd->flag & SD_TRANSPARENT) && !(sd->flag & SD_HAS_ONLY_VOLUME)) {
      weight = one_float3() - sd->closure_transparent_extinction;

      for (int i = 0; i < sd->num_closure; i++) {
        ShaderClosure *sc = &sd->closure[i];
        if (!CLOSURE_IS_BSDF_TRANSPARENT(sc->type)) {
          sc->type = NBUILTIN_CLOSURES;
        }
      }

      sd->flag &= ~(SD_CLOSURE_FLAGS - (SD_TRANSPARENT | SD_BSDF));
    }
    else {
      weight = one_float3();
    }
  }
  else {
    for (int i = 0; i < sd->num_closure; i++) {
      const ShaderClosure *sc = &sd->closure[i];
      if (CLOSURE_IS_HOLDOUT(sc->type)) {
        weight += sc->weight;
      }
    }
  }

  return weight;
}

/* Surface Evaluation */

template<uint node_feature_mask>
ccl_device void shader_eval_surface(INTEGRATOR_STATE_CONST_ARGS,
                                    ShaderData *ccl_restrict sd,
                                    ccl_global float *ccl_restrict buffer,
                                    int path_flag)
{
  PROFILING_INIT(kg, PROFILING_SHADER_EVAL);

  /* If path is being terminated, we are tracing a shadow ray or evaluating
   * emission, then we don't need to store closures. The emission and shadow
   * shader data also do not have a closure array to save GPU memory. */
  int max_closures;
  if (path_flag & (PATH_RAY_TERMINATE | PATH_RAY_SHADOW | PATH_RAY_EMISSION)) {
    max_closures = 0;
  }
  else {
    max_closures = kernel_data.integrator.max_closures;
  }

  sd->num_closure = 0;
  sd->num_closure_left = max_closures;

#ifdef __OSL__
  if (kg->osl) {
    if (sd->object == OBJECT_NONE && sd->lamp == LAMP_NONE) {
      OSLShader::eval_background(INTEGRATOR_STATE_PASS, sd, path_flag);
    }
    else {
      OSLShader::eval_surface(INTEGRATOR_STATE_PASS, sd, path_flag);
    }
  }
  else
#endif
  {
#ifdef __SVM__
    svm_eval_nodes<node_feature_mask, SHADER_TYPE_SURFACE>(
        INTEGRATOR_STATE_PASS, sd, buffer, path_flag);
#else
    if (sd->object == OBJECT_NONE) {
      sd->closure_emission_background = make_float3(0.8f, 0.8f, 0.8f);
      sd->flag |= SD_EMISSION;
    }
    else {
      DiffuseBsdf *bsdf = (DiffuseBsdf *)bsdf_alloc(
          sd, sizeof(DiffuseBsdf), make_float3(0.8f, 0.8f, 0.8f));
      if (bsdf != NULL) {
        bsdf->N = sd->N;
        sd->flag |= bsdf_diffuse_setup(bsdf);
      }
    }
#endif
  }

  if (NODES_FEATURE(BSDF) && (sd->flag & SD_BSDF_NEEDS_LCG)) {
    sd->lcg_state = lcg_state_init(INTEGRATOR_STATE(path, rng_hash),
                                   INTEGRATOR_STATE(path, rng_offset),
                                   INTEGRATOR_STATE(path, sample),
                                   0xb4bc3953);
  }
}

/* Volume */

#ifdef __VOLUME__

ccl_device_inline void _shader_volume_phase_multi_eval(const ShaderData *sd,
                                                       const float3 omega_in,
                                                       float *pdf,
                                                       int skip_phase,
                                                       BsdfEval *result_eval,
                                                       float sum_pdf,
                                                       float sum_sample_weight)
{
  for (int i = 0; i < sd->num_closure; i++) {
    if (i == skip_phase)
      continue;

    const ShaderClosure *sc = &sd->closure[i];

    if (CLOSURE_IS_PHASE(sc->type)) {
      float phase_pdf = 0.0f;
      float3 eval = volume_phase_eval(sd, sc, omega_in, &phase_pdf);

      if (phase_pdf != 0.0f) {
        bsdf_eval_accum(result_eval, false, eval, 1.0f);
        sum_pdf += phase_pdf * sc->sample_weight;
      }

      sum_sample_weight += sc->sample_weight;
    }
  }

  *pdf = (sum_sample_weight > 0.0f) ? sum_pdf / sum_sample_weight : 0.0f;
}

ccl_device void shader_volume_phase_eval(const KernelGlobals *kg,
                                         const ShaderData *sd,
                                         const float3 omega_in,
                                         BsdfEval *eval,
                                         float *pdf)
{
  PROFILING_INIT(kg, PROFILING_CLOSURE_VOLUME_EVAL);

  bsdf_eval_init(eval, false, zero_float3(), kernel_data.film.use_light_pass);

  _shader_volume_phase_multi_eval(sd, omega_in, pdf, -1, eval, 0.0f, 0.0f);
}

ccl_device int shader_volume_phase_sample(const KernelGlobals *kg,
                                          const ShaderData *sd,
                                          float randu,
                                          float randv,
                                          BsdfEval *phase_eval,
                                          float3 *omega_in,
                                          differential3 *domega_in,
                                          float *pdf)
{
  PROFILING_INIT(kg, PROFILING_CLOSURE_VOLUME_SAMPLE);

  int sampled = 0;

  if (sd->num_closure > 1) {
    /* pick a phase closure based on sample weights */
    float sum = 0.0f;

    for (sampled = 0; sampled < sd->num_closure; sampled++) {
      const ShaderClosure *sc = &sd->closure[sampled];

      if (CLOSURE_IS_PHASE(sc->type))
        sum += sc->sample_weight;
    }

    float r = randu * sum;
    float partial_sum = 0.0f;

    for (sampled = 0; sampled < sd->num_closure; sampled++) {
      const ShaderClosure *sc = &sd->closure[sampled];

      if (CLOSURE_IS_PHASE(sc->type)) {
        float next_sum = partial_sum + sc->sample_weight;

        if (r <= next_sum) {
          /* Rescale to reuse for BSDF direction sample. */
          randu = (r - partial_sum) / sc->sample_weight;
          break;
        }

        partial_sum = next_sum;
      }
    }

    if (sampled == sd->num_closure) {
      *pdf = 0.0f;
      return LABEL_NONE;
    }
  }

  /* todo: this isn't quite correct, we don't weight anisotropy properly
   * depending on color channels, even if this is perhaps not a common case */
  const ShaderClosure *sc = &sd->closure[sampled];
  int label;
  float3 eval = zero_float3();

  *pdf = 0.0f;
  label = volume_phase_sample(sd, sc, randu, randv, &eval, omega_in, domega_in, pdf);

  if (*pdf != 0.0f) {
    bsdf_eval_init(phase_eval, false, eval, kernel_data.film.use_light_pass);
  }

  return label;
}

ccl_device int shader_phase_sample_closure(const KernelGlobals *kg,
                                           const ShaderData *sd,
                                           const ShaderClosure *sc,
                                           float randu,
                                           float randv,
                                           BsdfEval *phase_eval,
                                           float3 *omega_in,
                                           differential3 *domega_in,
                                           float *pdf)
{
  PROFILING_INIT(kg, PROFILING_CLOSURE_VOLUME_SAMPLE);

  int label;
  float3 eval = zero_float3();

  *pdf = 0.0f;
  label = volume_phase_sample(sd, sc, randu, randv, &eval, omega_in, domega_in, pdf);

  if (*pdf != 0.0f)
    bsdf_eval_init(phase_eval, false, eval, kernel_data.film.use_light_pass);

  return label;
}

/* Volume Evaluation */

template<typename StackReadOp>
ccl_device_inline void shader_eval_volume(INTEGRATOR_STATE_CONST_ARGS,
                                          ShaderData *ccl_restrict sd,
                                          const int path_flag,
                                          StackReadOp stack_read)
{
  /* If path is being terminated, we are tracing a shadow ray or evaluating
   * emission, then we don't need to store closures. The emission and shadow
   * shader data also do not have a closure array to save GPU memory. */
  int max_closures;
  if (path_flag & (PATH_RAY_TERMINATE | PATH_RAY_SHADOW | PATH_RAY_EMISSION)) {
    max_closures = 0;
  }
  else {
    max_closures = kernel_data.integrator.max_closures;
  }

  /* reset closures once at the start, we will be accumulating the closures
   * for all volumes in the stack into a single array of closures */
  sd->num_closure = 0;
  sd->num_closure_left = max_closures;
  sd->flag = 0;
  sd->object_flag = 0;

  for (int i = 0;; i++) {
    const VolumeStack entry = stack_read(i);
    if (entry.shader == SHADER_NONE) {
      break;
    }

    /* setup shaderdata from stack. it's mostly setup already in
     * shader_setup_from_volume, this switching should be quick */
    sd->object = entry.object;
    sd->lamp = LAMP_NONE;
    sd->shader = entry.shader;

    sd->flag &= ~SD_SHADER_FLAGS;
    sd->flag |= kernel_tex_fetch(__shaders, (sd->shader & SHADER_MASK)).flags;
    sd->object_flag &= ~SD_OBJECT_FLAGS;

    if (sd->object != OBJECT_NONE) {
      sd->object_flag |= kernel_tex_fetch(__object_flag, sd->object);

#  ifdef __OBJECT_MOTION__
      /* todo: this is inefficient for motion blur, we should be
       * caching matrices instead of recomputing them each step */
      shader_setup_object_transforms(kg, sd, sd->time);
#  endif
    }

    /* evaluate shader */
#  ifdef __SVM__
#    ifdef __OSL__
    if (kg->osl) {
      OSLShader::eval_volume(INTEGRATOR_STATE_PASS, sd, path_flag);
    }
    else
#    endif
    {
      svm_eval_nodes<NODE_FEATURE_MASK_VOLUME, SHADER_TYPE_VOLUME>(
          INTEGRATOR_STATE_PASS, sd, NULL, path_flag);
    }
#  endif

    /* merge closures to avoid exceeding number of closures limit */
    if (i > 0)
      shader_merge_closures(sd);
  }
}

#endif /* __VOLUME__ */

/* Displacement Evaluation */

ccl_device void shader_eval_displacement(INTEGRATOR_STATE_CONST_ARGS, ShaderData *sd)
{
  sd->num_closure = 0;
  sd->num_closure_left = 0;

  /* this will modify sd->P */
#ifdef __SVM__
#  ifdef __OSL__
  if (kg->osl)
    OSLShader::eval_displacement(INTEGRATOR_STATE_PASS, sd);
  else
#  endif
  {
    svm_eval_nodes<NODE_FEATURE_MASK_DISPLACEMENT, SHADER_TYPE_DISPLACEMENT>(
        INTEGRATOR_STATE_PASS, sd, NULL, 0);
  }
#endif
}

/* Transparent Shadows */

#ifdef __TRANSPARENT_SHADOWS__
ccl_device bool shader_transparent_shadow(const KernelGlobals *kg, Intersection *isect)
{
  return (intersection_get_shader_flags(kg, isect) & SD_HAS_TRANSPARENT_SHADOW) != 0;
}
#endif /* __TRANSPARENT_SHADOWS__ */

ccl_device float shader_cryptomatte_id(const KernelGlobals *kg, int shader)
{
  return kernel_tex_fetch(__shaders, (shader & SHADER_MASK)).cryptomatte_id;
}

CCL_NAMESPACE_END
