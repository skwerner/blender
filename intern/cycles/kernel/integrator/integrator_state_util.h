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

#include "kernel/integrator/integrator_state.h"
#include "kernel/kernel_differential.h"

CCL_NAMESPACE_BEGIN

/* Ray */

ccl_device_forceinline void integrator_state_write_ray(INTEGRATOR_STATE_ARGS,
                                                       const Ray *ccl_restrict ray)
{
  INTEGRATOR_STATE_WRITE(ray, P) = ray->P;
  INTEGRATOR_STATE_WRITE(ray, D) = ray->D;
  INTEGRATOR_STATE_WRITE(ray, t) = ray->t;
  INTEGRATOR_STATE_WRITE(ray, time) = ray->time;
  INTEGRATOR_STATE_WRITE(ray, dP) = ray->dP;
  INTEGRATOR_STATE_WRITE(ray, dD) = ray->dD;
}

ccl_device_forceinline void integrator_state_read_ray(INTEGRATOR_STATE_CONST_ARGS,
                                                      Ray *ccl_restrict ray)
{
  ray->P = INTEGRATOR_STATE(ray, P);
  ray->D = INTEGRATOR_STATE(ray, D);
  ray->t = INTEGRATOR_STATE(ray, t);
  ray->time = INTEGRATOR_STATE(ray, time);
  ray->dP = INTEGRATOR_STATE(ray, dP);
  ray->dD = INTEGRATOR_STATE(ray, dD);
}

/* Shadow Ray */

ccl_device_forceinline void integrator_state_write_shadow_ray(INTEGRATOR_STATE_ARGS,
                                                              const Ray *ccl_restrict ray)
{
  INTEGRATOR_STATE_WRITE(shadow_ray, P) = ray->P;
  INTEGRATOR_STATE_WRITE(shadow_ray, D) = ray->D;
  INTEGRATOR_STATE_WRITE(shadow_ray, t) = ray->t;
  INTEGRATOR_STATE_WRITE(shadow_ray, time) = ray->time;
}

ccl_device_forceinline void integrator_state_read_shadow_ray(INTEGRATOR_STATE_CONST_ARGS,
                                                             Ray *ccl_restrict ray)
{
  ray->P = INTEGRATOR_STATE(shadow_ray, P);
  ray->D = INTEGRATOR_STATE(shadow_ray, D);
  ray->t = INTEGRATOR_STATE(shadow_ray, t);
  ray->time = INTEGRATOR_STATE(shadow_ray, time);
  ray->dP = differential_zero_compact();
  ray->dD = differential_zero_compact();
}

/* Intersection */

ccl_device_forceinline void integrator_state_write_isect(INTEGRATOR_STATE_ARGS,
                                                         const Intersection *ccl_restrict isect)
{
  INTEGRATOR_STATE_WRITE(isect, t) = isect->t;
  INTEGRATOR_STATE_WRITE(isect, u) = isect->u;
  INTEGRATOR_STATE_WRITE(isect, v) = isect->v;
  INTEGRATOR_STATE_WRITE(isect, object) = isect->object;
  INTEGRATOR_STATE_WRITE(isect, prim) = isect->prim;
  INTEGRATOR_STATE_WRITE(isect, type) = isect->type;
#ifdef __EMBREE__
  INTEGRATOR_STATE_WRITE(isect, Ng) = isect->Ng;
#endif
}

ccl_device_forceinline void integrator_state_read_isect(INTEGRATOR_STATE_CONST_ARGS,
                                                        Intersection *ccl_restrict isect)
{
  isect->prim = INTEGRATOR_STATE(isect, prim);
  isect->object = INTEGRATOR_STATE(isect, object);
  isect->type = INTEGRATOR_STATE(isect, type);
  isect->u = INTEGRATOR_STATE(isect, u);
  isect->v = INTEGRATOR_STATE(isect, v);
  isect->t = INTEGRATOR_STATE(isect, t);
#ifdef __EMBREE__
  isect->Ng = INTEGRATOR_STATE(isect, Ng);
#endif
}

/* Shadow Intersection */

ccl_device_forceinline void integrator_state_write_shadow_isect(
    INTEGRATOR_STATE_ARGS, const Intersection *ccl_restrict isect, const int index)
{
  INTEGRATOR_STATE_ARRAY_WRITE(shadow_isect, index, t) = isect->t;
  INTEGRATOR_STATE_ARRAY_WRITE(shadow_isect, index, u) = isect->u;
  INTEGRATOR_STATE_ARRAY_WRITE(shadow_isect, index, v) = isect->v;
  INTEGRATOR_STATE_ARRAY_WRITE(shadow_isect, index, object) = isect->object;
  INTEGRATOR_STATE_ARRAY_WRITE(shadow_isect, index, prim) = isect->prim;
  INTEGRATOR_STATE_ARRAY_WRITE(shadow_isect, index, type) = isect->type;
#ifdef __EMBREE__
  INTEGRATOR_STATE_ARRAY_WRITE(shadow_isect, index, Ng) = isect->Ng;
#endif
}

ccl_device_forceinline void integrator_state_read_shadow_isect(INTEGRATOR_STATE_CONST_ARGS,
                                                               Intersection *ccl_restrict isect,
                                                               const int index)
{
  isect->prim = INTEGRATOR_STATE_ARRAY(shadow_isect, index, prim);
  isect->object = INTEGRATOR_STATE_ARRAY(shadow_isect, index, object);
  isect->type = INTEGRATOR_STATE_ARRAY(shadow_isect, index, type);
  isect->u = INTEGRATOR_STATE_ARRAY(shadow_isect, index, u);
  isect->v = INTEGRATOR_STATE_ARRAY(shadow_isect, index, v);
  isect->t = INTEGRATOR_STATE_ARRAY(shadow_isect, index, t);
#ifdef __EMBREE__
  isect->Ng = INTEGRATOR_STATE_ARRAY(shadow_isect, index, Ng);
#endif
}

ccl_device_forceinline void integrator_state_copy_volume_stack_to_shadow(INTEGRATOR_STATE_ARGS)
{
  for (int i = 0; i < INTEGRATOR_VOLUME_STACK_SIZE; i++) {
    INTEGRATOR_STATE_ARRAY_WRITE(shadow_volume_stack, i, object) = INTEGRATOR_STATE_ARRAY(
        volume_stack, i, object);
    INTEGRATOR_STATE_ARRAY_WRITE(shadow_volume_stack, i, shader) = INTEGRATOR_STATE_ARRAY(
        volume_stack, i, shader);
  }
}

ccl_device_inline void integrator_state_copy_to_shadow_catcher(INTEGRATOR_STATE_ARGS)
{
  int index;

  /* Rely on the compiler to optimize out unused assignments and `while(false)`'s. */

#define KERNEL_STRUCT_BEGIN(name) \
  index = 0; \
  do {

#define KERNEL_STRUCT_MEMBER(parent_struct, type, name) \
  INTEGRATOR_SHADOW_CATCHER_STATE_WRITE(parent_struct, name) = INTEGRATOR_STATE(parent_struct, \
                                                                                name);

#define KERNEL_STRUCT_ARRAY_MEMBER(parent_struct, type, name) \
  INTEGRATOR_SHADOW_CATCHER_STATE_ARRAY_WRITE( \
      parent_struct, index, name) = INTEGRATOR_STATE_ARRAY(parent_struct, index, name);

#define KERNEL_STRUCT_END(name) \
  } \
  while (false) \
    ;

#define KERNEL_STRUCT_END_ARRAY(name, array_size) \
  ++index; \
  } \
  while (index < array_size) \
    ;

#include "kernel/integrator/integrator_state_template.h"

#undef KERNEL_STRUCT_BEGIN
#undef KERNEL_STRUCT_MEMBER
#undef KERNEL_STRUCT_ARRAY_MEMBER
#undef KERNEL_STRUCT_END
#undef KERNEL_STRUCT_END_ARRAY

  /* Make sure the device is aware of an extra kernel queued by the shadow catcher state. */
  INTEGRATOR_SHADOW_CATCHER_PATH_INIT();
}

CCL_NAMESPACE_END
