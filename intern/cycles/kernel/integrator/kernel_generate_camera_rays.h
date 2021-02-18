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

#include "kernel/kernel_camera.h"
#include "kernel/kernel_path_state.h"
#include "kernel/kernel_random.h"

CCL_NAMESPACE_BEGIN

ccl_device_inline void integrate_camera_sample(const KernelGlobals *ccl_restrict kg,
                                               const int sample,
                                               const int x,
                                               const int y,
                                               const uint rng_hash,
                                               Ray *ray)
{
  const int num_samples = 1;

  /* Filter sampling. */
  float filter_u, filter_v;

  if (sample == 0) {
    filter_u = 0.5f;
    filter_v = 0.5f;
  }
  else {
    path_rng_2D(kg, rng_hash, sample, num_samples, PRNG_FILTER_U, &filter_u, &filter_v);
  }

  /* Depth of field sampling. */
  float lens_u = 0.0f, lens_v = 0.0f;
  if (kernel_data.cam.aperturesize > 0.0f) {
    path_rng_2D(kg, rng_hash, sample, num_samples, PRNG_LENS_U, &lens_u, &lens_v);
  }

  /* Motion blur time sampling. */
  float time = 0.0f;
#ifdef __CAMERA_MOTION__
  if (kernel_data.cam.shuttertime != -1.0f)
    time = path_rng_1D(kg, rng_hash, sample, num_samples, PRNG_TIME);
#endif

  /* Generate camera ray. */
  camera_sample(kg, x, y, filter_u, filter_v, lens_u, lens_v, time, ray);
}

ccl_device void kernel_integrate_generate_camera_rays(INTEGRATOR_STATE_ARGS)
{
  /* TODO: fill in. */
  const int x = 0;
  const int y = 0;
  const int sample = 0;

  /* Initialize random number seed for path. */
  const uint rng_hash = path_rng_hash_init(kg, sample, x, y);

  {
    /* Generate camera ray. */
    Ray ray;
    integrate_camera_sample(kg, sample, x, y, rng_hash, &ray);
    if (ray.t == 0.0f) {
      INTEGRATOR_FLOW_END;
      return;
    }

    /* Write camera ray to state. */
    INTEGRATOR_STATE_WRITE(ray, P) = ray.P;
    INTEGRATOR_STATE_WRITE(ray, D) = ray.D;
    INTEGRATOR_STATE_WRITE(ray, t) = ray.t;
    INTEGRATOR_STATE_WRITE(ray, time) = ray.time;
    /* TODO: differentials. */
  }

  /* Initialize path state. */
  path_state_init(INTEGRATOR_STATE_PASS, sample, x, y, rng_hash);

  /* Continue with intersect_closest kernel. */
  INTEGRATOR_FLOW_QUEUE(intersect_closest);
}

CCL_NAMESPACE_END
