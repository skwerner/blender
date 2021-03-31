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

#include "integrator/adaptive_sampling.h"

CCL_NAMESPACE_BEGIN

AdaptiveSampling::AdaptiveSampling()
{
}

int AdaptiveSampling::align_samples(int sample, int num_samples) const
{
  if (!use) {
    return num_samples;
  }

  int end_sample = sample + num_samples;

  if (end_sample <= min_samples) {
    return num_samples;
  }

  /* Round down end sample to the nearest sample that needs filtering. */
  end_sample &= ~(adaptive_step - 1);

  if (end_sample <= sample) {
    /* In order to reach the next sample that needs filtering, we'd need
     * to increase num_samples. We don't do that in this function, so
     * just keep it as is and don't filter this time around. */
    return num_samples;
  }
  return end_sample - sample;
}

bool AdaptiveSampling::need_filter(int sample) const
{
  if (!use) {
    return false;
  }

  if (sample <= min_samples) {
    return false;
  }

  return (sample & (adaptive_step - 1)) == (adaptive_step - 1);
}

CCL_NAMESPACE_END
