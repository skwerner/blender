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

#include "testing/testing.h"

#include "integrator/adaptive_sampling.h"

CCL_NAMESPACE_BEGIN

TEST(AdaptiveSampling, schedule_samples)
{
  AdaptiveSampling adaptive_sampling;
  adaptive_sampling.use = true;
  adaptive_sampling.min_samples = 0;
  adaptive_sampling.adaptive_step = 4;

  for (int sample = 2; sample < 32; ++sample) {
    for (int num_samples = 8; num_samples < 32; ++num_samples) {
      const int num_samples_aligned = adaptive_sampling.align_samples(sample, num_samples);
      /* NOTE: `sample + num_samples_aligned` is the number of samples after rendering, so need
       * to convert this to the 0-based index of the last sample. */
      EXPECT_TRUE(adaptive_sampling.need_filter(sample + num_samples_aligned - 1));
    }
  }
}

TEST(AdaptiveSampling, align_samples)
{
  AdaptiveSampling adaptive_sampling;
  adaptive_sampling.use = true;
  adaptive_sampling.min_samples = 11 /* rounded of sqrt(128) */;
  adaptive_sampling.adaptive_step = 4;

  EXPECT_EQ(adaptive_sampling.align_samples(0, 4), 4);
  EXPECT_EQ(adaptive_sampling.align_samples(0, 7), 7);

  EXPECT_EQ(adaptive_sampling.align_samples(0, 15), 12);
  EXPECT_EQ(adaptive_sampling.align_samples(0, 16), 16);
  EXPECT_EQ(adaptive_sampling.align_samples(0, 17), 16);
  EXPECT_EQ(adaptive_sampling.align_samples(0, 20), 20);

  EXPECT_EQ(adaptive_sampling.align_samples(9, 8), 7);

  EXPECT_EQ(adaptive_sampling.align_samples(12, 6), 4);
}

TEST(AdaptiveSampling, need_filter)
{
  AdaptiveSampling adaptive_sampling;
  adaptive_sampling.use = true;
  adaptive_sampling.min_samples = 11 /* rounded of sqrt(128) */;
  adaptive_sampling.adaptive_step = 4;

  EXPECT_FALSE(adaptive_sampling.need_filter(0));
  EXPECT_FALSE(adaptive_sampling.need_filter(3));
  EXPECT_FALSE(adaptive_sampling.need_filter(7));
  EXPECT_FALSE(adaptive_sampling.need_filter(11));

  EXPECT_FALSE(adaptive_sampling.need_filter(14));
  EXPECT_TRUE(adaptive_sampling.need_filter(15));
  EXPECT_FALSE(adaptive_sampling.need_filter(16));

  EXPECT_FALSE(adaptive_sampling.need_filter(18));
  EXPECT_TRUE(adaptive_sampling.need_filter(19));
  EXPECT_FALSE(adaptive_sampling.need_filter(20));
}

CCL_NAMESPACE_END
