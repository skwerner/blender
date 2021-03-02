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

#include "integrator/tile.h"

#include "util/util_logging.h"
#include "util/util_math.h"

CCL_NAMESPACE_BEGIN

int2 tile_calculate_best_size(const int2 &image_size,
                              const int samples_num,
                              const int max_num_path_states)
{
  if (max_num_path_states == 1) {
    /* Simple case: avoid any calculation, which could cause rounding issues. */
    return make_int2(1, 1);
  }

  /* XXX: Return tile size which is known to work.
   * Proper tile wsize causes kernel to fail. Is there a mis-calculation of path integrator size
   * for CUDA on CPU and GPU? Is CPU-side allocation of GPU memory uses CPU-side structure size? */
  return make_int2(256, 256);

  const int64_t num_pixels = image_size.x * image_size.y;
  const int64_t num_pixel_samples = num_pixels * samples_num;

  if (max_num_path_states >= num_pixel_samples) {
    /* Image fully fits into the state (could be border render, for example). */
    return image_size;
  }

  /* TODO(sergey): Consider lowring the tile size when rendering multiple samples to improve
   * coherency across threads working on the tile. */

  const int tile_width = max(static_cast<int>(lround(sqrt(max_num_path_states))), 1);
  const int tile_height = max_num_path_states / tile_width;

  DCHECK_LE(tile_width * tile_height, max_num_path_states);

  return make_int2(tile_width, tile_height);
}

CCL_NAMESPACE_END
