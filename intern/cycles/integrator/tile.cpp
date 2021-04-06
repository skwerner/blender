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

std::ostream &operator<<(std::ostream &os, const TileSize &tile_size)
{
  os << "size: (" << tile_size.width << ", " << tile_size.height << ")";
  os << ", num_samples: " << tile_size.num_samples;
  return os;
}

ccl_device_inline uint round_down_to_power_of_two(uint x)
{
  if (is_power_of_two(x)) {
    return x;
  }

  return prev_power_of_two(x);
}

TileSize tile_calculate_best_size(const int2 &image_size,
                                  const int num_samples,
                                  const int max_num_path_states)
{
  if (max_num_path_states == 1) {
    /* Simple case: avoid any calculation, which could cause rounding issues. */
    return TileSize(1, 1, 1);
  }

  const int64_t num_pixels = image_size.x * image_size.y;
  const int64_t num_pixel_samples = num_pixels * num_samples;

  if (max_num_path_states >= num_pixel_samples) {
    /* Image fully fits into the state (could be border render, for example). */
    return TileSize(image_size.x, image_size.y, num_samples);
  }

  /* The idea here is to keep number of samples per tile as much as possible to improve coherency
   * across threads. */

  const int num_path_states_per_sample = max(max_num_path_states / num_samples, 1);

  TileSize tile_size;

  if (true) {
    /* Occupy as much of GPU threads as possible by the single tile.
     * This could cause non-optimal load due to "wasted" path states (due to non-integer division)
     * but currently it gives better performance. Possibly that coalescing will help with. */
    tile_size.width = max(static_cast<int>(lround(sqrt(num_path_states_per_sample))), 1);
    tile_size.height = max(num_path_states_per_sample / tile_size.width, 1);
  }
  else {
    /* Round down to the power of two, so that all path states are occupied. */
    /* TODO(sergey): Investigate why this is slower than the scheduling based on the code above and
     * use this scheduling strategy instead. */
    tile_size.width = round_down_to_power_of_two(
        max(static_cast<int>(lround(sqrt(num_path_states_per_sample))), 1));
    tile_size.height = tile_size.width;
  }

  tile_size.num_samples = min(num_samples,
                              max_num_path_states / (tile_size.width * tile_size.height));

  DCHECK_LE(tile_size.width * tile_size.height * tile_size.num_samples, max_num_path_states);

  return tile_size;
}

CCL_NAMESPACE_END
