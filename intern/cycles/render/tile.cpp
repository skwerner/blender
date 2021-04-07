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

#include "render/tile.h"

#include "util/util_algorithm.h"
#include "util/util_foreach.h"
#include "util/util_types.h"

CCL_NAMESPACE_BEGIN

TileManager::TileManager(int2 tile_size) : tile_size_(tile_size)
{
  BufferParams buffer_params;
  reset(buffer_params);
}

void TileManager::reset(BufferParams &params)
{
  params_ = params;

  state_.next_tile_index = 0;
  state_.num_tiles = 1;
}

bool TileManager::done()
{
  return state_.next_tile_index == state_.num_tiles;
}

bool TileManager::next()
{
  if (done()) {
    return false;
  }

  ++state_.next_tile_index;

  return true;
}

CCL_NAMESPACE_END
