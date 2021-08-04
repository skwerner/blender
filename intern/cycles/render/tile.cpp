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

void TileManager::reset(const BufferParams &params, int2 tile_size)
{
  tile_size_ = tile_size;
  buffer_params_ = params;

  num_tiles_x_ = divide_up(params.width, tile_size_.x);
  num_tiles_y_ = divide_up(params.height, tile_size_.y);

  state_.next_tile_index = 0;
  state_.num_tiles = num_tiles_x_ * num_tiles_y_;

  state_.current_tile = Tile();
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

  /* TODO(sergey): Consider using hilbert spiral, or. maybe, even configurable. Not sure this
   * brings a lot of value since this is only applicable to BIG tiles. */

  const int tile_y = state_.next_tile_index / num_tiles_x_;
  const int tile_x = state_.next_tile_index - tile_y * num_tiles_x_;

  state_.current_tile.x = tile_x * tile_size_.x;
  state_.current_tile.y = tile_y * tile_size_.y;
  state_.current_tile.width = tile_size_.x;
  state_.current_tile.height = tile_size_.y;

  state_.current_tile.width = min(state_.current_tile.width,
                                  buffer_params_.width - state_.current_tile.x);
  state_.current_tile.height = min(state_.current_tile.height,
                                   buffer_params_.height - state_.current_tile.y);

  ++state_.next_tile_index;

  return true;
}

const Tile &TileManager::get_current_tile() const
{
  return state_.current_tile;
}

CCL_NAMESPACE_END
