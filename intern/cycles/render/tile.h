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

#pragma once

#include "render/buffers.h"

CCL_NAMESPACE_BEGIN

/* Tile */

/* TODO(sergey): This will become the big tile. */

#if 0
class Tile {
 public:
  int x, y;
  int width, height;

  Tile()
  {
  }

  Tile(int x, int y, int width, int height) : x(x), y(y), width(width), height(height)
  {
  }
};
#endif

/* Tile Manager */

class TileManager {
 public:
  explicit TileManager(int2 tile_size);

  void reset(BufferParams &params);

  /* TODO(sergey): Big tile request API. */

  bool next();
  bool done();

 protected:
  int2 tile_size_;
  BufferParams params_;

  struct {
    int next_tile_index;
    int num_tiles;
  } state_;
};

CCL_NAMESPACE_END
