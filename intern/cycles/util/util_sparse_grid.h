/*
 * Copyright 2011-2017 Blender Foundation
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

#ifndef __UTIL_SPARSE_GRID_H__
#define __UTIL_SPARSE_GRID_H__

#include "util/util_types.h"
#include "util/util_vector.h"

CCL_NAMESPACE_BEGIN

static const int TILE_SIZE = 8;

struct SparseTile {
    float4 values[TILE_SIZE * TILE_SIZE * TILE_SIZE];
};

int compute_index(int x, int y, int z,
                  int width=TILE_SIZE,
                  int height=TILE_SIZE,
                  int depth=TILE_SIZE,
                  int min=0);

int3 compute_coordinates(int index,
                         int width=TILE_SIZE,
                         int height=TILE_SIZE,
                         int depth=TILE_SIZE,
                         int min=0);

int compute_tile_resolution(int res);

bool is_active(const int *offsets,
               size_t x, size_t y, size_t z,
               size_t width, size_t height, size_t depth);

int create_sparse_grid(float4 *dense_grid,
                       const size_t width,
                       const size_t height,
                       const size_t depth,
                       vector<SparseTile> *sparse_grid,
                       vector<int> *offsets);

float4 get_value(const SparseTile *grid, const int *offsets,
                 size_t x, size_t y, size_t z,
                 size_t width, size_t height, size_t depth);

float4 get_value(const SparseTile *grid, const int *offsets,
                 size_t tile_index,
                 size_t width, size_t height, size_t depth);

CCL_NAMESPACE_END

#endif /*__UTIL_SPARSE_GRID_H__*/
