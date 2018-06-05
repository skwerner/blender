/*
 * Copyright 2011-2018 Blender Foundation
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

#include "util/util_half.h"
#include "util/util_image.h"
#include "util/util_types.h"
#include "util/util_vector.h"

/* Functions to help generate and handle Sparse Grids.
 *
 * In this file, we use two different indexing systems:
 * in terms of voxels and in terms of tiles.
 *
 * The compute_*() functions work with either system,
 * so long as all args are in the same format.
 *
 * For all other functions:
 * - x, y, z, width, height, depth are measured by voxels.
 * - tix, tiy, tiz, tiw, tih, tid are measured by tiles.
 *
 */

CCL_NAMESPACE_BEGIN

/* For some type4 textures (e.g. color), they may still be
 * considered active even if at least one of their values
 * is beneath the threshold. Thus, we require a custom
 * function using ORs instead of ANDs.
 */
namespace {
	inline bool gt(float a, float b) { return a > b; }
	inline bool gt(uchar a, uchar b) { return a > b; }
	inline bool gt(half a, half b) { return a > b; }
	inline bool gt(float4 a, float4 b) { return a.x > b.x || a.y > b.y || a.z > b.z || a.w > b.w; }
	inline bool gt(uchar4 a, uchar4 b) { return a.x > b.x || a.y > b.y || a.z > b.z || a.w > b.w; }
	inline bool gt(half4 a, half4 b) { return a.x > b.x || a.y > b.y || a.z > b.z || a.w > b.w; }
}

static const int TILE_SIZE = 8;
static const float THRESHOLD = 0.001f;

template<typename T> struct SparseTile {
	T values[TILE_SIZE * TILE_SIZE * TILE_SIZE];
};

const inline int compute_index(const size_t x, const size_t y, const size_t z,
                               const size_t width, const size_t height, const size_t depth)
{
	if(x >= width || y >= height || z >= depth) {
		return -1;
	}
	return x + width * (y + z * height);
}

const inline int3 compute_coordinates(const size_t index, const size_t width,
                                      const size_t height, const size_t depth)
{
	if(index >= width * height * depth) {
		return make_int3(-1, -1, -1);
	}
	int x = index % width;
	int y = (index / width) % height;
	int z = index / (width * height);
	return make_int3(x, y, z);
}

const inline size_t compute_tile_resolution(const size_t res)
{
	return (res / TILE_SIZE) + !(res % TILE_SIZE == 0);
}

/* Sampling functions accept lookup coordinates in voxel format
 * and image resolution in tile format. This is because most
 * algorithms will sample one image multiple times, so it is
 * more efficient for the parent function itself to convert the
 * resolution to the tiled system only once.
 */

const inline bool tile_is_active(const int *offsets,
                                 int x, int y, int z,
                                 int tiw, int tih, int tid)
{
	int tix = x/TILE_SIZE, tiy = y/TILE_SIZE, tiz = z/TILE_SIZE;
	int dense_index = compute_index(tix, tiy, tiz, tiw, tih, tid);
	return dense_index < 0 ? false : offsets[dense_index] >= 0;
}

template<typename T>
const T get_value(const SparseTile<T> *grid, const int *offsets,
                  int x, int y, int z, int tiw, int tih, int tid)
{
	/* Get the 1D array index in the dense grid of the tile (x, y, z) is in. */
	int tix = x/TILE_SIZE, tiy = y/TILE_SIZE, tiz = z/TILE_SIZE;
	int dense_index = compute_index(tix, tiy, tiz, tiw, tih, tid);
	if(dense_index < 0) {
		return cast_from_float<T>(0.0f);
	}
	/* Get the index of the tile in the sparse grid. */
	int sparse_index = offsets[dense_index];
	if (sparse_index < 0) {
		return cast_from_float<T>(0.0f);
	}
	/* Look up voxel in the tile. */
	int in_tile_index = compute_index(x%TILE_SIZE, y%TILE_SIZE, z%TILE_SIZE,
	                                  TILE_SIZE, TILE_SIZE, TILE_SIZE);
	return grid[sparse_index].values[in_tile_index];
}

template<typename T>
int create_sparse_grid(const T *dense_grid,
                       int width, int height, int depth,
                       vector<SparseTile<T>> *sparse_grid,
                       vector<int> *offsets)
{
	if(!dense_grid) {
		return 0;
	}

	const T empty = cast_from_float<T>(0.0f);
	const T threshold = cast_from_float<T>(THRESHOLD);
	int active_tile_count = 0;
	int total_tile_count = compute_tile_resolution(width) *
	                       compute_tile_resolution(height) *
	                       compute_tile_resolution(depth);

	/* Resize vectors to tiled resolution. Have to overalloc
	 * sparse_grid because we don't know the number of
	 * active tiles yet. */
	sparse_grid->resize(total_tile_count);
	offsets->resize(total_tile_count);
	total_tile_count = 0;

	for(int z=0 ; z < depth ; z += TILE_SIZE) {
		for(int y=0 ; y < height ; y += TILE_SIZE) {
			for(int x=0 ; x < width ; x += TILE_SIZE) {

				SparseTile<T> tile;
				bool tile_is_empty = true;
				int c = 0;

				/* Populate the tile. */
				for(int k=z ; k < z+TILE_SIZE ; ++k) {
					for(int j=y ; j < y+TILE_SIZE ; ++j) {
						for(int i=x ; i < x+TILE_SIZE ; ++i) {
							int index = compute_index(i, j, k, width, height, depth);
							if(index < 0) {
								/* Out of bounds of original image
								 * store an empty voxel. */
								tile.values[c] = empty;
							}
							else {
								tile.values[c] = dense_grid[index];
								if(tile_is_empty) {
									if(gt(dense_grid[index], threshold)) {
										tile_is_empty = false;
									}
								}
							}
							++c;
						}
					}
				}

				/* Add tile if active. */
				if(tile_is_empty) {
					(*offsets)[total_tile_count] = -1;
				}
				else {
					(*sparse_grid)[active_tile_count] = tile;
					(*offsets)[total_tile_count] = active_tile_count;
					++active_tile_count;
				}
				++total_tile_count;
			}
		}
	}

	/* Return so that the parent function can resize
	 * sparse_grid appropriately. */
	return active_tile_count;
}

CCL_NAMESPACE_END

#endif /*__UTIL_SPARSE_GRID_H__*/
