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

namespace {
	inline bool gt(float a, float b) { return a > b; }
	inline bool gt(uchar a, uchar b) { return a > b; }
	inline bool gt(half a, half b) { return a > b; }
	inline bool gt(float4 a, float4 b) { return any(b < a); }
	inline bool gt(uchar4 a, uchar4 b) { return a.x > b.x || a.y > b.y || a.z > b.z || a.w > b.w; }
	inline bool gt(half4 a, half4 b) { return a.x > b.x || a.y > b.y || a.z > b.z || a.w > b.w; }
}

static const int TILE_SIZE = 8;

/* Sparse Tile Dimensions Information
 * For maintaining characteristics of each tile's dimensions.
 * A series of bools stored in the bits of the tile's dimension int. */
typedef enum SparseTileDimShift {
	ST_SHIFT_TRUNCATE_WIDTH = 0,
	ST_SHIFT_TRUNCATE_HEIGHT = 1,
	ST_SHIFT_X_PAD = 2,
	ST_SHIFT_Y_PAD = 3,
	ST_SHIFT_Z_PAD = 4,
} SparseTileDimShift;

const inline int compute_index(const size_t x, const size_t y, const size_t z,
                               const size_t width, const size_t height, const size_t depth)
{
	if(x >= width || y >= height || z >= depth) {
		return -1;
	}
	return x + width * (y + z * height);
}

const inline int compute_index(const size_t x, const size_t y, const size_t z,
                               const size_t width, const size_t height)
{
	return x + width * (y + z * height);
}

const inline int compute_index(const size_t x, const size_t y, const size_t z,
                               const int3 resolution)
{
	return compute_index(x, y, z, resolution.x, resolution.y, resolution.z);
}

const inline size_t get_tile_res(const size_t res)
{
	return (res / TILE_SIZE) + (res % TILE_SIZE != 0);
}

/* Finds the number of bits used by the largest
 * dimension of an image. Used for Morton order. */
const inline size_t compute_bit_count(size_t width, size_t height, size_t depth)
{
	size_t largest_dim = max(max(width, height), depth);
	size_t bit_count = 0, n = 1;
	while(largest_dim >= n) {
		n *= 2;
		++bit_count;
	}
	return bit_count;
}

const inline size_t compute_morton(size_t x, size_t y, size_t z, size_t bit_count)
{
	size_t morton_index = 0;
	for (size_t i = 0 ; i < bit_count ; ++i) {
		morton_index |= ((x >> i) & 1) << (i * 3 + 0);
		morton_index |= ((y >> i) & 1) << (i * 3 + 1);
		morton_index |= ((z >> i) & 1) << (i * 3 + 2);
	}
	return morton_index;
}

/* Do not call this function in the kernel. */
const inline int compute_index(const int *grid_info,
                               int x, int y, int z,
                               int bit_count, int ltw, int lth)
{
	/* Coordinates of (x, y, z)'s tile and
	 * coordinates of (x, y, z) with origin at tile start. */
	int tix = x / TILE_SIZE, itix = x % TILE_SIZE,
	    tiy = y / TILE_SIZE, itiy = y % TILE_SIZE,
	    tiz = z / TILE_SIZE, itiz = z % TILE_SIZE;
	/* Get the 1D array index in the dense grid of the tile (x, y, z) is in. */
	int dense_index = compute_morton(tix, tiy, tiz, bit_count) * 2;
	if(dense_index < 0) {
		return -1;
	}
	/* Get the offset and dimension info of target tile. */
	int sparse_index = grid_info[dense_index];
	int dims = grid_info[dense_index + 1];
	if (sparse_index < 0) {
		return -1;
	}
	/* If the tile is the last tile in a direction and the end is truncated, we
	 * have to recalulate itiN with the truncated length.  */
	int itiw = dims & (1 << ST_SHIFT_TRUNCATE_WIDTH) ? ltw : TILE_SIZE;
	int itih = dims & (1 << ST_SHIFT_TRUNCATE_HEIGHT) ? lth : TILE_SIZE;
	/* Look up voxel in the tile.
	 * Need to check whether or not a tile is padded on any of its 6 faces. */
	int in_tile_index = compute_index(itix, itiy, itiz, itiw, itih);
	return sparse_index + in_tile_index;
}

/* Do not call this function in the kernel. */
const inline int compute_index_cuda(const int *grid_info,
                                    int x, int y, int z,
                                    int width, int height, int depth,
                                    int tiw, int tih, int tid)
{
	int tix = x / TILE_SIZE, itix = x % TILE_SIZE,
		tiy = y / TILE_SIZE, itiy = y % TILE_SIZE,
		tiz = z / TILE_SIZE, itiz = z % TILE_SIZE;
	int dense_index = compute_index(tix, tiy, tiz, tiw, tih, tid) * 4;
	if(dense_index < 0) {
		return -1;
	}
	int tile_x = grid_info[dense_index];
	if(tile_x < 0) {
		return -1;
	}
	int tile_y = grid_info[dense_index + 1];
	int tile_z = grid_info[dense_index + 2];
	int dims = grid_info[dense_index + 3];
	int idx_x = tile_x + itix + (dims & (1 << ST_SHIFT_X_PAD));
	int idx_y = tile_y + itiy + (dims & (1 << ST_SHIFT_Y_PAD));
	int idx_z = tile_z + itiz + (dims & (1 << ST_SHIFT_Z_PAD));
	return compute_index(idx_x, idx_y, idx_z, width, height, depth);
}

template<typename T>
int create_sparse_grid(const T *dense_grid,
                       const int width, const int height,
                       const int depth, const float isovalue,
                       vector<T> *sparse_grid, vector<int> *grid_info)
{
	if(!dense_grid) {
		return 0;
	}

	const T threshold = cast_from_float<T>(isovalue);
	/* Get the minumum number of bits needed to represent each dimension. */
	size_t bit_count = compute_bit_count(width, height, depth);
	/* Resize vectors to tiled resolution. Have to overalloc
	 * sparse_grid because we don't know the number of
	 * active tiles yet. */
	sparse_grid->resize(width * height * depth);
	/* Overalloc of grid_info for morton order. */
	const size_t max_dim = max(max(width, height), depth);
	grid_info->resize(max_dim * max_dim * max_dim * 2);

	int voxel_count = 0;
	for(int z=0 ; z < depth ; z += TILE_SIZE) {
		for(int y=0 ; y < height ; y += TILE_SIZE) {
			for(int x=0 ; x < width ; x += TILE_SIZE) {

				bool is_active = false;
				int voxel = 0;

				/* Populate the tile. */
				for(int k=z ; k < min(z+TILE_SIZE, depth) ; ++k) {
					for(int j=y ; j < min(y+TILE_SIZE, height) ; ++j) {
						for(int i=x ; i < min(x+TILE_SIZE, width) ; ++i) {
							int index = compute_index(i, j, k, width, height);
							sparse_grid->at(voxel_count + voxel) = dense_grid[index];
							if(!is_active) {
								is_active = gt(dense_grid[index], threshold);
							}
							++voxel;
						}
					}
				}

				/* Compute tile index. */
				size_t tile = compute_morton(x/TILE_SIZE, y/TILE_SIZE,
				                             z/TILE_SIZE, bit_count) * 2;

				/* If tile is active, store tile's offset and dimension info. */
				if(is_active) {
					/* Store if the tile is the last tile in the X/Y direction
					 * and if its x/y resolution is not divisible by TILE_SIZE. */
					int dimensions = 0;
					dimensions |= ((x + TILE_SIZE > width) << ST_SHIFT_TRUNCATE_WIDTH);
					dimensions |= ((y + TILE_SIZE > height) << ST_SHIFT_TRUNCATE_HEIGHT);
					grid_info->at(tile) = voxel_count;
					grid_info->at(tile + 1) = dimensions;
					voxel_count += voxel;
				}
				else {
					grid_info->at(tile) = -1;
					grid_info->at(tile + 1) = 0;
				}
			}
		}
	}

	/* Return so that the parent function can resize
	 * sparse_grid appropriately. */
	return voxel_count;
}

template<typename T>
const bool check_tile_active(const T *dense_grid, T threshold,
                             int x, int y, int z,
                             int width, int height, int depth)
{
	for(int k=z ; k < min(z+TILE_SIZE, depth) ; ++k) {
		for(int j=y ; j < min(y+TILE_SIZE, height) ; ++j) {
			for(int i=x ; i < min(x+TILE_SIZE, width) ; ++i) {
				int index = compute_index(i, j, k, width, height);
				if(gt(dense_grid[index], threshold)) {
					return true;
				}
			}
		}
	}
	return false;
}

template<typename T>
int create_sparse_grid_cuda(const T *dense_grid,
                            int width, int height, int depth,
                            float isovalue,
                            vector<T> *sparse_grid,
                            vector<int> *grid_info)
{
	/* Total number of tiles in the grid (incl. inactive). */
	const int tile_count = get_tile_res(width) *
	                       get_tile_res(height) *
	                       get_tile_res(depth);

	if(!dense_grid || tile_count < 3) {
		return 0;
	}

	const T threshold = cast_from_float<T>(isovalue);

	/* Initial prepass to find active tiles. */
	grid_info->resize(tile_count * 4);
	int info_count = 0;
	for(int z = 0 ; z < depth ; z += TILE_SIZE) {
		for(int y = 0 ; y < height ; y += TILE_SIZE) {
			for(int x = 0 ; x < width ; x += TILE_SIZE) {
				int is_active = check_tile_active(dense_grid, threshold,
				                                  x, y, z,
				                                  width, height, depth) - 1;
				grid_info->at(info_count) = is_active;
				grid_info->at(info_count + 1) = is_active;
				grid_info->at(info_count + 2) = is_active;
				grid_info->at(info_count + 3) = 0;
				info_count += 4;
			}
		}
	}

	/* Have to overalloc sparse_grid because we don't know
	 * the number of active tiles with what padding yet. */
	sparse_grid->resize((TILE_SIZE+2)*(TILE_SIZE+2)*(TILE_SIZE+2)*tile_count);
	int voxel_count = 0;
	info_count = 0;

	/* Populate the sparse grid. */
	for(int z = 0 ; z < depth ; z += TILE_SIZE) {
		for(int y = 0 ; y < height ; y += TILE_SIZE) {
			for(int x = 0 ; x < width ; x += TILE_SIZE) {
				if(grid_info->at(info_count) < 0) {
					info_count += 4;
					continue;
				}

				/* Get the tile dimensions.
				 * Add a 1 voxel pad in the x direction only if the adjacent
				 * tile on that side is inactive. */
				int x_lhs = x;
				int x_rhs = x + TILE_SIZE;
				bool xpad = false;
				if(x_lhs > 1) {
					if(grid_info->at(info_count - 2) < 0) {
						--x_lhs;
						xpad = true;
					}
				}
				if(x_rhs < width) {
					if(grid_info->at(info_count + 2) < 0) {
						++x_rhs;
					}
				}
				x_rhs = min(x_rhs, width);

				int y_lhs = y - 1;
				int y_rhs = min(y + TILE_SIZE + 1, height);
				int z_lhs = z - 1;
				int z_rhs = min(z + TILE_SIZE + 1, depth);
				bool ypad = true, zpad = true;
				if(x_lhs < 0) {
					xpad = false;
					x_lhs = 0;
				}
				if(y_lhs < 0) {
					ypad = false;
					y_lhs = 0;
				}
				if(z_lhs < 0) {
					zpad = false;
					z_lhs = 0;
				}

				int voxel = 0;
				for(int k = z_lhs ; k < z_rhs ; ++k) {
					for(int j = y_lhs ; j < y_rhs ; ++j) {
						for(int i = x_lhs ; i < x_rhs ; ++i) {
							int index = compute_index(i, j, k, width, height);
							sparse_grid->at(voxel_count + voxel) = dense_grid[index];
							++voxel;
						}
					}
				}

				/* Set all tile dimensions and pad info.
				 * Store if each face of the LHS of a tile is padded.
				 * The tile will not be padded if it at the border of the grid.
				 * In the X direction, a tile will also not be padded if its
				 * adjacent tile is active. */
				int dimensions = 0;
				dimensions |= (xpad << ST_SHIFT_X_PAD);
				dimensions |= (ypad << ST_SHIFT_Y_PAD);
				dimensions |= (zpad << ST_SHIFT_Z_PAD);

				grid_info->at(info_count) = x;
				grid_info->at(info_count + 1) = y;
				grid_info->at(info_count + 2) = z;
				grid_info->at(info_count + 3) = dimensions;
				voxel_count += voxel;
				info_count += 4;
			}
		}
	}

	return voxel_count;
}

CCL_NAMESPACE_END

#endif /*__UTIL_SPARSE_GRID_H__*/
