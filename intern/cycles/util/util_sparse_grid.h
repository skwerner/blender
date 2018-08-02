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
#include "util/util_logging.h"
#include "util/util_string.h"
#include "util/util_texture.h"
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

/* Sparse Tile Dimensions Information
 * For maintaining characteristics of each tile's dimensions.
 * A series of bools stored in the bits of the tile's dimension int. */
typedef enum SparseTileDimShift {
	ST_SHIFT_X_PAD = 0,
	ST_SHIFT_Y_PAD = 1,
	ST_SHIFT_Z_PAD = 2,
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

const inline int get_tile_res(const size_t res)
{
	return (res / TILE_SIZE) + (res % TILE_SIZE != 0);
}

/* Do not call this function in the kernel. */
const inline int compute_index(const int *grid_info,
                               int x, int y, int z,
                               int width, int height, int depth)
{
	/* Coordinates of (x, y, z)'s tile and
	 * coordinates of (x, y, z) with origin at tile start. */
	int tix = x / TILE_SIZE, itix = x % TILE_SIZE,
	    tiy = y / TILE_SIZE, itiy = y % TILE_SIZE,
	    tiz = z / TILE_SIZE, itiz = z % TILE_SIZE,
	    tiw = get_tile_res(width),
	    tih = get_tile_res(height),
	    tid = get_tile_res(depth);
	/* Get the 1D array index in the dense grid of the tile (x, y, z) is in. */
	int dense_index = compute_index(tix, tiy, tiz, tiw, tih, tid);
	if(dense_index < 0) {
		return -1;
	}
	/* Get the offset of target tile. */
	int sparse_index = grid_info[dense_index];
	if (sparse_index < 0) {
		return -1;
	}
	/* If the tile is the last tile in a direction and the end is truncated, we
	 * have to recalulate itiN with the truncated length.  */
	int ltw = width % TILE_SIZE, lth = height % TILE_SIZE;
	int itiw = (x > width - ltw) ? ltw : TILE_SIZE;
	int itih = (y > height - lth) ? lth : TILE_SIZE;
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
bool create_sparse_grid(const T *dense_grid,
                        const int width,
                        const int height,
                        const int depth,
                        const int channels,
                        const string grid_name,
                        const float isovalue,
                        vector<T> *sparse_grid,
                        vector<int> *grid_info)
{
	assert(dense_grid != NULL);

	const T threshold = cast_from_float<T>(isovalue);
	const int tile_count = get_tile_res(width) *
	                       get_tile_res(height) *
	                       get_tile_res(depth);

	/* Initial prepass to find active tiles. */
	grid_info->resize(tile_count, -1);  /* 0 if active, -1 if inactive. */
	int tile = 0, voxel = 0, voxel_count = 0;

	for(int z = 0; z < depth; z += TILE_SIZE) {
		for(int y = 0; y < height; y += TILE_SIZE) {
			for(int x = 0; x < width; x += TILE_SIZE, ++tile) {
				bool is_active = false;
				const int max_i = min(x + TILE_SIZE, width);
				const int max_j = min(y + TILE_SIZE, height);
				const int max_k = min(z + TILE_SIZE, depth);

				voxel = 0;
				for(int k = z; k < max_k; ++k) {
					for(int j = y; j < max_j; ++j) {
						for(int i = x; i < max_i; ++i, ++voxel) {
							int index = compute_index(i, j, k, width, height) * channels;
							for(int c = 0; c < channels; ++c) {
								if(dense_grid[index + c] >= threshold) {
									is_active = true;
									break;
								}
							}
						}
					}
				}

				if(is_active) {
					grid_info->at(tile) = 0;
					voxel_count += voxel;
				}
			}
		}
	}

	/* Check memory savings. */
	int sparse_mem_use = tile_count * sizeof(int) + voxel_count * channels * sizeof(T);
	int dense_mem_use = width * height * depth * channels * sizeof(T);

	if(sparse_mem_use >= dense_mem_use) {
		VLOG(1) << "Memory of " << grid_name << " increased from "
		        << string_human_readable_size(dense_mem_use) << " to "
		        << string_human_readable_size(sparse_mem_use)
		        << ", not using sparse grid";
		return false;
	}

	VLOG(1) << "Memory of " << grid_name << " decreased from "
			<< string_human_readable_size(dense_mem_use) << " to "
			<< string_human_readable_size(sparse_mem_use);

	/* Populate the sparse grid. */
	sparse_grid->resize(voxel_count * channels);
	voxel = tile = 0;

	for(int z = 0; z < depth; z += TILE_SIZE) {
		for(int y = 0; y < height; y += TILE_SIZE) {
			for(int x = 0; x < width; x += TILE_SIZE, ++tile) {
				if(grid_info->at(tile) == -1) {
					continue;
				}

				grid_info->at(tile) = voxel / channels;

				/* Populate the tile. */
				const int max_i = min(x + TILE_SIZE, width);
				const int max_j = min(y + TILE_SIZE, height);
				const int max_k = min(z + TILE_SIZE, depth);

				for(int k = z; k < max_k; ++k) {
					for(int j = y; j < max_j; ++j) {
						for(int i = x; i < max_i; ++i) {
							int index = compute_index(i, j, k, width, height) * channels;
							for(int c = 0; c < channels; ++c, ++voxel) {
								sparse_grid->at(voxel) = dense_grid[index + c];
							}
						}
					}
				}
			}
		}
	}

	return true;
}

template<typename T>
int create_sparse_grid_cuda(const T *dense_grid,
                            const int width, const int height,
                            const int depth, const int channels,
                            const float isovalue,
                            vector<T> *sparse_grid, vector<int> *grid_info)
{
	/* Total number of tiles in the grid (incl. inactive). */
	const int tile_count = get_tile_res(width) *
	                       get_tile_res(height) *
	                       get_tile_res(depth);

	if(!dense_grid || tile_count < 3) {
		return -1;
	}

	const T threshold = cast_from_float<T>(isovalue);

	/* Initial prepass to find active tiles. */
	grid_info->resize(tile_count * 4);
	int info_count = 0;
	for(int z = 0 ; z < depth ; z += TILE_SIZE) {
		for(int y = 0 ; y < height ; y += TILE_SIZE) {
			for(int x = 0 ; x < width ; x += TILE_SIZE) {
				int is_active = check_tile_active(dense_grid, threshold,
				                                  x, y, z, width, height, depth,
				                                  channels) - 1;
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
