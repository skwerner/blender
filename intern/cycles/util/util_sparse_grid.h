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

/* Functions to help generate and handle Sparse Grids. */

CCL_NAMESPACE_BEGIN

const inline int compute_index(const int x, const int y, const int z,
                               const int width, const int height, const int depth)
{
	if(x < 0 || y < 0 || z < 0 || x >= width || y >= height || z >= depth) {
		return -1;
	}
	return x + width * (y + z * height);
}

const inline int compute_index(const int x, const int y, const int z,
                               const int3 resolution)
{
	return compute_index(x, y, z, resolution.x, resolution.y, resolution.z);
}

const inline int compute_index(const size_t x, const size_t y, const size_t z,
                               const size_t width, const size_t height)
{
	return x + width * (y + z * height);
}

const inline int get_tile_res(const size_t res)
{
	return (res >> TILE_INDEX_SHIFT) + ((res & TILE_INDEX_MASK) != 0);
}

const inline int compute_index(const int *offsets,
                               const int x, const int y, const int z,
                               const int width, const int height, const int depth)
{
	/* Get coordinates of voxel's tile in tiled image space and coordinates of
	 * voxel in tile space. */
	int tix = x >> TILE_INDEX_SHIFT, itix = x & TILE_INDEX_MASK,
	    tiy = y >> TILE_INDEX_SHIFT, itiy = y & TILE_INDEX_MASK,
	    tiz = z >> TILE_INDEX_SHIFT, itiz = z & TILE_INDEX_MASK;
	/* Get flat index of voxel's tile. */
	int tile = compute_index(tix, tiy, tiz,
	                         get_tile_res(width),
	                         get_tile_res(height),
	                         get_tile_res(depth));
	if(tile < 0) {
		return -1;
	}
	/* Get flat index (in image space) of the first voxel of the target tile. */
	int tile_start = offsets[tile];
	if (tile_start < 0) {
		return -1;
	}
	/* If the tile is the last tile in a direction and the end is truncated, we
	 * have to recalulate itiN with the truncated length. */
	int remainder_w = width & TILE_INDEX_MASK, remainder_h = height & TILE_INDEX_MASK;
	int itiw = (x > width - remainder_w) ? remainder_w : TILE_SIZE;
	int itih = (y > height - remainder_h) ? remainder_h : TILE_SIZE;
	/* Get flat index of voxel in tile space. */
	int in_tile_index = compute_index(itix, itiy, itiz, itiw, itih);
	return tile_start + in_tile_index;
}

const inline int compute_index_pad(const int *offsets,
                                   const int x, const int y, const int z,
                                   const int width, const int height, const int depth,
                                   const int sparse_width)
{
	/* Get coordinates of voxel's tile in tiled image space and coordinates of
	 * voxel in tile space. In-tile y and z coordinates assume a pad on
	 * the tile's minimum bound. */
	int tix = x >> TILE_INDEX_SHIFT, sx = (x & TILE_INDEX_MASK) + SPARSE_PAD,
	    tiy = y >> TILE_INDEX_SHIFT, sy = (y & TILE_INDEX_MASK) + SPARSE_PAD,
	    tiz = z >> TILE_INDEX_SHIFT, sz = (z & TILE_INDEX_MASK) + SPARSE_PAD;
	/* Get flat index of voxel's tile. */
	int tile = compute_index(tix, tiy, tiz,
	                         get_tile_res(width),
	                         get_tile_res(height),
	                         get_tile_res(depth));
	if(tile < 0) {
		return -1;
	}
	/* Get x-coordinate (in sparse image space) of the first voxel of the
	 * target tile. */
	int start_x = offsets[tile];
	if (start_x < 0) {
		return -1;
	}
	/* Check if tile's x min bound is padded. */
	if(x >= TILE_SIZE) {
		if(offsets[tile - 1] > -1) {
			sx -= SPARSE_PAD;
		}
	}
	/* Get flat index of voxel in sparse image space. */
	return compute_index(start_x + sx, sy, sz, sparse_width, PADDED_TILE);
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
	if(dense_grid == NULL) {
		return false;
	}

	const T threshold = util_image_cast_from_float<T>(isovalue);
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

const static int2 padded_tile_bound(const vector<int> *offsets, const int x,
                                    const int width, const int tile)
{
	/* Don't need x padding if neighbor tile in x direction is active. Always
	 * pad tile's min bound if the tile is at the min bound of the image.  */
	int min_bound = -SPARSE_PAD;
	if(x >= TILE_SIZE) {
		if(offsets->at(tile - 1) > -1) {
			min_bound = 0;
		}
	}

	int max_bound = TILE_SIZE + SPARSE_PAD;
	if(width - x < TILE_SIZE) {
		max_bound = width - x + SPARSE_PAD;
	}
	else if (offsets->at(tile + 1) > -1) {
		max_bound -= SPARSE_PAD;
	}

	return make_int2(min_bound, max_bound);
}

template<typename T>
bool create_sparse_grid_pad(const T *dense_grid,
                            const int width,
                            const int height,
                            const int depth,
                            const int channels,
                            const string grid_name,
                            const float isovalue,
                            vector<T> *sparse_grid,
                            vector<int> *grid_info,
                            int3 &sparse_resolution)
{
	/* For padded sparse grids, offsets only stores the x coordinate of the
	 * starting voxel in each padded tile (taking into account pad) because
	 * 1. CUDA expects coordinates instead of a flat index.
	 * 2. A padded sparse grid is stored as a long tile_count x 1 x 1 line of
	 *    tiles, so the starting voxel in each tile is always (X, 0, 0).
	 */
	const T threshold = util_image_cast_from_float<T>(isovalue);
	const int tile_count = get_tile_res(width) *
	                       get_tile_res(height) *
	                       get_tile_res(depth);

	if(!dense_grid || get_tile_res(width) < 3 ||
	   get_tile_res(height) < 3 || get_tile_res(depth) < 3)
	{
		return false;
	}

	/* Initial prepass to find active tiles. */
	grid_info->resize(tile_count, -1); /* 0 if active, -1 if inactive. */
	int tile = 0;

	for(int z = 0; z < depth; z += TILE_SIZE) {
		for(int y = 0; y < height; y += TILE_SIZE) {
			for(int x = 0; x < width; x += TILE_SIZE, ++tile) {
				const int max_i = min(x + TILE_SIZE, width);
				const int max_j = min(y + TILE_SIZE, height);
				const int max_k = min(z + TILE_SIZE, depth);

				for(int k = z; k < max_k; ++k) {
					for(int j = y; j < max_j; ++j) {
						for(int i = x; i < max_i; ++i) {
							int index = compute_index(i, j, k, width, height) * channels;
							for(int c = 0; c < channels; ++c) {
								if(threshold <= dense_grid[index + c]) {
									grid_info->at(tile) = 0;
									break;
								}
							}
						}
					}
				}
			}
		}
	}

	/* Second prepass to find sparse image dimensions and total number of voxels. */
	int sparse_width = tile = 0;
	const int sparse_height = PADDED_TILE;
	const int sparse_depth = PADDED_TILE;

	for(int z = 0; z < depth; z += TILE_SIZE) {
		for(int y = 0; y < height; y += TILE_SIZE) {
			for(int x = 0; x < width; x += TILE_SIZE, ++tile) {
				if(grid_info->at(tile) < 0) {
					continue;
				}
				grid_info->at(tile) = sparse_width;
				int2 bound_x = padded_tile_bound(grid_info, x, width, tile);
				sparse_width += bound_x.y - bound_x.x;
			}
		}
	}

	/* Check memory savings. */
	int sparse_mem_use = tile_count * sizeof(int) +
	                     sparse_width * sparse_height * sparse_depth *
	                     channels * sizeof(T);
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

	sparse_resolution = make_int3(sparse_width, sparse_height, sparse_depth);

	/* Populate the sparse grid. */
	sparse_grid->resize(sparse_width * sparse_height * sparse_depth * channels);
	tile = 0;

	for(int z = 0; z < depth; z += TILE_SIZE) {
		for(int y = 0; y < height; y += TILE_SIZE) {
			for(int x = 0; x < width; x += TILE_SIZE, ++tile) {
				if(grid_info->at(tile) < 0) {
					continue;
				}

				const int start_x = grid_info->at(tile);
				const int2 bound_x = padded_tile_bound(grid_info, x, width, tile);

				for(int k = -SPARSE_PAD; k < TILE_SIZE + SPARSE_PAD; ++k) {
					for(int j = -SPARSE_PAD; j < TILE_SIZE + SPARSE_PAD; ++j) {
						for(int i = bound_x.x; i < bound_x.y; ++i) {
							int dense_index = compute_index(i + x,
							                                j + y,
							                                k + z,
							                                width,
							                                height,
							                                depth) * channels;
							int sparse_index = compute_index(i - bound_x.x + start_x,
							                                 j + SPARSE_PAD,
							                                 k + SPARSE_PAD,
							                                 sparse_width,
							                                 sparse_height) * channels;

							if(dense_index < 0) {
								for(int c = 0; c < channels; ++c) {
									sparse_grid->at(sparse_index + c) = 0.0f;
								}
							}
							else {
								for(int c = 0; c < channels; ++c) {
									sparse_grid->at(sparse_index + c) = dense_grid[dense_index + c];
								}
							}
						}
					}
				}
			}
		}
	}

	return true;
}

CCL_NAMESPACE_END

#endif /*__UTIL_SPARSE_GRID_H__*/
