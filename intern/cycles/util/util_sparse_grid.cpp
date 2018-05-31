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

#include "util/util_sparse_grid.h"
#include "util/util_math.h"

/* Functions to help generate and handle Sparse Grids. */

CCL_NAMESPACE_BEGIN

int compute_index(int x, int y, int z, int width, int height, int depth, int min)
{
    if(x < min || y < min || z < min || x >= width || y >= height || z >= depth) {
		return min-1;
    }
	return x + width * (y + z * height);
}

int3 compute_coordinates(int index, int width, int height, int depth, int min)
{
    if(index < min || index >= width * height * depth) {
		return make_int3(min-1, min-1, min-1);
    }
    int x = index % width;
    int y = (index / width) % height;
    int z = index / (width * height);
    return make_int3(x, y, z);
}

int compute_tile_resolution(int res)
{
    if(res % TILE_SIZE == 0) {
        return res / TILE_SIZE;
    }
    return res / TILE_SIZE + 1;
}

bool is_active(const int *offsets,
               size_t x, size_t y, size_t z,
               size_t width, size_t height, size_t depth)
{
    if(offsets == NULL)
        return false;
    int tile_width = compute_tile_resolution(width);
    int tile_height = compute_tile_resolution(height);
    int tile_depth = compute_tile_resolution(depth);
    int tile_index = compute_index(x/TILE_SIZE, y/TILE_SIZE, z/TILE_SIZE,
                                   tile_width, tile_height, tile_depth);
    if(tile_index < 0 || tile_index > tile_width*tile_height*tile_depth)
        return false;
    int grid_index = offsets[tile_index];
    return (grid_index >= 0 && grid_index < width*height*depth);
}

float4 get_value(const SparseTile *grid, const int *offsets,
                 size_t x, size_t y, size_t z,
                 size_t width, size_t height, size_t depth)
{
    if(offsets == NULL) {
        return make_float4(0.0f);
    }
    int tile_width = compute_tile_resolution(width);
    int tile_height = compute_tile_resolution(height);
    int tile_depth = compute_tile_resolution(depth);
    /* Get the 1D array tile index of the tile the voxel (x, y, z) is in. */
    int tile_index = compute_index(x/TILE_SIZE, y/TILE_SIZE, z/TILE_SIZE,
                                   tile_width, tile_height, tile_depth);
    if(tile_index < 0 || tile_index > tile_width*tile_height*tile_depth) {
        return make_float4(0.0f);
    }
    /* Get the index of the tile in the sparse grid. */
    int grid_index = offsets[tile_index];
    if (grid_index < 0 || grid_index > width*height*depth) {
        return make_float4(0.0f);
    }
    /* Get tile and look up voxel in tile. */
    int voxel_index = compute_index(x%TILE_SIZE, y%TILE_SIZE, z%TILE_SIZE);
    return grid[grid_index].values[voxel_index];
}

float4 get_value(const SparseTile *grid, const int *offsets,
                 size_t tile_index,
                 size_t width, size_t height, size_t depth)
{
    int3 c = compute_coordinates(tile_index, width, height, depth);
    return get_value(grid, offsets, c.x, c.y, c.z, width, height, depth);
}

int create_sparse_grid(float4 *dense_grid,
                       const size_t width,
                       const size_t height,
                       const size_t depth,
                       vector<SparseTile> *sparse_grid,
                       vector<int> *offsets)
{

    /* Resize vectors to tiled resolution. */
    int active_tile_count = 0;
    int total_tile_count = compute_tile_resolution(width) *
                           compute_tile_resolution(height) *
                           compute_tile_resolution(depth);
    /* Overalloc grid because we don't know the
     * number of active tiles yet. */
    sparse_grid->resize(total_tile_count);
    offsets->resize(total_tile_count);
    total_tile_count = 0;

    for(int z=0 ; z < depth ; z += TILE_SIZE) {
        for(int y=0 ; y < height ; y += TILE_SIZE) {
            for(int x=0 ; x < width ; x += TILE_SIZE) {

                SparseTile tile;
                bool is_empty = true;
                int c = 0;

                /* Populate the tile. */
                for(int k=z ; k < z+TILE_SIZE ; ++k) {
                    for(int j=y ; j < y+TILE_SIZE ; ++j) {
                        for(int i=x ; i < x+TILE_SIZE ; ++i) {
                            int index = compute_index(i, j, k, width, height, depth);
                            if(index < 0) {
                                /* Out of bounds of original image, store an empty voxel. */
                                tile.values[c] = make_float4(0.0f);
                            }
                            else {
                                tile.values[c] = dense_grid[index];
                                if(is_empty) {
                                    if(dense_grid[index].x > 0.0f ||
                                       dense_grid[index].y > 0.0f ||
                                       dense_grid[index].z > 0.0f ||
                                       dense_grid[index].w > 0.0f) {
                                        /* All values are greater than the threshold. */
                                        is_empty = false;
                                    }
                                }
                            }
                            ++c;
                        }
                    }
                }

                /* Add tile if active. */
                if(is_empty) {
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
