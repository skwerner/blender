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

#ifndef __MESH_VOLUME_H__
#define __MESH_VOLUME_H__

#include "util/util_types.h"
#include "util/util_vector.h"

CCL_NAMESPACE_BEGIN

struct QuadData {
	int v0, v1, v2, v3;

	float3 normal;
};

struct VolumeParams {
	int3 resolution;
	float3 cell_size;
	float3 start_point;
	int pad_size;
};

/* Create a mesh from a volume.
 *
 * The way the algorithm works is as follows:
 *
 * - the coordinates of active voxels from a dense volume (or 3d image) are
 * gathered inside an auxialliary volume.
 * - each set of coordinates of an CUBE_SIZE cube are mapped to the same
 * coordinate of the auxilliary volume.
 * - quads are created between active and non-active voxels in the auxialliary
 * volume to generate a tight mesh around the volume.
 */
class VolumeMeshBuilder {
	/* Auxilliary volume that is used to check if a node already added. */
	vector<char> grid;

	/* The resolution of the auxilliary volume, set to be equal to 1/CUBE_SIZE
	 * of the original volume on each axis. */
	int3 res;

	size_t number_of_nodes;

	/* Offset due to padding in the original grid. Padding will transform the
	 * coordinates of the original grid from 0...res to -padding...res+padding,
	 * so some coordinates are negative, and we need to properly account for
	 * them. */
	int3 pad_offset;

	VolumeParams *params;

public:
	VolumeMeshBuilder(VolumeParams *volume_params);

	void add_node(int x, int y, int z);

	void add_node_with_padding(int x, int y, int z);

	void create_mesh(vector<float3> &vertices,
	                 vector<int> &indices,
	                 vector<float3> &face_normals);

private:
	void generate_vertices_and_quads(vector<int3> &vertices_is,
	                                 vector<QuadData> &quads);

	void deduplicate_vertices(vector<int3> &vertices,
	                          vector<QuadData> &quads);

	void convert_object_space(const vector<int3> &vertices,
	                          vector<float3> &out_vertices);

	void convert_quads_to_tris(const vector<QuadData> &quads,
	                           vector<int> &tris,
	                           vector<float3> &face_normals);
};

CCL_NAMESPACE_END

#endif /* __MESH_VOLUME_H__ */

