/*
 * Copyright 2016 Blender Foundation
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

#ifndef __VDB_GLOBALS_H__
#define __VDB_GLOBALS_H__

#ifdef WITH_OPENVDB

#include "vdb_intern.h"
#include "util/util_vector.h"
#include "util/util_thread.h"
#include "util/util_transform.h"

CCL_NAMESPACE_BEGIN

typedef openvdb::math::Ray<float> vdb_ray_t;
typedef openvdb::math::Transform vdb_transform_t;
#if 0
class OpenVDBTextureBase
{
public:
	virtual ~OpenVDBTextureBase() = 0;
	virtual bool lookup(float x, float y, float z, float *value) = 0;
};
#endif

template<class T> struct OpenVDBTexture // : public OpenVDBTextureBase
{
	OpenVDBTexture() : intersector(NULL) { ; }
	void init(typename T::Ptr &in) {
		grid = in;
		if(grid->hasUniformVoxels() && !grid->empty()) {
			intersector = new openvdb::tools::VolumeRayIntersector<T, T::TreeType::RootNodeType::ChildNodeType::LEVEL, vdb_ray_t>(*grid);
		}
		openvdb::CoordBBox bbox = grid->evalActiveVoxelBoundingBox();
		float3 min_p = make_float3(bbox.min().x(), bbox.min().y(), bbox.min().z());
		float3 max_p = make_float3(bbox.max().x()+1, bbox.max().y()+1, bbox.max().z()+1);
		float3 scale = max_p - min_p;
		tfm = transform_translate(-min_p)*transform_scale(scale);
	}
	typename T::Ptr grid;
	openvdb::tools::VolumeRayIntersector<T, T::TreeType::RootNodeType::ChildNodeType::LEVEL, vdb_ray_t> *intersector;
	Transform tfm;
};

typedef openvdb::tools::VolumeRayIntersector<openvdb::FloatGrid, openvdb::FloatGrid::TreeType::RootNodeType::ChildNodeType::LEVEL, vdb_ray_t> scalar_isector_t;
typedef openvdb::tools::VolumeRayIntersector<openvdb::Vec3SGrid, openvdb::Vec3SGrid::TreeType::RootNodeType::ChildNodeType::LEVEL, vdb_ray_t> vector_isector_t;

struct OpenVDBGlobals {
	vector<OpenVDBTexture<openvdb::FloatGrid> > scalar_grids;
	vector<OpenVDBTexture<openvdb::Vec3SGrid> > vector_grids;
	thread_mutex tex_paths_mutex;
};

CCL_NAMESPACE_END

#endif

#endif /* __VDB_GLOBALS_H__ */
