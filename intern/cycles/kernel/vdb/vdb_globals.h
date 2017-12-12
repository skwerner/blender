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

class OpenVDBTextureBase
{
public:
	static OpenVDBTextureBase* create_from_grid(openvdb::GridBase::Ptr grid, const Transform &tfm);
	virtual ~OpenVDBTextureBase() { ; }
	virtual bool hasUniformVoxels() const = 0;
};

struct OpenVDBGlobals {
	vector<OpenVDBTextureBase*> grids;
};

template<class T> class OpenVDBTexture : public OpenVDBTextureBase
{
public:
	OpenVDBTexture() : intersector(NULL) { ; }
	virtual ~OpenVDBTexture() { release(); }

	void init(typename T::Ptr &in, const Transform &in_tfm) {
		grid = in;
		if(grid->hasUniformVoxels() && !grid->empty()) {
			intersector = new openvdb::tools::VolumeRayIntersector<T, T::TreeType::RootNodeType::ChildNodeType::LEVEL, openvdb::math::Ray<float> >(*grid);
		}
		openvdb::CoordBBox bbox = grid->evalActiveVoxelBoundingBox();
		const openvdb::math::Transform &tran = grid->constTransform();
		const openvdb::BBoxd bbox_w = tran.indexToWorld(bbox);
		float3 min_p = make_float3(bbox_w.min().x(), bbox_w.min().y(), bbox_w.min().z());
		float3 max_p = make_float3(bbox_w.max().x(), bbox_w.max().y(), bbox_w.max().z());
		float3 scale = max_p - min_p;
		tfm = transform_translate(min_p) * transform_scale(scale) * in_tfm;
	}

	virtual bool hasUniformVoxels() const { return grid->hasUniformVoxels(); }

	typename T::Ptr grid;
	openvdb::tools::VolumeRayIntersector<T, T::TreeType::RootNodeType::ChildNodeType::LEVEL, openvdb::math::Ray<float> > *intersector;
	Transform tfm;

private:
	void release() {
		if(intersector) {
			delete intersector;
			intersector = NULL;
		}
		grid.reset();
	}
};

CCL_NAMESPACE_END

#endif

#endif /* __VDB_GLOBALS_H__ */
