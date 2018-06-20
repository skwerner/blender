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
	static OpenVDBTextureBase* create_from_grid(openvdb::GridBase::Ptr grid, const Transform &tfm, int3 resolution, int3 index_offset);
	virtual ~OpenVDBTextureBase() { ; }
	virtual bool hasUniformVoxels() const = 0;
	virtual int num_channels() const = 0;
};

struct OpenVDBGlobals {
	vector<OpenVDBTextureBase*> grids;
};

template<class T> class OpenVDBTexture : public OpenVDBTextureBase
{
public:
	OpenVDBTexture() : intersector(NULL) { ; }
	virtual ~OpenVDBTexture() { release(); }

	void init(typename T::Ptr &in, const Transform &in_tfm, int3 _resolution, int3 _index_offset) {
		grid = in;
		if(grid->hasUniformVoxels() && !grid->empty()) {
			intersector = new openvdb::tools::VolumeRayIntersector<T, T::TreeType::RootNodeType::ChildNodeType::LEVEL, openvdb::math::Ray<float> >(*grid);
		}

		tfm = in_tfm;

		resolution = _resolution;
		index_offset = _index_offset;
	}

	virtual bool hasUniformVoxels() const { return grid->hasUniformVoxels(); }

	virtual int num_channels() const
	{
		if(grid->template isType<openvdb::FloatGrid>()) {
			return 1;
		}
		else if(grid->template isType<openvdb::Vec3SGrid>()) {
			return 3;
		}
		return 0;
	}

	typename T::Ptr grid;
	openvdb::tools::VolumeRayIntersector<T, T::TreeType::RootNodeType::ChildNodeType::LEVEL, openvdb::math::Ray<float> > *intersector;
	Transform tfm; /* cycles world to vdb world */
	int3 resolution;
	int3 index_offset;

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
