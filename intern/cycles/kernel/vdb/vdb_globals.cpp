/*
 * Copyright 2017 Blender Foundation
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

#include "vdb_globals.h"

CCL_NAMESPACE_BEGIN

OpenVDBTextureBase *OpenVDBTextureBase::create_from_grid(openvdb::GridBase::Ptr grid, const Transform& tfm, int3 resolution, int3 index_offset)
{
	if(grid->isType<openvdb::FloatGrid>()) {
		OpenVDBTexture<openvdb::FloatGrid> *texture = new OpenVDBTexture<openvdb::FloatGrid>();
		openvdb::FloatGrid::Ptr float_grid = openvdb::gridPtrCast<openvdb::FloatGrid>(grid);
		texture->init(float_grid, tfm, resolution, index_offset);
		return texture;
	}
	else if(grid->isType<openvdb::Vec3SGrid>()) {
		OpenVDBTexture<openvdb::Vec3SGrid> *texture = new OpenVDBTexture<openvdb::Vec3SGrid>();
		openvdb::Vec3SGrid::Ptr vector_grid = openvdb::gridPtrCast<openvdb::Vec3SGrid>(grid);
		texture->init(vector_grid, tfm, resolution, index_offset);
		return texture;
	}
	return NULL;
}


CCL_NAMESPACE_END
