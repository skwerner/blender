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

#include "kernel/kernel_compat_cpu.h"
#include "kernel/kernel_types.h"
#include "kernel/split/kernel_split_data_types.h"
#include "kernel/kernel_globals.h"

#include "vdb_globals.h"
#include "vdb_intern.h"
#include "vdb_thread.h"

#include "util/util_vector.h"

CCL_NAMESPACE_BEGIN


typedef openvdb::math::Ray<float> vdb_ray_t;
typedef openvdb::math::Transform vdb_transform_t;

/* Manage thread-local data associated with volumes */

template <class T> class OpenVDBGenericThreadData {
public:
	OpenVDBGenericThreadData() : accessor(NULL), point_sampler(NULL), box_sampler(NULL), stag_point_sampler(NULL), stag_box_sampler(NULL), isector(NULL) {}
	typedef typename T::ConstAccessor accessor_t;
	typedef openvdb::tools::GridSampler<accessor_t, openvdb::tools::PointSampler> point_sampler_t;
	typedef openvdb::tools::GridSampler<accessor_t, openvdb::tools::BoxSampler> box_sampler_t;
	typedef openvdb::tools::VolumeRayIntersector<T, T::TreeType::RootNodeType::ChildNodeType::LEVEL, vdb_ray_t> isector_t;
	typedef openvdb::tools::GridSampler<accessor_t, openvdb::tools::StaggeredPointSampler> stag_point_sampler_t;
	typedef openvdb::tools::GridSampler<accessor_t, openvdb::tools::StaggeredBoxSampler> stag_box_sampler_t;

	void init(const OpenVDBTexture<T> &tex)
	{
		accessor = new accessor_t(tex.grid->getConstAccessor());
		point_sampler = new point_sampler_t(*accessor, tex.grid->transform());
		box_sampler = new box_sampler_t(*accessor, tex.grid->transform());
		stag_point_sampler = new stag_point_sampler_t(*accessor, tex.grid->transform());
		stag_box_sampler = new stag_box_sampler_t(*accessor, tex.grid->transform());
		if(tex.intersector) {
			isector = new isector_t(*tex.intersector);
		}
	}

	void free()
	{
		if(accessor) {
			delete accessor;
		}
		if(point_sampler) {
			delete point_sampler;
		}
		if(box_sampler) {
			delete box_sampler;
		}
		if(stag_point_sampler) {
			delete stag_point_sampler;
		}
		if(stag_box_sampler) {
			delete stag_box_sampler;
		}
		if(isector) {
			delete isector;
		}
	}

	accessor_t *accessor;
	point_sampler_t *point_sampler;
	box_sampler_t *box_sampler;
	stag_point_sampler_t *stag_point_sampler;
	stag_box_sampler_t *stag_box_sampler;
	isector_t *isector;
};

typedef OpenVDBGenericThreadData<openvdb::FloatGrid> OpenVDBScalarThreadData;
typedef OpenVDBGenericThreadData<openvdb::Vec3SGrid> OpenVDBVectorThreadData;

struct OpenVDBThreadData {
	std::vector<OpenVDBScalarThreadData> scalar_data;
	std::vector<OpenVDBVectorThreadData> vector_data;
};

void VDBVolume::thread_init(KernelGlobals *kg, OpenVDBGlobals *vdb_globals)
{
	kg->vdb = vdb_globals;
	
	OpenVDBThreadData *tdata = new OpenVDBThreadData;
	
	tdata->scalar_data.resize(vdb_globals->scalar_grids.size());
	tdata->vector_data.resize(vdb_globals->vector_grids.size());
	for (size_t i = 0; i < vdb_globals->scalar_grids.size(); ++i) {
		if(vdb_globals->scalar_grids[i].grid) {
			tdata->scalar_data[i].init(vdb_globals->scalar_grids[i]);
		}
	}
	for (size_t i = 0; i < vdb_globals->vector_grids.size(); ++i) {
		if(vdb_globals->scalar_grids[i].grid) {
			tdata->vector_data[i].init(vdb_globals->vector_grids[i]);
		}
	}
	kg->vdb_tdata = tdata;
}

void VDBVolume::thread_free(KernelGlobals *kg)
{
	OpenVDBThreadData *tdata = kg->vdb_tdata;
	kg->vdb_tdata = NULL;
	
	for (size_t i = 0; i < tdata->scalar_data.size(); ++i) {
		tdata->scalar_data[i].free();
	}
	for (size_t i = 0; i < tdata->vector_data.size(); ++i) {
		tdata->vector_data[i].free();
	}
	delete tdata;
}

bool VDBVolume::scalar_has_uniform_voxels(OpenVDBGlobals *vdb, int vdb_index)
{
	return vdb->scalar_grids[vdb_index].grid->hasUniformVoxels();
}

bool VDBVolume::vector_has_uniform_voxels(OpenVDBGlobals *vdb, int vdb_index)
{
	return vdb->vector_grids[vdb_index].grid->hasUniformVoxels();
}

float VDBVolume::sample_scalar(OpenVDBGlobals *vdb, OpenVDBThreadData *vdb_thread, int vdb_index,
                               float x, float y, float z, int sampling)
{
	OpenVDBScalarThreadData &data = vdb_thread->scalar_data[vdb_index];

	float3 pos = make_float3(x, y, z);
	pos = transform_point(&vdb->scalar_grids[vdb_index].tfm, pos);
	openvdb::Vec3d p(pos.x, pos.y, pos.z);

	switch (sampling) {
		case OPENVDB_SAMPLE_POINT:
			return data.point_sampler->isSample(p);
		case OPENVDB_SAMPLE_BOX:
			return data.box_sampler->isSample(p);
	}
	
	return 0.0f;
}

bool VDBVolume::sample_vector(OpenVDBGlobals *vdb, OpenVDBThreadData *vdb_thread, int vdb_index,
                                float x, float y, float z, float *r, float *g, float *b, int sampling)
{
	if(vdb_index >= vdb->vector_grids.size()) {
		return false;
	}
	bool staggered = (vdb->vector_grids[vdb_index].grid->getGridClass() == openvdb::GRID_STAGGERED);
	OpenVDBVectorThreadData &data = vdb_thread->vector_data[vdb_index];
	openvdb::Vec3s value;

	if (staggered) {
		switch (sampling) {
			case OPENVDB_SAMPLE_POINT:
				value = data.stag_point_sampler->wsSample(openvdb::Vec3d(x, y, z));
				break;
			case OPENVDB_SAMPLE_BOX:
				value = data.stag_box_sampler->wsSample(openvdb::Vec3d(x, y, z));
				break;
		}
	}
	else {
		switch (sampling) {
			case OPENVDB_SAMPLE_POINT:
				value = data.point_sampler->wsSample(openvdb::Vec3d(x, y, z));
				break;
			case OPENVDB_SAMPLE_BOX:
				value = data.box_sampler->wsSample(openvdb::Vec3d(x, y, z));
				break;
		}
	}

	*r = value.x();
	*g = value.y();
	*b = value.z();
	return true;
}

bool VDBVolume::intersect(OpenVDBThreadData *vdb_thread, int vdb_index,
                          const Ray *ray, float *isect)
{
	OpenVDBScalarThreadData &data = vdb_thread->scalar_data[vdb_index];
	
	vdb_ray_t::Vec3Type P(ray->P.x, ray->P.y, ray->P.z);
	vdb_ray_t::Vec3Type D(ray->D.x, ray->D.y, ray->D.z);
	D.normalize();
	
	vdb_ray_t vdb_ray(P, D, 1e-5f, ray->t);
	
	if(data.isector->setWorldRay(vdb_ray)) {
		// TODO(kevin): is this correct?
		*isect = static_cast<float>(vdb_ray.t1());
		
		return true;
	}
	
	return false;
}

bool VDBVolume::march(OpenVDBThreadData *vdb_thread, int vdb_index,
                      float *t0, float *t1)
{
	OpenVDBScalarThreadData &data = vdb_thread->scalar_data[vdb_index];
	
	float vdb_t0(*t0), vdb_t1(*t1);

	if(data.isector->march(vdb_t0, vdb_t1)) {
		*t0 = data.isector->getWorldTime(vdb_t0);
		*t1 = data.isector->getWorldTime(vdb_t1);

		return true;
	}

	return false;
}

CCL_NAMESPACE_END
