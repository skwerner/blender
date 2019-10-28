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

#ifdef WITH_OPENVDB
/* Must be included before Cycles #defines foreach() -
 * OpenVDB is using foreach as method name. */
#  include <openvdb/openvdb.h>
#  include <openvdb/tools/Interpolation.h>
#  include <openvdb/tools/RayIntersector.h>
#endif

#include "kernel/kernel_compat_cpu.h"
#include "kernel/kernel_types.h"
#include "kernel/split/kernel_split_data_types.h"
#include "kernel/kernel_globals.h"

#include "vdb_globals.h"
#include "vdb_intern.h"
#include "vdb_thread.h"

#include "util/util_vector.h"

using namespace openvdb;
using namespace openvdb::tools;

CCL_NAMESPACE_BEGIN

typedef math::Ray<float> vdb_ray_t;

class OpenVDBGridThreadDataBase {
 public:
  virtual ~OpenVDBGridThreadDataBase()
  {
    ;
  }
  static OpenVDBGridThreadDataBase *create_from_texture(const OpenVDBTextureBase *tex);

  virtual bool sample(float x, float y, float z, float *r, float *g, float *b, int sampling) = 0;
  virtual bool sample_index(int x, int y, int z, float *r, float *g, float *b) = 0;
  virtual bool intersect(const Ray *ray, float *isect) = 0;
  virtual bool march(float *t0, float *t1) = 0;
  virtual void index_offset(int *x, int *y, int *z) = 0;
};

template<class T> class OpenVDBGridThreadData : public OpenVDBGridThreadDataBase {
 public:
  OpenVDBGridThreadData(const OpenVDBTexture<T> *_tex)
      : accessor(NULL),
        point_sampler(NULL),
        box_sampler(NULL),
        stag_point_sampler(NULL),
        stag_box_sampler(NULL),
        isector(NULL),
        tex(_tex)
  {
    init(tex);
  }
  typedef typename T::ConstAccessor accessor_t;
  typedef GridSampler<accessor_t, PointSampler> point_sampler_t;
  typedef GridSampler<accessor_t, BoxSampler> box_sampler_t;
  typedef VolumeRayIntersector<T, T::TreeType::RootNodeType::ChildNodeType::LEVEL, vdb_ray_t>
      isector_t;
  typedef GridSampler<accessor_t, StaggeredPointSampler> stag_point_sampler_t;
  typedef GridSampler<accessor_t, StaggeredBoxSampler> stag_box_sampler_t;

  virtual ~OpenVDBGridThreadData()
  {
    free();
  }

  virtual bool sample(float x, float y, float z, float *r, float *g, float *b, int sampling)
  {
    return false;
  }

  virtual bool sample_index(int x, int y, int z, float *r, float *g, float *b)
  {
    return false;
  }

  virtual void index_offset(int *x, int *y, int *z)
  {
    *x = tex->index_offset.x;
    *y = tex->index_offset.y;
    *z = tex->index_offset.z;
  }

  virtual bool march(float *t0, float *t1)
  {
    float vdb_t0(*t0), vdb_t1(*t1);

    if (isector && isector->march(vdb_t0, vdb_t1)) {
      *t0 = isector->getWorldTime(vdb_t0);
      *t1 = isector->getWorldTime(vdb_t1);

      return true;
    }

    return false;
  }

  virtual bool intersect(const Ray *ray, float *isect)
  {
    if (!isector) {
      return false;
    }

    vdb_ray_t::Vec3Type P(ray->P.x, ray->P.y, ray->P.z);
    vdb_ray_t::Vec3Type D(ray->D.x, ray->D.y, ray->D.z);
    D.normalize();

    vdb_ray_t vdb_ray(P, D, 1e-5f, ray->t);

    if (isector->setWorldRay(vdb_ray)) {
      *isect = static_cast<float>(vdb_ray.t1());

      return true;
    }

    return false;
  }

 private:
  void init(const OpenVDBTexture<T> *tex)
  {
    accessor = new accessor_t(tex->grid->getConstAccessor());
    point_sampler = new point_sampler_t(*accessor, tex->grid->transform());
    box_sampler = new box_sampler_t(*accessor, tex->grid->transform());
    stag_point_sampler = new stag_point_sampler_t(*accessor, tex->grid->transform());
    stag_box_sampler = new stag_box_sampler_t(*accessor, tex->grid->transform());
    if (tex->intersector) {
      isector = new isector_t(*tex->intersector);
    }
  }

  void free()
  {
    if (accessor) {
      delete accessor;
    }
    if (point_sampler) {
      delete point_sampler;
    }
    if (box_sampler) {
      delete box_sampler;
    }
    if (stag_point_sampler) {
      delete stag_point_sampler;
    }
    if (stag_box_sampler) {
      delete stag_box_sampler;
    }
    if (isector) {
      delete isector;
    }
  }

  accessor_t *accessor;
  point_sampler_t *point_sampler;
  box_sampler_t *box_sampler;
  stag_point_sampler_t *stag_point_sampler;
  stag_box_sampler_t *stag_box_sampler;
  isector_t *isector;
  const OpenVDBTexture<T> *tex;
};

template<>
inline bool OpenVDBGridThreadData<FloatGrid>::sample_index(
    int x, int y, int z, float *r, float *g, float *b)
{
  float value = point_sampler->sampleVoxel(x, y, z);
  *r = *g = *b = value;
  return true;
}

template<>
inline bool OpenVDBGridThreadData<FloatGrid>::sample(
    float x, float y, float z, float *r, float *g, float *b, int sampling)
{
  float3 pos = make_float3(x, y, z);
  pos = transform_point(&tex->tfm, pos);
  Vec3d p(pos.x, pos.y, pos.z);

  float value = 0.0f;
  switch (sampling) {
    case VDBVolume::OPENVDB_SAMPLE_BOX:
      value = box_sampler->isSample(p);
      break;
    case VDBVolume::OPENVDB_SAMPLE_POINT:
    default:
      value = point_sampler->isSample(p);
  }
  *r = *g = *b = value;
  return true;
}

template<>
inline bool OpenVDBGridThreadData<Vec3SGrid>::sample(
    float x, float y, float z, float *r, float *g, float *b, int sampling)
{
  bool staggered = tex->grid->getGridClass() == GRID_STAGGERED;
  Vec3s value;

  float3 pos = make_float3(x, y, z);
  pos = transform_point(&tex->tfm, pos);
  Vec3d p(pos.x, pos.y, pos.z);

  if (staggered) {
    switch (sampling) {
      case VDBVolume::OPENVDB_SAMPLE_POINT:
        value = stag_point_sampler->isSample(p);
        break;
      case VDBVolume::OPENVDB_SAMPLE_BOX:
      default:
        value = stag_box_sampler->isSample(p);
        break;
    }
  }
  else {
    switch (sampling) {
      case VDBVolume::OPENVDB_SAMPLE_POINT:
        value = point_sampler->wsSample(p);
        break;
      case VDBVolume::OPENVDB_SAMPLE_BOX:
      default:
        value = box_sampler->wsSample(p);
    }
  }

  *r = value.x();
  *g = value.y();
  *b = value.z();
  return true;
}

template<>
inline bool OpenVDBGridThreadData<Vec3SGrid>::sample_index(
    int x, int y, int z, float *r, float *g, float *b)
{
  bool staggered = tex->grid->getGridClass() == GRID_STAGGERED;
  Vec3s value;

  if (staggered) {
    value = stag_point_sampler->sampleVoxel(x, y, z);
  }
  else {
    value = point_sampler->sampleVoxel(x, y, z);
  }

  *r = value.x();
  *g = value.y();
  *b = value.z();
  return true;
}

OpenVDBGridThreadDataBase *OpenVDBGridThreadDataBase::create_from_texture(
    const OpenVDBTextureBase *tex)
{
  const OpenVDBTexture<FloatGrid> *float_grid = dynamic_cast<const OpenVDBTexture<FloatGrid> *>(
      tex);
  if (float_grid) {
    return new OpenVDBGridThreadData<FloatGrid>(float_grid);
  }

  const OpenVDBTexture<Vec3SGrid> *vec_grid = dynamic_cast<const OpenVDBTexture<Vec3SGrid> *>(tex);
  if (float_grid) {
    return new OpenVDBGridThreadData<Vec3SGrid>(vec_grid);
  }

  assert(0);

  return NULL;
}

struct OpenVDBThreadData {
  std::vector<OpenVDBGridThreadDataBase *> data;
};

OpenVDBThreadData *VDBVolume::thread_init(OpenVDBGlobals *vdb_globals)
{
  OpenVDBThreadData *tdata = new OpenVDBThreadData;

  for (size_t i = 0; i < vdb_globals->grids.size(); ++i) {
    if (vdb_globals->grids[i]) {
      tdata->data.push_back(OpenVDBGridThreadDataBase::create_from_texture(vdb_globals->grids[i]));
    }
    else {
      tdata->data.push_back(NULL);
    }
  }

  return tdata;
}

void VDBVolume::thread_free(OpenVDBThreadData *tdata)
{
  if (!tdata) {
    return;
  }

  for (size_t i = 0; i < tdata->data.size(); ++i) {
    if (tdata->data[i]) {
      delete tdata->data[i];
    }
  }

  delete tdata;
}

bool VDBVolume::has_uniform_voxels(OpenVDBGlobals *vdb, int vdb_index)
{
  return vdb->grids[vdb_index]->hasUniformVoxels();
}

bool VDBVolume::sample(OpenVDBThreadData *vdb_thread,
                       int vdb_index,
                       float x,
                       float y,
                       float z,
                       float *r,
                       float *g,
                       float *b,
                       int sampling)
{
  if (vdb_thread && vdb_thread->data.size() > vdb_index && vdb_thread->data[vdb_index]) {
    return vdb_thread->data[vdb_index]->sample(x, y, z, r, g, b, sampling);
  }
  else {
    *r = TEX_IMAGE_MISSING_R;
    *g = TEX_IMAGE_MISSING_G;
    *b = TEX_IMAGE_MISSING_B;
    return false;
  }
}

bool VDBVolume::sample_index(OpenVDBThreadData *vdb_thread,
                             int vdb_index,
                             int x,
                             int y,
                             int z,
                             float *r,
                             float *g,
                             float *b)
{
  if (vdb_thread && vdb_thread->data.size() > vdb_index && vdb_thread->data[vdb_index]) {
    return vdb_thread->data[vdb_index]->sample_index(x, y, z, r, g, b);
  }
  else {
    *r = TEX_IMAGE_MISSING_R;
    *g = TEX_IMAGE_MISSING_G;
    *b = TEX_IMAGE_MISSING_B;
    return false;
  }
}

bool VDBVolume::intersect(OpenVDBThreadData *vdb_thread,
                          int vdb_index,
                          const Ray *ray,
                          float *isect)
{
  if (vdb_index < vdb_thread->data.size() && vdb_thread->data[vdb_index]) {
    return vdb_thread->data[vdb_index]->intersect(ray, isect);
  }
  return false;
}

bool VDBVolume::march(OpenVDBThreadData *vdb_thread, int vdb_index, float *t0, float *t1)
{
  if (vdb_index < vdb_thread->data.size() && vdb_thread->data[vdb_index]) {
    return vdb_thread->data[vdb_index]->march(t0, t1);
  }
  return false;
}

CCL_NAMESPACE_END
