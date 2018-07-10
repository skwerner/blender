#ifndef __IMAGE_OPENVDB_H__
#define __IMAGE_OPENVDB_H__

#include "device/device.h"
#include "render/mesh_volume.h"

#include "util/util_types.h"
#include "util/util_string.h"

CCL_NAMESPACE_BEGIN

bool openvdb_has_grid(const string& filepath, const string& grid_name);
int3 get_openvdb_resolution(const string& filepath);

device_memory *file_load_openvdb(Device *device,
                                 const string& filepath,
                                 const string& grid_name,
                                 const int3& resolution,
                                 const string& mem_name,
                                 const InterpolationType& interpolation,
                                 const ExtensionType& extension,
                                 const bool& is_vec,
                                 const int& /*texture_limit*/);

bool file_load_openvdb_dense(const string& filepath,
                             const string& grid_name,
                             const int3& resolution,
                             const int& /*texture_limit*/,
                             float *data);

bool file_load_openvdb_dense(const string& filepath,
                             const string& grid_name,
                             const int3& resolution,
                             const int& /*texture_limit*/,
                             float4 *data);

void build_openvdb_mesh_fl(VolumeMeshBuilder *builder,
                           void *v_accessor,
                           const int3 resolution,
                           const float isovalue);

void build_openvdb_mesh_vec(VolumeMeshBuilder *builder,
                            void *v_accessor,
                            const int3 resolution,
                            const float isovalue);

CCL_NAMESPACE_END

#endif /* __IMAGE_OPENVDB_H__ */
