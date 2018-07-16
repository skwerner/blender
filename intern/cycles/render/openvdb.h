#ifndef __IMAGE_OPENVDB_H__
#define __IMAGE_OPENVDB_H__

#include "device/device.h"
#include "render/mesh_volume.h"

#include "util/util_types.h"
#include "util/util_string.h"

CCL_NAMESPACE_BEGIN

bool openvdb_has_grid(const string& filepath, const string& grid_name);
int3 openvdb_get_resolution(const string& filepath);

device_memory *openvdb_load_device(Device *device,
                                   const string& filepath,
                                   const string& grid_name,
                                   const string& mem_name,
                                   const InterpolationType& interpolation,
                                   const ExtensionType& extension,
                                   const bool is_vec);

int openvdb_preprocess(const string& filepath, const string& grid_name,
                       const float threshold, int *grid_info, const bool is_vec);

bool openvdb_load_sparse(const string& filepath, const string& grid_name,
                         float *data, int *grid_info, const int channels);

bool openvdb_load_dense(const string& filepath, const string& grid_name,
                        float *data, const int channels);

void openvdb_build_mesh(VolumeMeshBuilder *builder, void *v_accessor,
                        const int3 resolution, const float threshold,
                        const bool is_vec);

CCL_NAMESPACE_END

#endif /* __IMAGE_OPENVDB_H__ */
