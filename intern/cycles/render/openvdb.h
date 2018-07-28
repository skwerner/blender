#ifndef __IMAGE_OPENVDB_H__
#define __IMAGE_OPENVDB_H__

#include "device/device.h"
#include "render/mesh_volume.h"

#include "util/util_types.h"
#include "util/util_string.h"

CCL_NAMESPACE_BEGIN

bool openvdb_has_grid(const string& filepath, const string& grid_name);
int3 openvdb_get_resolution(const string& filepath);

device_memory *openvdb_load_device_extern(Device *device,
                                          const string& filepath,
                                          const string& grid_name,
                                          const string& mem_name,
                                          const InterpolationType& interpolation,
                                          const ExtensionType& extension,
                                          const bool is_vec);

void openvdb_load_preprocess(const string& filepath,
                             const string& grid_name,
                             const int channels,
                             const float threshold,
                             vector<int> *sparse_index,
                             int &sparse_size);

void openvdb_load_image(const string& filepath,
                        const string& grid_name,
                        const int channels,
                        float *image,
                        vector<int> *sparse_index);

device_memory *openvdb_load_device_intern(Device *device,
                                          const float *data,
                                          const int3 resolution,
                                          const string& mem_name,
                                          const InterpolationType& interpolation,
                                          const ExtensionType& extension,
                                          const bool is_vec);

void openvdb_build_mesh(VolumeMeshBuilder *builder,
                        void *v_grid,
                        const float threshold,
                        const bool is_vec);

CCL_NAMESPACE_END

#endif /* __IMAGE_OPENVDB_H__ */
