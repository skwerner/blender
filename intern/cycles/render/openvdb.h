#ifndef __IMAGE_OPENVDB_H__
#define __IMAGE_OPENVDB_H__

#include "util/util_types.h"
#include "util/util_string.h"

CCL_NAMESPACE_BEGIN

bool openvdb_has_grid(const string& filepath, const string& grid_name);
int3 openvdb_get_resolution(const string& filepath);

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

CCL_NAMESPACE_END

#endif /* __IMAGE_OPENVDB_H__ */
