#ifndef __IMAGE_OPENVDB_H__
#define __IMAGE_OPENVDB_H__

#include "util/util_string.h"

CCL_NAMESPACE_BEGIN

/* Common OpenVDB grid types. */
enum OpenVDBGridType {
	OPENVDB_GRID_BOOL,
	OPENVDB_GRID_DOUBLE,
	OPENVDB_GRID_FLOAT,
	OPENVDB_GRID_INT32,
	OPENVDB_GRID_INT64,
	OPENVDB_GRID_VEC_DOUBLE, /* Vec3D */
	OPENVDB_GRID_VEC_UINT32, /* Vec3I */
	OPENVDB_GRID_VEC_FLOAT, /* Vec3S */

	OPENVDB_GRID_MISC
};

void openvdb_initialize();
bool openvdb_has_grid(const string& filepath, const string& grid_name);
int3 openvdb_get_resolution(const string& filepath);

void openvdb_load_preprocess(const string& filepath,
                             const string& grid_name,
                             const float threshold,
                             const bool use_pad,
                             vector<int> *sparse_index,
                             int &sparse_size);

void openvdb_load_image(const string& filepath,
                        const string& grid_name,
                        const vector<int> *sparse_indexes,
                        const int sparse_size,
                        const bool use_pad,
                        float *image);

CCL_NAMESPACE_END

#endif /* __IMAGE_OPENVDB_H__ */
