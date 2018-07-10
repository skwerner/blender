
#include <openvdb/openvdb.h>
#include "intern/openvdb_reader.h"
#include "openvdb_capi.h"
#include "render/openvdb.h"

#include "util/util_logging.h"
#include "util/util_path.h"

#include "device/device_memory_openvdb.h"

/* Functions that directly use the OpenVDB library throughout render. */

struct OpenVDBReader;

CCL_NAMESPACE_BEGIN

static struct OpenVDBReader *get_reader(const string& filepath,
                                        const string& grid_name)
{
	/* Logging is done when metadata is retrieved. */
	if(!path_exists(filepath) || path_is_directory(filepath)) {
		return NULL;
	}

	struct OpenVDBReader *reader = OpenVDBReader_create();
	OpenVDBReader_open(reader, filepath.c_str());

	/* If grid name is provided, we also check it's validity here. */
	if(!grid_name.empty()) {
		if (!OpenVDBReader_has_grid(reader, grid_name.c_str())) {
			VLOG(1) << filepath << " does not have grid " << grid_name;
			OpenVDBReader_free(reader);
			return NULL;
		}
	}

	return reader;
}

static void OpenVDB_import_grid_vector_fl4(OpenVDBReader *reader,
                                           const openvdb::Name &name,
                                           float4 *data,
                                           const int3& resolution)

{
	using namespace openvdb;

	Vec3SGrid::Ptr vgrid = gridPtrCast<Vec3SGrid>(reader->getGrid(name));
	Vec3SGrid::ConstAccessor acc = vgrid->getConstAccessor();
	math::Coord xyz;
	int &x = xyz[0], &y = xyz[1], &z = xyz[2];

	size_t index = 0;
	for (z = 0; z < resolution.z; ++z) {
		for (y = 0; y < resolution.y; ++y) {
			for (x = 0; x < resolution.x; ++x, index += 4) {
				math::Vec3s value = acc.getValue(xyz);
				(*data)[index + 0] = value.x();
				(*data)[index + 1] = value.y();
				(*data)[index + 2] = value.z();
				(*data)[index + 3] = 1.0f;
			}
		}
	}
}

bool openvdb_has_grid(const string& filepath, const string& grid_name)
{
	return get_reader(filepath, grid_name);
}

int3 get_openvdb_resolution(const string& filepath)
{
	struct OpenVDBReader *reader = get_reader(filepath, string());
	if(!reader) {
		return make_int3(0, 0, 0);
	}

	int res[3];
	OpenVDBReader_get_simple_bounds(reader, res);

	OpenVDBReader_free(reader);

	return make_int3(res[0], res[1], res[2]);
}

/* For now, since there is no official OpenVDB interpolation implementations
 * for CUDA or OpenCL, OpenVDB grids can only be saved for CPU rendering.
 * Otherwise, we convert the OpenVDB grids to dense arrays. */

/* to-do (gschua): handle texture limits. */

/* Thread must be locked before file_load_openvdb_cpu() is called. */
device_memory *file_load_openvdb(Device *device,
                                 const string& filepath,
                                 const string& grid_name,
                                 const int3& resolution,
                                 const string& mem_name,
                                 const InterpolationType& interpolation,
                                 const ExtensionType& extension,
                                 const bool& is_vec,
                                 const int& /*texture_limit*/)
{
	using namespace openvdb;

	struct OpenVDBReader *reader = get_reader(filepath, grid_name);
	if(!reader) {
		return NULL;
	}

	if(is_vec) {
		Vec3SGrid::Ptr grid = gridPtrCast<Vec3SGrid>(reader->getGrid(grid_name));
		Vec3SGrid::ConstAccessor accessor = grid->getConstAccessor();

		device_openvdb<Vec3SGrid, float4> *tex_img =
		        new device_openvdb<Vec3SGrid, float4>(device,
		                                              mem_name.c_str(),
		                                              MEM_TEXTURE,
		                                              grid,
		                                              accessor,
		                                              resolution);
		tex_img->interpolation = interpolation;
		tex_img->extension = extension;
		tex_img->grid_type = IMAGE_GRID_TYPE_OPENVDB;
		tex_img->copy_to_device();

		OpenVDBReader_free(reader);
		return tex_img;
	}
	else {
		FloatGrid::Ptr grid = gridPtrCast<FloatGrid>(reader->getGrid(grid_name));
		FloatGrid::ConstAccessor accessor = grid->getConstAccessor();

		device_openvdb<FloatGrid, float> *tex_img =
		        new device_openvdb<FloatGrid, float>(device,
		                                             mem_name.c_str(),
		                                             MEM_TEXTURE,
	                                                 grid,
		                                             accessor,
	                                                 resolution);

		tex_img->interpolation = interpolation;
		tex_img->extension = extension;
		tex_img->grid_type = IMAGE_GRID_TYPE_OPENVDB;
		tex_img->copy_to_device();

		OpenVDBReader_free(reader);
		return tex_img;
	}
}

bool file_load_openvdb_dense(const string& filepath,
                             const string& grid_name,
                             const int3& resolution,
                             const int& /*texture_limit*/,
                             float *data)
{
	struct OpenVDBReader *reader = get_reader(filepath, grid_name);
	if(!reader) {
		return false;
	}

	int res[3] = {resolution.x, resolution.y, resolution.z};
	OpenVDB_import_grid_fl(reader, grid_name.c_str(), &data, res);

	OpenVDBReader_free(reader);

	return true;
}

bool file_load_openvdb_dense(const string& filepath,
                             const string& grid_name,
                             const int3& resolution,
                             const int& /*texture_limit*/,
                             float4 *data)
{
	struct OpenVDBReader *reader = get_reader(filepath, grid_name);
	if(!reader) {
		return false;
	}

	OpenVDB_import_grid_vector_fl4(reader, grid_name, data, resolution);

	OpenVDBReader_free(reader);

	return true;
}

void build_openvdb_mesh_fl(VolumeMeshBuilder *builder,
                           void *v_accessor,
                           const int3 resolution,
                           const float isovalue)
{
	using namespace openvdb;

	FloatGrid::ConstAccessor *acc = static_cast<FloatGrid::ConstAccessor*>(v_accessor);
	math::Coord xyz;
	int &x = xyz[0], &y = xyz[1], &z = xyz[2];

	for (z = 0; z < resolution.z; ++z) {
		for (y = 0; y < resolution.y; ++y) {
			for (x = 0; x < resolution.x; ++x) {
				if(acc->getValue(xyz) >= isovalue) {
					builder->add_node_with_padding(x, y, z);
				}
			}
		}
	}
}

void build_openvdb_mesh_vec(VolumeMeshBuilder *builder,
                            void *v_accessor,
                            const int3 resolution,
                            const float isovalue)
{
	using namespace openvdb;

	Vec3SGrid::ConstAccessor *acc = static_cast<Vec3SGrid::ConstAccessor*>(v_accessor);
	math::Coord xyz;
	int &x = xyz[0], &y = xyz[1], &z = xyz[2];

	for (z = 0; z < resolution.z; ++z) {
		for (y = 0; y < resolution.y; ++y) {
			for (x = 0; x < resolution.x; ++x) {
				math::Vec3s val = acc->getValue(xyz);
				if(val.x() >= isovalue || val.y() >= isovalue || val.z() >= isovalue) {
					builder->add_node_with_padding(x, y, z);
				}
			}
		}
	}
}

CCL_NAMESPACE_END
