
#include <openvdb/openvdb.h>
#include <openvdb/tools/GridTransformer.h>
#include "intern/openvdb_reader.h"
#include "openvdb_capi.h"
#include "render/openvdb.h"

#include "util/util_logging.h"
#include "util/util_path.h"
#include "util/util_sparse_grid.h"

#include "device/device_memory_openvdb.h"

/* Functions that directly use the OpenVDB library throughout render. */

struct OpenVDBReader;

CCL_NAMESPACE_BEGIN

/* Comparison and assignment utilities. */

static const bool gte(openvdb::Vec3SGrid::ConstAccessor accessor,
                      openvdb::math::Coord c, const float f)
{
	openvdb::math::Vec3s v = accessor.getValue(c);
	return (v.x() >= f || v.y() >= f || v.z() >= f);
}

static const bool gte(openvdb::FloatGrid::ConstAccessor accessor,
                      openvdb::math::Coord c, const float f)
{
	return accessor.getValue(c) >= f;
}

static void copy(openvdb::Vec3SGrid::ConstAccessor accessor,
                 openvdb::math::Coord c, float *f)
{
	openvdb::math::Vec3s v = accessor.getValue(c);
	*(f + 0) = v.x();
	*(f + 1) = v.y();
	*(f + 2) = v.z();
	*(f + 3) = 1.0f;
}

static void copy(openvdb::FloatGrid::ConstAccessor accessor,
                 openvdb::math::Coord c, float *f)
{
	*f = accessor.getValue(c);
}

/* Misc internal helper functions.
 * Logging should be done by callers. */

/* Simple range shift for grids with non-zero background values. May have
 * strange results depending on the grid. */
static void shift_range(openvdb::Vec3SGrid::Ptr grid)
{
	using namespace openvdb;

	const math::Vec3s z(0.0f, 0.0f, 0.0f);
	const math::Vec3s background_value = grid->background();

	if(background_value != z) {
		for (Vec3SGrid::ValueOnIter iter = grid->beginValueOn(); iter; ++iter) {
		    iter.setValue(iter.getValue() - background_value);
		}
		tools::changeBackground(grid->tree(), z);
	}
}

static void shift_range(openvdb::FloatGrid::Ptr grid)
{
	using namespace openvdb;

	const float background_value = grid->background();

	if(background_value != 0.0f) {
		for (FloatGrid::ValueOnIter iter = grid->beginValueOn(); iter; ++iter) {
		    iter.setValue(iter.getValue() - background_value);
		}
		tools::changeBackground(grid->tree(), 0.0f);
	}
}

template<typename GridType>
static bool get_grid(const string& filepath,
                     const string& grid_name,
                     typename GridType::Ptr& grid,
                     typename GridType::ConstAccessor& accessor,
                     int3& resolution)
{
	using namespace openvdb;

	if(!path_exists(filepath) || path_is_directory(filepath)) {
		return false;
	}

	struct OpenVDBReader *reader = OpenVDBReader_create();
	OpenVDBReader_open(reader, filepath.c_str());

	if (!OpenVDBReader_has_grid(reader, grid_name.c_str())) {
		OpenVDBReader_free(reader);
		return false;
	}

	int min_bound[3], res[3];
	OpenVDBReader_get_bounds(reader, min_bound, NULL, res, NULL, NULL, NULL);

	/* In order to keep sampling uniform, we expect a volume's bound to begin at
	 * (0, 0, 0) in object space. External VDBs may have a non-zero origin, so
	 * all voxels must be translated. This process may be memory inefficient. */

	typename GridType::Ptr orig_grid = gridPtrCast<GridType>(reader->getGrid(grid_name));

	if(min_bound[0] == 0 && min_bound[1] == 0 && min_bound[2] == 0) {
		grid = orig_grid;
	}
	else {
		grid = GridType::create();
		math::Mat4d xform = math::Mat4d::identity();
		math::Vec3d translation(-min_bound[0], -min_bound[1], -min_bound[2]);
		xform.setTranslation(translation);
		tools::GridTransformer transformer(xform);
		transformer.transformGrid<tools::PointSampler, GridType>(*orig_grid, *grid);
	}

	/* Need to account for external grids with a non-zero background value and
	 * voxels below background value. */
	shift_range(grid);

	accessor = grid->getConstAccessor();
	resolution = make_int3(res[0], res[1], res[2]);

	OpenVDBReader_free(reader);
	return true;
}

/* Misc external helper functions. These all assume that the file exists and is
 * a valid .vdb file. Logging should be done by callers. */

bool openvdb_has_grid(const string& filepath, const string& grid_name)
{
	if(grid_name.empty()) {
		return false;
	}

	struct OpenVDBReader *reader = OpenVDBReader_create();
	OpenVDBReader_open(reader, filepath.c_str());

	bool has_grid = OpenVDBReader_has_grid(reader, grid_name.c_str());

	OpenVDBReader_free(reader);
	return has_grid;
}

int3 openvdb_get_resolution(const string& filepath)
{
	struct OpenVDBReader *reader = OpenVDBReader_create();
	OpenVDBReader_open(reader, filepath.c_str());

	int res[3];
	OpenVDBReader_get_bounds(reader, NULL, NULL, res, NULL, NULL, NULL);
	OpenVDBReader_free(reader);

	return make_int3(res[0], res[1], res[2]);
}

/* For now, since there is no official OpenVDB interpolation implementations
 * for CUDA or OpenCL, OpenVDB grids can only be saved for CPU rendering.
 * Otherwise, we convert the OpenVDB grids to arrays. */

/* Direct load external OpenVDB grid to device.
 * Thread must be locked before file_load_openvdb_cpu() is called. */
template<typename GridType>
static device_memory *openvdb_load_device_extern(Device *device,
                                                 const string& filepath,
                                                 const string& grid_name,
                                                 const string& mem_name,
                                                 const InterpolationType& interpolation,
                                                 const ExtensionType& extension)
{
	using namespace openvdb;

	typename GridType::Ptr grid = GridType::create();
	typename GridType::ConstAccessor accessor = grid->getConstAccessor();
	int3 resolution;

	if(!get_grid<GridType>(filepath, grid_name, grid, accessor, resolution)) {
		return NULL;
	}

	device_openvdb<GridType> *tex_img =
			new device_openvdb<GridType>(device, mem_name.c_str(),
	                                     MEM_TEXTURE, grid, accessor,
	                                     resolution);

	tex_img->interpolation = interpolation;
	tex_img->extension = extension;
	tex_img->grid_type = IMAGE_GRID_TYPE_OPENVDB;
	tex_img->copy_to_device();

	return tex_img;
}

device_memory *openvdb_load_device_extern(Device *device,
                                          const string& filepath,
                                          const string& grid_name,
                                          const string& mem_name,
                                          const InterpolationType& interpolation,
                                          const ExtensionType& extension,
                                          const bool is_vec)
{
	if(is_vec) {
		return openvdb_load_device_extern<openvdb::Vec3SGrid>(device, filepath,
		                                                      grid_name, mem_name,
		                                                      interpolation, extension);
	}
	else {
		return openvdb_load_device_extern<openvdb::FloatGrid>(device, filepath,
		                                                      grid_name, mem_name,
		                                                      interpolation, extension);
	}
}

/* Load OpenVDB file to sparse grid. Based on util/util_sparse_grid.h */
template<typename GridType>
static const bool openvdb_check_tile_active(typename GridType::ConstAccessor accessor,
                                            int x, int y, int z,
                                            float threshold, int3 resolution)
{
	using namespace openvdb;

	math::Coord ijk;
	int &i = ijk[0], &j = ijk[1], &k = ijk[2];

	const int max_i = min(x + TILE_SIZE, resolution.x);
	const int max_j = min(y + TILE_SIZE, resolution.y);
	const int max_k = min(z + TILE_SIZE, resolution.z);

	for(k = z; k < max_k; ++k) {
		for(j = y; j < max_j; ++j) {
			for(i = x; i < max_i; ++i) {
				if(gte(accessor, ijk, threshold)) {
					return true;
				}
			}
		}
	}
	return false;
}

template<typename GridType>
static bool openvdb_load_sparse(const string& filepath,
                                const string& grid_name,
                                const int channels,
                                const float threshold,
                                vector<float> *sparse_grid,
		                        vector<int> *grid_info)
{
	using namespace openvdb;

	typename GridType::Ptr grid = GridType::create();
	typename GridType::ConstAccessor accessor = grid->getConstAccessor();
	int3 resolution;

	if(!get_grid<GridType>(filepath, grid_name, grid, accessor, resolution)) {
		return false;
	}

	const int tile_count = get_tile_res(resolution.x) *
	                       get_tile_res(resolution.y) *
	                       get_tile_res(resolution.z);
	const int tile_pix_count = TILE_SIZE * TILE_SIZE * TILE_SIZE * channels;

	/* Initial prepass to find active tiles. */
	grid_info->resize(tile_count);
	int tile = 0, active_count = 0;

	for(int z = 0; z < resolution.z; z += TILE_SIZE) {
		for(int y = 0; y < resolution.y; y += TILE_SIZE) {
			for(int x = 0; x < resolution.x; x += TILE_SIZE, ++tile) {
				int is_active = openvdb_check_tile_active<GridType>(accessor,
				                                                    x, y, z,
				                                                    threshold,
				                                                    resolution);
				active_count += is_active;
				/* 0 if active, -1 if inactive. */
				grid_info->at(tile) = is_active - 1;
			}
		}
	}

	/* Check memory savings. */
	int sparse_mem_use = (tile_count * sizeof(int) +
	                      active_count * tile_pix_count * sizeof(float));
	int dense_mem_use = resolution.x * resolution.y *
	                    resolution.z * channels * sizeof(float);

	if(sparse_mem_use >= dense_mem_use) {
		VLOG(1) << "Memory of " << grid_name << " increased from "
		        << string_human_readable_size(dense_mem_use) << " to "
		        << string_human_readable_size(sparse_mem_use)
		        << ", not using sparse grid";
		return false;
	}

	VLOG(1) << "Memory of " << grid_name << " decreased from "
			<< string_human_readable_size(dense_mem_use) << " to "
			<< string_human_readable_size(sparse_mem_use);

	/* Populate the sparse grid. */
	sparse_grid->resize(active_count * tile_pix_count);
	float *sg = &(*sparse_grid)[0];

	int voxel = 0;
	tile = 0;

	for(int z = 0; z < resolution.z; z += TILE_SIZE) {
		for(int y = 0; y < resolution.y; y += TILE_SIZE) {
			for(int x = 0; x < resolution.x; x += TILE_SIZE, ++tile) {
				if(grid_info->at(tile) == -1) {
					continue;
				}

				grid_info->at(tile) = voxel / channels;

				/* Populate the tile. */
				const int max_i = min(x + TILE_SIZE, resolution.x);
				const int max_j = min(y + TILE_SIZE, resolution.y);
				const int max_k = min(z + TILE_SIZE, resolution.z);

				math::Coord ijk;
				int &i = ijk[0], &j = ijk[1], &k = ijk[2];

				for(k = z; k < max_k; ++k) {
					for(j = y; j < max_j; ++j) {
						for(i = x; i < max_i; ++i, ++voxel) {
							copy(accessor, ijk, sg + voxel);
						}
					}
				}
			}
		}
	}

	return true;
}

bool openvdb_load_sparse(const string& filepath,
                         const string& grid_name,
                         const int channels,
                         const float threshold,
                         vector<float> *sparse_grid,
                         vector<int> *grid_info)
{
	if(channels > 1) {
		return openvdb_load_sparse<openvdb::Vec3SGrid>(filepath, grid_name,
		                                               channels, threshold,
		                                               sparse_grid, grid_info);
	}
	else {
		return openvdb_load_sparse<openvdb::FloatGrid>(filepath, grid_name,
		                                               channels, threshold,
		                                               sparse_grid, grid_info);
	}
}

/* Load OpenVDB file to dense grid. */
template<typename GridType>
static bool openvdb_load_dense(const string& filepath, const string& grid_name,
	                           float *data, const int channels)
{
	using namespace openvdb;

	typename GridType::Ptr grid = GridType::create();
	typename GridType::ConstAccessor accessor = grid->getConstAccessor();
	int3 resolution;

	if(!get_grid<GridType>(filepath, grid_name, grid, accessor, resolution)) {
		return false;
	}

	math::Coord xyz;
	int &x = xyz[0], &y = xyz[1], &z = xyz[2];
	int index = 0;

	for (z = 0; z < resolution.z; ++z) {
		for (y = 0; y < resolution.y; ++y) {
			for (x = 0; x < resolution.x; ++x, index += channels) {
				copy(accessor, xyz, data + index);
			}
		}
	}

	return true;
}

bool openvdb_load_dense(const string& filepath, const string& grid_name,
                        float *data, const int channels)
{
	if(channels > 1) {
		return openvdb_load_dense<openvdb::Vec3SGrid>(filepath, grid_name,
		                                              data, channels);
	}
	else {
		return openvdb_load_dense<openvdb::FloatGrid>(filepath, grid_name,
		                                              data, channels);
	}
}

/* Convert internal volume data to OpenVDB grid, then load grid to device. */
static device_memory *openvdb_load_device_intern_vec(Device *device,
                                                     const float *data,
                                                     const int3 resolution,
                                                     const string& mem_name,
                                                     const InterpolationType& interpolation,
                                                     const ExtensionType& extension)
{
	using namespace openvdb;

	Vec3SGrid::Ptr grid = Vec3SGrid::create();
	Vec3SGrid::Accessor accessor = grid->getAccessor();

	const math::Vec3s zero_vec(0.0f, 0.0f, 0.0f);
	tools::changeBackground(grid->tree(), zero_vec);

	openvdb::math::Coord xyz;
	int &x = xyz[0], &y = xyz[1], &z = xyz[2];

	for (z = 0; z < resolution.z; ++z) {
		for (y = 0; y < resolution.y; ++y) {
			for (x = 0; x < resolution.x; ++x) {
				int index = (x + resolution.x * (y + z * resolution.y)) * 4;
				openvdb::math::Vec3s val(data[index + 0], data[index + 1], data[index + 2]);
				accessor.setValue(xyz, val);
			}
		}
	}

	Vec3SGrid::ConstAccessor c_accessor = grid->getConstAccessor();
	device_openvdb<Vec3SGrid> *tex_img =
			new device_openvdb<Vec3SGrid>(device, mem_name.c_str(), MEM_TEXTURE,
	                                      grid, c_accessor, resolution);

	tex_img->interpolation = interpolation;
	tex_img->extension = extension;
	tex_img->grid_type = IMAGE_GRID_TYPE_OPENVDB;
	tex_img->copy_to_device();

	return tex_img;
}

static device_memory *openvdb_load_device_intern_flt(Device *device,
                                                     const float *data,
                                                     const int3 resolution,
                                                     const string& mem_name,
                                                     const InterpolationType& interpolation,
                                                     const ExtensionType& extension)
{
	using namespace openvdb;

	FloatGrid::Ptr grid = FloatGrid::create();
	FloatGrid::Accessor accessor = grid->getAccessor();

	tools::changeBackground(grid->tree(), 0.0f);

	openvdb::math::Coord xyz;
	int &x = xyz[0], &y = xyz[1], &z = xyz[2];

	for (z = 0; z < resolution.z; ++z) {
		for (y = 0; y < resolution.y; ++y) {
			for (x = 0; x < resolution.x; ++x) {
				int index = x + resolution.x * (y + z * resolution.y);
				accessor.setValue(xyz, data[index]);
			}
		}
	}

	FloatGrid::ConstAccessor c_accessor = grid->getConstAccessor();
	device_openvdb<FloatGrid> *tex_img =
			new device_openvdb<FloatGrid>(device, mem_name.c_str(), MEM_TEXTURE,
	                                      grid, c_accessor, resolution);

	tex_img->interpolation = interpolation;
	tex_img->extension = extension;
	tex_img->grid_type = IMAGE_GRID_TYPE_OPENVDB;
	tex_img->copy_to_device();

	return tex_img;
}

device_memory *openvdb_load_device_intern(Device *device,
                                          const float *data,
                                          const int3 resolution,
                                          const string& mem_name,
                                          const InterpolationType& interpolation,
                                          const ExtensionType& extension,
                                          const bool is_vec)
{
	if(is_vec) {
		return openvdb_load_device_intern_vec(device, data, resolution,
		                                      mem_name, interpolation, extension);
	}
	else {
		return openvdb_load_device_intern_flt(device, data, resolution,
		                                      mem_name, interpolation, extension);
	}
}

/* Volume Mesh Builder functions. */

template<typename GridType>
static void openvdb_build_mesh(VolumeMeshBuilder *builder, void *v_accessor,
                               const int3 resolution, const float threshold)
{
	using namespace openvdb;

	typename GridType::ConstAccessor *acc =
	        static_cast<typename GridType::ConstAccessor*>(v_accessor);

	math::Coord xyz;
	int &x = xyz[0], &y = xyz[1], &z = xyz[2];

	for (z = 0; z < resolution.z; ++z) {
		for (y = 0; y < resolution.y; ++y) {
			for (x = 0; x < resolution.x; ++x) {
				if(gte(*acc, xyz, threshold)) {
					builder->add_node_with_padding(x, y, z);
				}
			}
		}
	}
}

void openvdb_build_mesh(VolumeMeshBuilder *builder, void *v_accessor,
                        const int3 resolution, const float threshold,
                        const bool is_vec)
{
	if(is_vec) {
		openvdb_build_mesh<openvdb::Vec3SGrid>(builder, v_accessor,
		                                       resolution, threshold);
	}
	else {
		openvdb_build_mesh<openvdb::FloatGrid>(builder, v_accessor,
		                                       resolution, threshold);
	}
}

CCL_NAMESPACE_END
