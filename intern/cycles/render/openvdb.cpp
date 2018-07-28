
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

/* Misc internal helper functions. */

static bool operator >=(const openvdb::math::Vec3s &a, const float &b)
{
	return a.x() >= b || a.y() >= b || a.z() >= b;
}

static void copy(float *des, const openvdb::math::Vec3s *src)
{
	*(des + 0) = src->x();
	*(des + 1) = src->y();
	*(des + 2) = src->z();
	*(des + 3) = 1.0f;
}

static void copy(float *des, const float *src)
{
	*des = *src;
}

static const int tile_index(openvdb::math::Coord start, int3 tiled_res)
{
	return compute_index(start.x() / TILE_SIZE, start.y() / TILE_SIZE,
	                     start.z() / TILE_SIZE, tiled_res.x, tiled_res.y);
}

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
                     int3 *resolution,
                     openvdb::math::Coord *minimum_bound)
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

	grid = gridPtrCast<GridType>(reader->getGrid(grid_name));

	/* Verify that leaf dimensions match internal tile dimensions. */
	typename GridType::TreeType::LeafCIter iter = grid->tree().cbeginLeaf();
	if(iter) {
		const math::Coord dim = iter.getLeaf()->getNodeBoundingBox().dim();

		if(dim[0] != TILE_SIZE || dim[1] != TILE_SIZE || dim[2] != TILE_SIZE) {
			VLOG(1) << "Cannot load grid " << grid->getName() << " from "
			        << filepath << ", leaf dimensions are "
			        << dim[0] << "x" << dim[1] << "x" << dim[2];
			OpenVDBReader_free(reader);
			return false;
		}
	}

	/* Need to account for external grids with a non-zero background value and
	 * voxels below background value. */
	shift_range(grid);

	int min_bound[3], res[3];
	OpenVDBReader_get_bounds(reader, min_bound, NULL, res, NULL, NULL, NULL);

	if(resolution) {
		*resolution = make_int3(res[0], res[1], res[2]);
	}
	if(minimum_bound) {
		*minimum_bound = math::Coord(min_bound[0], min_bound[1], min_bound[2]);
	}

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
	typename GridType::Ptr orig_grid = GridType::create();
	int3 resolution;
	math::Coord min_bound;

	if(!get_grid<GridType>(filepath, grid_name, orig_grid, &resolution, &min_bound)) {
		return NULL;
	}

	/* In order to keep sampling uniform, we expect a volume's bound to begin at
	 * (0, 0, 0) in object space. External VDBs may have a non-zero origin, so
	 * all voxels must be translated. This process may be memory inefficient. */

	if(min_bound.x() == 0 && min_bound.y() == 0 && min_bound.z() == 0) {
		grid = orig_grid;
	}
	else {
		math::Mat4d xform = math::Mat4d::identity();
		math::Vec3d translation(-min_bound.x(), -min_bound.y(), -min_bound.z());
		xform.setTranslation(translation);
		tools::GridTransformer transformer(xform);
		transformer.transformGrid<tools::PointSampler, GridType>(*orig_grid, *grid);
	}

	typename GridType::ConstAccessor accessor = grid->getConstAccessor();

	device_openvdb<GridType> *tex_img =
			new device_openvdb<GridType>(device, mem_name.c_str(), MEM_TEXTURE,
	                                     grid, accessor, resolution);

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

/* Load OpenVDB file to texture grid. */
template<typename GridType, typename T>
static void openvdb_load_preprocess(const string& filepath,
                                    const string& grid_name,
                                    const int channels,
                                    const float threshold,
                                    vector<int> *sparse_index,
                                    int &sparse_size)
{
	using namespace openvdb;

	typename GridType::Ptr grid = GridType::create();
	int3 resolution;
	math::Coord min_bound;

	if(!get_grid<GridType>(filepath, grid_name, grid, &resolution, &min_bound) ||
	   !(channels == 4 || channels == 1))
	{
		return;
	}

	const int3 tiled_res = make_int3(get_tile_res(resolution.x),
	                                 get_tile_res(resolution.y),
	                                 get_tile_res(resolution.z));
	const int3 last_tile = make_int3(resolution.x % TILE_SIZE,
	                                 resolution.y % TILE_SIZE,
	                                 resolution.z % TILE_SIZE);
	const int tile_count = tiled_res.x * tiled_res.y * tiled_res.z;
	const int tile_pix_count = TILE_SIZE * TILE_SIZE * TILE_SIZE * channels;

	sparse_index->resize(tile_count, -1); /* 0 if active, -1 if inactive. */
	int voxel_count = 0;

	for (typename GridType::TreeType::LeafCIter iter = grid->tree().cbeginLeaf(); iter; ++iter) {
		const typename GridType::TreeType::LeafNodeType *leaf = iter.getLeaf();
		const T *data = leaf->buffer().data();
		const math::Coord start = leaf->getNodeBoundingBox().getStart() - min_bound;

		for(int i = 0; i < tile_pix_count; ++i) {
			if(data[i] >= threshold) {
				sparse_index->at(tile_index(start, tiled_res)) = 0;

				/* Calculate how many voxels are in this tile. */
				int tile_width = (start.x() + TILE_SIZE > resolution.x) ? last_tile.x : TILE_SIZE;
				int tile_height = (start.y() + TILE_SIZE > resolution.y) ? last_tile.y : TILE_SIZE;
				int tile_depth = (start.z() + TILE_SIZE > resolution.z) ? last_tile.z : TILE_SIZE;
				voxel_count += tile_width * tile_height * tile_depth;

				break;
			}
		}
	}

	/* Check memory savings. */
	const int sparse_mem_use = tile_count * sizeof(int) + voxel_count * channels * sizeof(float);
	const int dense_mem_use = resolution.x * resolution.y * resolution.z * channels * sizeof(float);

	if(sparse_mem_use < dense_mem_use) {
		VLOG(1) << "Memory of " << grid_name << " decreased from "
				<< string_human_readable_size(dense_mem_use) << " to "
				<< string_human_readable_size(sparse_mem_use);
		sparse_size = voxel_count * channels;
	}
	else {
		VLOG(1) << "Memory of " << grid_name << " increased from "
		        << string_human_readable_size(dense_mem_use) << " to "
		        << string_human_readable_size(sparse_mem_use)
		        << ", not using sparse grid";
		sparse_size = -1;
		sparse_index->resize(0);
	}
}

void openvdb_load_preprocess(const string& filepath,
                             const string& grid_name,
                             const int channels,
                             const float threshold,
                             vector<int> *sparse_index,
                             int &sparse_size)
{
	if(channels > 1) {
		return openvdb_load_preprocess<openvdb::Vec3SGrid, openvdb::math::Vec3s>(
		            filepath, grid_name, channels, threshold, sparse_index, sparse_size);
	}
	else {
		return openvdb_load_preprocess<openvdb::FloatGrid, float>(
		            filepath, grid_name, channels, threshold, sparse_index, sparse_size);
	}
}


template<typename GridType, typename T>
static void openvdb_load_image(const string& filepath,
                               const string& grid_name,
                               const int channels,
                               float *image,
                               vector<int> *sparse_index)
{
	using namespace openvdb;

	typename GridType::Ptr grid = GridType::create();
	int3 resolution;
	math::Coord min_bound;

	if(!get_grid<GridType>(filepath, grid_name, grid, &resolution, &min_bound) ||
	   !(channels == 4 || channels == 1))
	{
		return;
	}

	bool make_sparse = false;
	if(sparse_index) {
		if(sparse_index->size() > 0) {
			make_sparse = true;
		}
	}

	if(make_sparse) {
		/* Load VDB as sparse image. */
		const int3 tiled_res = make_int3(get_tile_res(resolution.x),
		                                 get_tile_res(resolution.y),
		                                 get_tile_res(resolution.z));
		const int3 last_tile = make_int3(resolution.x % TILE_SIZE,
		                                 resolution.y % TILE_SIZE,
		                                 resolution.z % TILE_SIZE);
		int start_index = 0;

		for (typename GridType::TreeType::LeafCIter iter = grid->tree().cbeginLeaf(); iter; ++iter) {
			const typename GridType::TreeType::LeafNodeType *leaf = iter.getLeaf();

			const math::Coord start = leaf->getNodeBoundingBox().getStart() - min_bound;
			int tile = tile_index(start, tiled_res);
			if(sparse_index->at(tile) == -1) {
				continue;
			}
			sparse_index->at(tile) = start_index / channels;


			const T *vdb_tile = leaf->buffer().data();
			float *arr_tile = image + start_index;


			const int tile_width = (start.x() + TILE_SIZE > resolution.x) ? last_tile.x : TILE_SIZE;
			const int tile_height = (start.y() + TILE_SIZE > resolution.y) ? last_tile.y : TILE_SIZE;
			const int tile_depth = (start.z() + TILE_SIZE > resolution.z) ? last_tile.z : TILE_SIZE;

			/* Index computation by coordinates is reversed in VDB grids. */
			for(int k = 0; k < tile_depth; ++k) {
				for(int j = 0; j < tile_height; ++j) {
					for(int i = 0; i < tile_width; ++i) {
						int arr_index = compute_index(i, j, k, tile_width, tile_height);
						int vdb_index = compute_index(k, j, i, TILE_SIZE, TILE_SIZE);
						copy(arr_tile + arr_index, vdb_tile + vdb_index);
					}
				}
			}

			start_index += tile_width * tile_height * tile_depth;
		}
	}
	else {
		/* Load VDB as dense image. */
		for (typename GridType::TreeType::LeafCIter iter = grid->tree().cbeginLeaf(); iter; ++iter) {
			const typename GridType::TreeType::LeafNodeType *leaf = iter.getLeaf();
			const T *vdb_tile = leaf->buffer().data();
			const math::Coord start = leaf->getNodeBoundingBox().getStart() - min_bound;

			for (int k = 0; k < TILE_SIZE; ++k) {
				for (int j = 0; j < TILE_SIZE; ++j) {
					for (int i = 0; i < TILE_SIZE; ++i) {
						int arr_index = compute_index(start.x() + i,
													  start.y() + j,
													  start.z() + k,
													  resolution.x,
													  resolution.y);
						int vdb_index = compute_index(k, j, i, TILE_SIZE, TILE_SIZE);
						copy(image + arr_index, vdb_tile + vdb_index);
					}
				}
			}
		}
	}
}

void openvdb_load_image(const string& filepath,
                        const string& grid_name,
                        const int channels,
                        float *image,
                        vector<int> *sparse_index)
{
	if(channels > 1) {
		return openvdb_load_image<openvdb::Vec3SGrid, openvdb::math::Vec3s>(
		    filepath, grid_name, channels, image, sparse_index);
	}
	else {
		return openvdb_load_image<openvdb::FloatGrid, float>(
		    filepath, grid_name, channels, image, sparse_index);
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
static void openvdb_build_mesh(VolumeMeshBuilder *builder,
                               void *v_grid,
                               const float threshold)
{
	using namespace openvdb;

	typename GridType::Ptr grid = *static_cast<typename GridType::Ptr*>(v_grid);

	for (typename GridType::TreeType::LeafCIter iter = grid->tree().cbeginLeaf(); iter; ++iter) {
		const typename GridType::TreeType::LeafNodeType *leaf = iter.getLeaf();
		const float *data = (float*)leaf->buffer().data();
		const math::Coord start = leaf->getNodeBoundingBox().getStart();

		for (int k = 0; k < TILE_SIZE; ++k) {
			for (int j = 0; j < TILE_SIZE; ++j) {
				for (int i = 0; i < TILE_SIZE; ++i) {
					int vdb_index = compute_index(k, j, i, TILE_SIZE, TILE_SIZE);
					if(data[vdb_index] >= threshold) {
						builder->add_node_with_padding(start.x() + i,
						                               start.y() + j,
						                               start.z() + k);
					}
				}
			}
		}
	}
}

void openvdb_build_mesh(VolumeMeshBuilder *builder,
                        void *v_grid,
                        const float threshold,
                        const bool is_vec)
{
	if(is_vec) {
		openvdb_build_mesh<openvdb::Vec3SGrid>(builder, v_grid, threshold);
	}
	else {
		openvdb_build_mesh<openvdb::FloatGrid>(builder, v_grid, threshold);
	}
}

CCL_NAMESPACE_END
