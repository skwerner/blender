
#include <openvdb/openvdb.h>
#include <openvdb/tools/GridTransformer.h>

#include "render/openvdb.h"

#include "intern/openvdb_reader.h"
#include "intern/openvdb_dense_convert.h"
#include "openvdb_capi.h"

#include "util/util_logging.h"
#include "util/util_path.h"
#include "util/util_sparse_grid.h"

/* Functions that directly use the OpenVDB library throughout render. */

struct OpenVDBReader;

CCL_NAMESPACE_BEGIN

/* Misc internal helper functions. */

static bool operator >=(const openvdb::math::Vec3s &a, const float &b)
{
	return a.x() >= b || a.y() >= b || a.z() >= b;
}

static const int tile_index(openvdb::math::Coord start, const int tiled_res[3])
{
	return compute_index(start.x() / TILE_SIZE, start.y() / TILE_SIZE,
	                     start.z() / TILE_SIZE, tiled_res[0], tiled_res[1]);
}

/* Simple range shift for grids with non-zero background values. May have
 * strange results depending on the grid. */
static void shift_range(openvdb::Vec3SGrid::Ptr grid)
{
	using namespace openvdb;
	const math::Vec3s background_value = grid->background();
	if(background_value != math::Vec3s(0.0f, 0.0f, 0.0f)) {
		for (Vec3SGrid::ValueOnIter iter = grid->beginValueOn(); iter; ++iter) {
		    iter.setValue(iter.getValue() - background_value);
		}
		tools::changeBackground(grid->tree(), math::Vec3s(0.0f, 0.0f, 0.0f));
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
                     int resolution[3],
                     int min_bound[3])
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

	/* Retrieve bound data. */
	OpenVDBReader_get_bounds(reader, min_bound, NULL, resolution, NULL, NULL, NULL);

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
	int res[3], min_bound[3];

	if(!get_grid<GridType>(filepath, grid_name, grid, res, min_bound) ||
	   !(channels == 4 || channels == 1))
	{
		return;
	}

	const int tiled_res[3] = {get_tile_res(res[0]),
	                          get_tile_res(res[1]),
	                          get_tile_res(res[2])};
	const int remainder[3] = {res[0] % TILE_SIZE,
	                          res[1] % TILE_SIZE,
	                          res[2] % TILE_SIZE};
	const int tile_count = tiled_res[0] * tiled_res[1] * tiled_res[2];
	const int tile_pix_count = TILE_SIZE * TILE_SIZE * TILE_SIZE * channels;
	const math::Coord min(min_bound[0], min_bound[1], min_bound[2]);

	sparse_index->resize(tile_count, -1); /* 0 if active, -1 if inactive. */
	int voxel_count = 0;

	for (typename GridType::TreeType::LeafCIter iter = grid->tree().cbeginLeaf(); iter; ++iter) {
		const typename GridType::TreeType::LeafNodeType *leaf = iter.getLeaf();
		const T *data = leaf->buffer().data();
		const math::Coord start = leaf->getNodeBoundingBox().getStart() - min;

		for(int i = 0; i < tile_pix_count; ++i) {
			if(data[i] >= threshold) {
				sparse_index->at(tile_index(start, tiled_res)) = 0;

				/* Calculate how many voxels are in this tile. */
				const int tile_width = (start.x() + TILE_SIZE > res[0]) ? remainder[0] : TILE_SIZE;
				const int tile_height = (start.y() + TILE_SIZE > res[1]) ? remainder[1] : TILE_SIZE;
				const int tile_depth = (start.z() + TILE_SIZE > res[2]) ? remainder[2] : TILE_SIZE;

				voxel_count += tile_width * tile_height * tile_depth;
				break;
			}
		}
	}

	/* Check memory savings. */
	const int sparse_mem_use = tile_count * sizeof(int) + voxel_count * channels * sizeof(float);
	const int dense_mem_use = res[0] * res[1] * res[2] * channels * sizeof(float);

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
                               float *data,
                               vector<int> *sparse_index)
{
	using namespace openvdb;

	typename GridType::Ptr grid = GridType::create();
	int res[3], min_bound[3];

	if(!get_grid<GridType>(filepath, grid_name, grid, res, min_bound) ||
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
		const int tiled_res[3] = {get_tile_res(res[0]),
		                          get_tile_res(res[1]),
		                          get_tile_res(res[2])};
		const int remainder[3] = {res[0] % TILE_SIZE,
		                          res[1] % TILE_SIZE,
		                          res[2] % TILE_SIZE};
		const math::Coord min(min_bound[0], min_bound[1], min_bound[2]);
		int start_index = 0;

		for (typename GridType::TreeType::LeafCIter iter = grid->tree().cbeginLeaf(); iter; ++iter) {
			const typename GridType::TreeType::LeafNodeType *leaf = iter.getLeaf();

			const math::Coord start = leaf->getNodeBoundingBox().getStart() - min;
			int tile = tile_index(start, tiled_res);
			if(sparse_index->at(tile) == -1) {
				continue;
			}
			sparse_index->at(tile) = start_index / channels;

			const T *leaf_tile = leaf->buffer().data();
			float *data_tile = data + start_index;

			const int tile_width = (start.x() + TILE_SIZE > res[0]) ? remainder[0] : TILE_SIZE;
			const int tile_height = (start.y() + TILE_SIZE > res[1]) ? remainder[1] : TILE_SIZE;
			const int tile_depth = (start.z() + TILE_SIZE > res[2]) ? remainder[2] : TILE_SIZE;

			for(int k = 0; k < tile_depth; ++k) {
				for(int j = 0; j < tile_height; ++j) {
					for(int i = 0; i < tile_width; ++i) {
						int data_index = compute_index(i, j, k, tile_width, tile_height);
						/* Index computation by coordinates is reversed in VDB grids. */
						int leaf_index = compute_index(k, j, i, TILE_SIZE, TILE_SIZE);
						internal::copy(data_tile + data_index, leaf_tile + leaf_index);
					}
				}
			}

			start_index += tile_width * tile_height * tile_depth;
		}
	}
	else {
		/* Load VDB as dense image. */
		internal::OpenVDB_import_grid<GridType, T, float>(
		            NULL, NULL, &grid, &data, res, min_bound, channels, true);
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

CCL_NAMESPACE_END
