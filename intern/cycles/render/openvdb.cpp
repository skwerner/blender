
#include <openvdb/openvdb.h>
#include <openvdb/tools/GridTransformer.h>

#include "render/attribute.h"
#include "render/openvdb.h"

#include "util/util_logging.h"
#include "util/util_path.h"
#include "util/util_sparse_grid.h"

/* Functions that directly use the OpenVDB library throughout render. */

struct OpenVDBReader;

CCL_NAMESPACE_BEGIN

namespace {

/* Misc internal helper functions. */

bool operator >=(const openvdb::math::Vec3s &a, const float &b)
{
	return a.x() >= b || a.y() >= b || a.z() >= b;
}

void copy(float *des, const float *src)
{
	*des = *src;
}

void copy(float *des, const openvdb::math::Vec3s *src)
{
	*(des + 0) = src->x();
	*(des + 1) = src->y();
	*(des + 2) = src->z();
	*(des + 3) = 1.0f;
}

const int get_tile_index(const openvdb::math::Coord &start,
                         const openvdb::math::Coord &tiled_res)
{
	return compute_index(start.x() / TILE_SIZE,
	                     start.y() / TILE_SIZE,
	                     start.z() / TILE_SIZE,
	                     tiled_res.x(),
	                     tiled_res.y());
}

const int coord_product(const openvdb::math::Coord &c)
{
	return c.x() * c.y() * c.z();
}

const openvdb::math::Coord get_tile_dim(const openvdb::math::Coord &tile_min_bound,
                                        const openvdb::math::Coord &image_res,
                                        const openvdb::math::Coord &remainder)
{
	openvdb::math::Coord tile_dim;
	for(int i = 0; i < 3; ++i) {
		tile_dim[i] = (tile_min_bound[i] + TILE_SIZE > image_res[i]) ? remainder[i] : TILE_SIZE;
	}
	return tile_dim;
}

void expand_bbox(openvdb::io::File *vdb_file,
                 openvdb::math::CoordBBox *bbox,
                 AttributeStandard std)
{
	const char *grid_name = Attribute::standard_name(std);
	if(vdb_file->hasGrid(grid_name)) {
		bbox->expand(vdb_file->readGrid(grid_name)->evalActiveVoxelBoundingBox());
	}
}

void get_bounds(openvdb::io::File *vdb_file,
                openvdb::math::Coord &resolution,
                openvdb::math::Coord &min_bound)
{
	openvdb::math::CoordBBox bbox(openvdb::math::Coord(0, 0, 0),
	                              openvdb::math::Coord(0, 0, 0));

	/* Get the combined bounding box of all possible smoke grids in the file. */
	expand_bbox(vdb_file, &bbox, ATTR_STD_VOLUME_DENSITY);
	expand_bbox(vdb_file, &bbox, ATTR_STD_VOLUME_COLOR);
	expand_bbox(vdb_file, &bbox, ATTR_STD_VOLUME_FLAME);
	expand_bbox(vdb_file, &bbox, ATTR_STD_VOLUME_HEAT);
	expand_bbox(vdb_file, &bbox, ATTR_STD_VOLUME_TEMPERATURE);
	expand_bbox(vdb_file, &bbox, ATTR_STD_VOLUME_VELOCITY);

	resolution = bbox.dim();
	min_bound = bbox.getStart();
}

/* Simple range shift for grids with non-zero background values. May have
 * strange results depending on the grid. */
void shift_range(openvdb::Vec3SGrid::Ptr grid)
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

void shift_range(openvdb::FloatGrid::Ptr grid)
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

/* File and Grid IO */

void cleanup_file(openvdb::io::File *vdb_file)
{
	if(vdb_file) {
		vdb_file->close();
		delete vdb_file;
		vdb_file = NULL;
	}
}

openvdb::io::File *load_file(const string &filepath)
{
	if(!path_exists(filepath) || path_is_directory(filepath)) {
		return NULL;
	}

	openvdb::io::File *vdb_file = NULL;
	try {
		vdb_file = new openvdb::io::File(filepath);
		vdb_file->setCopyMaxBytes(0);
		vdb_file->open();
	}
	/* Mostly to catch exceptions related to Blosc not being supported. */
	catch (const openvdb::IoError &e) {
		std::cerr << e.what() << '\n';
		cleanup_file(vdb_file);
	}

	return vdb_file;
}

template<typename GridType>
bool get_grid(const string &filepath,
              const string &grid_name,
              typename GridType::Ptr &grid,
              openvdb::math::Coord &resolution,
              openvdb::math::Coord &min_bound)
{
	using namespace openvdb;

	io::File *vdb_file = load_file(filepath);

	if(!vdb_file) {
		return false;
	}
	if (!vdb_file->hasGrid(grid_name)) {
		cleanup_file(vdb_file);
		return false;
	}

	grid = gridPtrCast<GridType>(vdb_file->readGrid(grid_name));

	/* Verify that leaf dimensions match internal tile dimensions. */
	typename GridType::TreeType::LeafCIter iter = grid->tree().cbeginLeaf();
	if(iter) {
		const math::Coord dim = iter.getLeaf()->getNodeBoundingBox().dim();

		if(dim[0] != TILE_SIZE || dim[1] != TILE_SIZE || dim[2] != TILE_SIZE) {
			VLOG(1) << "Cannot load grid " << grid->getName() << " from "
			        << filepath << ", leaf dimensions are "
			        << dim[0] << "x" << dim[1] << "x" << dim[2];
			cleanup_file(vdb_file);
			return false;
		}
	}

	/* Need to account for external grids with a non-zero background value and
	 * voxels below background value. */
	shift_range(grid);

	/* Retrieve bound data. */
	get_bounds(vdb_file, resolution, min_bound);

	cleanup_file(vdb_file);
	return true;
}

/* Load OpenVDB grid to texture. */

template<typename GridType, typename T>
void image_load_preprocess(const string &filepath,
                           const string &grid_name,
                           const int channels,
                           const float threshold,
                           vector<int> *sparse_indexes,
                           int &sparse_size)
{
	using namespace openvdb;

	if(channels != 1 && channels != 4) {
		return;
	}

	typename GridType::Ptr grid = GridType::create();
	math::Coord resolution, min_bound, tiled_res, remainder;

	if(!get_grid<GridType>(filepath, grid_name, grid, resolution, min_bound)) {
		return;
	}

	for(int i = 0; i < 3; ++i) {
		tiled_res[i] = get_tile_res(resolution[i]);
		remainder[i] = resolution[i] % TILE_SIZE;
	}

	const int tile_count = coord_product(tiled_res);
	const int tile_pix_count = TILE_SIZE * TILE_SIZE * TILE_SIZE * channels;

	sparse_indexes->resize(tile_count, -1); /* 0 if active, -1 if inactive. */
	int voxel_count = 0;

	for (typename GridType::TreeType::LeafCIter iter = grid->tree().cbeginLeaf(); iter; ++iter) {
		const typename GridType::TreeType::LeafNodeType *leaf = iter.getLeaf();
		const T *data = leaf->buffer().data();

		for(int i = 0; i < tile_pix_count; ++i) {
			if(data[i] >= threshold) {
				const math::Coord tile_start = leaf->getNodeBoundingBox().getStart() - min_bound;
				sparse_indexes->at(get_tile_index(tile_start, tiled_res)) = 0;
				/* Calculate how many voxels are in this tile. */
				voxel_count += coord_product(get_tile_dim(tile_start, resolution, remainder));
				break;
			}
		}
	}

	/* Check memory savings. */
	const int sparse_mem_use = tile_count * sizeof(int) + voxel_count * channels * sizeof(float);
	const int dense_mem_use = coord_product(resolution) * channels * sizeof(float);

	VLOG(1) << grid_name << " memory usage: \n"
	        << "Dense: " << string_human_readable_size(dense_mem_use) << "\n"
	        << "Sparse: " << string_human_readable_size(sparse_mem_use) << "\n"
	        << "VDB Grid: " << string_human_readable_size(grid->memUsage());

	if(sparse_mem_use < dense_mem_use) {
		sparse_size = voxel_count * channels;
	}
	else {
		sparse_size = -1;
		sparse_indexes->resize(0);
	}
}

template<typename GridType, typename T>
void image_load_dense(const string &filepath,
                      const string &grid_name,
                      const int channels,
                      float *data)
{
	using namespace openvdb;

	if(channels != 1 && channels != 4) {
		return;
	}

	typename GridType::Ptr grid = GridType::create();
	math::Coord resolution, min_bound, tiled_res, remainder;

	if(!get_grid<GridType>(filepath, grid_name, grid, resolution, min_bound)) {
		return;
	}

	for(int i = 0; i < 3; ++i) {
		tiled_res[i] = get_tile_res(resolution[i]);
		remainder[i] = resolution[i] % TILE_SIZE;
	}

	memset(data, 0, coord_product(resolution) * channels * sizeof(float));

	for (typename GridType::TreeType::LeafCIter iter = grid->tree().cbeginLeaf(); iter; ++iter) {
		const typename GridType::TreeType::LeafNodeType *leaf = iter.getLeaf();
		const T *leaf_data = leaf->buffer().data();
		const math::Coord tile_start = leaf->getNodeBoundingBox().getStart() - min_bound;
		const math::Coord tile_dim = get_tile_dim(tile_start, resolution, remainder);

		for (int k = 0; k < tile_dim.z(); ++k) {
			for (int j = 0; j < tile_dim.y(); ++j) {
				for (int i = 0; i < tile_dim.x(); ++i) {
					int data_index = compute_index(tile_start.x() + i,
												   tile_start.y() + j,
												   tile_start.z() + k,
												   resolution.x(),
					                               resolution.y());
					/* Index computation by coordinates is reversed in VDB grids. */
					int leaf_index = compute_index(k, j, i, tile_dim.z(), tile_dim.y());
					copy(data + data_index, leaf_data + leaf_index);
				}
			}
		}
	}
}

template<typename GridType, typename T>
void image_load_sparse(const string &filepath,
                       const string &grid_name,
                       const int channels,
                       float *data,
                       vector<int> *sparse_indexes)
{
	using namespace openvdb;

	if(channels != 1 && channels != 4) {
		return;
	}

	typename GridType::Ptr grid = GridType::create();
	math::Coord resolution, min_bound, tiled_res, remainder;

	if(!get_grid<GridType>(filepath, grid_name, grid, resolution, min_bound)) {
		return;
	}

	for(int i = 0; i < 3; ++i) {
		tiled_res[i] = get_tile_res(resolution[i]);
		remainder[i] = resolution[i] % TILE_SIZE;
	}

	int voxel_count = 0;

	for (typename GridType::TreeType::LeafCIter iter = grid->tree().cbeginLeaf(); iter; ++iter) {
		const typename GridType::TreeType::LeafNodeType *leaf = iter.getLeaf();

		const math::Coord tile_start = leaf->getNodeBoundingBox().getStart() - min_bound;
		int tile_index = get_tile_index(tile_start, tiled_res);
		if(sparse_indexes->at(tile_index) == -1) {
			continue;
		}

		sparse_indexes->at(tile_index) = voxel_count / channels;
		const math::Coord tile_dim = get_tile_dim(tile_start, resolution, remainder);
		const T *leaf_tile = leaf->buffer().data();
		float *data_tile = data + voxel_count;

		for(int k = 0; k < tile_dim.z(); ++k) {
			for(int j = 0; j < tile_dim.y(); ++j) {
				for(int i = 0; i < tile_dim.x(); ++i, ++voxel_count) {
					int data_index = compute_index(i, j, k, tile_dim.x(), tile_dim.y());
					/* Index computation by coordinates is reversed in VDB grids. */
					int leaf_index = compute_index(k, j, i, TILE_SIZE, TILE_SIZE);
					copy(data_tile + data_index, leaf_tile + leaf_index);
				}
			}
		}
	}
}

} /* namespace */

/* Initializer, must be called if OpenVDB will be used. */
void openvdb_initialize()
{
	openvdb::initialize();
}

bool openvdb_has_grid(const string& filepath, const string& grid_name)
{
	if(grid_name.empty()) {
		return false;
	}
	openvdb::io::File *vdb_file = load_file(filepath);
	if(!vdb_file) {
		return false;
	}
	bool has_grid = vdb_file->hasGrid(grid_name);
	cleanup_file(vdb_file);
	return has_grid;
}

int3 openvdb_get_resolution(const string& filepath)
{
	openvdb::io::File *vdb_file = load_file(filepath);
	if(!vdb_file) {
		return make_int3(-1, -1, -1);
	}
	openvdb::math::Coord resolution, min_bound;
	get_bounds(vdb_file, resolution, min_bound);
	cleanup_file(vdb_file);
	return make_int3(resolution.x(), resolution.y(), resolution.z());
}

void openvdb_load_preprocess(const string& filepath,
                             const string& grid_name,
                             const int channels,
                             const float threshold,
                             vector<int> *sparse_indexes,
                             int &sparse_size)
{
	if(channels > 1) {
		image_load_preprocess<openvdb::Vec3SGrid, openvdb::math::Vec3s>(
		            filepath, grid_name, channels, threshold, sparse_indexes, sparse_size);
	}
	else {
		image_load_preprocess<openvdb::FloatGrid, float>(
		            filepath, grid_name, channels, threshold, sparse_indexes, sparse_size);
	}
}

void openvdb_load_image(const string& filepath,
                        const string& grid_name,
                        const int channels,
                        float *image,
                        vector<int> *sparse_indexes)
{
	bool make_sparse = false;
	if(sparse_indexes) {
		if(sparse_indexes->size() > 0) {
			make_sparse = true;
		}
	}

	if(make_sparse) {
		if(channels > 1) {
			image_load_sparse<openvdb::Vec3SGrid, openvdb::math::Vec3s>(
			            filepath, grid_name, channels, image, sparse_indexes);
		}
		else {
			image_load_sparse<openvdb::FloatGrid, float>(
			            filepath, grid_name, channels, image, sparse_indexes);
		}
	}
	else {
		if(channels > 1) {
			image_load_dense<openvdb::Vec3SGrid, openvdb::math::Vec3s>(
			            filepath, grid_name, channels, image);
		}
		else {
			image_load_dense<openvdb::FloatGrid, float>(
			            filepath, grid_name, channels, image);
		}
	}
}

CCL_NAMESPACE_END
