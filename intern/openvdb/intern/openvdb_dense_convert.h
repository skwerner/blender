/*
 * ***** BEGIN GPL LICENSE BLOCK *****
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 *
 * The Original Code is Copyright (C) 2015 Blender Foundation.
 * All rights reserved.
 *
 * The Original Code is: all of this file.
 *
 * Contributor(s): Kevin Dietrich
 *
 * ***** END GPL LICENSE BLOCK *****
 */

#ifndef __OPENVDB_DENSE_CONVERT_H__
#define __OPENVDB_DENSE_CONVERT_H__

#include "openvdb_reader.h"
#include "openvdb_writer.h"

#include <openvdb/tools/Clip.h>
#include <openvdb/tools/Dense.h>

#include <cstdio>

namespace internal {

static const int CUBE_SIZE = 8;

static const int compute_index(int x, int y, int z, int width, int height)
{
	return x + width * (y + z * height);
}

static void copy(float *des, const float *src)
{
	*des = *src;
}

static void copy(unsigned char *des, const int *src)
{
	*des = *src;
}

static void copy(float *des, const openvdb::math::Vec3s *src)
{
	*(des + 0) = src->x();
	*(des + 1) = src->y();
	*(des + 2) = src->z();
	*(des + 3) = 1.0f;
}

/* Verify that the name does not correspond to the old format, in which case we
 * need to replace the '_low' ending with ' low'. See T53802. */
openvdb::Name do_name_versionning(const openvdb::Name &name);

openvdb::Mat4R convertMatrix(const float mat[4][4]);

template <typename GridType, typename T>
typename GridType::Ptr OpenVDB_export_grid(
        OpenVDBWriter *writer,
        const openvdb::Name &name,
        const T *data,
        const int res[3],
		float fluid_mat[4][4],
		const float clipping,
        const openvdb::FloatGrid *mask)
{
	using namespace openvdb;

	math::CoordBBox bbox(Coord(0), Coord(res[0] - 1, res[1] - 1, res[2] - 1));

	typename GridType::Ptr grid = GridType::create(T(0));

	tools::Dense<const T, openvdb::tools::LayoutXYZ> dense_grid(bbox, data);
	tools::copyFromDense(dense_grid, grid->tree(), static_cast<T>(clipping));

	if(fluid_mat) {
		Mat4R mat = convertMatrix(fluid_mat);
		math::Transform::Ptr transform = math::Transform::createLinearTransform(mat);
		grid->setTransform(transform);
	}

	/* Avoid clipping against an empty grid. */
	if (mask && !mask->tree().empty()) {
		grid = tools::clip(*grid, *mask);
	}

	grid->setName(name);
	grid->setIsInWorldSpace(false);
	grid->setVectorType(openvdb::VEC_INVARIANT);

	if(writer) {
		writer->insert(grid);
	}

	return grid;
}

openvdb::GridBase::Ptr OpenVDB_export_vector_grid(OpenVDBWriter *writer,
		const openvdb::Name &name,
		const float *data_x, const float *data_y, const float *data_z,
		const int res[3],
		float fluid_mat[4][4],
		openvdb::VecType vec_type,
		const bool is_color,
		const float clipping,
		const openvdb::FloatGrid *mask);

template <typename GridType, typename GridDataType, typename DataType>
void OpenVDB_import_grid(
        OpenVDBReader *reader,
        const char *name,
        typename GridType::Ptr *grid_ptr,
        DataType **data,
        const int res[3],
        const int min_bound[3],
		const int channels,
		const bool weave)
{
	using namespace openvdb;

	/* Weave pattern: xyzxyz...xyz
	 * Normal pattern: xxx...yyy....zzz
	 * Normal data is an array of pointers while weave data is a pointer to a
	 * flat T array. */

	if(weave) {
		memset(data[0], 0, res[0] * res[1] * res[2] * channels * sizeof(DataType));
	}
	else {
		for(int c = 0; c < channels; ++c) {
			memset(data[c], 0, res[0] * res[1] * res[2] * sizeof(DataType));
		}
	}

	typename GridType::Ptr grid;

	if(reader && name) {
		openvdb::Name temp_name(name);

		if (!reader->hasGrid(temp_name)) {
			temp_name = do_name_versionning(temp_name);

			if (!reader->hasGrid(temp_name)) {
				std::fprintf(stderr, "OpenVDB grid %s not found in file!\n", temp_name.c_str());
				return;
			}
		}

		grid = gridPtrCast<GridType>(reader->getGrid(temp_name));
	}
	else if(grid_ptr) {
		grid = *grid_ptr;
	}
	else {
		return;
	}

	const int remainder[3] = {res[0] % CUBE_SIZE, res[1] % CUBE_SIZE, res[2] % CUBE_SIZE};
	const math::Coord min(min_bound[0], min_bound[1], min_bound[2]);

	for (typename GridType::TreeType::LeafCIter iter = grid->tree().cbeginLeaf(); iter; ++iter) {
		const typename GridType::TreeType::LeafNodeType *leaf = iter.getLeaf();
		const GridDataType *leaf_data = leaf->buffer().data();

		const math::Coord start = leaf->getNodeBoundingBox().getStart() - min;
		const int tile_width = (start.x() + CUBE_SIZE > res[0]) ? remainder[0] : CUBE_SIZE;
		const int tile_height = (start.y() + CUBE_SIZE > res[1]) ? remainder[1] : CUBE_SIZE;
		const int tile_depth = (start.z() + CUBE_SIZE > res[2]) ? remainder[2] : CUBE_SIZE;

		for (int k = 0; k < tile_depth; ++k) {
			for (int j = 0; j < tile_height; ++j) {
				for (int i = 0; i < tile_width; ++i) {
					int data_index = compute_index(start.x() + i,
												   start.y() + j,
												   start.z() + k,
												   res[0], res[1]);
					/* Index computation by coordinates is reversed in VDB grids. */
					int leaf_index = compute_index(k, j, i, tile_depth, tile_height);

					if(weave) {
						copy(data[0] + data_index, leaf_data + leaf_index);
					}
					else {
						DataType temp_value[4]; /* We don't expect channels > 4 */
						copy(temp_value, leaf_data + leaf_index);
						for(int c = 0; c < channels; ++c) {
							data[c][data_index] = temp_value[c];
						}
					}
				}
			}
		}
	}
}

}  /* namespace internal */

#endif /* __OPENVDB_DENSE_CONVERT_H__ */
