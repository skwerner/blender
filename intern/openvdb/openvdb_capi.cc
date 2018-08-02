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
 * Contributor(s): Kevin Dietrich
 *
 * ***** END GPL LICENSE BLOCK *****
 */

#include "openvdb_capi.h"
#include "openvdb_dense_convert.h"
#include "openvdb_util.h"

struct OpenVDBFloatGrid { int unused; };
struct OpenVDBIntGrid { int unused; };
struct OpenVDBVectorGrid { int unused; };

int OpenVDB_getVersionHex()
{
	return openvdb::OPENVDB_LIBRARY_VERSION;
}

const char *vdb_grid_name(const int grid)
{
	switch(grid) {
		case VDB_SMOKE_DENSITY:
			return "density";
		case VDB_SMOKE_COLOR:
			return "color";
		case VDB_SMOKE_FLAME:
			return "flame";
		case VDB_SMOKE_HEAT:
			return "heat";
		case VDB_SMOKE_TEMPERATURE:
			return "temperature";
		case VDB_SMOKE_VELOCITY:
			return "velocity";
		default:
			return "";
	}
}

OpenVDBFloatGrid *OpenVDB_export_grid_fl(
        OpenVDBWriter *writer,
        const char *name, float *data,
		const int res[3], float matrix[4][4], const float clipping,
        OpenVDBFloatGrid *mask)
{
	Timer(__func__);

	using openvdb::FloatGrid;

	FloatGrid *mask_grid = reinterpret_cast<FloatGrid *>(mask);
	FloatGrid::Ptr grid = internal::OpenVDB_export_grid<FloatGrid>(
	        writer,
	        name,
	        data,
	        res,
	        matrix,
			clipping,
	        mask_grid);

	return reinterpret_cast<OpenVDBFloatGrid *>(grid.get());
}

OpenVDBIntGrid *OpenVDB_export_grid_ch(
        OpenVDBWriter *writer,
        const char *name, unsigned char *data,
		const int res[3], float matrix[4][4], const float clipping,
        OpenVDBFloatGrid *mask)
{
	Timer(__func__);

	using openvdb::FloatGrid;
	using openvdb::Int32Grid;

	FloatGrid *mask_grid = reinterpret_cast<FloatGrid *>(mask);
	Int32Grid::Ptr grid = internal::OpenVDB_export_grid<Int32Grid>(
	        writer,
	        name,
	        data,
	        res,
	        matrix,
			clipping,
	        mask_grid);

	return reinterpret_cast<OpenVDBIntGrid *>(grid.get());
}

OpenVDBVectorGrid *OpenVDB_export_grid_vec(struct OpenVDBWriter *writer,
		const char *name,
		const float *data_x, const float *data_y, const float *data_z,
		const int res[3], float matrix[4][4], short vec_type, const float clipping,
		const bool is_color, OpenVDBFloatGrid *mask)
{
	Timer(__func__);

	using openvdb::GridBase;
	using openvdb::FloatGrid;
	using openvdb::VecType;

	FloatGrid *mask_grid = reinterpret_cast<FloatGrid *>(mask);
	typename GridBase::Ptr grid = internal::OpenVDB_export_vector_grid(
	        writer,
	        name,
	        data_x,
	        data_y,
	        data_z,
	        res,
	        matrix,
	        static_cast<VecType>(vec_type),
	        is_color,
			clipping,
	        mask_grid);

	return reinterpret_cast<OpenVDBVectorGrid *>(grid.get());
}

void OpenVDB_import_grid_fl(
        OpenVDBReader *reader,
        const char *name, float *data,
        const int res[3], const int min_bound[3])
{
	Timer(__func__);

	internal::OpenVDB_import_grid<openvdb::FloatGrid, float, float>(
	            reader, name, NULL, &data, res, min_bound, 1, false);
}

void OpenVDB_import_grid_ch(
        OpenVDBReader *reader,
        const char *name, unsigned char *data,
        const int res[3], const int min_bound[3])
{
	internal::OpenVDB_import_grid<openvdb::Int32Grid, int, unsigned char>(
	            reader, name, NULL, &data, res, min_bound, 1, false);
}

void OpenVDB_import_grid_vec(
        struct OpenVDBReader *reader,
        const char *name,
        float *data_x, float *data_y, float *data_z,
        const int res[3], const int min_bound[3])
{
	Timer(__func__);

	float *data[3] = {data_x, data_y, data_z};

	internal::OpenVDB_import_grid<openvdb::Vec3SGrid, openvdb::math::Vec3s, float>(
	            reader, name, NULL, data, res, min_bound, 3, false);
}

OpenVDBWriter *OpenVDBWriter_create()
{
	return new OpenVDBWriter();
}

void OpenVDBWriter_free(OpenVDBWriter *writer)
{
	delete writer;
}

void OpenVDBWriter_set_flags(OpenVDBWriter *writer, const int flag, const bool half)
{
	int compression_flags = openvdb::io::COMPRESS_ACTIVE_MASK;

#ifdef WITH_OPENVDB_BLOSC
	if (flag == 0) {
		compression_flags |= openvdb::io::COMPRESS_BLOSC;
	}
	else
#endif
	if (flag == 1) {
		compression_flags |= openvdb::io::COMPRESS_ZIP;
	}
	else {
		compression_flags = openvdb::io::COMPRESS_NONE;
	}

	writer->setFlags(compression_flags, half);
}

void OpenVDBWriter_add_meta_fl(OpenVDBWriter *writer, const char *name, const float value)
{
	writer->insertFloatMeta(name, value);
}

void OpenVDBWriter_add_meta_int(OpenVDBWriter *writer, const char *name, const int value)
{
	writer->insertIntMeta(name, value);
}

void OpenVDBWriter_add_meta_v3(OpenVDBWriter *writer, const char *name, const float value[3])
{
	writer->insertVec3sMeta(name, value);
}

void OpenVDBWriter_add_meta_v3_int(OpenVDBWriter *writer, const char *name, const int value[3])
{
	writer->insertVec3IMeta(name, value);
}

void OpenVDBWriter_add_meta_mat4(OpenVDBWriter *writer, const char *name, float value[4][4])
{
	writer->insertMat4sMeta(name, value);
}

void OpenVDBWriter_write(OpenVDBWriter *writer, const char *filename)
{
	writer->write(filename);
}

OpenVDBReader *OpenVDBReader_create()
{
	return new OpenVDBReader();
}

void OpenVDBReader_free(OpenVDBReader *reader)
{
	delete reader;
}

void OpenVDBReader_open(OpenVDBReader *reader, const char *filename)
{
	reader->open(filename);
}

bool OpenVDBReader_has_grid(OpenVDBReader *reader, const char *name)
{
	return reader->hasGrid(name);
}

bool OpenVDBReader_has_smoke_grid(OpenVDBReader *reader, const int grid)
{
	return reader->hasGrid(vdb_grid_name(grid));
}

void OpenVDBReader_get_meta_fl(OpenVDBReader *reader, const char *name, float *value)
{
	reader->floatMeta(name, *value);
}

void OpenVDBReader_get_meta_int(OpenVDBReader *reader, const char *name, int *value)
{
	reader->intMeta(name, *value);
}

void OpenVDBReader_get_meta_v3(OpenVDBReader *reader, const char *name, float value[3])
{
	reader->vec3sMeta(name, value);
}

void OpenVDBReader_get_meta_v3_int(OpenVDBReader *reader, const char *name, int value[3])
{
	reader->vec3IMeta(name, value);
}

void OpenVDBReader_get_meta_mat4(OpenVDBReader *reader, const char *name, float value[4][4])
{
	reader->mat4sMeta(name, value);
}

static bool OpenVDBReader_get_bbox(struct OpenVDBReader *reader,
                                   openvdb::math::CoordBBox *bbox,
                                   openvdb::BBoxd *bbox_world,
                                   openvdb::math::Vec3d *v_size)
{
	openvdb::math::Transform::Ptr tfm;
	bool is_valid = true; /* file is valid if all grids have the same tranform */
	bool has_smoke_grid = false;

	for(int type = 0; type < VDB_SMOKE_GRID_NUM; type++) {
		const char *grid_name = vdb_grid_name(type);

		if(reader->hasGrid(grid_name)) {
			has_smoke_grid = true;
			bbox->expand(reader->getGridBounds(grid_name));

			if(!tfm) {
				tfm = reader->getGridTranform(grid_name);
			}
			else {
				is_valid = (*tfm == *(reader->getGridTranform(grid_name)));
			}
		}
	}

	if(has_smoke_grid) {
		*bbox_world = tfm->indexToWorld(*bbox);
		*v_size = tfm->voxelSize();
	}

	return is_valid;
}

bool OpenVDBReader_get_bounds(struct OpenVDBReader *reader,
                              int res_min[3], int res_max[3], int res[3],
                              float bbox_min[3], float bbox_max[3], float voxel_size[3])
{
	using namespace openvdb;

	math::Coord coord(0, 0, 0);
	math::CoordBBox bbox(coord, coord);
	math::Vec3d vec3d(0, 0, 0);
	BBoxd bbox_world(vec3d, vec3d);

	bool is_valid = OpenVDBReader_get_bbox(reader, &bbox, &bbox_world, &vec3d);

	if(voxel_size) {
		voxel_size[0] = vec3d[0];
		voxel_size[1] = vec3d[1];
		voxel_size[2] = vec3d[2];
	}
	if(bbox_min) {
		vec3d = bbox_world.min();
		bbox_min[0] = vec3d[0];
		bbox_min[1] = vec3d[1];
		bbox_min[2] = vec3d[2];
	}
	if(bbox_max) {
		vec3d = bbox_world.max();
		bbox_max[0] = vec3d[0];
		bbox_max[1] = vec3d[1];
		bbox_max[2] = vec3d[2];
	}
	if(res_min) {
		coord = bbox.getStart();
		res_min[0] = coord[0];
		res_min[1] = coord[1];
		res_min[2] = coord[2];
	}
	if(res_max) {
		coord = bbox.getEnd();
		res_max[0] = coord[0];
		res_max[1] = coord[1];
		res_max[2] = coord[2];
	}
	if(res) {
		coord = bbox.dim();
		res[0] = coord[0];
		res[1] = coord[1];
		res[2] = coord[2];
	}

	return is_valid;
}
