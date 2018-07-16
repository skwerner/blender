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

#ifndef __OPENVDB_CAPI_H__
#define __OPENVDB_CAPI_H__

#ifdef __cplusplus
extern "C" {
#endif

struct OpenVDBReader;
struct OpenVDBWriter;
struct OpenVDBFloatGrid;
struct OpenVDBIntGrid;
struct OpenVDBVectorGrid;

int OpenVDB_getVersionHex(void);

enum {
	VEC_INVARIANT = 0,
	VEC_COVARIANT = 1,
	VEC_COVARIANT_NORMALIZE = 2,
	VEC_CONTRAVARIANT_RELATIVE = 3,
	VEC_CONTRAVARIANT_ABSOLUTE = 4,
};

enum {
	VDB_SMOKE_DENSITY,
	VDB_SMOKE_COLOR,
	VDB_SMOKE_FLAME,
	VDB_SMOKE_HEAT,
	VDB_SMOKE_TEMPERATURE,
	VDB_SMOKE_VELOCITY,

	VDB_SMOKE_GRID_NUM
};

const char *vdb_grid_name(const int grid);

struct OpenVDBFloatGrid *OpenVDB_export_grid_fl(
        struct OpenVDBWriter *writer,
        const char *name, float *data,
		const int res[3], float matrix[4][4], const float clipping,
        struct OpenVDBFloatGrid *mask);

struct OpenVDBIntGrid *OpenVDB_export_grid_ch(struct OpenVDBWriter *writer,
		const char *name, unsigned char *data,
		const int res[3], float matrix[4][4], const float clipping,
		struct OpenVDBFloatGrid *mask);

struct OpenVDBVectorGrid *OpenVDB_export_grid_vec(struct OpenVDBWriter *writer,
		const char *name,
		const float *data_x, const float *data_y, const float *data_z,
		const int res[3], float matrix[4][4], short vec_type, const float clipping,
		const bool is_color,
		struct OpenVDBFloatGrid *mask);

void OpenVDB_import_grid_fl(
        struct OpenVDBReader *reader,
        const char *name, float **data,
        const int res[3]);

void OpenVDB_import_grid_ch(
        struct OpenVDBReader *reader,
        const char *name, unsigned char **data,
        const int res[3]);

void OpenVDB_import_grid_vec(
        struct OpenVDBReader *reader,
        const char *name,
        float **data_x, float **data_y, float **data_z,
        const int res[3]);

struct OpenVDBWriter *OpenVDBWriter_create(void);
void OpenVDBWriter_free(struct OpenVDBWriter *writer);
void OpenVDBWriter_set_flags(struct OpenVDBWriter *writer, const int flag, const bool half);
void OpenVDBWriter_add_meta_fl(struct OpenVDBWriter *writer, const char *name, const float value);
void OpenVDBWriter_add_meta_int(struct OpenVDBWriter *writer, const char *name, const int value);
void OpenVDBWriter_add_meta_v3(struct OpenVDBWriter *writer, const char *name, const float value[3]);
void OpenVDBWriter_add_meta_v3_int(struct OpenVDBWriter *writer, const char *name, const int value[3]);
void OpenVDBWriter_add_meta_mat4(struct OpenVDBWriter *writer, const char *name, float value[4][4]);
void OpenVDBWriter_write(struct OpenVDBWriter *writer, const char *filename);

struct OpenVDBReader *OpenVDBReader_create(void);
void OpenVDBReader_free(struct OpenVDBReader *reader);
void OpenVDBReader_open(struct OpenVDBReader *reader, const char *filename);
bool OpenVDBReader_has_grid(struct OpenVDBReader *reader, const char *name);
bool OpenVDBReader_has_smoke_grid(struct OpenVDBReader *reader, const int grid);
void OpenVDBReader_get_meta_fl(struct OpenVDBReader *reader, const char *name, float *value);
void OpenVDBReader_get_meta_int(struct OpenVDBReader *reader, const char *name, int *value);
void OpenVDBReader_get_meta_v3(struct OpenVDBReader *reader, const char *name, float value[3]);
void OpenVDBReader_get_meta_v3_int(struct OpenVDBReader *reader, const char *name, int value[3]);
void OpenVDBReader_get_meta_mat4(struct OpenVDBReader *reader, const char *name, float value[4][4]);

bool OpenVDBReader_get_bounds(struct OpenVDBReader *reader,
                              int res_min[3], int res_max[3], int res[3],
                              float bbox_min[3], float bbox_max[3], float voxel_size[3]);

#ifdef __cplusplus
}
#endif

#endif /* __OPENVDB_CAPI_H__ */
