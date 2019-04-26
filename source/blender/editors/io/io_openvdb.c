
#include "DNA_object_types.h"
#include "DNA_smoke_types.h"
#include "DNA_space_types.h"

#include "BKE_context.h"
#include "BKE_global.h"
#include "BKE_main.h"
#include "BKE_modifier.h"
#include "BKE_object.h"
#include "BKE_pointcache.h"
#include "BKE_report.h"
#include "BKE_smoke.h"

#include "BLI_fileops.h"
#include "BLI_listbase.h"
#include "BLI_path_util.h"
#include "BLI_string.h"

#include "RNA_access.h"

#include "WM_api.h"
#include "WM_types.h"

#include "DEG_depsgraph.h"
#include "DEG_depsgraph_build.h"

#include "io_openvdb.h"
#include "openvdb_capi.h"

static int estimate_sample_level(const char *filepath)
{
	/* Estimate a sample level to avoid crashing when loading large volumes on
	 * low-memory computers. */

	int res[3], raw_res[3];
	int sample_level = 0, result = 0;

	struct OpenVDBReader *reader = OpenVDBReader_create();
	OpenVDBReader_open(reader, filepath);
	result = OpenVDBReader_get_bounds(reader, NULL, NULL, raw_res, NULL, NULL, NULL);
	OpenVDBReader_free(reader);

	if (!result) {
		return 1;
	}

	do {
		++sample_level;
		for(int i = 0; i < 3; ++i) {
			res[i] = raw_res[i] / sample_level + (raw_res[i] % sample_level != 0);
		}
	} while(res[0] * res[1] * res[2] > 40000000);

	return sample_level;
}

static void wm_openvdb_import_draw(bContext *UNUSED(C), wmOperator *op)
{
	PointerRNA ptr;

	RNA_pointer_create(NULL, op->type->srna, op->properties, &ptr);
}

static int wm_openvdb_import_exec(bContext *C, wmOperator *op)
{
	if (!RNA_struct_property_is_set(op->ptr, "filepath")) {
		BKE_report(op->reports, RPT_ERROR, "No filename given");
		return OPERATOR_CANCELLED;
	}

	char filepath[FILE_MAX];
	char filename[64];

	RNA_string_get(op->ptr, "filepath", filepath);
	BLI_split_file_part(filepath, filename, 64);

	Main *bmain = CTX_data_main(C);
	Scene *scene = CTX_data_scene(C);
	ViewLayer *view_layer = CTX_data_view_layer(C);
	Object *ob = BKE_object_add(bmain, scene, view_layer, OB_MESH, filename);

	BLI_path_abs(filepath, ID_BLEND_PATH(G.main, (ID *)ob));
	if (!BLI_exists(filepath)) {
		return OPERATOR_CANCELLED;
	}

	ModifierData *md = modifier_new(eModifierType_Smoke);
	BLI_addtail(&ob->modifiers, md);

	SmokeModifierData *smd = (SmokeModifierData *)md;
	smd->type = MOD_SMOKE_TYPE_DOMAIN;
	smokeModifier_createType(smd);

	smd->domain->flags |= MOD_SMOKE_FILE_LOAD | MOD_SMOKE_ADAPTIVE_DOMAIN;
	smd->domain->cache_file_format = PTCACHE_FILE_OPENVDB_EXTERN;
	smd->domain->multi_import = 0;
	smd->domain->sample_level = estimate_sample_level(filepath);
	BLI_strncpy(smd->domain->volume_filepath, filepath, sizeof(filepath));

	DEG_id_tag_update(&scene->id, ID_RECALC_BASE_FLAGS);
	DEG_relations_tag_update(bmain);

	return OPERATOR_FINISHED;
}

void WM_OT_openvdb_import(wmOperatorType *ot)
{
	ot->name = "Import OpenVDB";
	ot->description = "Load an external OpenVDB smoke file";
	ot->idname = "WM_OT_openvdb_import";

	ot->invoke = WM_operator_filesel;
	ot->exec = wm_openvdb_import_exec;
	ot->poll = WM_operator_winactive;
	ot->ui = wm_openvdb_import_draw;

	WM_operator_properties_filesel(ot, FILE_TYPE_FOLDER | FILE_TYPE_VOLUME,
	                               FILE_BLENDER, FILE_OPENFILE, WM_FILESEL_FILEPATH,
	                               FILE_DEFAULTDISPLAY, FILE_SORT_ALPHA);
}
