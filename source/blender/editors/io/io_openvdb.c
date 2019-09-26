#ifdef WITH_OPENVDB

/* needed for directory lookup */
#  ifndef WIN32
#    include <dirent.h>
#  else
#    include "BLI_winstuff.h"
#  endif

#  include "DNA_mesh_types.h"
#  include "DNA_modifier_types.h"
#  include "DNA_object_force_types.h"
#  include "DNA_object_types.h"
#  include "DNA_scene_types.h"
#  include "DNA_smoke_types.h"
#  include "DNA_space_types.h"

#  include "BKE_context.h"
#  include "BKE_global.h"
#  include "BKE_main.h"
#  include "BKE_material.h"
#  include "BKE_mesh.h"
#  include "BKE_modifier.h"
#  include "BKE_node.h"
#  include "BKE_object.h"
#  include "BKE_report.h"

#  include "BLI_fileops.h"
#  include "BLI_listbase.h"
#  include "BLI_path_util.h"
#  include "BLI_string.h"

#  include "DEG_depsgraph.h"

#  include "RNA_access.h"

#  include "WM_api.h"
#  include "WM_types.h"

#  include "io_openvdb.h"

#  include "openvdb_capi.h"

static void get_frame_range(Object *ob, char *path, int *r_start, int *r_end)
{
  char filepath[FILE_MAX];
  char head[FILE_MAX], tail[FILE_MAX];
  unsigned short numlen;

  BLI_strncpy(filepath, path, sizeof(filepath));

  if (BLI_path_is_rel(filepath)) {
    BLI_path_abs(filepath, ID_BLEND_PATH(G.main, (ID *)ob));
  }

  *r_start = *r_end = BLI_stringdec(filepath, head, tail, &numlen);

  /* Lower bound. */
  BLI_stringenc(filepath, head, tail, numlen, (*r_start - 1));

  while (*r_start > 0 && BLI_exists(filepath)) {
    (*r_start)--;
    BLI_stringenc(filepath, head, tail, numlen, (*r_start - 1));
  }

  /* Upper bound. */
  BLI_stringenc(filepath, head, tail, numlen, (*r_end + 1));

  while (BLI_exists(filepath)) {
    (*r_end)++;
    BLI_stringenc(filepath, head, tail, numlen, (*r_end + 1));
  }
}

static void wm_openvdb_import_draw(bContext *UNUSED(C), wmOperator *op)
{
  PointerRNA ptr;

  RNA_pointer_create(NULL, op->type->srna, op->properties, &ptr);
  // ui_openvdb_import_settings(op->layout, &ptr);
}

static int wm_openvdb_import_exec(bContext *C, wmOperator *op)
{
  if (!RNA_struct_property_is_set(op->ptr, "filepath")) {
    BKE_report(op->reports, RPT_ERROR, "No filename given");
    return OPERATOR_CANCELLED;
  }

  Main *bmain = CTX_data_main(C);
  Scene *scene = CTX_data_scene(C);
  ViewLayer *viewlayer = CTX_data_view_layer(C);
  PointCache *cache;
  char filepath[FILE_MAX];
  char filename[64];
  char cachename[64];
  RNA_string_get(op->ptr, "filepath", filepath);

  BLI_split_file_part(filepath, filename, 64);
  BLI_stringdec(filename, cachename, NULL, NULL);

  /* Set up a new object and mesh to apply the OpenVDB modifier to. */
  Mesh *mesh = BKE_mesh_add(bmain, cachename);
  Object *ob = BKE_object_add(bmain, scene, viewlayer, OB_MESH, cachename);
  ob->data = mesh;

  Material *material = BKE_material_add(bmain, "OpenVDB");
  BKE_object_material_slot_add(bmain, ob);
  assign_material(bmain, ob, material, ob->totcol, BKE_MAT_ASSIGN_EXISTING);

  if (!material->nodetree) {
    material->nodetree = ntreeAddTree(NULL, "Shader Nodetree", "ShaderNodeTree");
  }
  material->use_nodes = true;  /* Create a basic volumetric shader. */
  struct bNodeTree *tree = material->nodetree;
  struct bNode *volume_shader_node = nodeAddStaticNode(C, tree, SH_NODE_VOLUME_PRINCIPLED);
  struct bNode *output_node = nodeAddStaticNode(C, tree, SH_NODE_OUTPUT_MATERIAL);
  if (volume_shader_node && output_node) {
    volume_shader_node->locx = 0;
    volume_shader_node->locy = 300;
    volume_shader_node->flag |= NODE_SELECT;
    output_node->locx = 300;
    output_node->locy = 300;
    output_node->flag |= NODE_SELECT;
    bNodeSocket *from_socket = (bNodeSocket *)BLI_findlink(&volume_shader_node->outputs, 0);
    bNodeSocket *to_socket = (bNodeSocket *)BLI_findlink(&output_node->inputs, 1);
    nodeAddLink(tree, volume_shader_node, from_socket, output_node, to_socket);
    ntreeUpdateTree(bmain, tree);
  }

  ModifierData *md = modifier_new(eModifierType_OpenVDB);
  OpenVDBModifierData *vdbmd = (OpenVDBModifierData *)md;
  BLI_addtail(&ob->modifiers, md);

  cache = vdbmd->smoke->domain->point_cache[0];

  BLI_strncpy(vdbmd->filepath, filepath, 1024);

  get_frame_range(ob, vdbmd->filepath, &cache->startframe, &cache->endframe);

  vdbmd->frame_offset = cache->startframe - 1;
  cache->endframe -= cache->startframe - 1;
  cache->startframe = 1;

  /* Make sure to trigger updates. */
  PointerRNA ptr;
  PropertyRNA *prop;
  RNA_pointer_create(NULL, &RNA_OpenVDBModifier, vdbmd, &ptr);
  prop = RNA_struct_find_property(&ptr, "filepath");
  RNA_property_update(C, &ptr, prop);

  /* Try to find some default grids. */
  prop = RNA_struct_find_property(&ptr, "density");
  if (prop && vdbmd->numgrids > 0) {
    for (int i = 0; i < vdbmd->numgrids; ++i) {
      if (BLI_strcaseeq(vdbmd->grids[i], "density") ||
          BLI_strcaseeq(vdbmd->grids[i], "smoke")) {
        RNA_property_enum_set(&ptr, prop, i + 1);
        break;
      }
    }
  }

  return OPERATOR_FINISHED;
}

void WM_OT_openvdb_import(wmOperatorType *ot)
{
  ot->name = "Import OpenVDB";
  ot->description = "Load an OpenVDB cache";
  ot->idname = "WM_OT_openvdb_import";

  ot->invoke = WM_operator_filesel;
  ot->exec = wm_openvdb_import_exec;
  ot->poll = WM_operator_winactive;
  ot->ui = wm_openvdb_import_draw;

  WM_operator_properties_filesel(ot,
                                 FILE_TYPE_FOLDER | FILE_TYPE_OPENVDB,
                                 FILE_BLENDER,
                                 FILE_OPENFILE,
                                 WM_FILESEL_FILEPATH,
                                 FILE_DEFAULTDISPLAY,
                                 FILE_SORT_ALPHA);
}

#endif
