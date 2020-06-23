/*
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
 * The Original Code is Copyright (C) 2017 by Blender Foundation.
 * All rights reserved.
 */

/** \file
 * \ingroup draw
 *
 * \brief Contains procedural GPU hair drawing methods.
 */

#include "DRW_render.h"

#include "BLI_string_utils.h"
#include "BLI_utildefines.h"

#include "DNA_customdata_types.h"
#include "DNA_modifier_types.h"
#include "DNA_particle_types.h"

#include "BKE_duplilist.h"

#include "GPU_batch.h"
#include "GPU_shader.h"
#include "GPU_vertex_buffer.h"

#include "draw_hair_private.h"

typedef enum ParticleRefineShader {
  PART_REFINE_CATMULL_ROM = 0,
  PART_REFINE_MAX_SHADER,
} ParticleRefineShader;

static GPUVertBuf *g_dummy_vbo = NULL;
static GPUTexture *g_dummy_texture = NULL;
static GPUShader *g_refine_shaders[PART_REFINE_MAX_SHADER] = {NULL};
static DRWPass *g_tf_pass; /* XXX can be a problem with multiple DRWManager in the future */

extern char datatoc_common_hair_lib_glsl[];
extern char datatoc_common_hair_refine_vert_glsl[];
extern char datatoc_gpu_shader_3D_smooth_color_frag_glsl[];

static GPUShader *hair_refine_shader_get(ParticleRefineShader sh)
{
  if (g_refine_shaders[sh]) {
    return g_refine_shaders[sh];
  }

  char *vert_with_lib = BLI_string_joinN(datatoc_common_hair_lib_glsl,
                                         datatoc_common_hair_refine_vert_glsl);

  const char *var_names[1] = {"finalColor"};
  g_refine_shaders[sh] = DRW_shader_create_with_transform_feedback(
      vert_with_lib, NULL, "#define HAIR_PHASE_SUBDIV\n", GPU_SHADER_TFB_POINTS, var_names, 1);

  MEM_freeN(vert_with_lib);

  return g_refine_shaders[sh];
}

void DRW_hair_init(void)
{
  g_tf_pass = DRW_pass_create("Update Hair Pass", 0);

  if (g_dummy_vbo == NULL) {
    /* initialize vertex format */
    GPUVertFormat format = {0};
    uint dummy_id = GPU_vertformat_attr_add(&format, "dummy", GPU_COMP_F32, 4, GPU_FETCH_FLOAT);

    g_dummy_vbo = GPU_vertbuf_create_with_format(&format);

    float vert[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    GPU_vertbuf_data_alloc(g_dummy_vbo, 1);
    GPU_vertbuf_attr_fill(g_dummy_vbo, dummy_id, vert);
    /* Create vbo immediately to bind to texture buffer. */
    GPU_vertbuf_use(g_dummy_vbo);

    g_dummy_texture = GPU_texture_create_from_vertbuf(g_dummy_vbo);
  }
}

DRWShadingGroup *DRW_shgroup_hair_create_sub(Object *object,
                                             ParticleSystem *psys,
                                             ModifierData *md,
                                             DRWShadingGroup *shgrp_parent)
{
  /* TODO(fclem): Pass the scene as parameter */
  const DRWContextState *draw_ctx = DRW_context_state_get();
  Scene *scene = draw_ctx->scene;
  float dupli_mat[4][4];
  Object *dupli_parent = DRW_object_get_dupli_parent(object);
  DupliObject *dupli_object = DRW_object_get_dupli(object);

  int subdiv = scene->r.hair_subdiv;
  int thickness_res = (scene->r.hair_type == SCE_HAIR_SHAPE_STRAND) ? 1 : 2;

  ParticleHairCache *hair_cache;
  bool need_ft_update;
  if (psys) {
    /* Old particle hair. */
    need_ft_update = particles_ensure_procedural_data(
        object, psys, md, &hair_cache, subdiv, thickness_res);
  }
  else {
    /* New hair object. */
    need_ft_update = hair_ensure_procedural_data(object, &hair_cache, subdiv, thickness_res);
  }

  DRWShadingGroup *shgrp = DRW_shgroup_create_sub(shgrp_parent);

  /* TODO optimize this. Only bind the ones GPUMaterial needs. */
  for (int i = 0; i < hair_cache->num_uv_layers; i++) {
    for (int n = 0; n < MAX_LAYER_NAME_CT && hair_cache->uv_layer_names[i][n][0] != '\0'; n++) {
      DRW_shgroup_uniform_texture(shgrp, hair_cache->uv_layer_names[i][n], hair_cache->uv_tex[i]);
    }
  }
  for (int i = 0; i < hair_cache->num_col_layers; i++) {
    for (int n = 0; n < MAX_LAYER_NAME_CT && hair_cache->col_layer_names[i][n][0] != '\0'; n++) {
      DRW_shgroup_uniform_texture(
          shgrp, hair_cache->col_layer_names[i][n], hair_cache->col_tex[i]);
    }
  }

  /* Fix issue with certain driver not drawing anything if there is no texture bound to
   * "ac", "au", "u" or "c". */
  if (hair_cache->num_uv_layers == 0) {
    DRW_shgroup_uniform_texture(shgrp, "u", g_dummy_texture);
    DRW_shgroup_uniform_texture(shgrp, "au", g_dummy_texture);
  }
  if (hair_cache->num_col_layers == 0) {
    DRW_shgroup_uniform_texture(shgrp, "c", g_dummy_texture);
    DRW_shgroup_uniform_texture(shgrp, "ac", g_dummy_texture);
  }

  if (psys) {
    if ((dupli_parent != NULL) && (dupli_object != NULL)) {
      if (dupli_object->type & OB_DUPLICOLLECTION) {
        copy_m4_m4(dupli_mat, dupli_parent->obmat);
      }
      else {
        copy_m4_m4(dupli_mat, dupli_object->ob->obmat);
        invert_m4(dupli_mat);
        mul_m4_m4m4(dupli_mat, object->obmat, dupli_mat);
      }
    }
    else {
      unit_m4(dupli_mat);
    }
  }
  else {
    /* New hair object. */
    copy_m4_m4(dupli_mat, object->obmat);
  }

  /* Get hair shape parameters. */
  float hair_rad_shape, hair_rad_root, hair_rad_tip;
  bool hair_close_tip;
  if (psys) {
    /* Old particle hair. */
    ParticleSettings *part = psys->part;
    hair_rad_shape = part->shape;
    hair_rad_root = part->rad_root * part->rad_scale * 0.5f;
    hair_rad_tip = part->rad_tip * part->rad_scale * 0.5f;
    hair_close_tip = (part->shape_flag & PART_SHAPE_CLOSE_TIP) != 0;
  }
  else {
    /* TODO: implement for new hair object. */
    hair_rad_shape = 1.0f;
    hair_rad_root = 0.005f;
    hair_rad_tip = 0.0f;
    hair_close_tip = true;
  }

  DRW_shgroup_uniform_texture(shgrp, "hairPointBuffer", hair_cache->final[subdiv].proc_tex);
  DRW_shgroup_uniform_int(shgrp, "hairStrandsRes", &hair_cache->final[subdiv].strands_res, 1);
  DRW_shgroup_uniform_int_copy(shgrp, "hairThicknessRes", thickness_res);
  DRW_shgroup_uniform_float_copy(shgrp, "hairRadShape", hair_rad_shape);
  DRW_shgroup_uniform_vec4_array_copy(shgrp, "hairDupliMatrix", dupli_mat, 4);
  DRW_shgroup_uniform_float_copy(shgrp, "hairRadRoot", hair_rad_root);
  DRW_shgroup_uniform_float_copy(shgrp, "hairRadTip", hair_rad_tip);
  DRW_shgroup_uniform_bool_copy(shgrp, "hairCloseTip", hair_close_tip);
  /* TODO(fclem): Until we have a better way to cull the hair and render with orco, bypass
   * culling test. */
  GPUBatch *geom = hair_cache->final[subdiv].proc_hairs[thickness_res - 1];
  DRW_shgroup_call_no_cull(shgrp, geom, object);

  /* Transform Feedback subdiv. */
  if (need_ft_update) {
    int final_points_len = hair_cache->final[subdiv].strands_res * hair_cache->strands_len;
    if (final_points_len) {
      GPUShader *tf_shader = hair_refine_shader_get(PART_REFINE_CATMULL_ROM);

      DRWShadingGroup *tf_shgrp = DRW_shgroup_transform_feedback_create(
          tf_shader, g_tf_pass, hair_cache->final[subdiv].proc_buf);

      DRW_shgroup_uniform_texture(tf_shgrp, "hairPointBuffer", hair_cache->point_tex);
      DRW_shgroup_uniform_texture(tf_shgrp, "hairStrandBuffer", hair_cache->strand_tex);
      DRW_shgroup_uniform_texture(tf_shgrp, "hairStrandSegBuffer", hair_cache->strand_seg_tex);
      DRW_shgroup_uniform_int(
          tf_shgrp, "hairStrandsRes", &hair_cache->final[subdiv].strands_res, 1);
      DRW_shgroup_call_procedural_points(tf_shgrp, NULL, final_points_len);
    }
  }

  return shgrp;
}

void DRW_hair_update(void)
{
  /* TODO(fclem): replace by compute shader. */
  /* Just render using transform feedback. */
  DRW_draw_pass(g_tf_pass);
}

void DRW_hair_free(void)
{
  for (int i = 0; i < PART_REFINE_MAX_SHADER; i++) {
    DRW_SHADER_FREE_SAFE(g_refine_shaders[i]);
  }

  GPU_VERTBUF_DISCARD_SAFE(g_dummy_vbo);
  DRW_TEXTURE_FREE_SAFE(g_dummy_texture);
}
