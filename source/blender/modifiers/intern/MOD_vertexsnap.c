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
 * along with this program; if not, write to the Free Software  Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 *
 * The Original Code is Copyright (C) 2005 by the Blender Foundation.
 * All rights reserved.
 *
 * Contributor(s): Your name
 *
 * ***** END GPL LICENSE BLOCK *****
 *
 */

/** \file blender/modifiers/intern/MOD_Snap.c
 *  \ingroup modifiers
 */

#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"
#include "DNA_object_types.h"

#include "BLI_math.h"
#include "BLI_task.h"
#include "BLI_string.h"
#include "BLI_utildefines.h"

#include "MEM_guardedalloc.h"

#include "BKE_action.h"
#include "BKE_bvhutils.h"
#include "BKE_cdderivedmesh.h"
#include "BKE_deform.h"
#include "BKE_library.h"
#include "BKE_library_query.h"
#include "BKE_mesh.h"
#include "BKE_particle.h"

#include "MOD_util.h"

#include "DEG_depsgraph_query.h"


// #define DEBUG_TIME 1

#include "PIL_time.h"
#ifdef DEBUG_TIME
	#include "PIL_time_utildefines.h"
#endif


static void initData(ModifierData *md)
{
	VertexSnapModifierData* vmd = (VertexSnapModifierData*) md;
	vmd->blend = 1.0f;
	vmd->target = NULL;
	vmd->vertex_group[0] = 0;
	vmd->deform_space = MOD_VSNAP_LOCAL;
	vmd->bindings = NULL;
	vmd->total_bindings = 0;
	vmd->binding_type = MOD_VSNAP_BIND_INDEX;
	vmd->binding_distance = 64.0f;
	vmd->flags = 0;
}

static bool isDisabled(const struct Scene *scene, ModifierData *md, bool userRenderParams)
{
	/* disable if modifier there is no connected target object*/
	VertexSnapModifierData* vmd = (VertexSnapModifierData*)md;

	if ( vmd->target ) {
		if ( vmd->target->type == OB_MESH ) {
			return false;
		}
	}

	return true;
}

static void requiredDataMask(Object *UNUSED(ob),
                             ModifierData *md,
                             CustomData_MeshMasks *r_cddata_masks)
{
	VertexSnapModifierData* enmd = (VertexSnapModifierData*)md;

	/* Ask for vertexgroups if we need them. */
	if (enmd->vertex_group[0] != '\0') {
		r_cddata_masks->vmask |= ( CD_MASK_MDEFORMVERT );
	}
}

static inline void propagate_bindings(const VertexSnapModifierData* source, VertexSnapModifierData* target) {
	if (source->bindings) {
		target->bindings = MEM_dupallocN(source->bindings);
		target->total_bindings = source->total_bindings;
	}
	target->flags = source->flags;
}

static void freeData(ModifierData *md)
{
	VertexSnapModifierData* vmd = (VertexSnapModifierData*)md;

	if (vmd->bindings && vmd->total_bindings) {
		MEM_SAFE_FREE(vmd->bindings);
	}

	vmd->total_bindings = 0;
	vmd->flags = 0;
}

static void copyData(const ModifierData *md, ModifierData *target, const int flag)
{
	const VertexSnapModifierData* vmd  = (VertexSnapModifierData*)md;
	VertexSnapModifierData* target_vmd = (VertexSnapModifierData*)target;

	modifier_copyData_generic(md, target, flag);
	propagate_bindings(vmd, target_vmd);
}

static void foreachObjectLink(ModifierData *md, Object *ob, ObjectWalkFunc walk, void *userData)
{
	VertexSnapModifierData* vmd = (VertexSnapModifierData*)md;

	walk( userData, ob, &vmd->target, IDWALK_NOP );
}

static void updateDepsgraph(ModifierData *md,
							const ModifierUpdateDepsgraphContext *ctx)

{
	VertexSnapModifierData* vmd = (VertexSnapModifierData*)md;
	if (vmd->target != NULL) {
		DEG_add_object_relation(ctx->node, vmd->target,
				DEG_OB_COMP_GEOMETRY, "VertexSnap Modifier");
		DEG_add_object_relation(ctx->node, vmd->target,
				DEG_OB_COMP_TRANSFORM, "VertexSnap Modifier");
	}

	// make sure we're linked to our own transform
	// do we need this? Yes, we do.
	DEG_add_modifier_to_transform_relation(ctx->node, "VertexSnap Modifier");
}

/* binding calculations */
static inline void calculate_closest_point_bindings(ModifierData* md, Mesh* mesh, 
                                                    Mesh* target_mesh, const float (*world_matrix)[4],
																										const float (*target_inverse_matrix)[4]) {
	int index = 0;
	BVHTreeFromMesh bvh_tree = {NULL};
	BVHTreeNearest nearest = {0};
	VertexSnapModifierData* vmd = (VertexSnapModifierData*)md;
	freeData(md);

	if (!mesh) {
		// should never happen...
		modifier_setError(md, "Cannot bind-- no mesh data.");
		return;
	}

	if (!target_mesh) {
		modifier_setError(md, "Cannot bind-- no target mesh.");
		return;
	}

	vmd->bindings = (unsigned int*)MEM_mallocN(sizeof(unsigned int) * mesh->totvert, "VertexSnapModifier Bindings");
	BKE_bvhtree_from_mesh_get(&bvh_tree, target_mesh, BVHTREE_FROM_VERTS, 2);

	if (bvh_tree.tree == NULL) {
		freeData(md);
		modifier_setError((ModifierData*)md, "Out of memory");
		return;
	}

	//##!FIXME: Thread this binding
	for (index = 0; index < mesh->totvert; index++) {
		float world_coordinate[3];
		if (vmd->deform_space == MOD_VSNAP_WORLD) {
			mul_v3_m4v3(world_coordinate, world_matrix, mesh->mvert[index].co);
			mul_v3_m4v3(world_coordinate, target_inverse_matrix, world_coordinate);
		}
		else {
			copy_v3_v3(world_coordinate, mesh->mvert[index].co);
		}
		nearest.index = -1;
		nearest.dist_sq = 64.0f;
		BLI_bvhtree_find_nearest(bvh_tree.tree,
								 world_coordinate,
								 &nearest,
								 NULL,
								 &bvh_tree);

		// We're going to skip bindings whose
		// index is -1 later on.
		vmd->bindings[index] = nearest.index;
	}

	vmd->total_bindings = mesh->totvert;
	vmd->flags = 0;

	free_bvhtree_from_mesh(&bvh_tree);
}

static inline void calculate_closest_point_and_normal_bindings(ModifierData* md, Mesh* mesh, Mesh* target_mesh) {
	/*
		!TODO: 
		!check out BLI_bvhtree_range_query.
		!It seems to do what I want for looking
		!within a radius and applies the
		!callback function to each found point. 
	*/

}

typedef struct VertexSnapUserdata {
	//##!FIXME: How much of this is necessary now?
	/*const*/ VertexSnapModifierData* vmd;
	MDeformVert *dverts;
	MVert *target_mvert;
	float (*vertexCos)[3];
	float object_matrix[4][4];
	float target_matrix[4][4];
	float object_matrix_inv[4][4];
	int   deform_group_index;
} VertexSnapUserdata;


static void VertexSnapModifier_do_task(void *__restrict userdata,
                                       const int iter,
                                       const TaskParallelTLS *__restrict UNUSED(tls))
{
	VertexSnapUserdata     *data  = (VertexSnapUserdata *)userdata;
	VertexSnapModifierData* vmd   = data->vmd;
	MDeformVert *dverts           = data->dverts;
	MVert       *target_mvert     = data->target_mvert;
	float      (*vertexCos)[3]    = data->vertexCos;
	int          target_index     = iter;

	float blend = vmd->blend;
	const float deform_group_index = data->deform_group_index;

	if (dverts) {
		blend *= defvert_find_weight( &dverts[iter], deform_group_index);
	}

	if (!blend) {
		return;
	}

	if (vmd->binding_type != MOD_VSNAP_BIND_INDEX) {
		target_index = vmd->bindings[iter];
		if (target_index < 0) {
			// some sort of an error has happened in the bind--
			// skip this vertex
			return;
		}
	}

	if ( vmd->deform_space == MOD_VSNAP_WORLD ) {
		float object_co[3];
		float target_co[3];

		//!TODO: Test calculating in object space directly
		// calculate lerp in world space
		mul_v3_m4v3( object_co, data->object_matrix, vertexCos[iter] );
		mul_v3_m4v3( target_co, data->target_matrix, target_mvert[target_index].co );
		interp_v3_v3v3( object_co, object_co, target_co, blend);

		// remove the world matrix of the deforming object
		// after doing the lerp
		mul_v3_m4v3( vertexCos[iter], data->object_matrix_inv, object_co );

	} else {
		interp_v3_v3v3( vertexCos[iter], vertexCos[iter], target_mvert[target_index].co, blend);
	}
}


static void VertexSnapModifier_do(ModifierData *md,
                                  const ModifierEvalContext *ctx,
                                  Object *ob,
                                  Mesh *mesh,
                                  float (*vertexCos)[3],
                                  int numVerts)
{
	VertexSnapModifierData* vmd = (VertexSnapModifierData* )md;
	struct Object *target       = vmd->target;
	MDeformVert   *dverts       = NULL;
	int deform_group_index      = -1;
	const int vertex_count      = numVerts;
	const float blend           = vmd->blend;
	ModifierData *md_orig       = NULL;

	if ( blend == 0.0 )
		return;

	struct Mesh *target_mesh = BKE_modifier_get_evaluated_mesh_from_evaluated_object(target, false);
	if (!target_mesh) {
		modifier_setError(md, "Cannot get the target mesh object.");
		return;
	}

	/* 
		bindings calculations -- early exit if unbinding.
		Don't free the original in freeData-- it gets freed
		in a separate place. However, if you allocate bindings
		during a run you MUST copy to the orig_modifier_data or
		bind on the orig and copy the data over to the current
		modifier-- see propagate_bindings().
	*/

	if (vmd->flags == MOD_VSNAP_NEEDS_UNBIND) {
		md_orig = modifier_get_original(md);
		freeData(md_orig);
		return;
	}

	if (vmd->flags == MOD_VSNAP_NEEDS_BIND) {
		if (!DEG_is_active(ctx->depsgraph)) {
			modifier_setError(md, "Attempt to bind from inactive dependency graph");
			return;
		}

		md_orig = modifier_get_original(md);

		if (vmd->binding_type == MOD_VSNAP_BIND_CLOSEST) {
			calculate_closest_point_bindings(md_orig, mesh, target_mesh, ob->obmat, target->imat);
			propagate_bindings((VertexSnapModifierData*)md_orig, vmd);
		}
		else if (vmd->binding_type == MOD_VSNAP_BIND_NORMAL) {
			//!FIXME: Do this one
			modifier_setError(md_orig, "Tried to rebind, but type is not Closest.");
			return;
			calculate_closest_point_and_normal_bindings(md_orig, mesh, target_mesh);
		}
	}

	if (vmd->binding_type != MOD_VSNAP_BIND_INDEX && !vmd->bindings) {
		//##!FIXME: This shouldn't fire when it's in non-depsgraph mode
		modifier_setError(md, "Not bound.");
		return;
	}

	if (!(target && target != ob && target->type == OB_MESH)) {
		// this shouldn't happen
		modifier_setError(md, "Target %s is not a Mesh.", target->id.name + 2);
		return;
	}

	if (vmd->binding_type == MOD_VSNAP_BIND_INDEX && vertex_count != target_mesh->totvert) {
		modifier_setError(md, "Target vertex count is %d; should be %d.",
						target_mesh->totvert, vertex_count);
		return;
	}

	invert_m4_m4( ob->imat, ob->obmat );
	invert_m4_m4( target->imat, target->obmat );

	MOD_get_vgroup(ob, mesh, vmd->vertex_group, &dverts, &deform_group_index);

	VertexSnapUserdata data;
	data.vmd    = vmd;
	data.dverts = dverts;
	copy_m4_m4(data.object_matrix, ob->obmat);
	copy_m4_m4(data.object_matrix_inv, ob->imat);
	copy_m4_m4(data.target_matrix,target->obmat);
	data.target_mvert = target_mesh->mvert;
	data.vertexCos    = vertexCos;
	data.deform_group_index = deform_group_index;

	#ifdef DEBUG_TIME
		TIMEIT_START( vertex_snap_modifier ); 
	#endif

	TaskParallelSettings settings;
	BLI_parallel_range_settings_defaults(&settings);
	settings.use_threading = (vertex_count > 512);
	BLI_task_parallel_range(0, vertex_count, &data, VertexSnapModifier_do_task, &settings);

	#ifdef DEBUG_TIME
		TIMEIT_END( vertex_snap_modifier );
	#endif
}


static void deformVerts(struct ModifierData *md,

                        const struct ModifierEvalContext *ctx,
                        struct Mesh *mesh,
                        float (*vertexCos)[3],
                        int numVerts)
{
	Mesh *mesh_src = MOD_deform_mesh_eval_get(ctx->object, NULL, mesh, 
												NULL, numVerts, false, false);

	VertexSnapModifier_do(md, ctx, ctx->object, mesh_src, vertexCos, numVerts);
}

static void deformVertsEM(struct ModifierData *md,
                          const struct ModifierEvalContext *ctx,
                          struct BMEditMesh *editData,
                          struct Mesh *mesh,
                          float (*vertexCos)[3],
                          int numVerts)
{
	VertexSnapModifier_do(md, ctx, ctx->object, mesh, vertexCos, numVerts);
}


ModifierTypeInfo modifierType_VertexSnap = {
	/* name */              "VertexSnap",
	/* structName */        "VertexSnapModifierData",
	/* structSize */        sizeof(VertexSnapModifierData),
	/* type */              eModifierTypeType_OnlyDeform,
	/* flags */             eModifierTypeFlag_AcceptsMesh |
	                        eModifierTypeFlag_SupportsEditmode,

	/* copyData */          copyData,
	/* deformVerts */       deformVerts,
	/* deformMatrices */    NULL,
	/* deformVertsEM */     deformVertsEM,
	/* deformMatricesEM */  NULL,
	/* applyModifier */     NULL,
	/* initData */          initData,
	/* requiredDataMask */  requiredDataMask,
	/* freeData */          freeData,
	/* isDisabled */        isDisabled,
	/* updateDepsgraph */   updateDepsgraph,
	/* dependsOnTime */     NULL,
	/* dependsOnNormals */  NULL,
	/* foreachObjectLink */ foreachObjectLink,
	/* foreachIDLink */     NULL,
	/* foreachTexLink */    NULL,
	/* freeRuntimeData */   NULL
};

