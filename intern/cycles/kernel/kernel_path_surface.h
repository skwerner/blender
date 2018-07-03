/*
 * Copyright 2011-2013 Blender Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "util/util_logging.h"

CCL_NAMESPACE_BEGIN

ccl_device void accum_light_contribution(KernelGlobals *kg,
                                         ShaderData *sd,
                                         ShaderData* emission_sd,
                                         LightSample *ls,
                                         ccl_addr_space PathState *state,
                                         Ray *light_ray,
                                         BsdfEval *L_light,
                                         PathRadiance *L,
                                         bool *is_lamp,
                                         float terminate,
                                         float3 throughput,
                                         float scale)
{
	if(direct_emission(kg, sd, emission_sd, ls, state, light_ray, L_light, is_lamp, terminate)) {
		/* trace shadow ray */
		float3 shadow;

		if(!shadow_blocked(kg, sd, emission_sd, state, light_ray, &shadow)) {
			/* accumulate */
			path_radiance_accum_light(L, state, throughput*scale, L_light, shadow, scale, *is_lamp);
		}
		else {
			path_radiance_accum_total_light(L, state, throughput*scale, L_light);
		}
	}
}

/* Decides whether to go down both childen or only one in the tree traversal */
ccl_device bool split(KernelGlobals *kg, ShaderData * sd, int node_offset,
                      float randu, float randv)
{
	/* early exists if never/always splitting */
	const float threshold = 1.0f - kernel_data.integrator.splitting_threshold;
	if(threshold == 1.0f){
		return false;
	} else if(threshold == 0.0f){
		return true;
	}

	/* extract bounding box of cluster */
	const float4 node1   = kernel_tex_fetch(__light_tree_nodes, node_offset + 1);
	const float4 node2   = kernel_tex_fetch(__light_tree_nodes, node_offset + 2);
	const float3 bboxMin = make_float3( node1[0], node1[1], node1[2]);
	const float3 bboxMax = make_float3( node1[3], node2[0], node2[1]);

	/* if P is inside bounding box then split */
	const float3 P = sd->P;
	const bool x_inside = (P[0] >= bboxMin[0] && P[0] <= bboxMax[0]);
	const bool y_inside = (P[1] >= bboxMin[1] && P[1] <= bboxMax[1]);
	const bool z_inside = (P[2] >= bboxMin[2] && P[2] <= bboxMax[2]);
	if(x_inside && y_inside && z_inside){
		return true; // Split
	}

	/* solid angle */

	/* approximate solid angle of bbox with solid angle of sphere */
	// From PBRT, todo: do for visible faces of bbox instead?
	const float3 centroid       = 0.5f * (bboxMax + bboxMin);
	const float  radius_squared = len_squared(bboxMax-centroid);
	const float  dist_squared   = len_squared(centroid-P);

	/* (---r---C       )
	 *  \     /
	 *   \ th/ <--- d
	 *    \ /
	 *     P
	 * sin(th) = r/d <=> sin^2(th) = r^2 / d^2 */
	const float sin_theta_max_squared = radius_squared / dist_squared;
	const float cos_theta_max = safe_sqrtf(max(0.0f,1.0f-sin_theta_max_squared));
	const float solid_angle = (dist_squared <= radius_squared)
	                          ? M_2PI_F : M_2PI_F * (1.0f - cos_theta_max);

	/* BSDF peak */

	/* TODO: Instead of randomly picking a BSDF, it might be better to
	 * loop over the BSDFs for the point and see if there are any specular ones.
	 * If so, pick one of these, otherwise, skip BSDF peak calculations. */
	const ShaderClosure *sc = shader_bsdf_pick(sd, &randu);
	if(sc == NULL) {
		return false; // TODO: handle this
	}

	float bsdf_peak = 1.0f;

	/* only sample BSDF if "highly specular" */
	/* TODO: This does not work as I expect, but it might be related to that we
	 * currently consider non-specular BSDFs here too */
	if(bsdf_get_roughness_squared(sc) < 0.25f) {
		float3 eval;
		float bsdf_pdf;
		float3 bsdf_omega_in;
		differential3 bsdf_domega_in;

		bsdf_pdf = 0.0f;
		bsdf_sample(kg, sd, sc, randu, randv, &eval, &bsdf_omega_in,
		            &bsdf_domega_in, &bsdf_pdf);

		/* TODO: More efficient to:
		 *  1. Only sample direction
		 *  2. If sampled direction points towards cluster
		 *       - Compute conservative cosine with vector to cluster center
		 *       - Evaluate simplified GGX for direction sampled direction or
		 *         vector to cluster?
		*/

		if(bsdf_pdf != 0.0f && !is_zero(eval)){

			/* check if sampled direction is pointing towards the cluster */
			const float3 P_to_centroid = normalize(centroid - P);
			const float theta     = acosf(dot(bsdf_omega_in, P_to_centroid));
			const float theta_max = acosf(cos_theta_max);
			if(theta <= theta_max){

				eval /= bsdf_pdf;
				const float BSDF = min(max3(eval), 1.0f);

				/* conservative cosine between dir to cluster's center and N */
				const float cosNI = dot(P_to_centroid, sd->N);
				const float NI    = acosf(cosNI);
				/* TODO: Do something better than clamp here.
				 * The problem: conservative_cosNI = cos(M_PI_2_F - theta_max)
				 * for NI > PI/2 instead of 0 */
				const float conservative_NI = clamp(NI - theta_max,
				                                    0.0, M_PI_2_F - theta_max);
				const float conservative_cosNI = cosf(conservative_NI);

				bsdf_peak = BSDF * conservative_cosNI;
			}
		}
	}

	/* TODO: how to make it so bsdf_peak makes it more probable to split? */
	const float heuristic = solid_angle * bsdf_peak;

	/* normalize heuristic */
	const float normalized_heuristic = heuristic  * M_1_PI_F * 0.5f;

	/* if heuristic is larger than the threshold then split */
	return normalized_heuristic > threshold;
}

/* Recursive tree traversal and accumulating contribution to L for each leaf. */
ccl_device void accum_light_tree_contribution(KernelGlobals *kg, float randu,
                                              float randv, int offset,
                                              float pdf_factor, bool can_split,
                                              float3 throughput, PathRadiance *L,
                                              ccl_addr_space PathState * state,
                                              ShaderData *sd, ShaderData *emission_sd,
                                              int *num_lights)
{
	/* TODO: sort out randu and randv rescaling */
	light_distribution_sample(kg, &randu);
	light_distribution_sample(kg, &randv);

	float3 P = sd->P;
	float time = sd->time;
	int bounce = state->bounce;

	/* read in first part of node of light BVH */
	int secondChildOffset, distribution_id, nemitters;
	update_parent_node(kg, offset, &secondChildOffset, &distribution_id, &nemitters);

	/* Found a leaf - Choose which light to use */
	if(nemitters > 0){ // Found a leaf

		if(nemitters == 1){
			(*num_lights)++; // used for debugging purposes
			// Distribution_id is the index
			/* consider this as having picked a light. */
			LightSample ls;
			light_point_sample(kg, randu, randv, time, P, bounce, distribution_id, &ls);

			/* combine pdfs */
			ls.pdf *= pdf_factor;

			if(ls.pdf == 0.0f){
				return;
			}

			Ray light_ray;
			BsdfEval L_light;
			bool is_lamp;
			float terminate = path_state_rng_light_termination(kg, state);
			accum_light_contribution(kg, sd, emission_sd, &ls, state,
			                         &light_ray, &L_light, L, &is_lamp,
			                         terminate, throughput, 1.0f);

		} // TODO: do else, i.e. with several lights per node

		return;
	} else { // Interior node, choose which child(ren) to go down

		int child_offsetL = offset + 4;
		int child_offsetR = 4*secondChildOffset;

		/* choose whether to go down both(split) or only one of the children */
		if(can_split && split(kg, sd, offset, randu, randv)){
			/* go down both child nodes */
			accum_light_tree_contribution(kg, randu, randv, child_offsetL,
			                              pdf_factor, true, throughput, L,
			                              state, sd, emission_sd, num_lights);
			accum_light_tree_contribution(kg, randu, randv, child_offsetR,
			                              pdf_factor, true, throughput, L,
			                              state, sd, emission_sd, num_lights);
		} else {
			/* go down one of the child nodes */

			/* calculate probability of going down left node */
			float I_L = calc_node_importance(kg, P, child_offsetL);
			float I_R = calc_node_importance(kg, P, child_offsetR);
			float P_L = I_L / ( I_L + I_R);
			light_distribution_sample(kg, &randu);
			if(randu <= P_L){ // Going down left node
				offset = child_offsetL;
				pdf_factor *= P_L;
			} else { // Going down right node
				offset = child_offsetR;
				pdf_factor *= 1.0f - P_L;
			}

			accum_light_tree_contribution(kg, randu, randv, offset, pdf_factor,
			                              false, throughput, L, state, sd,
			                              emission_sd, num_lights);
		}
	}
}

#if defined(__BRANCHED_PATH__) || defined(__SUBSURFACE__) || defined(__SHADOW_TRICKS__) || defined(__BAKING__)
/* branched path tracing: connect path directly to position on one or more lights and add it to L */
ccl_device_noinline void kernel_branched_path_surface_connect_light(
        KernelGlobals *kg,
        ShaderData *sd,
        ShaderData *emission_sd,
        ccl_addr_space PathState *state,
        float3 throughput,
        float num_samples_adjust,
        PathRadiance *L,
        int sample_all_lights)
{
#ifdef __EMISSION__
	/* sample illumination from lights to find path contribution */
	if(!(sd->flag & SD_BSDF_HAS_EVAL))
		return;

	Ray light_ray;
	BsdfEval L_light;
	bool is_lamp;

#  ifdef __OBJECT_MOTION__
	light_ray.time = sd->time;
#  endif

	bool use_light_bvh = kernel_data.integrator.use_light_bvh;
	bool use_splitting = kernel_data.integrator.splitting_threshold != 0.0f;
	if(use_light_bvh && use_splitting){

		int index;
		float randu, randv;
		path_state_rng_2D(kg, state, PRNG_LIGHT_U, &randu, &randv);

		/* sample light group distribution */
		int   group      = light_group_distribution_sample(kg, &randu);
		float group_prob = kernel_tex_fetch(__light_group_sample_prob, group);
		float pdf = 1.0f;
		if(group == LIGHTGROUP_TREE){
			/* accumulate contribution to L from potentially several lights */
			int num_lights = 0;
			accum_light_tree_contribution(kg, randu, randv, 0, group_prob, true,
			                              throughput, L, state, sd, emission_sd,
			                              &num_lights);
			if(num_lights > 10){ // Debug print
				VLOG(1) << "Sampled " << num_lights << " lights!";
			}

			/* have accumulated all the contributions so return */
			return;
		} else if(group == LIGHTGROUP_DISTANT) {
			/* pick a single distant light */
			light_distant_sample(kg, sd->P, &randu, &index, &pdf);
		} else if(group == LIGHTGROUP_BACKGROUND) {
			/* pick a single background light */
			light_background_sample(kg, sd->P, &randu, &index, &pdf);
		} else {
			kernel_assert(false);
		}

		/* sample a point on the given distant/background light */
		LightSample ls;
		light_point_sample(kg, randu, randv, sd->time, sd->P, state->bounce, index, &ls);

		/* combine pdfs */
		ls.pdf *= group_prob;

		if(ls.pdf == 0.0f) return;

		/* accumulate the contribution of this distant/background light to L */
		float terminate = path_state_rng_light_termination(kg, state);
		accum_light_contribution(kg, sd, emission_sd, &ls, state, &light_ray,
		                         &L_light, L, &is_lamp, terminate, throughput,
		                         num_samples_adjust);

	} else if(sample_all_lights) {
		/* lamp sampling */
		for(int i = 0; i < kernel_data.integrator.num_all_lights; i++) {
			if(UNLIKELY(light_select_reached_max_bounces(kg, i, state->bounce)))
				continue;

			int num_samples = ceil_to_int(num_samples_adjust*light_select_num_samples(kg, i));
			float num_samples_inv = num_samples_adjust/num_samples;
			uint lamp_rng_hash = cmj_hash(state->rng_hash, i);

			for(int j = 0; j < num_samples; j++) {
				float light_u, light_v;
				path_branched_rng_2D(kg, lamp_rng_hash, state, j, num_samples, PRNG_LIGHT_U, &light_u, &light_v);
				float terminate = path_branched_rng_light_termination(kg, lamp_rng_hash, state, j, num_samples);

				LightSample ls;
				if(lamp_light_sample(kg, i, light_u, light_v, sd->P, &ls)) {
					accum_light_contribution(kg, sd, emission_sd, &ls, state,
					                         &light_ray, &L_light, L, &is_lamp,
					                         terminate, throughput,
					                         num_samples_inv);
				}
			}
		}

		/* mesh light sampling */
		if(kernel_data.integrator.pdf_triangles != 0.0f) {
			int num_samples = ceil_to_int(num_samples_adjust*kernel_data.integrator.mesh_light_samples);
			float num_samples_inv = num_samples_adjust/num_samples;

			for(int j = 0; j < num_samples; j++) {
				float light_u, light_v;
				path_branched_rng_2D(kg, state->rng_hash, state, j, num_samples, PRNG_LIGHT_U, &light_u, &light_v);
				float terminate = path_branched_rng_light_termination(kg, state->rng_hash, state, j, num_samples);

				/* only sample triangle lights */
				if(kernel_data.integrator.num_all_lights)
					light_u = 0.5f*light_u;

				kernel_assert(!kernel_data.integrator.use_light_bvh);

				LightSample ls;
				if(light_sample(kg, light_u, light_v, sd->time, sd->P, state->bounce, &ls)) {
					/* Same as above, probability needs to be corrected since the sampling was forced to select a mesh light. */
					if(kernel_data.integrator.num_all_lights)
						ls.pdf *= 2.0f;

					accum_light_contribution(kg, sd, emission_sd, &ls, state,
					                         &light_ray, &L_light, L, &is_lamp,
					                         terminate, throughput, num_samples_inv);
				}
			}
		}
	}
	else {
		/* sample one light at random */
		float light_u, light_v;
		path_state_rng_2D(kg, state, PRNG_LIGHT_U, &light_u, &light_v);
		float terminate = path_state_rng_light_termination(kg, state);

		LightSample ls;
		if(light_sample(kg, light_u, light_v, sd->time, sd->P, state->bounce, &ls)) {
			/* sample random light */
			accum_light_contribution(kg, sd, emission_sd, &ls, state, &light_ray,
			                         &L_light, L, &is_lamp, terminate, throughput,
			                         num_samples_adjust);
		}
	}
#endif
}

/* branched path tracing: bounce off or through surface to with new direction stored in ray */
ccl_device bool kernel_branched_path_surface_bounce(
        KernelGlobals *kg,
        ShaderData *sd,
        const ShaderClosure *sc,
        int sample,
        int num_samples,
        ccl_addr_space float3 *throughput,
        ccl_addr_space PathState *state,
        PathRadianceState *L_state,
        ccl_addr_space Ray *ray,
        float sum_sample_weight)
{
	/* sample BSDF */
	float bsdf_pdf;
	BsdfEval bsdf_eval;
	float3 bsdf_omega_in;
	differential3 bsdf_domega_in;
	float bsdf_u, bsdf_v;
	path_branched_rng_2D(kg, state->rng_hash, state, sample, num_samples, PRNG_BSDF_U, &bsdf_u, &bsdf_v);
	int label;

	label = shader_bsdf_sample_closure(kg, sd, sc, bsdf_u, bsdf_v, &bsdf_eval,
		&bsdf_omega_in, &bsdf_domega_in, &bsdf_pdf);

	if(bsdf_pdf == 0.0f || bsdf_eval_is_zero(&bsdf_eval))
		return false;

	/* modify throughput */
	path_radiance_bsdf_bounce(kg, L_state, throughput, &bsdf_eval, bsdf_pdf, state->bounce, label);

#ifdef __DENOISING_FEATURES__
	state->denoising_feature_weight *= sc->sample_weight / (sum_sample_weight * num_samples);
#endif

	/* modify path state */
	path_state_next(kg, state, label);

	/* setup ray */
	ray->P = ray_offset(sd->P, (label & LABEL_TRANSMIT)? -sd->Ng: sd->Ng);
	ray->D = normalize(bsdf_omega_in);
	ray->t = FLT_MAX;
#ifdef __RAY_DIFFERENTIALS__
	ray->dP = sd->dP;
	ray->dD = bsdf_domega_in;
#endif
#ifdef __OBJECT_MOTION__
	ray->time = sd->time;
#endif

#ifdef __VOLUME__
	/* enter/exit volume */
	if(label & LABEL_TRANSMIT)
		kernel_volume_stack_enter_exit(kg, sd, state->volume_stack);
#endif

	/* branch RNG state */
	path_state_branch(state, sample, num_samples);

	/* set MIS state */
	state->min_ray_pdf = fminf(bsdf_pdf, FLT_MAX);
	state->ray_pdf = bsdf_pdf;
#ifdef __LAMP_MIS__
	state->ray_t = 0.0f;
#endif

	return true;
}

#endif

/* path tracing: connect path directly to position on a light and add it to L */
ccl_device_inline void kernel_path_surface_connect_light(KernelGlobals *kg,
	ShaderData *sd, ShaderData *emission_sd, float3 throughput, ccl_addr_space PathState *state,
	PathRadiance *L)
{
#ifdef __EMISSION__
	if(!(kernel_data.integrator.use_direct_light && (sd->flag & SD_BSDF_HAS_EVAL)))
		return;

#ifdef __SHADOW_TRICKS__
	if(state->flag & PATH_RAY_SHADOW_CATCHER) {
		kernel_branched_path_surface_connect_light(kg,
		                                           sd,
		                                           emission_sd,
		                                           state,
		                                           throughput,
		                                           1.0f,
		                                           L,
		                                           1);
		return;
	}
#endif

	/* sample illumination from lights to find path contribution */
	float light_u, light_v;
	path_state_rng_2D(kg, state, PRNG_LIGHT_U, &light_u, &light_v);

	Ray light_ray;
	BsdfEval L_light;
	bool is_lamp;

#ifdef __OBJECT_MOTION__
	light_ray.time = sd->time;
#endif

	LightSample ls;
	if(light_sample(kg, light_u, light_v, sd->time, sd->P, state->bounce, &ls)) {
		float terminate = path_state_rng_light_termination(kg, state);
		accum_light_contribution(kg, sd, emission_sd, &ls, state, &light_ray,
		                         &L_light, L, &is_lamp, terminate, throughput,
		                         1.0f);
	}
#endif
}

/* path tracing: bounce off or through surface to with new direction stored in ray */
ccl_device bool kernel_path_surface_bounce(KernelGlobals *kg,
                                           ShaderData *sd,
                                           ccl_addr_space float3 *throughput,
                                           ccl_addr_space PathState *state,
                                           PathRadianceState *L_state,
                                           ccl_addr_space Ray *ray)
{
	/* no BSDF? we can stop here */
	if(sd->flag & SD_BSDF) {
		/* sample BSDF */
		float bsdf_pdf;
		BsdfEval bsdf_eval;
		float3 bsdf_omega_in;
		differential3 bsdf_domega_in;
		float bsdf_u, bsdf_v;
		path_state_rng_2D(kg, state, PRNG_BSDF_U, &bsdf_u, &bsdf_v);
		int label;

		label = shader_bsdf_sample(kg, sd, bsdf_u, bsdf_v, &bsdf_eval,
			&bsdf_omega_in, &bsdf_domega_in, &bsdf_pdf);

		if(bsdf_pdf == 0.0f || bsdf_eval_is_zero(&bsdf_eval))
			return false;

		/* modify throughput */
		path_radiance_bsdf_bounce(kg, L_state, throughput, &bsdf_eval, bsdf_pdf, state->bounce, label);

		/* set labels */
		if(!(label & LABEL_TRANSPARENT)) {
			state->ray_pdf = bsdf_pdf;
#ifdef __LAMP_MIS__
			state->ray_t = 0.0f;
#endif
			state->min_ray_pdf = fminf(bsdf_pdf, state->min_ray_pdf);
		}

		/* update path state */
		path_state_next(kg, state, label);

		/* setup ray */
		ray->P = ray_offset(sd->P, (label & LABEL_TRANSMIT)? -sd->Ng: sd->Ng);
		ray->D = normalize(bsdf_omega_in);

		if(state->bounce == 0)
			ray->t -= sd->ray_length; /* clipping works through transparent */
		else
			ray->t = FLT_MAX;

#ifdef __RAY_DIFFERENTIALS__
		ray->dP = sd->dP;
		ray->dD = bsdf_domega_in;
#endif

#ifdef __VOLUME__
		/* enter/exit volume */
		if(label & LABEL_TRANSMIT)
			kernel_volume_stack_enter_exit(kg, sd, state->volume_stack);
#endif
		return true;
	}
#ifdef __VOLUME__
	else if(sd->flag & SD_HAS_ONLY_VOLUME) {
		if(!path_state_volume_next(kg, state)) {
			return false;
		}

		if(state->bounce == 0)
			ray->t -= sd->ray_length; /* clipping works through transparent */
		else
			ray->t = FLT_MAX;

		/* setup ray position, direction stays unchanged */
		ray->P = ray_offset(sd->P, -sd->Ng);
#ifdef __RAY_DIFFERENTIALS__
		ray->dP = sd->dP;
#endif

		/* enter/exit volume */
		kernel_volume_stack_enter_exit(kg, sd, state->volume_stack);
		return true;
	}
#endif
	else {
		/* no bsdf or volume? */
		return false;
	}
}

CCL_NAMESPACE_END

