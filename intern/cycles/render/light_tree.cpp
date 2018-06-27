/*
 * Copyright 2011-2018 Blender Foundation
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

#include <algorithm>

#include "render/light.h"
#include "render/light_tree.h"
#include "render/mesh.h"
#include "render/object.h"
#include "render/scene.h"

#include "util/util_foreach.h"
#include "util/util_logging.h"

CCL_NAMESPACE_BEGIN


LightTree::LightTree(const vector<Primitive>& prims_,
                     const vector<Object*>& objects_,
                     const vector<Light*>& lights_,
                     const unsigned int maxPrimsInNode_)
    :  objects(objects_), lights(lights_), maxPrimsInNode(maxPrimsInNode_)
{

	if (prims_.empty()) return;

	/* move all primitives except background and distant lights into local
	 * primitives array */
	primitives.reserve(prims_.size());
	vector<Primitive> distant_lights;
	vector<Primitive> background_lights;
	foreach(Primitive prim, prims_ ){

		/* put distant lights into its own array */
		if (prim.prim_id < 0){
			const Light *lamp = lights[prim.lamp_id];
			if (lamp->type == LIGHT_DISTANT){
				distant_lights.push_back(prim);
				continue;
			} else if(lamp->type == LIGHT_BACKGROUND){
				background_lights.push_back(prim);
				continue;
			}
		}

		primitives.push_back(prim);
	}

	/* initialize buildData array */
	vector<BVHPrimitiveInfo> buildData;
	buildData.reserve(primitives.size());
	for(int i = 0; i < primitives.size(); ++i){
		BoundBox bbox = get_bbox(primitives[i]);
		Orientation bcone = get_bcone(primitives[i]);
		float energy = get_energy(primitives[i]);
		buildData.push_back(BVHPrimitiveInfo(i, bbox, bcone, energy));
	}

	/* recursively build BVH tree */
	unsigned int totalNodes = 0;
	vector<Primitive> orderedPrims;
	orderedPrims.reserve(primitives.size());
	BVHBuildNode *root = recursive_build(0, primitives.size(), buildData,
	                                     totalNodes, orderedPrims);

	primitives.swap(orderedPrims);
	orderedPrims.clear();
	buildData.clear();

	/* add background lights to the primitives array */
	for(int i = 0; i < background_lights.size(); ++i){
		primitives.push_back(background_lights[i]);
	}

	/* add distant lights to the end of primitives array */
	for(int i = 0; i < distant_lights.size(); ++i){
		primitives.push_back(distant_lights[i]);
	}

	VLOG(1) << "Total BVH nodes: " << totalNodes;

	if(!root) return;

	/* convert to linear representation of the tree */
	nodes.resize(totalNodes);
	int offset = 0;
	flattenBVHTree(*root, offset);

	assert(offset == totalNodes);
}

int LightTree::flattenBVHTree(const BVHBuildNode &node, int &offset){

	CompactNode& compactNode = nodes[offset];
	compactNode.bounds_w = node.bbox;
	compactNode.bounds_o = node.bcone;

	int myOffset = offset++;
	if (node.nPrimitives > 0){

		assert( !node.children[0] && !node.children[1] );

		compactNode.energy = node.energy;
		compactNode.prim_id = node.firstPrimOffset;
		compactNode.nemitters = node.nPrimitives;
	} else {

		/* create interior compact node */
		compactNode.nemitters = 0;
		assert( node.children[0] && node.children[1] );
		flattenBVHTree(*node.children[0], offset);
		compactNode.secondChildOffset = flattenBVHTree(*node.children[1],
		        offset);
		compactNode.energy = node.energy;
	}

	return myOffset;
}


BoundBox LightTree::get_bbox(const Primitive& prim)
{
	BoundBox bbox = BoundBox::empty;
	if( prim.prim_id >= 0 ){
		/* extract bounding box from emissive triangle */
		const Object* object = objects[prim.object_id];
		const Mesh* mesh = object->mesh;
		const int triangle_id = prim.prim_id - mesh->tri_offset;
		const Mesh::Triangle triangle = mesh->get_triangle(triangle_id);
		const float3 *vpos = &mesh->verts[0];
		triangle.bounds_grow(vpos, bbox);

	} else {
		/* extract bounding box from lamp based on light type */
		Light* lamp = lights[prim.lamp_id];

		if (lamp->type == LIGHT_POINT || lamp->type == LIGHT_SPOT){
			float radius = lamp->size;
			bbox.grow(lamp->co + make_float3(radius));
			bbox.grow(lamp->co - make_float3(radius));
		} else if(lamp->type == LIGHT_AREA){
			/*     p2--------p3
			*    /         /
			*   /         /
			*  p0--------p1
			*/
			const float3& p0 = lamp->co;
			const float3 axisu = lamp->axisu*(lamp->sizeu*lamp->size);
			const float3 axisv = lamp->axisv*(lamp->sizev*lamp->size);
			const float3 p1 = p0 + axisu;
			const float3 p2 = p0 + axisv;
			const float3 p3 = p0 + axisu + axisv;
			bbox.grow(p0);
			bbox.grow(p1);
			bbox.grow(p2);
			bbox.grow(p3);
		} else {
			assert(false);
		}
	}
	return bbox;
}

Orientation LightTree::get_bcone(const Primitive& prim){
	Orientation bcone;
	if (prim.prim_id >= 0){
		/* extract bounding cone from emissive triangle */
		const Object* object = objects[prim.object_id];
		const Mesh* mesh = object->mesh;
		const int triangle_id = prim.prim_id - mesh->tri_offset;
		const Mesh::Triangle triangle = mesh->get_triangle(triangle_id);
		const float3 *vpos = &mesh->verts[0];
		bcone.axis = triangle.compute_normal(vpos);
		bcone.theta_o = 0.0f;
		bcone.theta_e = M_PI_2_F;
	} else {
		Light* lamp = lights[prim.lamp_id];
		bcone.axis = lamp->dir / len(lamp->dir);
		if (lamp->type == LIGHT_POINT) {
			bcone.theta_o = M_PI_F;
			bcone.theta_e = M_PI_2_F;
		} else if (lamp->type == LIGHT_SPOT){
			bcone.theta_o = 0;
			bcone.theta_e = lamp->spot_angle * 0.5f;
		} else if (lamp->type == LIGHT_AREA){
			bcone.theta_o = 0;
			bcone.theta_e = M_PI_2_F;
		}

	}

	return bcone;

}

float LightTree::get_energy(const Primitive &prim){
	float3 emission = make_float3(0.0f);
	Shader *shader = NULL;

	if (prim.prim_id >= 0){
		/* extract shader from emissive triangle */
		const Object* object = objects[prim.object_id];
		const Mesh* mesh = object->mesh;
		const int triangle_id = prim.prim_id - mesh->tri_offset;

		int shader_index = mesh->shader[triangle_id];
		shader = mesh->used_shaders.at(shader_index);

		/* get emission from shader */
		bool is_constant_emission = shader->is_constant_emission(&emission);
		if(!is_constant_emission){
			return 0.0f;
		}

		const Transform& tfm = objects[prim.object_id]->tfm;
		float area = mesh->compute_triangle_area(triangle_id, tfm);

		emission *= area * M_PI_F;

	} else {
		const Light* light = lights[prim.lamp_id];

		/* get emission from shader */
		shader = light->shader;
		bool is_constant_emission = shader->is_constant_emission(&emission);
		if(!is_constant_emission){
			return 0.0f;
		}

		/* calculate the total emission by integrating the emission over the
		 * the entire sphere of directions. */
		if (light->type == LIGHT_POINT){
			emission *= M_4PI_F;
		} else if (light->type == LIGHT_SPOT){
			/* The emission is only non-zero within the cone and if spot_smooth
			 * is non-zero there will be a falloff. In this case, approximate
			 * the integral by considering a smaller cone without falloff. */
			float spot_angle = light->spot_angle * 0.5f;
			float spot_falloff_angle = spot_angle * (1.0f - light->spot_smooth);
			float spot_middle_angle = (spot_angle + spot_falloff_angle) * 0.5f;
			emission *= M_2PI_F * (1.0f - cosf(spot_middle_angle));
		} else if (light->type == LIGHT_AREA){
			float3 axisu = light->axisu*(light->sizeu*light->size);
			float3 axisv = light->axisv*(light->sizev*light->size);
			float area = len(axisu)*len(axisv);
			emission *= area * M_PI_F;
		} else {
			// TODO: All light types are not supported yet
			assert(false);
		}
	}

	return rgb_to_luminance(emission);

}

Orientation LightTree::aggregate_bounding_cones(
        const vector<Orientation> &bcones) {

	if(bcones.size() == 1){
		return bcones[0];
	}

	/* use average of all axes as axis for now */
	Orientation bcone;
	for(unsigned int i = 0; i < bcones.size(); ++i){
		bcone.axis += bcones[i].axis;
	}

	const float length = len(bcone.axis);
	if (length == 0){
		bcone.axis = make_float3(0.0f, 0.0f, 0.0f); // NOTE: 0.0, 0.0, 0.0 here for now
	} else {
		bcone.axis /= length;
	}

	float max_theta_o = 0.0f;
	float max_theta_e = 0.0f;
	for(unsigned int i = 0; i < bcones.size(); ++i){
		float theta = acosf(dot(bcone.axis, bcones[i].axis));
		float theta_o = min(theta + bcones[i].theta_o, M_PI_F);
		float theta_e_full = theta_o + bcones[i].theta_e;
		if (theta_o > max_theta_o) {
			max_theta_o = theta_o;
		}
		if (theta_e_full > max_theta_e) {
			max_theta_e = theta_e_full;
		}
	}

	max_theta_e -= max_theta_o;
	bcone.theta_o = max_theta_o;
	bcone.theta_e = max_theta_e;

	return bcone;
}

float LightTree::calculate_cone_measure(const Orientation &bcone) {
	// http://www.wolframalpha.com/input/?i=integrate+cos(w-x)sin(w)dw+from+x+to+x%2By

	return M_2PI_F * (1.0f-cosf(bcone.theta_o) +
	                  0.5f * bcone.theta_e * sinf(bcone.theta_o) +
	                  0.25f * cosf(bcone.theta_o) -
	                  0.25f * cosf(bcone.theta_o + 2.0f*bcone.theta_e ));
}

void LightTree::split_saoh(const BoundBox &centroidBbox,
                           const vector<BVHPrimitiveInfo> &buildData,
                           const int start, const int end, const int nBuckets,
                           const float node_energy, const float node_M_Omega,
                           const BoundBox &node_bbox,
                           float &min_cost, int &min_dim, int &min_bucket){

	struct BucketInfo {
		BucketInfo(): count(0), energy(0.0f){
			bounds = BoundBox::empty;
		}

		int count;
		float energy; // total energy
		BoundBox bounds; // bounds of all primitives
		Orientation bcone;
	};

	min_cost = -1;
	min_cost = std::numeric_limits<float>::max();
	min_bucket = -1;

	for (int dim = 0; dim < 3; ++dim){

		BucketInfo buckets[nBuckets];
		vector<Orientation> bucketBcones[nBuckets];

		/* calculate total energy in each bucket and a bbox of it */
		const float extent = centroidBbox.max[dim] - centroidBbox.min[dim];
		if (extent == 0.0f){ // All dims cannot be zero
			continue;
		}

		const float invExtent = 1.0f / extent;
		for (unsigned int i = start; i < end; ++i)
		{
			int bucket_id = (int)((float)nBuckets *
			                      (buildData[i].centroid[dim] - centroidBbox.min[dim]) *
			                      invExtent);
			if (bucket_id == nBuckets) bucket_id = nBuckets - 1;
			buckets[bucket_id].count++;
			buckets[bucket_id].energy += buildData[i].energy;
			buckets[bucket_id].bounds.grow(buildData[i].bbox);
			bucketBcones[bucket_id].push_back(buildData[i].bcone);
		}

		for(unsigned int i = 0; i < nBuckets; ++i){
			if (buckets[i].count != 0){
				buckets[i].bcone = aggregate_bounding_cones(bucketBcones[i]);
			}
		}

		/* compute costs for splitting at bucket boundaries */
		float cost[nBuckets-1];
		BoundBox bbox_L,bbox_R;
		float energy_L, energy_R;
		vector<Orientation> bcones_L, bcones_R;

		for (int i = 0; i < nBuckets-1; ++i) {
			bbox_L = BoundBox::empty;
			bbox_R = BoundBox::empty;
			energy_L = 0;
			energy_R = 0;
			bcones_L.clear();
			bcones_R.clear();

			for (int j = 0; j <= i; ++j){
				if (buckets[j].count != 0){
					energy_L += buckets[j].energy;
					bbox_L.grow(buckets[j].bounds);
					bcones_L.push_back(buckets[j].bcone);
				}
			}

			for (int j = i+1; j < nBuckets; ++j){
				if (buckets[j].count != 0){
					energy_R += buckets[j].energy;
					bbox_R.grow(buckets[j].bounds);
					bcones_R.push_back(buckets[j].bcone);
				}
			}

			Orientation bcone_L = aggregate_bounding_cones(bcones_L);
			Orientation bcone_R = aggregate_bounding_cones(bcones_R);
			float M_Omega_L = calculate_cone_measure(bcone_L);
			float M_Omega_R = calculate_cone_measure(bcone_R);

			cost[i] = (energy_L*M_Omega_L*bbox_L.area() +
			           energy_R*M_Omega_R*bbox_R.area()) /
			        (node_energy*node_M_Omega*node_bbox.area());

		}

		/* update minimum cost, dim and bucket */
		for (int i = 0; i < nBuckets-1; ++i){
			if (cost[i] < min_cost){
				min_cost = cost[i];
				min_dim = dim;
				min_bucket = i;
			}
		}
	}
}

BVHBuildNode* LightTree::recursive_build(const unsigned int start,
                                         const unsigned int end,
                                         vector<BVHPrimitiveInfo> &buildData,
                                         unsigned int &totalNodes,
                                         vector<Primitive> &orderedPrims)
{
	if(buildData.size() == 0) return NULL;

	totalNodes++;
	BVHBuildNode *node = new BVHBuildNode();

	/* compute bounds for emissive primitives in node */
	BoundBox node_bbox = BoundBox::empty;
	vector<Orientation> bcones;
	bcones.reserve(end-start);
	float node_energy = 0.0f;
	for (unsigned int i = start; i < end; ++i){
		node_bbox.grow(buildData[i].bbox);
		bcones.push_back(buildData[i].bcone);
		node_energy += buildData[i].energy;
	}

	Orientation node_bcone = aggregate_bounding_cones(bcones);
	bcones.clear();
	const float node_M_Omega = calculate_cone_measure(node_bcone);

	assert(end >= start);
	unsigned int nPrimitives = end - start;
	if(nPrimitives == 1){
		/* create leaf */
		int firstPrimOffset = orderedPrims.size();
		int prim = buildData[start].primitiveNumber;
		orderedPrims.push_back(primitives[prim]);

		node->init_leaf(firstPrimOffset, nPrimitives, node_bbox, node_bcone,
		                node_energy);
		return node;
	} else {
		/* compute bounds for primitive centroids */
		BoundBox centroidBbox = BoundBox::empty;
		for (unsigned int i = start; i < end; ++i){
			centroidBbox.grow(buildData[i].centroid);
		}

		float3 diag = centroidBbox.size();
		int maxDim;
		if(diag[0] > diag[1] && diag[0] > diag[2]){
			maxDim = 0;
		} else if ( diag[1] > diag[2] ) {
			maxDim = 1;
		} else {
			maxDim = 2;
		}

		/* checks special case if all lights are in the same place */
		if (centroidBbox.max[maxDim] == centroidBbox.min[maxDim]){
			/* create leaf */
			int firstPrimOffset = orderedPrims.size();
			for (int i = start; i < end; ++i) {
				int prim = buildData[i].primitiveNumber;
				orderedPrims.push_back(primitives[prim]);
			}

			node->init_leaf(firstPrimOffset, nPrimitives, node_bbox, node_bcone,
			                node_energy);

			return node;
		} else {

			/* find dimension and bucket with smallest SAOH cost */
			const int nBuckets = 12;
			float min_cost;
			int min_dim, min_bucket;
			split_saoh(centroidBbox, buildData, start, end, nBuckets,
			           node_energy, node_M_Omega, node_bbox,
			           min_cost, min_dim, min_bucket);
			assert(total_min_dim != -1);

			int mid = 0;
			if (nPrimitives > maxPrimsInNode || min_cost < nPrimitives){
				/* partition primitives */
				BVHPrimitiveInfo *midPtr =
				        std::partition(&buildData[start], &buildData[end-1]+1,
				        CompareToBucket(min_bucket, nBuckets,
				                        min_dim, centroidBbox));
				mid = midPtr - &buildData[0];
			} else {
				/* create leaf */
				int firstPrimOffset = orderedPrims.size();
				for (int i = start; i < end; ++i) {
					int prim = buildData[i].primitiveNumber;
					orderedPrims.push_back(primitives[prim]);
				}

				node->init_leaf(firstPrimOffset, nPrimitives, node_bbox,
				                node_bcone, node_energy);
				return node;
			}

			/* build children */
			/* the order of execution of arguments is platform dependent so
			 * force a depth first going down left child first like this. */
			BVHBuildNode *left = recursive_build( start, mid, buildData,
			                                      totalNodes, orderedPrims );
			BVHBuildNode *right = recursive_build( mid, end, buildData,
			                                       totalNodes, orderedPrims);
			node->init_interior( min_dim, left, right, node_bcone, node_energy);
		}
	}

	return node;
}


CCL_NAMESPACE_END
