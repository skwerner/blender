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

    // Move all primitives into local primitives array
    primitives.reserve(prims_.size());
    foreach(Primitive prim, prims_ ){
        primitives.push_back(prim);
    }

    // Initialize buildData array
    vector<BVHPrimitiveInfo> buildData;
    buildData.reserve(primitives.size());
    for(int i = 0; i < primitives.size(); i++){
        BoundBox bbox = get_bbox(primitives[i]);
        buildData.push_back(BVHPrimitiveInfo(i, bbox));
    }

    // Recursively build BVH tree
    unsigned int totalNodes = 0;
    vector<Primitive> orderedPrims;
    orderedPrims.reserve(primitives.size());
    BVHBuildNode *root = recursive_build(buildData, 0, primitives.size(),
                                         &totalNodes, orderedPrims);
    primitives.swap(orderedPrims);
    orderedPrims.clear();
    buildData.clear();

    VLOG(1) << "Total BVH nodes: " << totalNodes;

    // Convert to linear representation of the tree
    nodes.resize(totalNodes);
    int offset = 0;
    flattenBVHTree(root, &offset);

    assert(offset == totalNodes);
}

int LightTree::flattenBVHTree(BVHBuildNode *node, int *offset){

    CompactNode& compactNode = nodes[*offset];
    compactNode.bounds_w = node->bbox;

    int myOffset = (*offset)++;
    if (node->nPrimitives > 0){

        assert( !node->children[0] && !node->children[1] );

        compactNode.energy = 1.0f; // TODO: Figure out what to put here
        compactNode.prim_id = node->firstPrimOffset;
        compactNode.nemitters = node->nPrimitives;

        /* TODO: Make general. This is specific for point lights and does not
         * consider several light sources within same node */
        compactNode.bounds_o.axis = make_float3(1.0f,0.0f,0.0f);
        compactNode.bounds_o.theta_o = (float)M_PI;
        compactNode.bounds_o.theta_e = (float)(M_PI/2);
    } else {

        /* create interior compact node */
        compactNode.nemitters = 0;
        assert( node->children[0] && node->children[1] );
        flattenBVHTree(node->children[0], offset);
        compactNode.secondChildOffset = flattenBVHTree(node->children[1],
                                                       offset);
        compactNode.energy = 1.0f; // TODO: Figure out what to put here

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
        assert(prim.object_id == -1);
        int lamp_id = -prim.prim_id-1;
        Light* lamp = lights[lamp_id];

        /* TODO: Handle all possible light sources here. */
        bbox.grow(lamp->co);
    }

    return bbox;
}

BVHBuildNode* LightTree::recursive_build(vector<BVHPrimitiveInfo> &buildData,
                                         unsigned int start,
                                         unsigned int end,
                                         unsigned int *totalNodes,
                                         vector<Primitive> &orderedPrims)
{
    (*totalNodes)++;
    BVHBuildNode *node = new BVHBuildNode();

    /* compute bounds for emissive primitives in node */
    BoundBox bbox = BoundBox::empty;
    for (unsigned int i = start; i < end; ++i){
        bbox.grow(buildData[i].bbox);
    }

    assert(end >= start);
    unsigned int nPrimitives = end - start;
    if(nPrimitives <= maxPrimsInNode){
        /* create leaf */
        int firstPrimOffset = orderedPrims.size();
        for (int i = start; i < end; ++i){
            int prim = buildData[i].primitiveNumber;
            orderedPrims.push_back(primitives[prim]);
        }

        node->init_leaf(firstPrimOffset, nPrimitives, bbox);
        return node;
    } else {
        /* compute bounds for primitive centroids and choose split dimension */
        BoundBox centroidBbox = BoundBox::empty;
        for (unsigned int i = start; i < end; ++i){
            centroidBbox.grow(buildData[i].centroid);
        }
        float3 diag = centroidBbox.size();
        int dim;
        if(diag[0] > diag[1] && diag[0] > diag[2]){
            dim = 0;
        } else if ( diag[1] > diag[2] ) {
            dim = 1;
        } else {
            dim = 2;
        }

        /* checks special case if all lights are in the same place */
        if (centroidBbox.max[dim] == centroidBbox.min[dim]){
            /* create leaf */
            int firstPrimOffset = orderedPrims.size();
            for (int i = start; i < end; ++i) {
                int prim = buildData[i].primitiveNumber;
                orderedPrims.push_back(primitives[prim]);
            }

            node->init_leaf(firstPrimOffset, nPrimitives, bbox);
            return node;
        } else {

            /* partition primitives based on split method: simplest possible */

            // -------------- SPLITTING CODE -----------------------------------
            float pmid = 0.5f * (centroidBbox.max[dim] + centroidBbox.min[dim]);
            BVHPrimitiveInfo *midPtr = std::partition(&buildData[start],
                                                      &buildData[end-1]+1,
                                                      CompareToMid(dim,pmid));
            int mid = midPtr - &buildData[0];
            // -----------------------------------------------------------------

            /* build children */
            node->init_interior( dim,
                                 recursive_build( buildData, start, mid,
                                                  totalNodes, orderedPrims ),
                                 recursive_build( buildData, mid, end,
                                                  totalNodes, orderedPrims));
        }
    }

    return node;
}


CCL_NAMESPACE_END
