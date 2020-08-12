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

#ifndef __LIGHT_TREE_H__
#define __LIGHT_TREE_H__

#include "util/util_boundbox.h"

CCL_NAMESPACE_BEGIN

class Light;
class Object;
class Scene;

#define LIGHT_TREE_NODE_SIZE 4

/* Data structure to represent orientation bounds. It consists of two bounding
 * cones represented by a direction(axis) and two angles out from this axis.
 * This can be thought of as two cones.
 */
struct Orientation {
  Orientation()
  {
    axis = make_float3(0.0f, 0.0f, 0.0f);
    theta_o = 0;
    theta_e = 0;
  }

  Orientation(const float3 &a, float o, float e) : axis(a), theta_o(o), theta_e(e)
  {
  }

  /* orientation/direction of the cones */
  float3 axis;

  /* angle bounding light orientations */
  float theta_o;

  /* angle bounding the directions light can be emitted in */
  float theta_e;
};

/* Temporary data structure for nodes during construction.
 * After the construction is complete a final step converts the tree consisting
 * of these nodes into a tree consisting of CompactNode:s. */
struct BVHBuildNode {

  BVHBuildNode()
  {
    children[0] = children[1] = NULL;
    bbox = BoundBox::empty;
  }

  /* initializes this node as a leaf node */
  void init_leaf(
      uint first, uint n, const BoundBox &b, const Orientation &c, double e, double e_var)
  {
    first_prim_offset = first;
    num_lights = n;
    bbox = b;
    bcone = c;
    energy = (float)e;
    energy_variance = (float)e_var;
    is_leaf = true;
  }

  /* initializes this node as an interior node */
  void init_interior(
      BVHBuildNode *c0, BVHBuildNode *c1, const Orientation &c, uint n, double e, double e_var)
  {
    bbox = merge(c0->bbox, c1->bbox);
    bcone = c;

    children[0] = c0;
    children[1] = c1;

    num_lights = n;
    energy = (float)e;
    energy_variance = (float)e_var;
    is_leaf = false;
  }

  /* spatial and orientation bounds */
  BoundBox bbox;
  Orientation bcone;

  /* total energy and energy variance for the lights in the node */
  float energy, energy_variance;

  /* pointers to the two children */
  BVHBuildNode *children[2];

  /* each leaf node contains one or more lights. lights that are contained in
   * the same node are stored next to each other in the ordered primitives
   * array. this offset points to the first of these lights. num_lights below
   * can be used to find the last light for this node */
  uint first_prim_offset;

  /* total number of lights contained in this node */
  uint num_lights;

  /* if this node is a leaf or not */
  bool is_leaf;
};

// TODO: Have this struct in kernel_types.h instead?
/* A more memory efficient representation of BVHBuildNode above. This is the
 * structure of the nodes on the device. */
struct CompactNode {

  CompactNode()
      : right_child_offset(-1),
        first_prim_offset(-1),
        num_lights(-1),
        bounds_s(BoundBox::empty),
        energy(0.0f),
        energy_variance(0.0f)
  {
    bounds_o.axis = make_float3(0.0f);
    bounds_o.theta_o = 0.0f;
    bounds_o.theta_e = 0.0f;
  }

  /* All compact nodes are stored in a single array. interior nodes can find
   * their two child nodes as follows:
   * - the left child node is directly after its parent in the nodes array
   * - the right child node is at the offset below in the nodes array.
   *
   * This offset is default initialized to -1 and will only change if this is
   * and interior node. this therefore used to see if a node is a leaf/interior
   * node as well. */
  int right_child_offset;

  /* see comment in BVHBuildNode for the variable with the same name */
  int first_prim_offset;

  /* total number of lights contained in this node */
  int num_lights;

  /* spatial and orientation bounds */
  BoundBox bounds_s;
  Orientation bounds_o;

  /* total energy and energy variance for the lights in the node */
  float energy, energy_variance;
};

/* Helper struct that is only used during the construction of the tree */
struct BVHPrimitiveInfo {

  BVHPrimitiveInfo()
  {
    bbox = BoundBox::empty;
  }

  BVHPrimitiveInfo(uint offset, const BoundBox &bounds, const Orientation &oBounds, float e)
      : primitive_offset(offset), bbox(bounds), centroid(bounds.center()), energy(e)
  {
    bcone.axis = oBounds.axis;
    bcone.theta_o = oBounds.theta_o;
    bcone.theta_e = oBounds.theta_e;
  }

  /* this primitives offset into the unordered primtives array */
  uint primitive_offset;

  /* spatial and orientation bounds */
  BoundBox bbox;
  float3 centroid;
  Orientation bcone;

  /* total energy of this emissive primitive */
  float energy;
};

/* A custom pointer struct that points to an emissive triangle or a lamp. */
class Primitive {
 public:
  /* If prim_id >= 0 then the primitive is a triangle and prim_id is a global
   * triangle index.
   * If prim_id < 0 then the primitive is a lamp and -prim_id-1 is an index
   * into the lights array on the device. */
  int prim_id;
  union {
    /* which object the triangle belongs to */
    int object_id;
    /* index for this lamp in the scene->lights array */
    int lamp_id;
  };
  Primitive(int prim, int object) : prim_id(prim), object_id(object)
  {
  }
};

/* Compare operator that returns true if the given light is in a lower
 * bucket than a given split_bucket. This is used to partition lights into lights
 * for the left and right child during tree construction. */
struct CompareToBucket {
  CompareToBucket(int split, int num, int d, const BoundBox &b) : centroid_bbox(b)
  {
    split_bucket = split;
    num_buckets = num;
    dim = d;
    inv_extent = 1.0f / (centroid_bbox.max[dim] - centroid_bbox.min[dim]);
  }

  bool operator()(const BVHPrimitiveInfo &p) const
  {
    int bucket_id = (int)((float)num_buckets * (p.centroid[dim] - centroid_bbox.min[dim]) *
                          inv_extent);
    if (bucket_id == num_buckets) {
      bucket_id = num_buckets - 1;
    }

    return bucket_id <= split_bucket;
  }

  /* everything lower or equal to the split_bucket is considered to be in one
   * child and everything above will be considered to belong to the other. */
  int split_bucket;

  /* the total number of buckets that are considered for this dimension(dim) */
  int num_buckets;

  /* the construction creates candidate splits along the three dimensions.
   * this variable stores which dimension is currently being split along.*/
  int dim;

  /* storing the inverse extend of the bounding box along the current
   * dimension to only have to do the division once instead of everytime the
   * operator() is called. */
  float inv_extent;

  /* bound for the centroids of all lights of the current node being split */
  const BoundBox &centroid_bbox;
};

/* This class takes a set of lights as input and organizes them into a light
 * hierarchy. This hierarchy is represented as a Bounding Volume Hierarchy(BVH).
 * This is the process to acheive this:
 * 1. For each given light, important information is gathered
 *    - Bounding box of the light
 *    - Bounding cones of the light
 *    - The energy of the light
 *    This first calculated and then stored as BVHPrimitiveInfo for each light.
 * 2. A top-down recursive build algorithm creates a BVH consisting of
 *    BVHBuildNode:s which each are allocated randomly on the heap with new.
 *    This step also reorders the given array of lights such that lights
 *    belonging to the same node are next to each other in the primitives array.
 * 3. A final step converts this BVH into a more memory efficient layout where
 *    each BVHBuildNode is converted to a CompactNode and all of these nodes
 *    are placed next to each other in memory in a single nodes array.
 *
 *  This structure is based on PBRTs geometry BVH implementation.
 **/
class LightTree {
 public:
  LightTree(const vector<Primitive> &prims_, Scene *scene_, const uint max_lights_in_node_);

  /* returns the ordered emissive primitives */
  const vector<Primitive> &get_primitives() const
  {
    return primitives;
  }

  /* returns the array of nodes */
  const vector<CompactNode> &get_nodes() const
  {
    return nodes;
  }

  /* computes the bounding box for the given light */
  BoundBox compute_bbox(const Primitive &prim);

  /* computes the orientation bounds for the given light. */
  Orientation compute_bcone(const Primitive &prim);

  /* computes the emitted energy for the given light. this is done by
   * integrating the constant emission over the angles of the sphere it emits
   * light in. */
  float compute_energy(const Primitive &prim);

 private:
  /* the top-down recursive build algorithm mentioned in step 2 above */
  BVHBuildNode *recursive_build(const uint start,
                                const uint end,
                                vector<BVHPrimitiveInfo> &build_data,
                                uint &total_nodes,
                                vector<Primitive> &ordered_prims);

  /* returns the union of two orientation bounds and returns the result */
  Orientation cone_union(const Orientation &a, const Orientation &b);

  /* returns the union of several orientation bounds */
  Orientation combine_bounding_cones(const vector<Orientation> &bcones);

  /* calculates the cone measure in the surface area orientation heuristic */
  float calculate_cone_measure(const Orientation &bcone);

  /* takes a node and the lights contained in it as input and returns a way to
   * split the node into two child nodes. This is done as follows:
   * 1. A bounding box of all lights centroid is constructed
   * 2. A set of candidate splits(proposed left and right child nodes) are
   *    created.
   *    - This is done by partitioning the bounding box into two regions.
   *      All lights in the same region belongs to the same child node. This
   *      is done for several partions of the bounding box.
   * 3. Each such candidate is evaluated using the Surface Area Orientation
   *    Heuristic(SAOH).
   * 4. The candidate split with the minimum cost(heuristic) is returned */
  void split_saoh(const BoundBox &centroid_bbox,
                  const vector<BVHPrimitiveInfo> &build_data,
                  const int start,
                  const int end,
                  const int num_buckets,
                  const float node_M_Omega,
                  const BoundBox &node_bbox,
                  float &min_cost,
                  int &min_dim,
                  int &min_bucket);

  /* this method performs step 3 above. */
  int flattenBVHTree(const BVHBuildNode &node, int &offset);

  /* contains all the lights in the scene. when the constructor has finished,
   * these will be ordered such that lights belonging to the same node will be
   * next to each other in this array. */
  vector<Primitive> primitives;

  /* the maximum number of allowed lights in each leaf node */
  uint max_lights_in_node;

  /* pointer to the scene. this is used to access the objects, the lights and
   * the shader manager of the scene.*/
  Scene *scene;

  /* the nodes of the light hierarchy */
  vector<CompactNode> nodes;
};

CCL_NAMESPACE_END

#endif  // __LIGHT_TREE_H__
