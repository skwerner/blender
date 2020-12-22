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

#include "render/light_tree.h"
#include "render/light.h"
#include "render/mesh.h"
#include "render/object.h"

#include "util/util_foreach.h"
#include "util/util_logging.h"

CCL_NAMESPACE_BEGIN

LightTree::LightTree(const vector<Primitive> &prims_,
                     Scene *scene_,
                     const uint max_lights_in_node_)
    : max_lights_in_node(max_lights_in_node_), scene(scene_)
{

  if (prims_.empty())
    return;

  /* background and distant lights are not added to the light tree and are
   * considered seperately. so here all primitives except background and
   * distant lights are moved into a local primitives array */
  primitives.reserve(prims_.size());
  vector<Primitive> distant_lights;
  vector<Primitive> background_lights;
  foreach (Primitive prim, prims_) {

    /* put background and distant lights into their own arrays */
    if (prim.prim_id < 0) {
      const Light *lamp = scene->lights[prim.lamp_id];
      if (lamp->get_light_type() == LIGHT_DISTANT) {
        distant_lights.push_back(prim);
        continue;
      }
      else if (lamp->get_light_type() == LIGHT_BACKGROUND) {
        background_lights.push_back(prim);
        continue;
      }
    }

    primitives.push_back(prim);
  }

  /* initialize build_data array that stores the energy and spatial and
   * orientation bounds for each light. */
  vector<BVHPrimitiveInfo> build_data;
  build_data.reserve(primitives.size());
  for (int i = 0; i < primitives.size(); ++i) {
    BoundBox bbox = compute_bbox(primitives[i]);
    Orientation bcone = compute_bcone(primitives[i]);
    float energy = compute_energy(primitives[i]);

    build_data.push_back(BVHPrimitiveInfo(i, bbox, bcone, energy));
  }

  /* recursively build BVH tree */
  uint total_nodes = 0;
  vector<Primitive> ordered_prims;
  ordered_prims.reserve(primitives.size());
  BVHBuildNode *root = recursive_build(
      0, primitives.size(), build_data, total_nodes, ordered_prims);

  /* order the primitives array so lights belonging to the same node are
   * next to each other */
  primitives.swap(ordered_prims);
  ordered_prims.clear();
  build_data.clear();

  /* add background lights to the primitives array */
  for (int i = 0; i < background_lights.size(); ++i) {
    primitives.push_back(background_lights[i]);
  }

  /* add distant lights to the end of primitives array */
  for (int i = 0; i < distant_lights.size(); ++i) {
    primitives.push_back(distant_lights[i]);
  }

  VLOG(1) << "Total BVH nodes: " << total_nodes;

  if (!root)
    return;

  /* convert to linear representation of the tree */
  nodes.resize(total_nodes);
  int offset = 0;
  flattenBVHTree(*root, offset);

  assert(offset == total_nodes);
}

int LightTree::flattenBVHTree(const BVHBuildNode &node, int &offset)
{

  CompactNode &compact_node = nodes[offset];
  compact_node.bounds_s = node.bbox;
  compact_node.bounds_o = node.bcone;

  int my_offset = offset++;
  if (node.is_leaf) {
    /* create leaf node */
    assert(!node.children[0] && !node.children[1]);
    compact_node.energy = node.energy;
    compact_node.energy_variance = node.energy_variance;
    compact_node.first_prim_offset = node.first_prim_offset;
    compact_node.num_lights = node.num_lights;
  }
  else {
    /* create interior compact node */
    compact_node.num_lights = node.num_lights;
    compact_node.energy = node.energy;
    compact_node.energy_variance = node.energy_variance;
    assert(node.children[0] && node.children[1]);
    flattenBVHTree(*node.children[0], offset);
    compact_node.right_child_offset = flattenBVHTree(*node.children[1], offset);
    compact_node.energy = node.energy;
  }

  return my_offset;
}

BoundBox LightTree::compute_bbox(const Primitive &prim)
{
  BoundBox bbox = BoundBox::empty;
  if (prim.prim_id >= 0) {
    /* extract bounding box from emissive triangle */
    const Object *object = scene->objects[prim.object_id];
    const Mesh *mesh = (Mesh *)object->get_geometry();
    const int triangle_id = prim.prim_id - mesh->prim_offset;
    const Mesh::Triangle triangle = mesh->get_triangle(triangle_id);

    const array<float3> &verts = mesh->get_verts();
    float3 p0 = verts[triangle.v[0]];
    float3 p1 = verts[triangle.v[1]];
    float3 p2 = verts[triangle.v[2]];

    /* instanced mesh lights have not applied their transform at this point.
     * in this case, these points have to be transformed to get the proper
     * spatial bound. */
    if (!mesh->transform_applied) {
      const Transform &tfm = object->get_tfm();
      p0 = transform_point(&tfm, p0);
      p1 = transform_point(&tfm, p1);
      p2 = transform_point(&tfm, p2);
    }

    bbox.grow(p0);
    bbox.grow(p1);
    bbox.grow(p2);
  }
  else {
    /* extract bounding box from lamp based on light type */
    Light *lamp = scene->lights[prim.lamp_id];
    if (lamp->get_light_type() == LIGHT_POINT || lamp->get_light_type() == LIGHT_SPOT) {
      float radius = lamp->get_size();
      bbox.grow(lamp->get_co() + make_float3(radius));
      bbox.grow(lamp->get_co() - make_float3(radius));
    }
    else if (lamp->get_light_type() == LIGHT_AREA) {
      const float3 center = lamp->get_co();
      const float3 half_axisu = 0.5f * lamp->get_axisu() * (lamp->get_sizeu() * lamp->get_size());
      const float3 half_axisv = 0.5f * lamp->get_axisv() * (lamp->get_sizev() * lamp->get_size());
      const float3 p0 = center - half_axisu - half_axisv;
      const float3 p1 = center - half_axisu + half_axisv;
      const float3 p2 = center + half_axisu - half_axisv;
      const float3 p3 = center + half_axisu + half_axisv;

      bbox.grow(p0);
      bbox.grow(p1);
      bbox.grow(p2);
      bbox.grow(p3);
    }
    else {
      /* LIGHT_DISTANT and LIGHT_BACKGROUND are handled separately */
      assert(false);
    }
  }

  return bbox;
}

Orientation LightTree::compute_bcone(const Primitive &prim)
{
  Orientation bcone;
  if (prim.prim_id >= 0) {
    /* extract bounding cone from emissive triangle */
    const Object *object = scene->objects[prim.object_id];
    const Mesh *mesh = (Mesh *)object->get_geometry();
    const int triangle_id = prim.prim_id - mesh->prim_offset;
    const Mesh::Triangle triangle = mesh->get_triangle(triangle_id);

    const array<float3> &verts = mesh->get_verts();
    
    float3 p0 = verts[triangle.v[0]];
    float3 p1 = verts[triangle.v[1]];
    float3 p2 = verts[triangle.v[2]];

    if (!mesh->transform_applied) {
      const Transform &tfm = object->get_tfm();
      p0 = transform_point(&tfm, p0);
      p1 = transform_point(&tfm, p1);
      p2 = transform_point(&tfm, p2);
    }

    float3 normal = make_float3(1.0f, 0.0f, 0.0f);
    const float3 norm = cross(p1 - p0, p2 - p0);
    const float normlen = len(norm);
    if (normlen != 0.0f) {
      normal = norm / normlen;
    }

    bcone.axis = normal;
    bcone.theta_o = 0.0f;
    bcone.theta_e = M_PI_2_F;
  }
  else {
    Light *lamp = scene->lights[prim.lamp_id];
    bcone.axis = lamp->get_dir() / len(lamp->get_dir());
    if (lamp->get_light_type() == LIGHT_POINT) {
      bcone.theta_o = M_PI_F;
      bcone.theta_e = M_PI_2_F;
    }
    else if (lamp->get_light_type() == LIGHT_SPOT) {
      bcone.theta_o = 0;
      bcone.theta_e = lamp->get_spot_angle() * 0.5f;
    }
    else if (lamp->get_light_type() == LIGHT_AREA) {
      bcone.theta_o = 0;
      bcone.theta_e = M_PI_2_F;
    }
  }

  return bcone;
}

float LightTree::compute_energy(const Primitive &prim)
{
  float3 emission = make_float3(0.0f);
  Shader *shader = NULL;

  if (prim.prim_id >= 0) {
    /* extract shader from emissive triangle */
    const Object *object = scene->objects[prim.object_id];
    const Mesh *mesh = (Mesh *)object->get_geometry();
    const int triangle_id = prim.prim_id - mesh->prim_offset;

    int shader_index = mesh->get_shader()[triangle_id];
    shader = static_cast<Shader *>(mesh->get_used_shaders()[shader_index]);

    /* get emission from shader */
    bool is_constant_emission = shader->is_constant_emission(&emission);
    if (!is_constant_emission) {
      emission = make_float3(1.0f);
    }

    const Transform &tfm = scene->objects[prim.object_id]->get_tfm();
    float area = mesh->compute_triangle_area(triangle_id, tfm);

    emission *= area * 4;
  }
  else {
    const Light *light = scene->lights[prim.lamp_id];

    emission = light->get_strength();

    /* calculate the max emission in a single direction. */
    if (light->get_light_type() == LIGHT_POINT) {
      emission /= M_PI_F;
    }
    else if (light->get_light_type() == LIGHT_SPOT) {
      emission /= M_PI_F;
    }
    else if (light->get_light_type() == LIGHT_AREA) {
    }
    else {
      /* LIGHT_DISTANT and LIGHT_BACKGROUND are handled separately */
      assert(false);
    }
  }

  return scene->shader_manager->linear_rgb_to_gray(emission);
}

Orientation LightTree::combine_bounding_cones(const vector<Orientation> &bcones)
{

  if (bcones.size() == 0) {
    return Orientation();
  }
  else if (bcones.size() == 1) {
    return bcones[0];
  }

  Orientation cone = bcones[0];
  for (int i = 1; i < bcones.size(); ++i) {
    cone = cone_union(cone, bcones[i]);
  }

  return cone;
}

/* Algorithm 1 */
Orientation LightTree::cone_union(const Orientation &cone1, const Orientation &cone2)
{
  const Orientation *a = &cone1;
  const Orientation *b = &cone2;
  if (b->theta_o > a->theta_o) {
    a = &cone2;
    b = &cone1;
  }

  float theta_d = safe_acosf(dot(a->axis, b->axis));

  float theta_e = fmaxf(a->theta_e, b->theta_e);
  if (fminf(theta_d + b->theta_o, M_PI_F) <= a->theta_o) {
    return Orientation(a->axis, a->theta_o, theta_e);
  }

  float theta_o = (a->theta_o + theta_d + b->theta_o) * 0.5f;
  if (M_PI_F <= theta_o) {
    return Orientation(a->axis, M_PI_F, theta_e);
  }

  float theta_r = theta_o - a->theta_o;
  float3 axis = rotate_around_axis(a->axis, cross(a->axis, b->axis), theta_r);
  axis = normalize(axis);
  return Orientation(axis, theta_o, theta_e);
}

float LightTree::calculate_cone_measure(const Orientation &bcone)
{
  /* eq. 1 */
  float theta_w = fminf(bcone.theta_o + bcone.theta_e, M_PI_F);
  return M_2PI_F *
         (1.0f - cosf(bcone.theta_o) + 0.5f * (theta_w - bcone.theta_o) * sinf(bcone.theta_o) +
          0.25f * cosf(bcone.theta_o) - 0.25f * cosf(bcone.theta_o - 2.0f * theta_w));
}

void LightTree::split_saoh(const BoundBox &centroid_bbox,
                           const vector<BVHPrimitiveInfo> &build_data,
                           const int start,
                           const int end,
                           const int num_buckets,
                           const float node_M_Omega,
                           const BoundBox &node_bbox,
                           float &min_cost,
                           int &min_dim,
                           int &min_bucket)
{

  struct BucketInfo {
    BucketInfo() : count(0), energy(0.0f)
    {
      bounds = BoundBox::empty;
    }

    int count;
    float energy;     // total energy
    BoundBox bounds;  // bounds of all primitives
    Orientation bcone;
  };

  min_cost = std::numeric_limits<float>::max();
  min_bucket = -1;

  const float extent_max = max3(centroid_bbox.size());
  for (int dim = 0; dim < 3; ++dim) {

    vector<BucketInfo> buckets(num_buckets);
    vector<vector<Orientation>> bucket_bcones(num_buckets);

    /* calculate total energy in each bucket and a bbox of it */
    const float extent = centroid_bbox.max[dim] - centroid_bbox.min[dim];
    if (extent == 0.0f) {  // All dims cannot be zero
      continue;
    }

    const float extent_inv = 1.0f / extent;
    for (unsigned int i = start; i < end; ++i) {
      int bucket_id = (int)((float)num_buckets *
                            (build_data[i].centroid[dim] - centroid_bbox.min[dim]) * extent_inv);
      if (bucket_id == num_buckets)
        bucket_id = num_buckets - 1;
      buckets[bucket_id].count++;
      buckets[bucket_id].energy += build_data[i].energy;
      buckets[bucket_id].bounds.grow(build_data[i].bbox);
      bucket_bcones[bucket_id].push_back(build_data[i].bcone);
    }

    for (unsigned int i = 0; i < num_buckets; ++i) {
      if (buckets[i].count != 0) {
        buckets[i].bcone = combine_bounding_cones(bucket_bcones[i]);
      }
    }

    /* compute costs for splitting at bucket boundaries */
    vector<float> cost(num_buckets - 1);
    BoundBox bbox_L, bbox_R;
    float energy_L, energy_R;
    vector<Orientation> bcones_L, bcones_R;

    for (int i = 0; i < num_buckets - 1; ++i) {
      bbox_L = BoundBox::empty;
      bbox_R = BoundBox::empty;
      energy_L = 0;
      energy_R = 0;
      bcones_L.clear();
      bcones_R.clear();

      /* L corresponds to all buckets up to and including i */
      for (int j = 0; j <= i; ++j) {
        if (buckets[j].count != 0) {
          energy_L += buckets[j].energy;
          bbox_L.grow(buckets[j].bounds);
          bcones_L.push_back(buckets[j].bcone);
        }
      }

      /* R corresponds to bucket i+1 and all after */
      for (int j = i + 1; j < num_buckets; ++j) {
        if (buckets[j].count != 0) {
          energy_R += buckets[j].energy;
          bbox_R.grow(buckets[j].bounds);
          bcones_R.push_back(buckets[j].bcone);
        }
      }

      /* eq. 2 */
      const Orientation bcone_L = combine_bounding_cones(bcones_L);
      const Orientation bcone_R = combine_bounding_cones(bcones_R);
      const float M_Omega_L = calculate_cone_measure(bcone_L);
      const float M_Omega_R = calculate_cone_measure(bcone_R);
      const float K = extent_max * extent_inv;

      cost[i] = K * (energy_L * M_Omega_L * bbox_L.area() + energy_R * M_Omega_R * bbox_R.area()) /
                (node_M_Omega * node_bbox.area());
    }

    /* update minimum cost, dim and bucket */
    for (int i = 0; i < num_buckets - 1; ++i) {
      if (cost[i] < min_cost) {
        min_cost = cost[i];
        min_dim = dim;
        min_bucket = i;
      }
    }
  }
}

BVHBuildNode *LightTree::recursive_build(const uint start,
                                         const uint end,
                                         vector<BVHPrimitiveInfo> &build_data,
                                         uint &total_nodes,
                                         vector<Primitive> &ordered_prims)
{
  if (build_data.size() == 0)
    return NULL;

  total_nodes++;
  BVHBuildNode *node = new BVHBuildNode();

  /* compute bounds and energy for all emissive primitives in node */
  BoundBox node_bbox = BoundBox::empty;
  vector<Orientation> bcones;
  bcones.reserve(end - start);
  double node_energy = 0.0;
  double node_energy_sum_squared = 0.0;
  uint num_lights = end - start;

  for (unsigned int i = start; i < end; ++i) {
    const BVHPrimitiveInfo &light = build_data.at(i);
    node_bbox.grow(light.bbox);
    bcones.push_back(light.bcone);

    double energy = (double)light.energy;
    node_energy += energy;
    node_energy_sum_squared += energy * energy;
  }

  /* pre-calculate energy variance for the splitting heuristic */
  double node_energy_mean = node_energy / (double)num_lights;
  double node_energy_variance = node_energy_sum_squared / (double)num_lights -  // E[e^2]
                                node_energy_mean * node_energy_mean;            // E[e]^2
  node_energy_variance = max(node_energy_variance, 0.0);

  Orientation node_bcone = combine_bounding_cones(bcones);
  bcones.clear();
  const float node_M_Omega = calculate_cone_measure(node_bcone);

  if (num_lights == 1) {
    /* create leaf */
    int first_prim_offset = ordered_prims.size();
    int prim = build_data.at(start).primitive_offset;
    ordered_prims.push_back(primitives.at(prim));

    node->init_leaf(
        first_prim_offset, num_lights, node_bbox, node_bcone, node_energy, node_energy_variance);
    return node;
  }
  else {
    /* compute spatial bound for primitive centroids */
    BoundBox centroid_bbox = BoundBox::empty;
    for (unsigned int i = start; i < end; ++i) {
      centroid_bbox.grow(build_data.at(i).centroid);
    }

    /* find dimension of bounding box with maximum extent */
    float3 diag = centroid_bbox.size();
    int max_dim;
    if (diag[0] > diag[1] && diag[0] > diag[2]) {
      max_dim = 0;
    }
    else if (diag[1] > diag[2]) {
      max_dim = 1;
    }
    else {
      max_dim = 2;
    }

    /* checks special case if all lights are in the same place */
    if (centroid_bbox.max[max_dim] == centroid_bbox.min[max_dim]) {
      /* create leaf */
      int first_prim_offset = ordered_prims.size();
      for (int i = start; i < end; ++i) {
        int prim = build_data.at(i).primitive_offset;
        ordered_prims.push_back(primitives.at(prim));
      }

      node->init_leaf(
          first_prim_offset, num_lights, node_bbox, node_bcone, node_energy, node_energy_variance);

      return node;
    }
    else {

      /* find dimension and bucket with smallest SAOH cost */
      const int num_buckets = 12;
      float min_cost;
      int min_dim, min_bucket;
      split_saoh(centroid_bbox,
                 build_data,
                 start,
                 end,
                 num_buckets,
                 node_M_Omega,
                 node_bbox,
                 min_cost,
                 min_dim,
                 min_bucket);
      assert(min_dim != -1);

      int mid = 0;
      if (num_lights > max_lights_in_node || min_cost < (float)node_energy) {
        /* partition primitives */
        BVHPrimitiveInfo *mid_ptr = std::partition(
            &build_data[start],
            &build_data[end - 1] + 1,
            CompareToBucket(min_bucket, num_buckets, min_dim, centroid_bbox));
        mid = mid_ptr - &build_data[0];
      }
      else {
        /* create leaf */
        int first_prim_offset = ordered_prims.size();
        for (int i = start; i < end; ++i) {
          int prim = build_data.at(i).primitive_offset;
          ordered_prims.push_back(primitives.at(prim));
        }

        node->init_leaf(first_prim_offset,
                        num_lights,
                        node_bbox,
                        node_bcone,
                        node_energy,
                        node_energy_variance);
        return node;
      }

      /* build children */
      BVHBuildNode *left = recursive_build(start, mid, build_data, total_nodes, ordered_prims);
      BVHBuildNode *right = recursive_build(mid, end, build_data, total_nodes, ordered_prims);
      node->init_interior(left, right, node_bcone, num_lights, node_energy, node_energy_variance);
    }
  }

  return node;
}

CCL_NAMESPACE_END
