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

#include <stdio.h>

#include <algorithm>
#include <iterator>
#include <sstream>

#include "graph/node_xml.h"

#include "render/background.h"
#include "render/camera.h"
#include "render/film.h"
#include "render/graph.h"
#include "render/integrator.h"
#include "render/light.h"
#include "render/mesh.h"
#include "render/nodes.h"
#include "render/object.h"
#include "render/osl.h"
#include "render/scene.h"
#include "render/session.h"
#include "render/shader.h"

#include "subd/subd_patch.h"
#include "subd/subd_split.h"

#include "util/util_foreach.h"
#include "util/util_path.h"
#include "util/util_projection.h"
#include "util/util_transform.h"
#include "util/util_xml.h"

#include "app/cycles_xml.h"

CCL_NAMESPACE_BEGIN

#define CYCLES_XML_VERSION_MAJOR 1
#define CYCLES_XML_VERSION_MINOR 0

/* XML reading state */

struct XMLReadState : public XMLReader {
  Scene *scene;      /* scene pointer */
  Transform tfm;     /* current transform state */
  bool smooth;       /* smooth normal state */
  Shader *shader;    /* current shader */
  string base;       /* base path to current file*/
  float dicing_rate; /* current dicing rate */
  int version_major; /* File format version */
  int version_minor;

  XMLReadState()
      : scene(NULL),
        smooth(false),
        shader(NULL),
        dicing_rate(1.0f),
        version_major(CYCLES_XML_VERSION_MAJOR),
        version_minor(CYCLES_XML_VERSION_MINOR)
  {
    tfm = transform_identity();
  }
};

/* Attribute Reading */

static bool xml_read_int(int *value, xml_node node, const char *name)
{
  xml_attribute attr = node.attribute(name);

  if (attr) {
    *value = atoi(attr.value());
    return true;
  }

  return false;
}

static bool xml_read_int_array(vector<int> &value, xml_node node, const char *name)
{
  xml_attribute attr = node.attribute(name);

  if (attr) {
    vector<string> tokens;
    string_split(tokens, attr.value());

    foreach (const string &token, tokens)
      value.push_back(atoi(token.c_str()));

    return true;
  }

  return false;
}

static void xml_write_int_array(const array<int> &value, xml_node node, const char *name)
{
  xml_attribute attr = node.append_attribute(name);

  if (attr) {
    string tokens;
    foreach (int i, value) {
      tokens += std::to_string(i) + " ";
    }
    attr.set_value(tokens.c_str());
  }
}

static void xml_write_bool_array(const array<bool> &value, xml_node node, const char *name)
{
  xml_attribute attr = node.append_attribute(name);

  if (attr) {
    string tokens;
    foreach (bool b, value) {
      tokens += b ? "true " : "false ";
    }
    attr.set_value(tokens.c_str());
  }
}

static bool xml_read_float(float *value, xml_node node, const char *name)
{
  xml_attribute attr = node.attribute(name);

  if (attr) {
    *value = (float)atof(attr.value());
    return true;
  }

  return false;
}

static bool xml_read_float_array(vector<float> &value, xml_node node, const char *name)
{
  xml_attribute attr = node.attribute(name);

  if (attr) {
    vector<string> tokens;
    string_split(tokens, attr.value());

    foreach (const string &token, tokens)
      value.push_back((float)atof(token.c_str()));

    return true;
  }

  return false;
}

static void xml_write_float_array(const vector<float> &value, xml_node node, const char *name)
{
  std::string value_string;
  foreach (float f, value) {
    value_string += std::to_string(f) + " ";
  }
  node.append_attribute(name) = value_string.c_str();
}

static bool xml_read_float3(float3 *value, xml_node node, const char *name)
{
  vector<float> array;

  if (xml_read_float_array(array, node, name) && array.size() == 3) {
    *value = make_float3(array[0], array[1], array[2]);
    return true;
  }

  return false;
}

static bool xml_read_float3_array(vector<float3> &value, xml_node node, const char *name)
{
  vector<float> array;

  if (xml_read_float_array(array, node, name)) {
    for (size_t i = 0; i < array.size(); i += 3)
      value.push_back(make_float3(array[i + 0], array[i + 1], array[i + 2]));

    return true;
  }

  return false;
}

static bool xml_read_float4(float4 *value, xml_node node, const char *name)
{
  vector<float> array;

  if (xml_read_float_array(array, node, name) && array.size() == 4) {
    *value = make_float4(array[0], array[1], array[2], array[3]);
    return true;
  }

  return false;
}

static bool xml_read_string(string *str, xml_node node, const char *name)
{
  xml_attribute attr = node.attribute(name);

  if (attr) {
    *str = attr.value();
    return true;
  }

  return false;
}

static bool xml_equal_string(xml_node node, const char *name, const char *value)
{
  xml_attribute attr = node.attribute(name);

  if (attr)
    return string_iequals(attr.value(), value);

  return false;
}

static xml_node xml_write_transform(const Transform &tfm, xml_node node)
{
  xml_node tfm_node = node.append_child("Transform");
  vector<float> matrix;
  const float *tfm_f = (const float *)(&tfm);
  for (int i = 0; i < 12; ++i) {
    matrix.push_back(tfm_f[i]);
  }
  xml_write_float_array(matrix, tfm_node, "matrix");

  return tfm_node;
}

/* Camera */

static void xml_read_camera(XMLReadState &state, xml_node node)
{
  Camera *cam = state.scene->camera;

  int width = -1, height = -1;
  xml_read_int(&width, node, "width");
  xml_read_int(&height, node, "height");

  cam->set_full_width(width);
  cam->set_full_height(height);

  xml_read_node(state, cam, node);

  cam->set_matrix(state.tfm);

  cam->need_flags_update = true;
  cam->update(state.scene);
}

static void xml_write_camera(Camera *cam, xml_node node)
{
  xml_node transform = xml_write_transform(cam->get_matrix(), node);
  xml_write_node(cam, transform);
}

/* Shader */

static void xml_read_shader_graph(XMLReadState &state, Shader *shader, xml_node graph_node)
{
  xml_read_node(state, shader, graph_node);

  ShaderGraph *graph = new ShaderGraph();

  /* local state, shader nodes can't link to nodes outside the shader graph */
  XMLReader graph_reader;
  graph_reader.node_map[ustring("output")] = graph->output();

  for (xml_node node = graph_node.first_child(); node; node = node.next_sibling()) {
    ustring node_name(node.name());

    if (node_name == "connect") {
      /* connect nodes */
      ustring from_node_name(node.attribute("from").value());
      ustring from_socket_name(node.attribute("output").value());
      ustring to_node_name(node.attribute("to").value());
      ustring to_socket_name(node.attribute("input").value());

      /* find nodes and sockets */
      ShaderOutput *output = NULL;
      ShaderInput *input = NULL;

      if (graph_reader.node_map.find(from_node_name) != graph_reader.node_map.end()) {
        ShaderNode *fromnode = (ShaderNode *)graph_reader.node_map[from_node_name];

        foreach (ShaderOutput *out, fromnode->outputs)
          if (string_iequals(out->socket_type.name.string(), from_socket_name.string()))
            output = out;

        if (!output)
          fprintf(stderr,
                  "Unknown output socket name \"%s\" on \"%s\".\n",
                  from_node_name.c_str(),
                  from_socket_name.c_str());
      }
      else
        fprintf(stderr, "Unknown shader node name \"%s\".\n", from_node_name.c_str());

      if (graph_reader.node_map.find(to_node_name) != graph_reader.node_map.end()) {
        ShaderNode *tonode = (ShaderNode *)graph_reader.node_map[to_node_name];

        foreach (ShaderInput *in, tonode->inputs)
          if (string_iequals(in->socket_type.name.string(), to_socket_name.string()))
            input = in;

        if (!input)
          fprintf(stderr,
                  "Unknown input socket name \"%s\" on \"%s\".\n",
                  to_socket_name.c_str(),
                  to_node_name.c_str());
      }
      else
        fprintf(stderr, "Unknown shader node name \"%s\".\n", to_node_name.c_str());

      /* connect */
      if (output && input)
        graph->connect(output, input);

      continue;
    }

    ShaderNode *snode = NULL;

#ifdef WITH_OSL
    if (node_name == "osl_shader") {
      ShaderManager *manager = state.scene->shader_manager;

      if (manager->use_osl()) {
        std::string filepath;

        if (xml_read_string(&filepath, node, "src")) {
          if (path_is_relative(filepath)) {
            filepath = path_join(state.base, filepath);
          }

          snode = OSLShaderManager::osl_node(graph, manager, filepath, "");

          if (!snode) {
            fprintf(stderr, "Failed to create OSL node from \"%s\".\n", filepath.c_str());
            continue;
          }
        }
        else {
          fprintf(stderr, "OSL node missing \"src\" attribute.\n");
          continue;
        }
      }
      else {
        fprintf(stderr, "OSL node without using --shadingsys osl.\n");
        continue;
      }
    }
    else
#endif
    {
      /* exception for name collision */
      if (node_name == "background")
        node_name = "background_shader";

      const NodeType *node_type = NodeType::find(node_name);

      if (!node_type) {
        fprintf(stderr, "Unknown shader node \"%s\".\n", node.name());
        continue;
      }
      else if (node_type->type != NodeType::SHADER) {
        fprintf(stderr, "Node type \"%s\" is not a shader node.\n", node_type->name.c_str());
        continue;
      }
      else if (node_type->create == NULL) {
        fprintf(stderr, "Can't create abstract node type \"%s\".\n", node_type->name.c_str());
        continue;
      }

      snode = (ShaderNode *)node_type->create(node_type);
    }

    xml_read_node(graph_reader, snode, node);

    if (node_name == "image_texture") {
      ImageTextureNode *img = (ImageTextureNode *)snode;
      ustring filename(path_join(state.base, img->get_filename().string()));
      img->set_filename(filename);
    }
    else if (node_name == "environment_texture") {
      EnvironmentTextureNode *env = (EnvironmentTextureNode *)snode;
      ustring filename(path_join(state.base, env->get_filename().string()));
      env->set_filename(filename);
    }

    if (snode) {
      /* add to graph */
      graph->add(snode);
    }
  }

  shader->set_graph(graph);
  shader->tag_update(state.scene);
}

static void xml_write_shader_graph(const ShaderGraph *graph, xml_node node)
{
  /* Write the individual nodes first, then write the connections. */
  foreach (ShaderNode *n, graph->nodes) {
    /* Skip the output node, that one gets added by default. */
    if (n->name != "output") {
      xml_node shader_node = xml_write_node(n, node);
      /* Make sure the nodes have unique names. */
      shader_node.attribute("name") = ustring::format("%d.%s", n->id, n->name.c_str()).c_str();
    }
  }

  foreach (ShaderNode *n, graph->nodes) {
    foreach (ShaderInput *in, n->inputs) {
      if (in->link && in->link->parent) {
        xml_node connection = node.append_child("connect");
        connection.append_attribute("from") =
            ustring::format("%d.%s", in->link->parent->id, in->link->parent->name.c_str()).c_str();
        if (n->name == "output") {
          connection.append_attribute("to") = n->name.c_str();
        }
        else {
          connection.append_attribute(
              "to") = ustring::format("%d.%s", n->id, n->name.c_str()).c_str();
        }
        connection.append_attribute("input") = in->socket_type.name.c_str();
        connection.append_attribute("output") = in->link->socket_type.name.c_str();
      }
    }
  }
}

static void xml_read_shader(XMLReadState &state, xml_node node)
{
  Shader *shader = new Shader();
  xml_read_shader_graph(state, shader, node);
  state.scene->shaders.push_back(shader);
}

static void xml_write_shader(const Shader *shader, xml_node node)
{
  xml_node shader_node = xml_write_node(shader, node);
  if (shader->graph) {
    xml_write_shader_graph(shader->graph, shader_node);
  }
}

/* Background */

static void xml_read_background(XMLReadState &state, xml_node node)
{
  /* Background Settings */
  xml_read_node(state, state.scene->background, node);

  /* Background Shader */
  Shader *shader = state.scene->default_background;
  xml_read_shader_graph(state, shader, node);
}

static void xml_write_background(Background *background, xml_node node)
{
  /* Background Settings */
  xml_node background_node = xml_write_node(background, node);

  if (background->get_shader()) {
    /* Background Shader */
    xml_write_shader(background->get_shader(), background_node);
  }
}

/* Mesh */

static Mesh *xml_add_mesh(Scene *scene, const Transform &tfm)
{
  /* create mesh */
  Mesh *mesh = new Mesh();
  scene->geometry.push_back(mesh);

  /* create object*/
  Object *object = new Object();
  object->set_geometry(mesh);
  object->set_tfm(tfm);
  scene->objects.push_back(object);

  return mesh;
}

static void xml_read_mesh(const XMLReadState &state, xml_node node)
{
  /* add mesh */
  Mesh *mesh = xml_add_mesh(state.scene, state.tfm);
  array<Node *> used_shaders = mesh->get_used_shaders();
  used_shaders.push_back_slow(state.shader);
  mesh->set_used_shaders(used_shaders);

  /* read state */
  int shader = 0;
  bool smooth = state.smooth;

  /* read vertices and polygons */
  vector<float3> P;
  vector<float> UV;
  vector<int> verts, nverts;

  xml_read_float3_array(P, node, "P");
  xml_read_int_array(verts, node, "verts");
  xml_read_int_array(nverts, node, "nverts");

  if (xml_equal_string(node, "subdivision", "catmull-clark")) {
    mesh->set_subdivision_type(Mesh::SUBDIVISION_CATMULL_CLARK);
  }
  else if (xml_equal_string(node, "subdivision", "linear")) {
    mesh->set_subdivision_type(Mesh::SUBDIVISION_LINEAR);
  }

  array<float3> P_array;
  P_array = P;

  if (mesh->get_subdivision_type() == Mesh::SUBDIVISION_NONE) {
    /* create vertices */

    mesh->set_verts(P_array);

    size_t num_triangles = 0;
    for (size_t i = 0; i < nverts.size(); i++)
      num_triangles += nverts[i] - 2;
    mesh->reserve_mesh(mesh->get_verts().size(), num_triangles);

    /* create triangles */
    int index_offset = 0;

    for (size_t i = 0; i < nverts.size(); i++) {
      for (int j = 0; j < nverts[i] - 2; j++) {
        int v0 = verts[index_offset];
        int v1 = verts[index_offset + j + 1];
        int v2 = verts[index_offset + j + 2];

        assert(v0 < (int)P.size());
        assert(v1 < (int)P.size());
        assert(v2 < (int)P.size());

        mesh->add_triangle(v0, v1, v2, shader, smooth);
      }

      index_offset += nverts[i];
    }

    if (xml_read_float_array(UV, node, "UV")) {
      ustring name = ustring("UVMap");
      Attribute *attr = mesh->attributes.add(ATTR_STD_UV, name);
      float2 *fdata = attr->data_float2();

      /* loop over the triangles */
      index_offset = 0;
      for (size_t i = 0; i < nverts.size(); i++) {
        for (int j = 0; j < nverts[i] - 2; j++) {
          int v0 = index_offset;
          int v1 = index_offset + j + 1;
          int v2 = index_offset + j + 2;

          assert(v0 * 2 + 1 < (int)UV.size());
          assert(v1 * 2 + 1 < (int)UV.size());
          assert(v2 * 2 + 1 < (int)UV.size());

          fdata[0] = make_float2(UV[v0 * 2], UV[v0 * 2 + 1]);
          fdata[1] = make_float2(UV[v1 * 2], UV[v1 * 2 + 1]);
          fdata[2] = make_float2(UV[v2 * 2], UV[v2 * 2 + 1]);
          fdata += 3;
        }

        index_offset += nverts[i];
      }
    }
  }
  else {
    /* create vertices */
    mesh->set_verts(P_array);

    size_t num_ngons = 0;
    size_t num_corners = 0;
    for (size_t i = 0; i < nverts.size(); i++) {
      num_ngons += (nverts[i] == 4) ? 0 : 1;
      num_corners += nverts[i];
    }
    mesh->reserve_subd_faces(nverts.size(), num_ngons, num_corners);

    /* create subd_faces */
    int index_offset = 0;

    for (size_t i = 0; i < nverts.size(); i++) {
      mesh->add_subd_face(&verts[index_offset], nverts[i], shader, smooth);
      index_offset += nverts[i];
    }

    /* uv map */
    if (xml_read_float_array(UV, node, "UV")) {
      ustring name = ustring("UVMap");
      Attribute *attr = mesh->subd_attributes.add(ATTR_STD_UV, name);
      float3 *fdata = attr->data_float3();

#if 0
      if (subdivide_uvs) {
        attr->flags |= ATTR_SUBDIVIDED;
      }
#endif

      index_offset = 0;
      for (size_t i = 0; i < nverts.size(); i++) {
        for (int j = 0; j < nverts[i]; j++) {
          *(fdata++) = make_float3(UV[index_offset++]);
        }
      }
    }

    /* setup subd params */
    float dicing_rate = state.dicing_rate;
    xml_read_float(&dicing_rate, node, "dicing_rate");
    dicing_rate = std::max(0.1f, dicing_rate);

    mesh->set_subd_dicing_rate(dicing_rate);
    mesh->set_subd_objecttoworld(state.tfm);
  }

  /* we don't yet support arbitrary attributes, for now add vertex
   * coordinates as generated coordinates if requested */
  if (mesh->need_attribute(state.scene, ATTR_STD_GENERATED)) {
    Attribute *attr = mesh->attributes.add(ATTR_STD_GENERATED);
    memcpy(
        attr->data_float3(), mesh->get_verts().data(), sizeof(float3) * mesh->get_verts().size());
  }
}

static const unsigned char base64_table[65] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/* https://stackoverflow.com/questions/342409/how-do-i-base64-encode-decode-in-c */
static string base64_encode(const unsigned char *src, size_t len)
{
  unsigned char *out, *pos;
  const unsigned char *end, *in;

  size_t olen;

  olen = 4 * ((len + 2) / 3); /* 3-byte blocks to 4-byte */

  if (olen < len) {
    return string(); /* integer overflow */
  }
  string outStr;
  outStr.resize(olen);
  out = (unsigned char *)&outStr[0];

  end = src + len;
  in = src;
  pos = out;
  while (end - in >= 3) {
    *pos++ = base64_table[in[0] >> 2];
    *pos++ = base64_table[((in[0] & 0x03) << 4) | (in[1] >> 4)];
    *pos++ = base64_table[((in[1] & 0x0f) << 2) | (in[2] >> 6)];
    *pos++ = base64_table[in[2] & 0x3f];
    in += 3;
  }

  if (end - in) {
    *pos++ = base64_table[in[0] >> 2];
    if (end - in == 1) {
      *pos++ = base64_table[(in[0] & 0x03) << 4];
      *pos++ = '=';
    }
    else {
      *pos++ = base64_table[((in[0] & 0x03) << 4) | (in[1] >> 4)];
      *pos++ = base64_table[(in[1] & 0x0f) << 2];
    }
    *pos++ = '=';
  }

  return outStr;
}

static void xml_write_attribute(const Attribute &attr, xml_node attribute)
{
  attribute.append_attribute("name") = attr.name.c_str();
  attribute.append_attribute("standard") = attr.std;
  attribute.append_attribute("type.basetype") = attr.type.basetype;
  attribute.append_attribute("type.aggregate") = attr.type.aggregate;
  attribute.append_attribute("type.vecsemantics") = attr.type.vecsemantics;
  attribute.append_attribute("element") = attr.element;
  string encoded = base64_encode((const unsigned char *)attr.data(), attr.buffer.size());
  attribute.append_child(pugi::node_pcdata).set_value(encoded.c_str());
}

static void xml_write_mesh(Mesh *mesh, xml_node node)
{
  xml_node mesh_node = xml_write_node(mesh, node);
  mesh_node.attribute("name") = ustring::format("%s:%zx", mesh->name.c_str(), mesh).c_str();

  /* Subdivison surfaces must be written explicitly. */
  xml_write_int_array(mesh->get_subd_start_corner(), mesh_node, "subd.start_corners");
  xml_write_int_array(mesh->get_subd_num_corners(), mesh_node, "subd.num_corners");
  xml_write_int_array(mesh->get_subd_shader(), mesh_node, "subd.shader");
  xml_write_bool_array(mesh->get_subd_smooth(), mesh_node, "subd.smooth");
  xml_write_int_array(mesh->get_subd_ptex_offset(), mesh_node, "subd.ptex_offset");

  if (mesh->get_subd_params()) {
    mesh_node.append_attribute("subd_params.ptex").set_value(mesh->get_subd_params()->ptex);
    mesh_node.append_attribute("subd_params.test_steps").set_value(mesh->get_subd_params()->test_steps);
    mesh_node.append_attribute("subd_params.split_threshold")
        .set_value(mesh->get_subd_params()->split_threshold);
    mesh_node.append_attribute("subd_params.dicing_rate")
        .set_value(mesh->get_subd_params()->dicing_rate);
    mesh_node.append_attribute("subd_params.max_level").set_value(mesh->get_subd_params()->max_level);
  }

  std::stringstream ss;
  for (size_t i = 0; i < mesh->get_used_shaders().size(); ++i) {
    const Shader *shader = static_cast<const Shader *>(mesh->get_used_shaders()[i]);
    ss << shader->name.c_str();
    if (i != mesh->get_used_shaders().size() - 1) {
      ss << " ";
    }
  }
  mesh_node.append_attribute("used_shaders") = ss.str().c_str();

  /* Write mesh attributes. Encode as base64, which saves meory but takes away human readability. */
  foreach (const Attribute &attr, mesh->attributes.attributes) {
    xml_write_attribute(attr, mesh_node.append_child("attribute"));
  }
  foreach (const Attribute &attr, mesh->subd_attributes.attributes) {
    xml_write_attribute(attr, mesh_node.append_child("subd_attribute"));
  }
}

static void xml_write_geometry(Geometry *geom, xml_node node)
{
  if(geom->is_mesh()) {
    xml_write_mesh(static_cast<Mesh*>(geom), node);
  }
  else if (geom->is_hair()) {
    // TODO
  }
  else if (geom->is_volume()) {
    // TODO
  }
}

/* Light */

static void xml_read_light(XMLReadState &state, xml_node node)
{
  Light *light = new Light();

  light->set_shader(state.shader);
  xml_read_node(state, light, node);

  state.scene->lights.push_back(light);
}

/* Transform */

static void xml_read_transform(xml_node node, Transform &tfm)
{
  if (node.attribute("matrix")) {
    vector<float> matrix;
    if (xml_read_float_array(matrix, node, "matrix") && matrix.size() == 16) {
      ProjectionTransform projection = *(ProjectionTransform *)&matrix[0];
      tfm = tfm * projection_to_transform(projection_transpose(projection));
    }
  }

  if (node.attribute("translate")) {
    float3 translate = make_float3(0.0f, 0.0f, 0.0f);
    xml_read_float3(&translate, node, "translate");
    tfm = tfm * transform_translate(translate);
  }

  if (node.attribute("rotate")) {
    float4 rotate = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    xml_read_float4(&rotate, node, "rotate");
    tfm = tfm * transform_rotate(DEG2RADF(rotate.x), make_float3(rotate.y, rotate.z, rotate.w));
  }

  if (node.attribute("scale")) {
    float3 scale = make_float3(0.0f, 0.0f, 0.0f);
    xml_read_float3(&scale, node, "scale");
    tfm = tfm * transform_scale(scale);
  }
}

/* Object */

static void xml_write_object(const Object *object, xml_node node)
{
  xml_node object_node = xml_write_node(object, node);
  if (object->get_geometry()) {
    object_node.attribute(
        "mesh") = ustring::format("%s:%zx", object->get_geometry()->name.c_str(), object->get_geometry()).c_str();
  }
}

/* State */

static void xml_read_state(XMLReadState &state, xml_node node)
{
  /* read shader */
  string shadername;

  if (xml_read_string(&shadername, node, "shader")) {
    bool found = false;

    foreach (Shader *shader, state.scene->shaders) {
      if (shader->name == shadername) {
        state.shader = shader;
        found = true;
        break;
      }
    }

    if (!found)
      fprintf(stderr, "Unknown shader \"%s\".\n", shadername.c_str());
  }

  xml_read_float(&state.dicing_rate, node, "dicing_rate");

  /* read smooth/flat */
  if (xml_equal_string(node, "interpolation", "smooth"))
    state.smooth = true;
  else if (xml_equal_string(node, "interpolation", "flat"))
    state.smooth = false;
}

/* Scene */

static void xml_read_include(XMLReadState &state, const string &src);

static void xml_read_scene(XMLReadState &state, xml_node scene_node)
{
  for (xml_node node = scene_node.first_child(); node; node = node.next_sibling()) {
    if (string_iequals(node.name(), "film")) {
      xml_read_node(state, state.scene->film, node);
    }
    else if (string_iequals(node.name(), "integrator")) {
      xml_read_node(state, state.scene->integrator, node);
    }
    else if (string_iequals(node.name(), "camera")) {
      xml_read_camera(state, node);
    }
    else if (string_iequals(node.name(), "shader")) {
      xml_read_shader(state, node);
    }
    else if (string_iequals(node.name(), "background")) {
      xml_read_background(state, node);
    }
    else if (string_iequals(node.name(), "mesh")) {
      xml_read_mesh(state, node);
    }
    else if (string_iequals(node.name(), "light")) {
      xml_read_light(state, node);
    }
    else if (string_iequals(node.name(), "transform")) {
      XMLReadState substate = state;

      xml_read_transform(node, substate.tfm);
      xml_read_scene(substate, node);
    }
    else if (string_iequals(node.name(), "state")) {
      XMLReadState substate = state;

      xml_read_state(substate, node);
      xml_read_scene(substate, node);
    }
    else if (string_iequals(node.name(), "include")) {
      string src;

      if (xml_read_string(&src, node, "src"))
        xml_read_include(state, src);
    }
    else
      fprintf(stderr, "Unknown node \"%s\".\n", node.name());
  }
}

static void xml_write_scene(const Scene *scene, const SessionParams *params, xml_node &node)
{
  xml_write_node(params, node);
  xml_write_node(scene->film, node);
  xml_write_node(scene->integrator, node);
  xml_write_camera(scene->camera, node);

  /* Write shaders first, the following elements will use them. */
  foreach (Shader *shader, scene->shaders) {
    xml_write_shader(shader, node);
  }

  xml_write_background(scene->background, node);

  foreach (Light *light, scene->lights) {
    xml_write_node(light, node);
  }

  foreach (Geometry *geom, scene->geometry) {
    xml_write_geometry(geom, node);
  }

  foreach (Object *object, scene->objects) {
    xml_write_object(object, node);
  }
}

/* Include */

static void xml_read_include(XMLReadState &state, const string &src)
{
  /* open XML document */
  xml_document doc;
  xml_parse_result parse_result;

  string path = path_join(state.base, src);
  parse_result = doc.load_file(path.c_str());

  if (parse_result) {
    XMLReadState substate = state;
    substate.base = path_dirname(path);

    xml_node cycles = doc.child("cycles");
    xml_read_scene(substate, cycles);
  }
  else {
    fprintf(stderr, "%s read error: %s\n", src.c_str(), parse_result.description());
    exit(EXIT_FAILURE);
  }
}

/* File */

void xml_read_file(Scene *scene, const char *filepath)
{
  XMLReadState state;

  state.scene = scene;
  state.tfm = transform_identity();
  state.shader = scene->default_surface;
  state.smooth = false;
  state.dicing_rate = 1.0f;
  state.base = path_dirname(filepath);

  xml_read_include(state, path_filename(filepath));

  scene->params.bvh_type = SceneParams::BVH_STATIC;
}

/* Writes the scene to an XML file. This may end up using quite some memory, as pugixml has to keep everything
   in RAM before it is finally writing the data to a file. This should eventually be optimized with a better
   XML library or even with a different file format than XML.
   An alternative would be to write separate files for different parts of the scene and use the inlcude feature.
 */
void xml_write_file(const Scene *scene, const SessionParams *params, const char *filepath)
{
  xml_document doc;

  xml_node cycles = doc.append_child("cycles");
  cycles.append_attribute("version") =
      ustring::format("%d.%d", CYCLES_XML_VERSION_MAJOR, CYCLES_XML_VERSION_MINOR).c_str();

  xml_write_scene(scene, params, cycles);

  doc.save_file(filepath);
}

CCL_NAMESPACE_END
