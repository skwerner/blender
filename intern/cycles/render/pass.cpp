/*
 * Copyright 2011-2021 Blender Foundation
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

#include "render/pass.h"

#include "util/util_algorithm.h"
#include "util/util_logging.h"

CCL_NAMESPACE_BEGIN

static bool compare_pass_order(const Pass &a, const Pass &b)
{
  if (a.components == b.components)
    return (a.type < b.type);
  return (a.components > b.components);
}

const NodeEnum *Pass::get_type_enum()
{
  static NodeEnum pass_type_enum;

  if (pass_type_enum.empty()) {
    pass_type_enum.insert("combined", PASS_COMBINED);
    pass_type_enum.insert("depth", PASS_DEPTH);
    pass_type_enum.insert("normal", PASS_NORMAL);
    pass_type_enum.insert("roughness", PASS_ROUGHNESS);
    pass_type_enum.insert("uv", PASS_UV);
    pass_type_enum.insert("object_id", PASS_OBJECT_ID);
    pass_type_enum.insert("material_id", PASS_MATERIAL_ID);
    pass_type_enum.insert("motion", PASS_MOTION);
    pass_type_enum.insert("motion_weight", PASS_MOTION_WEIGHT);
    pass_type_enum.insert("render_time", PASS_RENDER_TIME);
    pass_type_enum.insert("cryptomatte", PASS_CRYPTOMATTE);
    pass_type_enum.insert("aov_color", PASS_AOV_COLOR);
    pass_type_enum.insert("aov_value", PASS_AOV_VALUE);
    pass_type_enum.insert("adaptive_aux_buffer", PASS_ADAPTIVE_AUX_BUFFER);
    pass_type_enum.insert("sample_count", PASS_SAMPLE_COUNT);
    pass_type_enum.insert("mist", PASS_MIST);
    pass_type_enum.insert("emission", PASS_EMISSION);
    pass_type_enum.insert("background", PASS_BACKGROUND);
    pass_type_enum.insert("ambient_occlusion", PASS_AO);
    pass_type_enum.insert("shadow", PASS_SHADOW);
    pass_type_enum.insert("diffuse_direct", PASS_DIFFUSE_DIRECT);
    pass_type_enum.insert("diffuse_indirect", PASS_DIFFUSE_INDIRECT);
    pass_type_enum.insert("diffuse_color", PASS_DIFFUSE_COLOR);
    pass_type_enum.insert("glossy_direct", PASS_GLOSSY_DIRECT);
    pass_type_enum.insert("glossy_indirect", PASS_GLOSSY_INDIRECT);
    pass_type_enum.insert("glossy_color", PASS_GLOSSY_COLOR);
    pass_type_enum.insert("transmission_direct", PASS_TRANSMISSION_DIRECT);
    pass_type_enum.insert("transmission_indirect", PASS_TRANSMISSION_INDIRECT);
    pass_type_enum.insert("transmission_color", PASS_TRANSMISSION_COLOR);
    pass_type_enum.insert("volume_direct", PASS_VOLUME_DIRECT);
    pass_type_enum.insert("volume_indirect", PASS_VOLUME_INDIRECT);
    pass_type_enum.insert("bake_primitive", PASS_BAKE_PRIMITIVE);
    pass_type_enum.insert("bake_differential", PASS_BAKE_DIFFERENTIAL);
  }

  return &pass_type_enum;
}

NODE_DEFINE(Pass)
{
  NodeType *type = NodeType::add("pass", create);

  const NodeEnum *pass_type_enum = get_type_enum();
  SOCKET_ENUM(type, "Type", *pass_type_enum, PASS_COMBINED);
  SOCKET_STRING(name, "Name", ustring());

  return type;
}

Pass::Pass() : Node(get_node_type())
{
}

PassInfo Pass::get_info(PassType type)
{
  PassInfo pass_info;

  pass_info.type = type;
  pass_info.use_filter = true;
  pass_info.use_exposure = false;
  pass_info.divide_type = PASS_NONE;
  pass_info.is_unaligned = false;

  switch (type) {
    case PASS_NONE:
      pass_info.num_components = 0;
      break;
    case PASS_COMBINED:
      pass_info.num_components = 4;
      pass_info.use_exposure = true;
      break;
    case PASS_DEPTH:
      pass_info.num_components = 1;
      pass_info.use_filter = false;
      break;
    case PASS_MIST:
      pass_info.num_components = 1;
      break;
    case PASS_NORMAL:
      pass_info.num_components = 4;
      break;
    case PASS_ROUGHNESS:
      pass_info.num_components = 1;
      break;
    case PASS_UV:
      pass_info.num_components = 4;
      break;
    case PASS_MOTION:
      pass_info.num_components = 4;
      pass_info.divide_type = PASS_MOTION_WEIGHT;
      break;
    case PASS_MOTION_WEIGHT:
      pass_info.num_components = 1;
      break;
    case PASS_OBJECT_ID:
    case PASS_MATERIAL_ID:
      pass_info.num_components = 1;
      pass_info.use_filter = false;
      break;

    case PASS_EMISSION:
    case PASS_BACKGROUND:
      pass_info.num_components = 4;
      pass_info.use_exposure = true;
      break;
    case PASS_AO:
      pass_info.num_components = 4;
      break;
    case PASS_SHADOW:
      pass_info.num_components = 4;
      pass_info.use_exposure = false;
      break;
    case PASS_RENDER_TIME:
      /* This pass is handled entirely on the host side. */
      pass_info.num_components = 0;
      break;

    case PASS_DIFFUSE_COLOR:
    case PASS_GLOSSY_COLOR:
    case PASS_TRANSMISSION_COLOR:
      pass_info.num_components = 4;
      break;
    case PASS_DIFFUSE_DIRECT:
    case PASS_DIFFUSE_INDIRECT:
      pass_info.num_components = 4;
      pass_info.use_exposure = true;
      pass_info.divide_type = PASS_DIFFUSE_COLOR;
      break;
    case PASS_GLOSSY_DIRECT:
    case PASS_GLOSSY_INDIRECT:
      pass_info.num_components = 4;
      pass_info.use_exposure = true;
      pass_info.divide_type = PASS_GLOSSY_COLOR;
      break;
    case PASS_TRANSMISSION_DIRECT:
    case PASS_TRANSMISSION_INDIRECT:
      pass_info.num_components = 4;
      pass_info.use_exposure = true;
      pass_info.divide_type = PASS_TRANSMISSION_COLOR;
      break;
    case PASS_VOLUME_DIRECT:
    case PASS_VOLUME_INDIRECT:
      pass_info.num_components = 4;
      pass_info.use_exposure = true;
      break;

    case PASS_CRYPTOMATTE:
      pass_info.num_components = 4;
      break;

    case PASS_DENOISING_COLOR:
      pass_info.num_components = 3;
      pass_info.use_exposure = true;
      pass_info.is_unaligned = true;
      break;
    case PASS_DENOISING_NORMAL:
      pass_info.num_components = 3;
      pass_info.is_unaligned = true;
      break;
    case PASS_DENOISING_ALBEDO:
      pass_info.num_components = 3;
      pass_info.is_unaligned = true;
      break;

    case PASS_SHADOW_CATCHER:
      pass_info.num_components = 4;
      pass_info.use_exposure = true;
      break;
    case PASS_SHADOW_CATCHER_MATTE:
      pass_info.num_components = 4;
      pass_info.use_exposure = true;
      break;

    case PASS_ADAPTIVE_AUX_BUFFER:
      pass_info.num_components = 4;
      break;
    case PASS_SAMPLE_COUNT:
      pass_info.num_components = 1;
      pass_info.use_exposure = false;
      break;

    case PASS_AOV_COLOR:
      pass_info.num_components = 4;
      break;
    case PASS_AOV_VALUE:
      pass_info.num_components = 1;
      break;

    case PASS_BAKE_PRIMITIVE:
    case PASS_BAKE_DIFFERENTIAL:
      pass_info.num_components = 4;
      pass_info.use_exposure = false;
      pass_info.use_filter = false;
      break;

    case PASS_CATEGORY_LIGHT_END:
    case PASS_CATEGORY_DATA_END:
    case PASS_CATEGORY_BAKE_END:
    case PASS_NUM:
      LOG(DFATAL) << "Unexpected pass type is used " << type;
      pass_info.num_components = 0;
      break;
  }

  return pass_info;
}

void Pass::add(PassType type, vector<Pass> &passes, const char *name, bool is_auto)
{
  for (Pass &pass : passes) {
    if (pass.type != type) {
      continue;
    }

    /* An empty name is used as a placeholder to signal that any pass of
     * that type is fine (because the content always is the same).
     * This is important to support divide_type: If the pass that has a
     * divide_type is added first, a pass for divide_type with an empty
     * name will be added. Then, if a matching pass with a name is later
     * requested, the existing placeholder will be renamed to that.
     * If the divide_type is explicitly allocated with a name first and
     * then again as part of another pass, the second one will just be
     * skipped because that type already exists. */

    /* If no name is specified, any pass of the correct type will match. */
    if (name == NULL) {
      pass.is_auto &= is_auto;
      return;
    }

    /* If we already have a placeholder pass, rename that one. */
    if (pass.name.empty()) {
      pass.name = name;
      pass.is_auto &= is_auto;
      return;
    }

    /* If neither existing nor requested pass have placeholder name, they
     * must match. */
    if (name == pass.name) {
      pass.is_auto &= is_auto;
      return;
    }
  }

  const PassInfo pass_info = get_info(type);

  Pass pass;
  pass.type = type;
  pass.components = pass_info.num_components;
  pass.filter = pass_info.use_filter;
  pass.exposure = pass_info.use_exposure;
  pass.divide_type = pass_info.divide_type;
  pass.is_auto = is_auto;

  if (name) {
    pass.name = name;
  }

  passes.push_back(pass);

  /* Order from by components, to ensure alignment so passes with size 4
   * come first and then passes with size 1. Note this must use stable sort
   * so cryptomatte passes remain in the right order. */
  stable_sort(&passes[0], &passes[0] + passes.size(), compare_pass_order);

  if (pass.divide_type != PASS_NONE) {
    Pass::add(pass.divide_type, passes, nullptr, is_auto);
  }
}

bool Pass::equals_exact(const vector<Pass> &A, const vector<Pass> &B)
{
  if (A.size() != B.size())
    return false;

  for (int i = 0; i < A.size(); i++)
    if (A[i].type != B[i].type || A[i].name != B[i].name)
      return false;

  return true;
}

/* Get first index which is greater than the given one which correspongs to a non-auto pass.
 * If there are only runtime passes after the given index, -1 is returned. */
static const int get_next_no_auto_pass_index(const vector<Pass> &passes, int index)
{
  ++index;

  while (index < passes.size()) {
    if (!passes[index].is_auto) {
      return index;
    }
  }

  return -1;
}

bool Pass::equals_no_auto(const vector<Pass> &A, const vector<Pass> &B)
{
  int index_a = -1, index_b = -1;

  while (true) {
    index_a = get_next_no_auto_pass_index(A, index_a);
    index_b = get_next_no_auto_pass_index(A, index_b);

    if (index_a == -1 && index_b == -1) {
      break;
    }

    if (index_a == -1 || index_b == -1) {
      return false;
    }

    const Pass &pass_a = A[index_a];
    const Pass &pass_b = B[index_b];

    if (pass_a.type != pass_b.type || pass_a.name != pass_b.name) {
      return false;
    }
  }

  return true;
}

bool Pass::contains(const vector<Pass> &passes, PassType type)
{
  for (size_t i = 0; i < passes.size(); i++)
    if (passes[i].type == type)
      return true;

  return false;
}

void Pass::remove_auto(vector<Pass> &passes, PassType type)
{
  const size_t num_passes = passes.size();

  size_t i = 0;
  while (i < num_passes) {
    if (passes[i].type == type) {
      break;
    }
    ++i;
  }

  if (i >= num_passes) {
    /* Pass does not exist. */
    return;
  }

  if (!passes[i].is_auto) {
    /* Pass is not automatically created, can not remove. */
    return;
  }

  passes.erase(passes.begin() + i);
}

void Pass::remove_all_auto(vector<Pass> &passes)
{
  vector<Pass> new_passes;

  for (const Pass &pass : passes) {
    if (!pass.is_auto) {
      new_passes.push_back(pass);
    }
  }

  passes = new_passes;
}

const Pass *Pass::find(const vector<Pass> &passes, const string &name)
{
  for (const Pass &pass : passes) {
    if (pass.name == name) {
      return &pass;
    }
  }

  return nullptr;
}

const Pass *Pass::find(const vector<Pass> &passes, PassType type)
{
  for (const Pass &pass : passes) {
    if (pass.type == type) {
      return &pass;
    }
  }

  return nullptr;
}

int Pass::get_offset(const vector<Pass> &passes, PassType type)
{
  int pass_offset = 0;

  for (const Pass &pass : passes) {
    if (pass.type == type) {
      return pass_offset;
    }
    pass_offset += pass.components;
  }

  return PASS_UNUSED;
}

int Pass::get_offset(const vector<Pass> &passes, const Pass &pass)
{
  int pass_offset = 0;

  for (const Pass &current_pass : passes) {
    if (current_pass.type == pass.type && current_pass.name == pass.name) {
      return pass_offset;
    }
    pass_offset += current_pass.components;
  }

  return PASS_UNUSED;
}

CCL_NAMESPACE_END
