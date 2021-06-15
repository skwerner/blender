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

/* TODO(sergey): Should be able to de-duplicate with `Pass::get_type_enum` somehow.
 * The latter one should also help with solving fragile nature of
 * `enum_view3d_shading_render_pass`. */
const char *pass_type_as_string(const PassType type)
{
  switch (type) {
    case PASS_NONE:
      return "NONE";

    case PASS_COMBINED:
      return "COMBINED";
    case PASS_EMISSION:
      return "EMISSION";
    case PASS_BACKGROUND:
      return "BACKGROUND";
    case PASS_AO:
      return "AO";
    case PASS_SHADOW:
      return "SHADOW";
    case PASS_DIFFUSE_DIRECT:
      return "DIFFUSE_DIRECT";
    case PASS_DIFFUSE_INDIRECT:
      return "DIFFUSE_INDIRECT";
    case PASS_GLOSSY_DIRECT:
      return "GLOSSY_DIRECT";
    case PASS_GLOSSY_INDIRECT:
      return "GLOSSY_INDIRECT";
    case PASS_TRANSMISSION_DIRECT:
      return "TRANSMISSION_DIRECT";
    case PASS_TRANSMISSION_INDIRECT:
      return "TRANSMISSION_INDIRECT";
    case PASS_VOLUME_DIRECT:
      return "VOLUME_DIRECT";
    case PASS_VOLUME_INDIRECT:
      return "VOLUME_INDIRECT";

    case PASS_DEPTH:
      return "DEPTH";
    case PASS_NORMAL:
      return "NORMAL";
    case PASS_ROUGHNESS:
      return "ROUGHNESS";
    case PASS_UV:
      return "UV";
    case PASS_OBJECT_ID:
      return "OBJECT_ID";
    case PASS_MATERIAL_ID:
      return "MATERIAL_ID";
    case PASS_MOTION:
      return "MOTION";
    case PASS_MOTION_WEIGHT:
      return "MOTION_WEIGHT";
    case PASS_RENDER_TIME:
      return "RENDER_TIME";
    case PASS_CRYPTOMATTE:
      return "CRYPTOMATTE";
    case PASS_AOV_COLOR:
      return "AOV_COLOR";
    case PASS_AOV_VALUE:
      return "AOV_VALUE";
    case PASS_ADAPTIVE_AUX_BUFFER:
      return "ADAPTIVE_AUX_BUFFER";
    case PASS_SAMPLE_COUNT:
      return "SAMPLE_COUNT";
    case PASS_DIFFUSE_COLOR:
      return "DIFFUSE_COLOR";
    case PASS_GLOSSY_COLOR:
      return "GLOSSY_COLOR";
    case PASS_TRANSMISSION_COLOR:
      return "TRANSMISSION_COLOR";
    case PASS_MIST:
      return "MIST";
    case PASS_DENOISING_NORMAL:
      return "DENOISING_NORMAL";
    case PASS_DENOISING_ALBEDO:
      return "DENOISING_ALBEDO";
    case PASS_SHADOW_CATCHER:
      return "SHADOW_CATCHER";
    case PASS_SHADOW_CATCHER_MATTE:
      return "SHADOW_CATCHER_MATTE";

    case PASS_BAKE_PRIMITIVE:
      return "BAKE_PRIMITIVE";
    case PASS_BAKE_DIFFERENTIAL:
      return "BAKE_DIFFERENTIAL";

    case PASS_CATEGORY_LIGHT_END:
    case PASS_CATEGORY_DATA_END:
    case PASS_CATEGORY_BAKE_END:
    case PASS_NUM:
      LOG(DFATAL) << "Invalid value for the pass type " << static_cast<int>(type)
                  << " (value is reserved for an internal use only).";
      return "UNKNOWN";
  }

  LOG(DFATAL) << "Unhandled pass type " << static_cast<int>(type) << ", not supposed to happen.";

  return "UNKNOWN";
}

const char *pass_mode_as_string(PassMode mode)
{
  switch (mode) {
    case PassMode::NOISY:
      return "NOISY";
    case PassMode::DENOISED:
      return "DENOISED";
  }

  LOG(DFATAL) << "Unhandled pass mode " << static_cast<int>(mode) << ", should never happen.";
  return "UNKNOWN";
}

std::ostream &operator<<(std::ostream &os, PassMode mode)
{
  os << pass_mode_as_string(mode);
  return os;
}

static bool compare_pass_order(const Pass &a, const Pass &b)
{
  const int num_components_a = a.get_info().num_components;
  const int num_components_b = b.get_info().num_components;

  if (num_components_a == num_components_b) {
    return (a.type < b.type);
  }

  return num_components_a > num_components_b;
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

const NodeEnum *Pass::get_mode_enum()
{
  static NodeEnum pass_mode_enum;

  if (pass_mode_enum.empty()) {
    pass_mode_enum.insert("noisy", static_cast<int>(PassMode::NOISY));
    pass_mode_enum.insert("denoised", static_cast<int>(PassMode::DENOISED));
  }

  return &pass_mode_enum;
}

NODE_DEFINE(Pass)
{
  NodeType *type = NodeType::add("pass", create);

  const NodeEnum *pass_type_enum = get_type_enum();
  const NodeEnum *pass_mode_enum = get_mode_enum();

  SOCKET_ENUM(type, "Type", *pass_type_enum, PASS_COMBINED);
  SOCKET_ENUM(mode, "Mode", *pass_mode_enum, static_cast<int>(PassMode::DENOISED));
  SOCKET_STRING(name, "Name", ustring());

  return type;
}

Pass::Pass() : Node(get_node_type())
{
}

const PassInfo &Pass::get_info() const
{
  return info_;
}

bool Pass::is_written() const
{
  return is_written_;
}

PassInfo Pass::get_info(PassType type)
{
  PassInfo pass_info;

  pass_info.use_filter = true;
  pass_info.use_exposure = false;
  pass_info.divide_type = PASS_NONE;
  pass_info.is_aligned = true;
  pass_info.use_compositing = false;
  pass_info.use_denoising_albedo = true;

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

    case PASS_DENOISING_NORMAL:
      pass_info.num_components = 3;
      pass_info.is_aligned = false;
      break;
    case PASS_DENOISING_ALBEDO:
      pass_info.num_components = 3;
      pass_info.is_aligned = false;
      break;

    case PASS_SHADOW_CATCHER:
      pass_info.num_components = 4;
      pass_info.use_exposure = true;
      pass_info.use_compositing = true;
      pass_info.use_denoising_albedo = false;
      break;
    case PASS_SHADOW_CATCHER_MATTE:
      pass_info.num_components = 4;
      pass_info.use_exposure = true;
      /* Without shadow catcher approximation compositing is not needed.
       * Since we don't know here whether approximation is used or not, leave the decision up to
       * the caller which will know that. */
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

  if (pass_info.divide_type != PASS_NONE) {
    pass_info.use_compositing = true;
  }

  return pass_info;
}

void Pass::add(vector<Pass> &passes, PassType type, const char *name)
{
  add_internal(passes, type, PassMode::NOISY, Pass::FLAG_NONE, name);
}

void Pass::add_denoising_read(vector<Pass> &passes, PassType type, const char *name)
{
  add_internal(passes, type, PassMode::DENOISED, Pass::FLAG_READ_ONLY, name);
}

void Pass::add_denoising_write(vector<Pass> &passes, PassType type, const char *name)
{
  add_internal(passes, type, PassMode::DENOISED, Pass::FLAG_NONE, name);
}

/* Check whether the pass is a placeholder for the given configuration.
 *
 * An empty name is used as a placeholder to signal that any pass of that type is fine (because the
 * content always is the same). This is important to support divide_type:
 * - If the pass that has a `divide_type` is added first, a pass for `divide_type` with an empty
 *   name will be added. Then, if a matching pass with a name is later requested, the existing
 *   placeholder will be renamed to that.
 * - If the `divide_type` is explicitly allocated with a name first and then again as part of
 *   another pass, the second one will just be skipped because that type already exists. */
static bool pass_placeholder_match(Pass &pass, PassType type, PassMode mode, const char *name)
{
  if (pass.type != type || pass.mode != mode) {
    return false;
  }

  /* If no name is specified, any pass of the correct type will match. */
  if (name == nullptr) {
    return true;
  }

  /* If we already have a placeholder pass, rename that one. */
  if (pass.name.empty()) {
    return true;
  }

  /* If neither existing nor requested pass have placeholder name, they must match. */
  if (name == pass.name) {
    return true;
  }

  return false;
}

void Pass::add_internal(vector<Pass> &passes, PassType type, int flags, const char *name)
{
  add_internal(passes, type, PassMode::NOISY, flags, name);
}

void Pass::add_internal(
    vector<Pass> &passes, PassType type, PassMode mode, int flags, const char *name)
{
  const bool is_auto = (flags & Pass::FLAG_AUTO);
  const bool is_written = (flags & Pass::FLAG_READ_ONLY) == 0;

  for (Pass &pass : passes) {
    if (!pass_placeholder_match(pass, type, mode, name)) {
      continue;
    }

    if (name && pass.name.empty()) {
      pass.name = name;
    }

    pass.is_auto_ &= is_auto;
    pass.is_written_ |= is_written;

    return;
  }

  Pass pass;
  pass.type = type;
  pass.mode = mode;

  if (name) {
    pass.name = name;
  }

  pass.info_ = get_info(type);
  pass.is_auto_ = is_auto;
  pass.is_written_ = is_written;

  passes.push_back(pass);

  /* Order from by components, to ensure alignment so passes with size 4 come first and then passes
   * with size 1. Note this must use stable sort so cryptomatte passes remain in the right order.
   */
  stable_sort(&passes[0], &passes[0] + passes.size(), compare_pass_order);

  if (pass.info_.divide_type != PASS_NONE) {
    Pass::add_internal(passes, pass.info_.divide_type, mode, flags);
  }
}

bool Pass::contains(const vector<Pass> &passes, PassType type, PassMode mode)
{
  return Pass::find(passes, type, mode) != nullptr;
}

void Pass::remove_all_auto(vector<Pass> &passes)
{
  vector<Pass> new_passes;

  for (const Pass &pass : passes) {
    if (!pass.is_auto_) {
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

const Pass *Pass::find(const vector<Pass> &passes, PassType type, PassMode mode)
{
  for (const Pass &pass : passes) {
    if (pass.type != type || pass.mode != mode) {
      continue;
    }

    return &pass;
  }

  return nullptr;
}

int Pass::get_offset(const vector<Pass> &passes, const Pass &pass)
{
  int pass_offset = 0;

  for (const Pass &current_pass : passes) {
    /* Note that pass name is allowed to be empty. This is why we check for type and mode. */
    if (current_pass.type == pass.type && current_pass.mode == pass.mode &&
        current_pass.name == pass.name) {
      if (current_pass.is_written()) {
        return pass_offset;
      }
      else {
        return PASS_UNUSED;
      }
    }
    if (current_pass.is_written()) {
      pass_offset += current_pass.get_info().num_components;
    }
  }

  return PASS_UNUSED;
}

std::ostream &operator<<(std::ostream &os, const Pass &pass)
{
  os << "type: " << pass_type_as_string(pass.type);
  os << ", name: \"" << pass.name << "\"";
  os << ", mode: " << pass.mode;
  os << ", is_written: " << string_from_bool(pass.is_written());

  return os;
}

CCL_NAMESPACE_END
