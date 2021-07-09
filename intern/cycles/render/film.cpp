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

#include "render/film.h"
#include "device/device.h"
#include "render/camera.h"
#include "render/integrator.h"
#include "render/mesh.h"
#include "render/scene.h"
#include "render/stats.h"
#include "render/tables.h"

#include "util/util_algorithm.h"
#include "util/util_foreach.h"
#include "util/util_math.h"
#include "util/util_math_cdf.h"
#include "util/util_time.h"

CCL_NAMESPACE_BEGIN

/* Pixel Filter */

static float filter_func_box(float /*v*/, float /*width*/)
{
  return 1.0f;
}

static float filter_func_gaussian(float v, float width)
{
  v *= 6.0f / width;
  return expf(-2.0f * v * v);
}

static float filter_func_blackman_harris(float v, float width)
{
  v = M_2PI_F * (v / width + 0.5f);
  return 0.35875f - 0.48829f * cosf(v) + 0.14128f * cosf(2.0f * v) - 0.01168f * cosf(3.0f * v);
}

static vector<float> filter_table(FilterType type, float width)
{
  vector<float> filter_table(FILTER_TABLE_SIZE);
  float (*filter_func)(float, float) = NULL;

  switch (type) {
    case FILTER_BOX:
      filter_func = filter_func_box;
      break;
    case FILTER_GAUSSIAN:
      filter_func = filter_func_gaussian;
      width *= 3.0f;
      break;
    case FILTER_BLACKMAN_HARRIS:
      filter_func = filter_func_blackman_harris;
      width *= 2.0f;
      break;
    default:
      assert(0);
  }

  /* Create importance sampling table. */

  /* TODO(sergey): With the even filter table size resolution we can not
   * really make it nice symmetric importance map without sampling full range
   * (meaning, we would need to sample full filter range and not use the
   * make_symmetric argument).
   *
   * Current code matches exactly initial filter table code, but we should
   * consider either making FILTER_TABLE_SIZE odd value or sample full filter.
   */

  util_cdf_inverted(FILTER_TABLE_SIZE,
                    0.0f,
                    width * 0.5f,
                    function_bind(filter_func, _1, width),
                    true,
                    filter_table);

  return filter_table;
}

/* Film */

NODE_DEFINE(Film)
{
  NodeType *type = NodeType::add("film", create);

  SOCKET_FLOAT(exposure, "Exposure", 0.8f);
  SOCKET_FLOAT(pass_alpha_threshold, "Pass Alpha Threshold", 0.0f);

  static NodeEnum filter_enum;
  filter_enum.insert("box", FILTER_BOX);
  filter_enum.insert("gaussian", FILTER_GAUSSIAN);
  filter_enum.insert("blackman_harris", FILTER_BLACKMAN_HARRIS);

  SOCKET_ENUM(filter_type, "Filter Type", filter_enum, FILTER_BOX);
  SOCKET_FLOAT(filter_width, "Filter Width", 1.0f);

  SOCKET_FLOAT(mist_start, "Mist Start", 0.0f);
  SOCKET_FLOAT(mist_depth, "Mist Depth", 100.0f);
  SOCKET_FLOAT(mist_falloff, "Mist Falloff", 1.0f);

  SOCKET_BOOLEAN(use_light_visibility, "Use Light Visibility", false);

  const NodeEnum *pass_type_enum = Pass::get_type_enum();
  SOCKET_ENUM(display_pass, "Display Pass", *pass_type_enum, PASS_COMBINED);

  SOCKET_BOOLEAN(show_active_pixels, "Show Active Pixels", false);

  static NodeEnum cryptomatte_passes_enum;
  cryptomatte_passes_enum.insert("none", CRYPT_NONE);
  cryptomatte_passes_enum.insert("object", CRYPT_OBJECT);
  cryptomatte_passes_enum.insert("material", CRYPT_MATERIAL);
  cryptomatte_passes_enum.insert("asset", CRYPT_ASSET);
  cryptomatte_passes_enum.insert("accurate", CRYPT_ACCURATE);
  SOCKET_ENUM(cryptomatte_passes, "Cryptomatte Passes", cryptomatte_passes_enum, CRYPT_NONE);

  SOCKET_INT(cryptomatte_depth, "Cryptomatte Depth", 0);

  SOCKET_BOOLEAN(use_approximate_shadow_catcher, "Use Approximate Shadow Catcher", false);

  return type;
}

Film::Film() : Node(get_node_type()), filter_table_offset_(TABLE_OFFSET_INVALID)
{

  use_light_visibility = false;
  cryptomatte_passes = CRYPT_NONE;
  display_pass = PASS_COMBINED;
  show_active_pixels = false;
}

Film::~Film()
{
}

void Film::add_default(Scene *scene)
{
  Pass::add(scene->passes, PASS_COMBINED);
}

void Film::device_update(Device *device, DeviceScene *dscene, Scene *scene)
{
  if (!is_modified())
    return;

  scoped_callback_timer timer([scene](double time) {
    if (scene->update_stats) {
      scene->update_stats->film.times.add_entry({"update", time});
    }
  });

  device_free(device, dscene, scene);

  KernelFilm *kfilm = &dscene->data.film;

  const Pass *display_pass = get_actual_display_pass(scene, get_display_pass());
  const Pass *display_pass_denoised = get_actual_display_pass(
      scene, get_display_pass(), PassMode::DENOISED);

  /* update __data */
  kfilm->exposure = exposure;
  kfilm->pass_flag = 0;

  kfilm->display_pass_type = display_pass->type;
  kfilm->display_pass_offset = PASS_UNUSED;
  kfilm->display_pass_denoised_offset = PASS_UNUSED;
  kfilm->show_active_pixels = show_active_pixels;
  kfilm->use_approximate_shadow_catcher = get_use_approximate_shadow_catcher();

  kfilm->light_pass_flag = 0;
  kfilm->pass_stride = 0;
  kfilm->use_light_pass = use_light_visibility;
  kfilm->pass_aov_value_num = 0;
  kfilm->pass_aov_color_num = 0;

  /* Mark with PASS_UNUSED to avoid mask test in the kernel. */
  kfilm->pass_background = PASS_UNUSED;
  kfilm->pass_emission = PASS_UNUSED;
  kfilm->pass_ao = PASS_UNUSED;
  kfilm->pass_diffuse_direct = PASS_UNUSED;
  kfilm->pass_diffuse_indirect = PASS_UNUSED;
  kfilm->pass_glossy_direct = PASS_UNUSED;
  kfilm->pass_glossy_indirect = PASS_UNUSED;
  kfilm->pass_transmission_direct = PASS_UNUSED;
  kfilm->pass_transmission_indirect = PASS_UNUSED;
  kfilm->pass_volume_direct = PASS_UNUSED;
  kfilm->pass_volume_indirect = PASS_UNUSED;
  kfilm->pass_volume_direct = PASS_UNUSED;
  kfilm->pass_volume_indirect = PASS_UNUSED;
  kfilm->pass_shadow = PASS_UNUSED;

  /* Mark passes as unused so that the kernel knows the pass is inaccessible. */
  kfilm->pass_denoising_normal = PASS_UNUSED;
  kfilm->pass_denoising_albedo = PASS_UNUSED;
  kfilm->pass_sample_count = PASS_UNUSED;
  kfilm->pass_adaptive_aux_buffer = PASS_UNUSED;
  kfilm->pass_shadow_catcher = PASS_UNUSED;
  kfilm->pass_shadow_catcher_matte = PASS_UNUSED;

  bool have_cryptomatte = false;

  for (size_t i = 0; i < scene->passes.size(); i++) {
    Pass &pass = scene->passes[i];

    if (pass.type == PASS_NONE || !pass.is_written()) {
      continue;
    }

    if (pass.mode == PassMode::DENOISED) {
      if (&pass == display_pass_denoised) {
        kfilm->display_pass_denoised_offset = kfilm->pass_stride;
      }

      /* Generally we only storing offsets of the noisy passes. The display pass is an exception
       * since it is a read operation and not a write. */
      kfilm->pass_stride += pass.get_info().num_components;
      continue;
    }

    /* Can't do motion pass if no motion vectors are available. */
    if (pass.type == PASS_MOTION || pass.type == PASS_MOTION_WEIGHT) {
      if (scene->need_motion() != Scene::MOTION_PASS) {
        kfilm->pass_stride += pass.get_info().num_components;
        continue;
      }
    }

    int pass_flag = (1 << (pass.type % 32));
    if (pass.type <= PASS_CATEGORY_LIGHT_END) {
      if (pass.type != PASS_COMBINED) {
        kfilm->use_light_pass = 1;
      }
      kfilm->light_pass_flag |= pass_flag;
    }
    else if (pass.type <= PASS_CATEGORY_DATA_END) {
      kfilm->pass_flag |= pass_flag;
    }
    else {
      assert(pass.type <= PASS_CATEGORY_BAKE_END);
    }

    switch (pass.type) {
      case PASS_COMBINED:
        kfilm->pass_combined = kfilm->pass_stride;
        break;
      case PASS_DEPTH:
        kfilm->pass_depth = kfilm->pass_stride;
        break;
      case PASS_NORMAL:
        kfilm->pass_normal = kfilm->pass_stride;
        break;
      case PASS_ROUGHNESS:
        kfilm->pass_roughness = kfilm->pass_stride;
        break;
      case PASS_UV:
        kfilm->pass_uv = kfilm->pass_stride;
        break;
      case PASS_MOTION:
        kfilm->pass_motion = kfilm->pass_stride;
        break;
      case PASS_MOTION_WEIGHT:
        kfilm->pass_motion_weight = kfilm->pass_stride;
        break;
      case PASS_OBJECT_ID:
        kfilm->pass_object_id = kfilm->pass_stride;
        break;
      case PASS_MATERIAL_ID:
        kfilm->pass_material_id = kfilm->pass_stride;
        break;

      case PASS_MIST:
        kfilm->pass_mist = kfilm->pass_stride;
        break;
      case PASS_EMISSION:
        kfilm->pass_emission = kfilm->pass_stride;
        break;
      case PASS_BACKGROUND:
        kfilm->pass_background = kfilm->pass_stride;
        break;
      case PASS_AO:
        kfilm->pass_ao = kfilm->pass_stride;
        break;
      case PASS_SHADOW:
        kfilm->pass_shadow = kfilm->pass_stride;
        break;

      case PASS_DIFFUSE_COLOR:
        kfilm->pass_diffuse_color = kfilm->pass_stride;
        break;
      case PASS_GLOSSY_COLOR:
        kfilm->pass_glossy_color = kfilm->pass_stride;
        break;
      case PASS_TRANSMISSION_COLOR:
        kfilm->pass_transmission_color = kfilm->pass_stride;
        break;
      case PASS_DIFFUSE_INDIRECT:
        kfilm->pass_diffuse_indirect = kfilm->pass_stride;
        break;
      case PASS_GLOSSY_INDIRECT:
        kfilm->pass_glossy_indirect = kfilm->pass_stride;
        break;
      case PASS_TRANSMISSION_INDIRECT:
        kfilm->pass_transmission_indirect = kfilm->pass_stride;
        break;
      case PASS_VOLUME_INDIRECT:
        kfilm->pass_volume_indirect = kfilm->pass_stride;
        break;
      case PASS_DIFFUSE_DIRECT:
        kfilm->pass_diffuse_direct = kfilm->pass_stride;
        break;
      case PASS_GLOSSY_DIRECT:
        kfilm->pass_glossy_direct = kfilm->pass_stride;
        break;
      case PASS_TRANSMISSION_DIRECT:
        kfilm->pass_transmission_direct = kfilm->pass_stride;
        break;
      case PASS_VOLUME_DIRECT:
        kfilm->pass_volume_direct = kfilm->pass_stride;
        break;

      case PASS_BAKE_PRIMITIVE:
        kfilm->pass_bake_primitive = kfilm->pass_stride;
        break;
      case PASS_BAKE_DIFFERENTIAL:
        kfilm->pass_bake_differential = kfilm->pass_stride;
        break;

      case PASS_RENDER_TIME:
        break;
      case PASS_CRYPTOMATTE:
        kfilm->pass_cryptomatte = have_cryptomatte ?
                                      min(kfilm->pass_cryptomatte, kfilm->pass_stride) :
                                      kfilm->pass_stride;
        have_cryptomatte = true;
        break;

      case PASS_DENOISING_NORMAL:
        kfilm->pass_denoising_normal = kfilm->pass_stride;
        break;
      case PASS_DENOISING_ALBEDO:
        kfilm->pass_denoising_albedo = kfilm->pass_stride;
        break;

      case PASS_SHADOW_CATCHER:
        kfilm->pass_shadow_catcher = kfilm->pass_stride;
        break;
      case PASS_SHADOW_CATCHER_MATTE:
        kfilm->pass_shadow_catcher_matte = kfilm->pass_stride;
        break;

      case PASS_ADAPTIVE_AUX_BUFFER:
        kfilm->pass_adaptive_aux_buffer = kfilm->pass_stride;
        break;
      case PASS_SAMPLE_COUNT:
        kfilm->pass_sample_count = kfilm->pass_stride;
        break;

      case PASS_AOV_COLOR:
        if (kfilm->pass_aov_color_num == 0) {
          kfilm->pass_aov_color = kfilm->pass_stride;
        }
        kfilm->pass_aov_color_num++;
        break;
      case PASS_AOV_VALUE:
        if (kfilm->pass_aov_value_num == 0) {
          kfilm->pass_aov_value = kfilm->pass_stride;
        }
        kfilm->pass_aov_value_num++;
        break;
      default:
        assert(false);
        break;
    }

    if (&pass == display_pass) {
      kfilm->display_pass_offset = kfilm->pass_stride;
    }

    kfilm->pass_stride += pass.get_info().num_components;
  }

  kfilm->pass_stride = align_up(kfilm->pass_stride, 4);

  /* When displaying the normal/uv pass in the viewport we need to disable
   * transparency.
   *
   * We also don't need to perform light accumulations. Later we want to optimize this to suppress
   * light calculations. */
  if (display_pass->type == PASS_NORMAL || display_pass->type == PASS_UV ||
      display_pass->type == PASS_ROUGHNESS) {
    kfilm->use_light_pass = 0;
  }
  else {
    kfilm->pass_alpha_threshold = pass_alpha_threshold;
  }

  /* update filter table */
  vector<float> table = filter_table(filter_type, filter_width);
  scene->lookup_tables->remove_table(&filter_table_offset_);
  filter_table_offset_ = scene->lookup_tables->add_table(dscene, table);
  kfilm->filter_table_offset = (int)filter_table_offset_;

  /* mist pass parameters */
  kfilm->mist_start = mist_start;
  kfilm->mist_inv_depth = (mist_depth > 0.0f) ? 1.0f / mist_depth : 0.0f;
  kfilm->mist_falloff = mist_falloff;

  kfilm->cryptomatte_passes = cryptomatte_passes;
  kfilm->cryptomatte_depth = cryptomatte_depth;

  clear_modified();
}

void Film::device_free(Device * /*device*/, DeviceScene * /*dscene*/, Scene *scene)
{
  scene->lookup_tables->remove_table(&filter_table_offset_);
}

void Film::assign_and_tag_passes_update(Scene *scene, const vector<Pass> &passes)
{
  if (Pass::contains(scene->passes, PASS_UV) != Pass::contains(passes, PASS_UV)) {
    scene->geometry_manager->tag_update(scene, GeometryManager::UV_PASS_NEEDED);

    foreach (Shader *shader, scene->shaders)
      shader->need_update_uvs = true;
  }
  else if (Pass::contains(scene->passes, PASS_MOTION) != Pass::contains(passes, PASS_MOTION)) {
    scene->geometry_manager->tag_update(scene, GeometryManager::MOTION_PASS_NEEDED);
  }
  else if (Pass::contains(scene->passes, PASS_AO) != Pass::contains(passes, PASS_AO)) {
    scene->integrator->tag_update(scene, Integrator::AO_PASS_MODIFIED);
  }

  scene->passes = passes;
}

int Film::get_aov_offset(Scene *scene, string name, bool &is_color)
{
  int num_color = 0, num_value = 0;
  foreach (const Pass &pass, scene->passes) {
    if (pass.type == PASS_AOV_COLOR) {
      num_color++;
    }
    else if (pass.type == PASS_AOV_VALUE) {
      num_value++;
    }
    else {
      continue;
    }

    if (pass.name == name) {
      is_color = (pass.type == PASS_AOV_COLOR);
      return (is_color ? num_color : num_value) - 1;
    }
  }

  return -1;
}

const Pass *Film::get_actual_display_pass(Scene *scene, PassType pass_type, PassMode pass_mode)
{
  const Pass *pass = Pass::find(scene->passes, pass_type, pass_mode);
  return get_actual_display_pass(scene, pass);
}

const Pass *Film::get_actual_display_pass(Scene *scene, const Pass *pass)
{
  if (!pass) {
    return nullptr;
  }

  if (!pass->is_written()) {
    if (pass->mode == PassMode::DENOISED) {
      pass = Pass::find(scene->passes, pass->type);
      if (!pass) {
        return nullptr;
      }
    }
    else {
      return nullptr;
    }
  }

  if (pass->type == PASS_COMBINED && scene->has_shadow_catcher()) {
    const Pass *shadow_catcher_matte_pass = Pass::find(
        scene->passes, PASS_SHADOW_CATCHER_MATTE, pass->mode);
    if (shadow_catcher_matte_pass) {
      pass = shadow_catcher_matte_pass;
    }
  }

  return pass;
}

uint Film::get_kernel_features(const Scene *scene) const
{
  uint kernel_features = 0;

  for (const Pass &pass : scene->passes) {
    if (!pass.is_written()) {
      continue;
    }

    if ((pass.type == PASS_COMBINED && pass.mode == PassMode::DENOISED) ||
        pass.type == PASS_DENOISING_NORMAL || pass.type == PASS_DENOISING_ALBEDO) {
      kernel_features |= KERNEL_FEATURE_DENOISING;
    }

    if (pass.type != PASS_NONE && pass.type != PASS_COMBINED &&
        pass.type <= PASS_CATEGORY_LIGHT_END) {
      kernel_features |= KERNEL_FEATURE_LIGHT_PASSES;

      if (pass.type == PASS_SHADOW) {
        kernel_features |= KERNEL_FEATURE_SHADOW_PASS;
      }
    }

    if (pass.type == PASS_AO) {
      kernel_features |= KERNEL_FEATURE_NODE_RAYTRACE;
    }
  }

  return kernel_features;
}

CCL_NAMESPACE_END
