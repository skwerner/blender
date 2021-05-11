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

#pragma once

#include "render/film.h"
#include "util/util_string.h"
#include "util/util_vector.h"

CCL_NAMESPACE_BEGIN

class RenderBuffers;

/* Helper class which allows to access pass data.
 * Is designed in a way that it is created once when the pass data is known, and then pixels gets
 * progressively update from various render buffers. */
class PassAccessor {
 public:
  PassAccessor(const vector<Pass> &passes,
               const string &pass_name,
               int num_components,
               float exposure,
               int num_samples);

  bool is_valid() const;

  /* Get pass data from the given render buffers, perform needed filtering, and store result into
   * the pixels.
   * The result is stored sequentially starting from the very beginning of the pixels memory. */
  bool get_render_tile_pixels(RenderBuffers *render_buffers, float *pixels);

#if 0
  bool set_pass_rect(PassType type, int components, float *pixels);
#endif

  int get_pass_offset(PassType type) const;

  /* NOTE: Leaves pass and offset unchanged if the pass is not found. */
  bool get_pass_by_name(const string &name, const Pass **r_pass, int *r_offset = nullptr) const;
  bool get_pass_by_type(const PassType type, const Pass **r_pass, int *r_offset = nullptr) const;

 protected:
  const vector<Pass> &passes_;

  int pass_offset_ = -1;
  const Pass *pass_ = nullptr;

  int num_components_ = 0;

  float exposure_ = 0.0f;
  int num_samples_ = 0;

  bool approximate_shadow_in_matte_ = false;
};

CCL_NAMESPACE_END
