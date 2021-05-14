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

#include "render/pass.h"
#include "util/util_string.h"
#include "util/util_vector.h"

CCL_NAMESPACE_BEGIN

class Film;
class RenderBuffers;

/* Helper class which allows to access pass data.
 * Is designed in a way that it is created once when the pass data is known, and then pixels gets
 * progressively update from various render buffers. */
class PassAccessor {
 public:
  PassAccessor(const Film *film,
               const vector<Pass> &passes,
               const Pass *pass,
               int num_components,
               float exposure,
               int num_samples);

  /* Get pass data from the given render buffers, perform needed filtering, and store result into
   * the pixels.
   * The result is stored sequentially starting from the very beginning of the pixels memory. */
  bool get_render_tile_pixels(RenderBuffers *render_buffers, float *pixels);

#if 0
  bool set_pass_rect(PassType type, int components, float *pixels);
#endif

  /* Returns PASS_UNUSED if there is no pass with the given type. */
  int get_pass_offset(PassType type) const;

 protected:
  const Film *film_;

  const vector<Pass> &passes_;

  int pass_offset_ = -1;
  const Pass *pass_ = nullptr;

  int num_components_ = 0;

  float exposure_ = 0.0f;
  int num_samples_ = 0;
};

CCL_NAMESPACE_END
