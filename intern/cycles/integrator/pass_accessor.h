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
  class PassAccessInfo {
   public:
    PassAccessInfo() = default;
    PassAccessInfo(const Pass &pass, const Film &film, const vector<Pass> &passes);

    PassType type = PASS_NONE;
    int offset = -1;

    /* For the shadow catcher matte pass: whether to approximate shadow catcher pass into its
     * matte pass, so that both artificial objects and shadows can be alpha-overed onto a backdrop.
     */
    bool use_approximate_shadow_catcher = false;
  };

  PassAccessor(const PassAccessInfo &pass_access_info,
               int num_pixel_components,
               float exposure,
               int num_samples);

  /* Get pass data from the given render buffers, perform needed filtering, and store result into
   * the pixels.
   * The result is stored sequentially starting from the very beginning of the pixels memory. */
  bool get_render_tile_pixels(RenderBuffers *render_buffers, float *pixels);

#if 0
  bool set_pass_rect(PassType type, int components, float *pixels);
#endif

 protected:
  PassAccessInfo pass_access_info_;

  int num_components_ = 0;

  float exposure_ = 0.0f;
  int num_samples_ = 0;
};

CCL_NAMESPACE_END
