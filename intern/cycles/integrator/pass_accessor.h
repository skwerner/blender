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
#include "util/util_half.h"
#include "util/util_string.h"
#include "util/util_types.h"
#include "util/util_vector.h"

CCL_NAMESPACE_BEGIN

class Film;
class RenderBuffers;
class BufferParams;
struct KernelFilmConvert;

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

    bool show_active_pixels = false;
  };

  class Destination {
   public:
    Destination() = default;
    Destination(float *pixels, int num_components);
    Destination(const PassType pass_type, half4 *pixels);

    /* Destination will be initialized with the number of components which is native for the given
     * pass type. */
    explicit Destination(const PassType pass_type);

    /* CPU-side pointers. only usable by the `PassAccessorCPU`. */
    float *pixels = nullptr;
    half4 *pixels_half_rgba = nullptr;

    /* Device-side pointers. */
    device_ptr d_pixels_half_rgba;

    int num_components = 0;
  };

  class Source {
   public:
    Source() = default;
    Source(const float *pixels, int num_components);

    /* CPU-side pointers. only usable by the `PassAccessorCPU`. */
    const float *pixels = nullptr;
    int num_components = 0;
  };

  PassAccessor(const PassAccessInfo &pass_access_info, float exposure, int num_samples);

  virtual ~PassAccessor() = default;

  /* Get pass data from the given render buffers, perform needed filtering, and store result into
   * the pixels.
   * The result is stored sequentially starting from the very beginning of the pixels memory. */
  bool get_render_tile_pixels(const RenderBuffers *render_buffers,
                              const Destination &destination) const;
  bool get_render_tile_pixels(const RenderBuffers *render_buffers,
                              const BufferParams &buffer_params,
                              const Destination &destination) const;
  /* Set pass data for the given render buffers. Used for baking to read from passes. */
  bool set_render_tile_pixels(RenderBuffers *render_buffers, const Source &source);

#if 0
  bool set_pass_rect(PassType type, int components, float *pixels);
#endif

 protected:
  virtual void init_kernel_film_convert(KernelFilmConvert *kfilm_convert,
                                        const BufferParams &buffer_params,
                                        const Destination &destination) const;

#define DECLARE_PASS_ACCESSOR(pass) \
  virtual void get_pass_##pass(const RenderBuffers *render_buffers, \
                               const BufferParams &buffer_params, \
                               const Destination &destination) const = 0;

  /* Float (scalar) passes. */
  DECLARE_PASS_ACCESSOR(depth)
  DECLARE_PASS_ACCESSOR(mist)
  DECLARE_PASS_ACCESSOR(sample_count)
  DECLARE_PASS_ACCESSOR(float)

  /* Float3 passes. */
  DECLARE_PASS_ACCESSOR(divide_even_color)
  DECLARE_PASS_ACCESSOR(float3)

  /* Float4 passes. */
  DECLARE_PASS_ACCESSOR(motion)
  DECLARE_PASS_ACCESSOR(cryptomatte)
  DECLARE_PASS_ACCESSOR(shadow_catcher)
  DECLARE_PASS_ACCESSOR(shadow_catcher_matte_with_shadow)
  DECLARE_PASS_ACCESSOR(float4)

  /* Float3 or Float4 passes. */
  DECLARE_PASS_ACCESSOR(shadow)

#undef DECLARE_PASS_ACCESSOR

  PassAccessInfo pass_access_info_;

  float exposure_ = 0.0f;
  int num_samples_ = 0;
};

CCL_NAMESPACE_END
