/*
 * Copyright 2021 Blender Foundation
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

#include "util/util_half.h"
#include "util/util_thread.h"
#include "util/util_types.h"

CCL_NAMESPACE_BEGIN

class BufferParams;

class GPUDisplayParams {
 public:
  /* Offset of the display within a viewport.
   * For example, set to a lower-bottom corner of border render in Blender's viewport. */
  int2 offset = make_int2(0, 0);

  /* Full viewport size. */
  int2 full_size = make_int2(0, 0);

  /* Effective vieport size.
   * In the case of border render, size of the border rectangle.
   *
   * NOTE: This is size of viewport. The size in pixels of texture needed to draw this viewport due
   * to resolution divider used during viewport navigation. */
  int2 size = make_int2(0, 0);

  bool modified(const GPUDisplayParams &other) const
  {
    return !(offset == other.offset && full_size == other.full_size && size == other.size);
  }
};

class GPUDisplay {
 public:
  GPUDisplay() = default;
  virtual ~GPUDisplay() = default;

  /* Reset the display for its new configuration.
   * Will invalidate the texture as well. */
  virtual void reset(BufferParams &buffer_params);

  /* Copy rendered pixels from path tracer to a GPU texture.
   *
   * NOTE: The caller must acquire GPUDisplay::mutex prior of using this function.
   *
   * The reason for this is is to allow use of this function for partial updates from different
   * devices. In this case the caller will acquire the lock once, update all the slices and release
   * the lock once. This will ensure that draw() will never use partially updated texture. */
  /* TODO(sergey): Specify parameters which will allow to do partial updates, which will be needed
   * to update the texture from multiple devices. */
  /* TODO(sergey): Do we need to support uint8 data type? */
  virtual void copy_pixels_to_texture(const half4 *rgba_pixels, int width, int height) = 0;

  /* Access CUDA buffer which can be used to define GPU-side texture without extra data copy. */
  /* TODO(sergey): Depending on a point of view, might need to be called "set" instead, so that
   * the render session sets CUDA buffer to a display owned by viewport. */
  /* TODO(sergey): Need proper return value. */
  virtual void get_cuda_buffer() = 0;

  /* Returns true if drawing was performed. */
  virtual bool draw() = 0;

  thread_mutex mutex;

 protected:
  GPUDisplayParams params_;
};

CCL_NAMESPACE_END
