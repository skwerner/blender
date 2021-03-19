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

/* GPUDisplay class takes care of drawing render result in a viewport. The render result is stored
 * in a GPU-side texture, which is updated from a path tracer and drawn by an application.
 *
 * The base GPUDisplay does some special texture state tracking, which allows render Session to
 * make decisions on whether reset for an updated state is possible or not. This state should only
 * be tracked in a base class and a particular implementation should not worry about it.
 *
 * The subclasses should only implement the pure virtual methods, which allows them to not worry
 * about parent method calls, which helps them to be as small and reliable as possible. */

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

  /* Reset the display for the new state of render session. Is called whenever session is reset,
   * which happens on changes like viewport navigation or viewport dimension change.
   *
   * This call will configure parameters for a changed buffer and reset the texture state.
   *
   * NOTE: This call acquires the GPUDisplay::mutex lock. */
  void reset(BufferParams &buffer_params);

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
  void copy_pixels_to_texture(const half4 *rgba_pixels, int width, int height);

  /* Access CUDA buffer which can be used to define GPU-side texture without extra data copy. */
  /* TODO(sergey): Depending on a point of view, might need to be called "set" instead, so that
   * the render session sets CUDA buffer to a display owned by viewport. */
  /* TODO(sergey): Need proper return value. */
  virtual void get_cuda_buffer() = 0;

  /* Draw the current state of the texture.
   *
   * Returns truth if this call did draw an updated state of the texture.
   *
   * NOTE: This call acquires the GPUDisplay::mutex lock. */
  bool draw();

  thread_mutex mutex;

 protected:
  /* Implementation-specific calls which subclasses are to implement.
   * These `do_foo()` method corresponds to their `foo()` calls, but they are purely virtual to
   * simplify their particular implementation. */
  virtual void do_copy_pixels_to_texture(const half4 *rgba_pixels, int width, int height) = 0;
  virtual void do_draw() = 0;

  GPUDisplayParams params_;

 private:
  /* State of the texture, which is needed for an integration with render session and interactive
   * updates and navigation. */
  struct {
    /* Denotes whether possibly existing state of GPU side texture is still usable.
     * It will not be usable in cases like render border did change (in this case we don't want
     * previous texture to be rendered at all).
     *
     * However, if only navigation or object in scene did change, then the outdated state of the
     * texture is still usable for draw, preventing display viewport flickering on navigation and
     * object modifications. */
    bool is_usable = false;

    /* Texture is considered outdated after `reset()` until the next call of
     * `copy_pixels_to_texture()`. */
    bool is_outdated = true;
  } texture_state_;
};

CCL_NAMESPACE_END
