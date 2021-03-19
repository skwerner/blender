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

#include "render/gpu_display.h"

#include "render/buffers.h"
#include "util/util_logging.h"

CCL_NAMESPACE_BEGIN

void GPUDisplay::reset(BufferParams &buffer_params)
{
  thread_scoped_lock lock(mutex);

  const GPUDisplayParams old_params = params_;

  params_.offset = make_int2(buffer_params.full_x, buffer_params.full_y);
  params_.full_size = make_int2(buffer_params.full_width, buffer_params.full_height);
  params_.size = make_int2(buffer_params.width, buffer_params.height);

  /* If the parameters did change tag texture as unusable. This avoids drawing old texture content
   * in an updated configuration of the viewport. For example, avoids drawing old frame when render
   * border did change.
   * If the parameters did not change, allow drawing the current state of the texture, which will
   * not count as an up-to-date redraw. This will avoid flickering when doping camera navigation by
   * showing a previously rendered frame for until the new one is ready. */
  if (old_params.modified(params_)) {
    texture_state_.is_usable = false;
  }

  texture_state_.is_outdated = true;
}

void GPUDisplay::copy_pixels_to_texture(const half4 *rgba_pixels, int width, int height)
{
  texture_state_.is_outdated = false;
  texture_state_.is_usable = true;

  do_copy_pixels_to_texture(rgba_pixels, width, height);
}

half4 *GPUDisplay::map_texture_buffer(int width, int height)
{
  DCHECK(!is_mapped_);

  half4 *mapped_rgba_pixels = do_map_texture_buffer(width, height);

  if (mapped_rgba_pixels) {
    is_mapped_ = true;
  }

  return mapped_rgba_pixels;
}

void GPUDisplay::unmap_texture_buffer()
{
  DCHECK(is_mapped_);

  is_mapped_ = false;

  texture_state_.is_outdated = false;
  texture_state_.is_usable = true;

  do_unmap_texture_buffer();
}

bool GPUDisplay::draw()
{
  thread_scoped_lock lock(mutex);

  if (texture_state_.is_usable) {
    do_draw();
  }

  return !texture_state_.is_outdated;
}

CCL_NAMESPACE_END
