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

#include <stdlib.h>

#include "device/device.h"
#include "render/buffers.h"

#include "util/util_foreach.h"
#include "util/util_hash.h"
#include "util/util_math.h"
#include "util/util_opengl.h"
#include "util/util_time.h"
#include "util/util_types.h"

CCL_NAMESPACE_BEGIN

/* Convert part information to an index of `BufferParams::pass_offset_`. */

static int pass_type_mode_to_index(PassType pass_type, PassMode mode)
{
  int index = static_cast<int>(pass_type) * 2;

  if (mode == PassMode::DENOISED) {
    ++index;
  }

  return index;
}

static int pass_to_index(const Pass *pass)
{
  return pass_type_mode_to_index(pass->get_type(), pass->get_mode());
}

/* Buffer Params */

BufferParams::BufferParams()
{
  width = 0;
  height = 0;

  full_x = 0;
  full_y = 0;
  full_width = 0;
  full_height = 0;

  reset_pass_offset();
}

void BufferParams::update_passes(const vector<Pass *> &passes)
{
  update_offset_stride();
  reset_pass_offset();

  pass_stride = 0;
  for (const Pass *pass : passes) {
    const int index = pass_to_index(pass);

    if (pass->is_written()) {
      if (pass_offset_[index] == PASS_UNUSED) {
        pass_offset_[index] = pass_stride;
      }

      pass_stride += pass->get_info().num_components;
    }
  }
}

void BufferParams::reset_pass_offset()
{
  for (int i = 0; i < kNumPassOffsets; ++i) {
    pass_offset_[i] = PASS_UNUSED;
  }
}

int BufferParams::get_pass_offset(PassType pass_type, PassMode mode) const
{
  if (pass_type == PASS_NONE || pass_type == PASS_UNUSED) {
    return PASS_UNUSED;
  }

  const int index = pass_type_mode_to_index(pass_type, mode);
  return pass_offset_[index];
}

void BufferParams::update_offset_stride()
{
  offset = -(full_x + full_y * width);
  stride = width;
}

bool BufferParams::modified(const BufferParams &other) const
{
  if (!(width == other.width && height == other.height && full_x == other.full_x &&
        full_y == other.full_y && full_width == other.full_width &&
        full_height == other.full_height && offset == other.offset && stride == other.stride &&
        pass_stride == other.pass_stride)) {
    return true;
  }

  return memcmp(pass_offset_, other.pass_offset_, sizeof(pass_offset_)) != 0;
}

/* Render Buffers */

RenderBuffers::RenderBuffers(Device *device) : buffer(device, "RenderBuffers", MEM_READ_WRITE)
{
}

RenderBuffers::~RenderBuffers()
{
  buffer.free();
}

void RenderBuffers::reset(const BufferParams &params_)
{
  DCHECK(params_.pass_stride != -1);

  params = params_;

  /* re-allocate buffer */
  buffer.alloc(params.width * params.pass_stride, params.height);
}

void RenderBuffers::zero()
{
  buffer.zero_to_device();
}

bool RenderBuffers::copy_from_device()
{
  DCHECK(params.pass_stride != -1);

  if (!buffer.device_pointer)
    return false;

  buffer.copy_from_device(0, params.width * params.pass_stride, params.height);

  return true;
}

void RenderBuffers::copy_to_device()
{
  buffer.copy_to_device();
}

void render_buffers_host_copy_denoised(RenderBuffers *dst,
                                       const BufferParams &dst_params,
                                       const RenderBuffers *src,
                                       const BufferParams &src_params,
                                       const size_t src_offset)
{
  DCHECK_EQ(dst_params.width, src_params.width);
  /* TODO(sergey): More sanity checks to avoid buffer overrun. */

  /* Create a map of pass ofsets to be copied.
   * Assume offsets are different to allow copying passes between buffers with different set of
   * passes. */

  struct {
    int dst_offset;
    int src_offset;
  } pass_offsets[PASS_NUM];

  int num_passes = 0;

  for (int i = 0; i < PASS_NUM; ++i) {
    const PassType pass_type = static_cast<PassType>(i);

    const int dst_pass_offset = dst_params.get_pass_offset(pass_type, PassMode::DENOISED);
    if (dst_pass_offset == PASS_UNUSED) {
      continue;
    }

    const int src_pass_offset = src_params.get_pass_offset(pass_type, PassMode::DENOISED);
    if (src_pass_offset == PASS_UNUSED) {
      continue;
    }

    pass_offsets[num_passes].dst_offset = dst_pass_offset;
    pass_offsets[num_passes].src_offset = src_pass_offset;
    ++num_passes;
  }

  /* Copy passes. */
  /* TODO(sergey): Make it more reusable, allowing implement copy of noisy passes. */

  const int64_t dst_width = dst_params.width;
  const int64_t dst_height = dst_params.height;
  const int64_t dst_pass_stride = dst_params.pass_stride;
  const int64_t dst_num_pixels = dst_width * dst_height;

  const int64_t src_pass_stride = src_params.pass_stride;
  const int64_t src_offset_in_floats = src_offset * src_pass_stride;

  const float *src_pixel = src->buffer.data() + src_offset_in_floats;
  float *dst_pixel = dst->buffer.data();

  for (int i = 0; i < dst_num_pixels;
       ++i, src_pixel += src_pass_stride, dst_pixel += dst_pass_stride) {
    for (int pass_offset_idx = 0; pass_offset_idx < num_passes; ++pass_offset_idx) {
      const int dst_pass_offset = pass_offsets[pass_offset_idx].dst_offset;
      const int src_pass_offset = pass_offsets[pass_offset_idx].src_offset;

      /* TODO(sergey): Support non-RGBA passes. */
      dst_pixel[dst_pass_offset + 0] = src_pixel[src_pass_offset + 0];
      dst_pixel[dst_pass_offset + 1] = src_pixel[src_pass_offset + 1];
      dst_pixel[dst_pass_offset + 2] = src_pixel[src_pass_offset + 2];
      dst_pixel[dst_pass_offset + 3] = src_pixel[src_pass_offset + 3];
    }
  }
}

CCL_NAMESPACE_END
