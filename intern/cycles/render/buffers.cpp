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

static int pass_to_index(const Pass &pass)
{
  return pass_type_mode_to_index(pass.type, pass.mode);
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

void BufferParams::update_passes(vector<Pass> &passes)
{
  update_offset_stride();
  reset_pass_offset();

  pass_stride = 0;
  for (const Pass &pass : passes) {
    const int index = pass_to_index(pass);

    if (pass_offset_[index] == PASS_UNUSED) {
      pass_offset_[index] = pass_stride;
    }

    if (pass.is_written()) {
      pass_stride += pass.get_info().num_components;
    }
  }

  pass_stride = align_up(pass_stride, 4);
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

/* Render Buffer Task */

RenderTile::RenderTile()
{
  x = 0;
  y = 0;
  w = 0;
  h = 0;

  sample = 0;
  start_sample = 0;
  num_samples = 0;
  resolution = 0;

  offset = 0;
  stride = 0;

  buffer = 0;

  buffers = NULL;
  stealing_state = NO_STEALING;
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
  buffer.zero_to_device();
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

CCL_NAMESPACE_END
