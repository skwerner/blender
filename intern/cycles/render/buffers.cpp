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

/* Buffer Params */

BufferParams::BufferParams()
{
  width = 0;
  height = 0;

  full_x = 0;
  full_y = 0;
  full_width = 0;
  full_height = 0;
}

void BufferParams::update_passes(vector<Pass> &passes)
{
  update_offset_stride();

  pass_sample_count = PASS_UNUSED;
  pass_denoising_color = PASS_UNUSED;
  pass_denoising_normal = PASS_UNUSED;
  pass_denoising_albedo = PASS_UNUSED;

  pass_stride = 0;
  for (const Pass &pass : passes) {
    switch (pass.type) {
      case PASS_SAMPLE_COUNT:
        pass_sample_count = pass_stride;
        break;

      case PASS_DENOISING_COLOR:
        pass_denoising_color = pass_stride;
        break;
      case PASS_DENOISING_NORMAL:
        pass_denoising_normal = pass_stride;
        break;
      case PASS_DENOISING_ALBEDO:
        pass_denoising_albedo = pass_stride;
        break;

      default:
        break;
    }

    pass_stride += pass.components;
  }

  pass_stride = align_up(pass_stride, 4);
}

void BufferParams::update_offset_stride()
{
  offset = -(full_x + full_y * width);
  stride = width;
}

bool BufferParams::modified(const BufferParams &params) const
{
  return !(width == params.width && height == params.height && full_x == params.full_x &&
           full_y == params.full_y && full_width == params.full_width &&
           full_height == params.full_height && offset == params.offset &&
           stride == params.stride && pass_stride == params.pass_stride &&
           pass_denoising_color == params.pass_denoising_color &&
           pass_denoising_normal == params.pass_denoising_normal &&
           pass_denoising_albedo == params.pass_denoising_albedo);
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

CCL_NAMESPACE_END
