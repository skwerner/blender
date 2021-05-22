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

#ifndef __BUFFERS_H__
#define __BUFFERS_H__

#include "device/device_memory.h"

#include "render/film.h"

#include "kernel/kernel_types.h"

#include "util/util_half.h"
#include "util/util_string.h"
#include "util/util_thread.h"
#include "util/util_types.h"

CCL_NAMESPACE_BEGIN

class Device;
struct DeviceDrawParams;
struct float4;

/* Buffer Parameters
 * Size of render buffer and how it fits in the full image (border render). */

class BufferParams {
 public:
  /* width/height of the physical buffer */
  int width;
  int height;

  /* offset into and width/height of the full buffer */
  int full_x;
  int full_y;
  int full_width;
  int full_height;

  /* Runtime fields, only valid after `update_passes()` or `update_offset_stride()`. */
  int offset = -1, stride = -1;

  /* Runtime fields, only valid after `update_passes()`. */
  int pass_stride = -1;

  /* Offsets of passes needed for the rendering functionality like adaptive sampling and denoising.
   * Pre-calculated so that they are available in areas where list of passes is not accessible. */
  int pass_sample_count = PASS_UNUSED;
  int pass_denoising_color = PASS_UNUSED;
  int pass_denoising_normal = PASS_UNUSED;
  int pass_denoising_albedo = PASS_UNUSED;

  /* functions */
  BufferParams();

  /* Pre-calculate all fields which depends on the passes. */
  void update_passes(vector<Pass> &passes);

  void update_offset_stride();

  bool modified(const BufferParams &params) const;
};

/* Render Buffers */

class RenderBuffers {
 public:
  /* buffer parameters */
  BufferParams params;

  /* float buffer */
  device_vector<float> buffer;

  explicit RenderBuffers(Device *device);
  ~RenderBuffers();

  void reset(const BufferParams &params);
  void zero();

  bool copy_from_device();
};

/* Render Tile
 * Rendering task on a buffer */

class RenderTile {
 public:
  typedef enum { PATH_TRACE = (1 << 0), BAKE = (1 << 1), DENOISE = (1 << 2) } Task;

  Task task;
  int x, y, w, h;
  int start_sample;
  int num_samples;
  int sample;
  int resolution;
  int offset;
  int stride;
  int tile_index;

  device_ptr buffer;
  int device_size;

  typedef enum { NO_STEALING = 0, CAN_BE_STOLEN = 1, WAS_STOLEN = 2 } StealingState;
  StealingState stealing_state;

  RenderBuffers *buffers;

  RenderTile();

  int4 bounds() const
  {
    return make_int4(x,      /* xmin */
                     y,      /* ymin */
                     x + w,  /* xmax */
                     y + h); /* ymax */
  }
};

/* Render Tile Neighbors
 * Set of neighboring tiles used for denoising. Tile order:
 *  0 1 2
 *  3 4 5
 *  6 7 8 */

class RenderTileNeighbors {
 public:
  static const int SIZE = 9;
  static const int CENTER = 4;

  RenderTile tiles[SIZE];
  RenderTile target;

  RenderTileNeighbors(const RenderTile &center)
  {
    tiles[CENTER] = center;
  }

  int4 bounds() const
  {
    return make_int4(tiles[3].x,               /* xmin */
                     tiles[1].y,               /* ymin */
                     tiles[5].x + tiles[5].w,  /* xmax */
                     tiles[7].y + tiles[7].h); /* ymax */
  }

  void set_bounds_from_center()
  {
    tiles[3].x = tiles[CENTER].x;
    tiles[1].y = tiles[CENTER].y;
    tiles[5].x = tiles[CENTER].x + tiles[CENTER].w;
    tiles[7].y = tiles[CENTER].y + tiles[CENTER].h;
  }
};

CCL_NAMESPACE_END

#endif /* __BUFFERS_H__ */
