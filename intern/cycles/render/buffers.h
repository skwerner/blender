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
  int width = 0;
  int height = 0;

  /* offset into and width/height of the full buffer */
  int full_x = 0;
  int full_y = 0;
  int full_width = 0;
  int full_height = 0;

  /* Runtime fields, only valid after `update_passes()` or `update_offset_stride()`. */
  int offset = -1, stride = -1;

  /* Runtime fields, only valid after `update_passes()`. */
  int pass_stride = -1;

  /* functions */
  BufferParams();

  /* Pre-calculate all fields which depends on the passes. */
  void update_passes(vector<Pass *> &passes);

  /* Returns PASS_UNUSED if there is no such pass in the buffer. */
  int get_pass_offset(PassType type, PassMode mode = PassMode::NOISY) const;

  void update_offset_stride();

  bool modified(const BufferParams &other) const;

 protected:
  void reset_pass_offset();

  /* Multipled by 2 to be able to store noisy and denoised pass types. */
  static constexpr int kNumPassOffsets = PASS_NUM * 2;

  /* Indexed by pass type, indicates offset of the corresponding pass in the buffer. */
  int pass_offset_[kNumPassOffsets];
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
  void copy_to_device();
};

/* Copy denoised passes form source to destination.
 *
 * Buffer parameters are provided explicitly, allowing to copy pixelks between render buffers which
 * content corresponds to a render result at a non-unit resolution divider.
 *
 * `src_offset` allows to offset source pixel index which is used when a fraction of the source
 * buffer is to be copied.
 *
 * Copy happens of the number of pixels in the destination. */
void render_buffers_host_copy_denoised(RenderBuffers *dst,
                                       const BufferParams &dst_params,
                                       const RenderBuffers *src,
                                       const BufferParams &src_params,
                                       const size_t src_offset = 0);

CCL_NAMESPACE_END

#endif /* __BUFFERS_H__ */
