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

/* Constant Globals */

#pragma once

#include "kernel/kernel_profiling.h"
#include "kernel/kernel_types.h"

#include "util/util_atomic.h"

CCL_NAMESPACE_BEGIN

#define KERNEL_TEX(type, name) typedef type name##_t;
#include "kernel/kernel_textures.h"

typedef ccl_addr_space struct KernelGlobals {
  ccl_constant KernelData *data;
  ccl_global char *buffers[8];

#define KERNEL_TEX(type, name) TextureInfo name;
#include "kernel/kernel_textures.h"

#ifdef __SPLIT_KERNEL__
  SplitData split_data;
  SplitParams split_param_data;
#endif
} KernelGlobals;

#define KERNEL_BUFFER_PARAMS \
  ccl_global char *buffer0, ccl_global char *buffer1, ccl_global char *buffer2, \
      ccl_global char *buffer3, ccl_global char *buffer4, ccl_global char *buffer5, \
      ccl_global char *buffer6, ccl_global char *buffer7

#define KERNEL_BUFFER_ARGS buffer0, buffer1, buffer2, buffer3, buffer4, buffer5, buffer6, buffer7

ccl_device_inline void kernel_set_buffer_pointers(KernelGlobals *kg, KERNEL_BUFFER_PARAMS)
{
#ifdef __SPLIT_KERNEL__
  if (ccl_local_id(0) + ccl_local_id(1) == 0)
#endif
  {
    kg->buffers[0] = buffer0;
    kg->buffers[1] = buffer1;
    kg->buffers[2] = buffer2;
    kg->buffers[3] = buffer3;
    kg->buffers[4] = buffer4;
    kg->buffers[5] = buffer5;
    kg->buffers[6] = buffer6;
    kg->buffers[7] = buffer7;
  }

#ifdef __SPLIT_KERNEL__
  ccl_barrier(CCL_LOCAL_MEM_FENCE);
#endif
}

ccl_device_inline void kernel_set_buffer_info(KernelGlobals *kg)
{
#ifdef __SPLIT_KERNEL__
  if (ccl_local_id(0) + ccl_local_id(1) == 0)
#endif
  {
    ccl_global TextureInfo *info = (ccl_global TextureInfo *)kg->buffers[0];

#define KERNEL_TEX(type, name) kg->name = *(info++);
#include "kernel/kernel_textures.h"
  }

#ifdef __SPLIT_KERNEL__
  ccl_barrier(CCL_LOCAL_MEM_FENCE);
#endif
}

CCL_NAMESPACE_END
