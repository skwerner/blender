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

CCL_NAMESPACE_BEGIN

/* For CUDA, constant memory textures must be globals, so we can't put them
 * into a struct. As a result we don't actually use this struct and use actual
 * globals and simply pass along a NULL pointer everywhere, which we hope gets
 * optimized out. */

__constant__ KernelData __data;
typedef struct KernelGlobals {
  /* NOTE: Keep the size in sync with SHADOW_STACK_MAX_HITS. */
  Intersection hits_stack[64];
} KernelGlobals;

#define KERNEL_TEX(type, name) const __constant__ __device__ type *name;
#include "kernel/kernel_textures.h"

CCL_NAMESPACE_END
