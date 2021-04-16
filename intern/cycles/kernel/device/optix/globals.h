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

#include "kernel/integrator/integrator_path_state.h"
#include "kernel/integrator/integrator_state.h"

CCL_NAMESPACE_BEGIN

typedef struct ShaderParams {
  KernelShaderEvalInput *input;
  float4 *output;
  int type;
  int filter;
  int sx;
  int offset;
  int sample;
} ShaderParams;

typedef struct KernelParams {
  IntegratorState __integrator_state;
  IntegratorQueueCounter *__integrator_queue_counter;
  int *__integrator_sort_key;
  int *__integrator_sort_key_counter;
  int *path_index_array;

  //KernelWorkTile tile;
  KernelData data;
  ShaderParams shader;
#define KERNEL_TEX(type, name) const type *name;
#include "kernel/kernel_textures.h"
} KernelParams;

typedef struct KernelGlobals {
#ifdef __VOLUME__
  VolumeState volume_state;
#endif
  Intersection hits_stack[64];
} KernelGlobals;

#ifdef __NVCC__
extern "C" __constant__ KernelParams __params;
#endif

CCL_NAMESPACE_END
