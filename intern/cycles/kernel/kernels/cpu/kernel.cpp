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

#include "util/util_kernel_isa.h"

#include "kernel/kernel.h"
#define KERNEL_ARCH cpu
#include "kernel/kernels/cpu/kernel_cpu_impl.h"

CCL_NAMESPACE_BEGIN

/* Memory Copy */

void kernel_const_copy(KernelGlobals *kg, const char *name, void *host, size_t)
{
  if (strcmp(name, "__data") == 0) {
    kg->__data = *(KernelData *)host;
  }
  else {
    assert(0);
  }
}

void kernel_global_memory_copy(KernelGlobals *kg, const char *name, void *mem, size_t size)
{
  if (0) {
  }

#define KERNEL_TEX(type, tname) \
  else if (strcmp(name, #tname) == 0) \
  { \
    kg->tname.data = (type *)mem; \
    kg->tname.width = size; \
  }
#include "kernel/kernel_textures.h"
  else {
    assert(0);
  }
}

CCL_NAMESPACE_END
