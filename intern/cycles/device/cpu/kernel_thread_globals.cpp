/*
 * Copyright 2011-2021 Blender Foundation
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

#include "device/cpu/kernel_thread_globals.h"

// clang-format off
#include "kernel/osl/osl_shader.h"
#include "kernel/osl/osl_globals.h"
// clang-format on

CCL_NAMESPACE_BEGIN

/* TODO(sergey): Consider making more available function. Maybe `util_memory.h`? */
static void safe_free(void *mem)
{
  if (mem == nullptr) {
    return;
  }
  free(mem);
}

/* Get number of elements in a bound array. */
/* TODO(sergey): Make this function more re-usable. */
template<class T, int N> constexpr inline int ARRAY_SIZE(T (&/*array*/)[N])
{
  return N;
}

CPUKernelThreadGlobals::CPUKernelThreadGlobals()
{
  reset_runtime_memory();
}

CPUKernelThreadGlobals::CPUKernelThreadGlobals(const KernelGlobals &kernel_globals,
                                               void *osl_globals_memory)
    : KernelGlobals(kernel_globals)
{
  reset_runtime_memory();

  decoupled_volume_steps_index = 0;
  coverage_asset = nullptr;
  coverage_object = nullptr;
  coverage_material = nullptr;

#ifdef WITH_OSL
  OSLShader::thread_init(this, reinterpret_cast<OSLGlobals *>(osl_globals_memory));
#else
  (void)osl_globals_memory;
#endif
}

CPUKernelThreadGlobals::CPUKernelThreadGlobals(CPUKernelThreadGlobals &&other) noexcept
    : KernelGlobals(std::move(other))
{
  other.reset_runtime_memory();
}

CPUKernelThreadGlobals::~CPUKernelThreadGlobals()
{
  safe_free(transparent_shadow_intersections);

  const int decoupled_count = ARRAY_SIZE(decoupled_volume_steps);
  for (int i = 0; i < decoupled_count; ++i) {
    safe_free(decoupled_volume_steps[i]);
  }
#ifdef WITH_OSL
  OSLShader::thread_free(this);
#endif
}

CPUKernelThreadGlobals &CPUKernelThreadGlobals::operator=(CPUKernelThreadGlobals &&other)
{
  if (this == &other) {
    return *this;
  }

  *static_cast<KernelGlobals *>(this) = *static_cast<KernelGlobals *>(&other);

  other.reset_runtime_memory();

  return *this;
}

void CPUKernelThreadGlobals::reset_runtime_memory()
{
  transparent_shadow_intersections = nullptr;

#ifdef WITH_OSL
  osl = nullptr;
#endif

  memset(decoupled_volume_steps, 0, sizeof(decoupled_volume_steps));
}

CCL_NAMESPACE_END
