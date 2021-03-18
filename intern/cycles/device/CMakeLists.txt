# Copyright 2011-2020 Blender Foundation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set(INC
  ..
  ../../glew-mx
)

set(INC_SYS
  ${GLEW_INCLUDE_DIR}
  ../../../extern/clew/include
)

if(WITH_CUDA_DYNLOAD)
  list(APPEND INC
    ../../../extern/cuew/include
  )
  add_definitions(-DWITH_CUDA_DYNLOAD)
else()
  list(APPEND INC_SYS
    ${CUDA_TOOLKIT_INCLUDE}
  )
  add_definitions(-DCYCLES_CUDA_NVCC_EXECUTABLE="${CUDA_NVCC_EXECUTABLE}")
endif()

set(SRC
  device.cpp
  device_denoising.cpp
  device_kernel.cpp
  device_memory.cpp
  device_queue.cpp
  device_task.cpp
)

set(SRC_CPU
  cpu/device.cpp
  cpu/device.h
  cpu/device_impl.cpp
  cpu/device_impl.h
  cpu/kernel.cpp
  cpu/kernel.h
  cpu/kernel_function.h
  cpu/kernel_thread_globals.cpp
  cpu/kernel_thread_globals.h
  cpu/queue.cpp
  cpu/queue.h
)

set(SRC_CUDA
  cuda/device.cpp
  cuda/device.h
  cuda/device_impl.cpp
  cuda/device_impl.h
  cuda/kernel.cpp
  cuda/kernel.h
  cuda/queue.cpp
  cuda/queue.h
  cuda/util.cpp
  cuda/util.h
)

set(SRC_DUMMY
  dummy/device.cpp
  dummy/device.h
)

set(SRC_MULTI
  multi/device.cpp
  multi/device.h
)

set(SRC_OPENCL
  opencl/device.cpp
  opencl/device.h
  opencl/device_impl.cpp
  opencl/device_impl.h
  opencl/memory_manager.h
  opencl/memory_manager.cpp
  opencl/util.cpp
)

set(SRC_OPTIX
  optix/device.cpp
  optix/device.h
)

set(SRC_HEADERS
  device.h
  device_denoising.h
  device_memory.h
  device_kernel.h
  device_queue.h
  device_task.h
)

set(LIB
  cycles_render
  cycles_kernel
  cycles_util
  ${CYCLES_GL_LIBRARIES}
)

if(WITH_CUDA_DYNLOAD)
  list(APPEND LIB
    extern_cuew
  )
else()
  list(APPEND LIB
    ${CUDA_CUDA_LIBRARY}
  )
endif()

add_definitions(${GL_DEFINITIONS})

if(WITH_CYCLES_DEVICE_OPENCL)
  list(APPEND LIB
    extern_clew
  )
  add_definitions(-DWITH_OPENCL)
endif()
if(WITH_CYCLES_DEVICE_CUDA)
  add_definitions(-DWITH_CUDA)
endif()
if(WITH_CYCLES_DEVICE_OPTIX)
  add_definitions(-DWITH_OPTIX)
endif()
if(WITH_CYCLES_DEVICE_MULTI)
  add_definitions(-DWITH_MULTI)
endif()

if(WITH_OPENIMAGEDENOISE)
  list(APPEND LIB
    ${OPENIMAGEDENOISE_LIBRARIES}
  )
endif()

include_directories(${INC})
include_directories(SYSTEM ${INC_SYS})

cycles_add_library(cycles_device "${LIB}"
  ${SRC}
  ${SRC_CPU}
  ${SRC_CUDA}
  ${SRC_DUMMY}
  ${SRC_MULTI}
  ${SRC_OPENCL}
  ${SRC_OPTIX}
  ${SRC_HEADERS}
)

