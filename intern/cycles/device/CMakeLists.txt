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
  cpu/device_cpu.cpp
  cpu/device_cpu.h
  cpu/device_queue.cpp
  cpu/device_queue.h
  cpu/kernel.cpp
  cpu/kernel.h
  cpu/kernel_function.h
  cpu/kernel_thread_globals.cpp
  cpu/kernel_thread_globals.h
)

set(SRC_CUDA
  cuda/device_cuda.cpp
  cuda/device_cuda.h
  cuda/device_cuda_impl.cpp
  cuda/device_cuda_impl.h
)

set(SRC_DUMMY
  dummy/device_dummy.cpp
  dummy/device_dummy.h
)

set(SRC_MULTI
  multi/device_multi.cpp
  multi/device_multi.h
)

set(SRC_OPENCL
  opencl/device_opencl.cpp
  opencl/device_opencl.h
  opencl/device_opencl_impl.cpp
  opencl/device_opencl_impl.h
  opencl/memory_manager.h
  opencl/memory_manager.cpp
  opencl/opencl_util.cpp
)

set(SRC_OPTIX
  optix/device_optix.cpp
  optix/device_optix.h
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
  add_definitions(-DWITH_OPENIMAGEDENOISE)
  add_definitions(-DOIDN_STATIC_LIB)
  list(APPEND INC_SYS
    ${OPENIMAGEDENOISE_INCLUDE_DIRS}
  )
  list(APPEND LIB
    ${OPENIMAGEDENOISE_LIBRARIES}
    ${TBB_LIBRARIES}
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

