# Copyright 2011-2021 Blender Foundation
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
)

set(SRC
  path_trace.cpp
  tile.cpp
  path_trace_work.cpp
  path_trace_work_tiled.cpp
  work_scheduler.cpp
)

set(SRC_HEADERS
  path_trace.h
  tile.h
  path_trace_work.h
  path_trace_work_tiled.h
  work_scheduler.h
)

set(LIB
  # NOTE: Is required for RenderBuffers access. Might consider moving files around a bit to
  # avoid such cyclic dependency.
  cycles_render

  cycles_util
)

include_directories(${INC})
include_directories(SYSTEM ${INC_SYS})

cycles_add_library(cycles_integrator "${LIB}" ${SRC} ${SRC_HEADERS})
