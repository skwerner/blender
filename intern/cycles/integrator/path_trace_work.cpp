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

#include "integrator/path_trace_work.h"

#include "integrator/path_trace_work_tiled.h"

CCL_NAMESPACE_BEGIN

unique_ptr<PathTraceWork> PathTraceWork::create(Device *render_device, RenderBuffers *buffers)
{
  return make_unique<PathTraceWorkTiled>(render_device, buffers);
}

PathTraceWork::PathTraceWork(Device *render_device, RenderBuffers *buffers)
    : render_device_(render_device), buffers_(buffers)
{
}

PathTraceWork::~PathTraceWork()
{
}

CCL_NAMESPACE_END
