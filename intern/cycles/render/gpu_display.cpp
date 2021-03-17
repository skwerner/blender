/*
 * Copyright 2021 Blender Foundation
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

#include "render/gpu_display.h"

#include "render/buffers.h"

CCL_NAMESPACE_BEGIN

void GPUDisplay::reset(BufferParams &buffer_params)
{
  params_.offset = make_int2(buffer_params.full_x, buffer_params.full_y);
  params_.full_size = make_int2(buffer_params.full_width, buffer_params.full_height);
  params_.size = make_int2(buffer_params.width, buffer_params.height);
}

CCL_NAMESPACE_END
