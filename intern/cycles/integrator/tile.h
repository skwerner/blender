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

#pragma once

#include "util/util_types.h"

CCL_NAMESPACE_BEGIN

/* Calculate tile size which is best suitable for rendering image of a given size with given number
 * of active path states.
 * Will attempt to provide best guess to keep path tracing threads of a device as localized as
 * possible, and have as many threads active for every tile as possible. */
int2 tile_calculate_best_size(const int2 &image_size,
                              const int samples_num,
                              const int max_num_path_states);

CCL_NAMESPACE_END
