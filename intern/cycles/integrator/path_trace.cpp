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

#include "integrator/path_trace.h"

CCL_NAMESPACE_BEGIN

PathTrace::PathTrace(Device *device) : device_(device)
{
}

void PathTrace::render_samples(int samples_num)
{
  /* NOTE: Currently assume this is a single device.
   *
   * TODO(sergey): Support `DeviceMulti`, by creating per-device thread which will call the
   * `render_samples_on_device` for every device, allowing the kernel graph to track state of
   * deviced independently between each other. */
  render_samples_on_device(device_, samples_num);
}

void PathTrace::render_samples_on_device(Device * /*device*/, int samples_num)
{
  /* TODO(sergey): Determine possible queue/wavefront size. For the ideal performance the wavefront
   * should be as big as it is needed to have all device threads occupied. In practice, its size
   * might need to be smaller if the state does not fit into the memory. */

  /* Number of samples which happens in parallel for the work group on the device. */
  const int parallel_samples_num = 1;

  /* Number of samples which were path traces for the big tile. */
  int traced_samples_num = 0;

  while (traced_samples_num < samples_num) {
    /* TODO(sergey): Invoke "Generate Camera Rays" kernel. */

    /* TODO(sergey): Evaluate the path trace graph for the scheduled   */

    /* TODO(sergey): For the initial implementation: wait for the graph to be fully evaluated, so
     * that the number of samples can be advanced forward.
     *
     * For the final implementation can do something smarter, like re-generating camera rays if the
     * wavefront becomes too small but there are still a lot of samples to be calculated. */

    traced_samples_num += parallel_samples_num;
  }
}

CCL_NAMESPACE_END
