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

#pragma once

#include "device/device_memory.h"
#include "render/buffers.h"

CCL_NAMESPACE_BEGIN

enum DenoiserType {
  DENOISER_OPTIX = 2,
  DENOISER_OPENIMAGEDENOISE = 4,
  DENOISER_NUM,

  DENOISER_NONE = 0,
  DENOISER_ALL = ~0,
};

/* COnstruct human-readable string which denotes the denoiser type. */
const char *denoiserTypeToHumanReadable(DenoiserType type);

typedef int DenoiserTypeMask;

class DenoiseParams {
 public:
  /* Apply denoiser to image. */
  bool use;

  /* Output denoising data passes (possibly without applying the denoiser). */
  bool store_passes;

  /* Denoiser type. */
  DenoiserType type;

  /* Viewport start sample. */
  int start_sample;

  /* Extra passes which are used by the denoiser (the color pass is always used). */
  bool use_pass_albedo;
  bool use_pass_normal;

  DenoiseParams()
  {
    use = false;
    store_passes = false;

    type = DENOISER_OPENIMAGEDENOISE;

    /* Default to color + albedo only, since normal input does not always have the desired effect
     * when denoising with OptiX. */
    use_pass_albedo = true;
    use_pass_normal = false;

    start_sample = 0;
  }

  bool modified(const DenoiseParams &other) const
  {
    return !(use == other.use && store_passes == other.store_passes && type == other.type &&
             start_sample == other.start_sample && use_pass_albedo == other.use_pass_albedo &&
             use_pass_normal == other.use_pass_normal);
  }
};

/* All the parameters needed to perform buffer denoising on a device.
 * Is not really a task in its canonical terms (as in, is not an asynchronous running task). Is
 * more like a wrapper for all the arguments and parameters needed to perform denoising. Is a
 * single place where they are all listed, so that it's not required to modify all device methods
 * when these parameters do change. */
class DeviceDenoiseTask {
 public:
  DenoiseParams params;

  int num_samples;

  RenderBuffers *render_buffers;
  BufferParams buffer_params;
};

CCL_NAMESPACE_END
