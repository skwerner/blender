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

#include "integrator/denoiser.h"

CCL_NAMESPACE_BEGIN

/* Implemenation of non-local-means (NLM) denoiser.
 *
 * The word "implementation" is used in a context that this class provides higher-level API used
 * by denoising process. The actual functions needed for the denoising algorithm are implemented
 * as kernels and are accessed via device API. */
class NLMDenoiser : public Denoiser {
 public:
  NLMDenoiser(Device *device, const DenoiseParams &params);

  virtual void denoise_buffer(const DenoiserBufferParams &buffer_params,
                              RenderBuffers *render_buffers,
                              const int num_samples) override;
};

CCL_NAMESPACE_END
