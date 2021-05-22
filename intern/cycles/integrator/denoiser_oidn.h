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
#include "util/util_thread.h"
#include "util/util_unique_ptr.h"

CCL_NAMESPACE_BEGIN

/* Implementation of denoising API which uses OpenImageDenoise library. */
class OIDNDenoiser : public Denoiser {
 public:
  /* Forwardly declared state which might be using compile-flag specific fields, such as
   * OpenImageDenoise device and filter handles. */
  class State;

  OIDNDenoiser(Device *device, const DenoiseParams &params);
  ~OIDNDenoiser();

  virtual void load_kernels(Progress *progress) override;

  virtual void denoise_buffer(const BufferParams &buffer_params,
                              RenderBuffers *render_buffers,
                              const int num_samples) override;

  virtual DeviceInfo get_denoiser_device_info() const override;

 protected:
  /* Make sure all lazily-initializable resources are initialized and are ready for use by the
   * denoising process. */
  void initialize();

  /* We only perform one denoising at a time, since OpenImageDenoise itself is multithreaded.
   * Use this mutex whenever images are passed to the OIDN and needs to be denoised. */
  static thread_mutex mutex_;

  unique_ptr<State> state_;
};

CCL_NAMESPACE_END
