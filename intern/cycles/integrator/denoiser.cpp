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

#include "integrator/denoiser.h"

#include "device/device.h"
#include "integrator/denoiser_oidn.h"
#include "integrator/denoiser_optix.h"
#include "render/buffers.h"
#include "util/util_logging.h"

CCL_NAMESPACE_BEGIN

unique_ptr<Denoiser> Denoiser::create(Device *device, const DenoiseParams &params)
{
  DCHECK(params.use);

  switch (params.type) {
    case DENOISER_OPTIX:
      return make_unique<OptiXDenoiser>(device, params);

    case DENOISER_OPENIMAGEDENOISE:
      return make_unique<OIDNDenoiser>(device, params);

    case DENOISER_NUM:
    case DENOISER_NONE:
    case DENOISER_ALL:
      /* pass */
      break;
  }

  LOG(FATAL) << "Unhandled denoiser type " << params.type << ", should never happen.";

  return nullptr;
}

Denoiser::Denoiser(Device *device, const DenoiseParams &params) : device_(device), params_(params)
{
  DCHECK(params.use);
}

void Denoiser::set_params(const DenoiseParams &params)
{
  DCHECK_EQ(params.type, params_.type);

  if (params.type == params_.type) {
    params_ = params;
  }
  else {
    LOG(ERROR) << "Attempt to change denoiser type.";
  }
}

const DenoiseParams &Denoiser::get_params() const
{
  return params_;
}

CCL_NAMESPACE_END
