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

#ifdef WITH_OPTIX

#  include "device/optix/queue.h"
#  include "device/optix/device_impl.h"

#  include "util/util_time.h"

#  undef __KERNEL_CPU__
#  define __KERNEL_OPTIX__
#  include "kernel/device/optix/globals.h"

CCL_NAMESPACE_BEGIN

/* CUDADeviceQueue */

OptiXDeviceQueue::OptiXDeviceQueue(OptiXDevice *device) : CUDADeviceQueue(device)
{
}

void OptiXDeviceQueue::init_execution()
{
  CUDADeviceQueue::init_execution();
}

bool OptiXDeviceQueue::enqueue(DeviceKernel kernel, const int work_size, void *args[])
{
  if (kernel == DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST ||
      kernel == DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW ||
      kernel == DEVICE_KERNEL_INTEGRATOR_INTERSECT_SUBSURFACE) {
    if (cuda_device_->have_error()) {
      return false;
    }

    const CUDAContextScope scope(cuda_device_);

    OptiXDevice *const optix_device = static_cast<OptiXDevice *>(cuda_device_);

    device_ptr launch_params_ptr = optix_device->launch_params.device_pointer;

    cuda_device_assert(
        cuda_device_,
        cuMemcpyHtoDAsync(launch_params_ptr + offsetof(KernelParams, path_index_array),
                          args[0],  // &d_path_index
                          sizeof(device_ptr),
                          cuda_stream_));
    cuda_device_assert(cuda_device_, cuStreamSynchronize(cuda_stream_));

    OptixShaderBindingTable sbt_params = {};
    sbt_params.raygenRecord = optix_device->sbt_data.device_pointer +
                              (OptiXDevice::PG_RGEN_INTEGRATOR_INTERSECT_CLOSEST + kernel -
                               DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST) *
                                  sizeof(OptiXDevice::SbtRecord);
    sbt_params.missRecordBase = optix_device->sbt_data.device_pointer +
                                OptiXDevice::PG_MISS * sizeof(OptiXDevice::SbtRecord);
    sbt_params.missRecordStrideInBytes = sizeof(OptiXDevice::SbtRecord);
    sbt_params.missRecordCount = 1;
    sbt_params.hitgroupRecordBase = optix_device->sbt_data.device_pointer +
                                    OptiXDevice::PG_HITD * sizeof(OptiXDevice::SbtRecord);
    sbt_params.hitgroupRecordStrideInBytes = sizeof(OptiXDevice::SbtRecord);
#  if OPTIX_ABI_VERSION >= 36
    sbt_params.hitgroupRecordCount = 5; /* PG_HITD(_MOTION), PG_HITS(_MOTION), PG_HITL */
#  else
    sbt_params.hitgroupRecordCount = 3; /* PG_HITD, PG_HITS, PG_HITL */
#  endif
    sbt_params.callablesRecordBase = optix_device->sbt_data.device_pointer +
                                     OptiXDevice::PG_CALL * sizeof(OptiXDevice::SbtRecord);
    sbt_params.callablesRecordCount = 3;
    sbt_params.callablesRecordStrideInBytes = sizeof(OptiXDevice::SbtRecord);

    /* Launch the ray generation program. */
    optix_device_assert(optix_device,
                        optixLaunch(optix_device->pipelines[OptiXDevice::PIP_PATH_TRACE],
                                    cuda_stream_,
                                    launch_params_ptr,
                                    optix_device->launch_params.data_elements,
                                    &sbt_params,
                                    work_size,
                                    1,
                                    1));

    return !(optix_device->have_error());
  }
  else {
    return CUDADeviceQueue::enqueue(kernel, work_size, args);
  }
}

CCL_NAMESPACE_END

#endif /* WITH_OPTIX */
