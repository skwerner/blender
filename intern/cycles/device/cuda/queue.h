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

#ifdef WITH_CUDA

#  include "device/device_kernel.h"
#  include "device/device_memory.h"
#  include "device/device_queue.h"

#  include "device/cuda/util.h"

CCL_NAMESPACE_BEGIN

class CUDADevice;
class device_memory;

/* Base class for CUDA queues. */
class CUDADeviceQueue : public DeviceQueue {
 public:
  CUDADeviceQueue(CUDADevice *device);
  ~CUDADeviceQueue();

  virtual void init_execution() override;

  virtual bool enqueue(DeviceKernel kernel, const int work_size, void *args[]) override;

  virtual bool synchronize() override;

  virtual void zero_to_device(device_memory &mem) override;
  virtual void copy_to_device(device_memory &mem) override;
  virtual void copy_from_device(device_memory &mem) override;

 protected:
  CUDADevice *cuda_device_;
  CUstream cuda_stream_;
  double last_sync_time_;
};

CCL_NAMESPACE_END

#endif /* WITH_CUDA */
