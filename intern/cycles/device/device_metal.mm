/*
 * Copyright 2019 Blender Foundation
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

#ifdef WITH_METAL

#  include <climits>
#  include <limits.h>
#  include <stdio.h>
#  include <stdlib.h>
#  include <string.h>

#  include "device/device.h"
#  include "device/device_denoising.h"
#  include "device/device_intern.h"
#  include "device/device_split_kernel.h"

#  include "render/buffers.h"

#  include "kernel/filter/filter_defines.h"

#  include "util/util_debug.h"
#  include "util/util_foreach.h"
#  include "util/util_logging.h"
#  include "util/util_map.h"
#  include "util/util_md5.h"
#  include "util/util_opengl.h"
#  include "util/util_path.h"
#  include "util/util_string.h"
#  include "util/util_system.h"
#  include "util/util_types.h"
#  include "util/util_time.h"

#  include "kernel/split/kernel_split_data_types.h"

#  import <Metal/Metal.h>

CCL_NAMESPACE_BEGIN

class MetalDevice;

class MetalSplitKernel : public DeviceSplitKernel {
  MetalDevice *device;

 public:
  explicit MetalSplitKernel(MetalDevice *device);

  virtual uint64_t state_buffer_size(device_memory &kg, device_memory &data, size_t num_threads);

  virtual bool enqueue_split_kernel_data_init(const KernelDimensions &dim,
                                              RenderTile &rtile,
                                              int num_global_elements,
                                              device_memory &kernel_globals,
                                              device_memory &kernel_data_,
                                              device_memory &split_data,
                                              device_memory &ray_state,
                                              device_memory &queue_index,
                                              device_memory &use_queues_flag,
                                              device_memory &work_pool_wgs);

  virtual SplitKernelFunction *get_split_kernel_function(const string &kernel_name,
                                                         const DeviceRequestedFeatures &);
  virtual int2 split_kernel_local_size();
  virtual int2 split_kernel_global_size(device_memory &kg, device_memory &data, DeviceTask *task);
};

class MetalDevice : public Device {
 public:
  DedicatedTaskPool task_pool;
  bool first_error;
  MetalSplitKernel *split_kernel;
  id<MTLDevice> device;
  id<MTLCommandQueue> command_queue;
  id<MTLCommandBuffer> command_buffer;
  id<MTLComputeCommandEncoder> command_encoder;

  virtual bool show_samples() const
  {
    /* The MetalDevice only processes one tile at a time, so showing samples is fine. */
    return true;
  }

  MetalDevice(DeviceInfo &info, Stats &stats, Profiler &profiler, bool background_)
      : Device(info, stats, profiler, background_)
  {
    NSArray<id<MTLDevice>> *mtl_devices = MTLCopyAllDevices();

    vector<DeviceInfo> display_devices;
    for (id<MTLDevice> mtl_device in mtl_devices) {
      if (info.id ==
          string_printf("METAL_%s_%16llu", mtl_device.name.UTF8String, mtl_device.registryID)) {
        device = mtl_device;
        break;
      }
    }
    command_queue = [device newCommandQueue];
    command_buffer = [command_queue commandBuffer];
    command_encoder = [command_buffer computeCommandEncoder];
  }

  ~MetalDevice()
  {
    if (command_encoder) {
      [command_encoder endEncoding];
    }
  }

  virtual BVHLayoutMask get_bvh_layout_mask() const
  {
    return BVH_LAYOUT_BVH2;
  }

  bool use_adaptive_compilation()
  {
    return false;
  }

  bool use_split_kernel()
  {
    return false;
  }

  bool load_kernels(const DeviceRequestedFeatures &requested_features)
  {
    VLOG(1) << "load_kernels, not currently supported.";
    return false;
  }

  void mem_alloc(device_memory &mem)
  {
    mem.device_pointer = (device_ptr)
        [device newBufferWithLength:mem.memory_size() options:MTLResourceStorageModeManaged];
  }

  void mem_copy_to(device_memory &mem)
  {
  }

  void mem_copy_from(device_memory &mem, int y, int w, int h, int elem)
  {
  }

  void mem_zero(device_memory &mem)
  {
  }

  void mem_free(device_memory &mem)
  {
    if (mem.device_pointer) {
      id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)mem.device_pointer;
      [buffer release];
      mem.device_pointer = NULL;
    }
  }

  virtual device_ptr mem_alloc_sub_ptr(device_memory &mem, int offset, int /*size*/)
  {
    return NULL;
  }

  void const_copy_to(const char *name, void *host, size_t size)
  {
  }

  void tex_alloc(device_memory &mem)
  {
  }

  void tex_free(device_memory &mem)
  {
  }

  bool denoising_set_tiles(device_ptr *buffers, DenoisingTask *task)
  {
    return false;
  }

  bool denoising_non_local_means(device_ptr image_ptr,
                                 device_ptr guide_ptr,
                                 device_ptr variance_ptr,
                                 device_ptr out_ptr,
                                 DenoisingTask *task)
  {
    return false;
  }

  bool denoising_construct_transform(DenoisingTask *task)
  {
    return false;
  }

  bool denoising_reconstruct(device_ptr color_ptr,
                             device_ptr color_variance_ptr,
                             device_ptr output_ptr,
                             DenoisingTask *task)
  {
    return false;
  }

  bool denoising_combine_halves(device_ptr a_ptr,
                                device_ptr b_ptr,
                                device_ptr mean_ptr,
                                device_ptr variance_ptr,
                                int r,
                                int4 rect,
                                DenoisingTask *task)
  {
    return false;
  }

  bool denoising_divide_shadow(device_ptr a_ptr,
                               device_ptr b_ptr,
                               device_ptr sample_variance_ptr,
                               device_ptr sv_variance_ptr,
                               device_ptr buffer_variance_ptr,
                               DenoisingTask *task)
  {
    return false;
  }

  bool denoising_get_feature(int mean_offset,
                             int variance_offset,
                             device_ptr mean_ptr,
                             device_ptr variance_ptr,
                             DenoisingTask *task)
  {
    return false;
  }

  bool denoising_detect_outliers(device_ptr image_ptr,
                                 device_ptr variance_ptr,
                                 device_ptr depth_ptr,
                                 device_ptr output_ptr,
                                 DenoisingTask *task)
  {
    return false;
  }

  void denoise(RenderTile &rtile, DenoisingTask &denoising, const DeviceTask &task)
  {
  }

  void path_trace(DeviceTask &task, RenderTile &rtile, device_vector<WorkTile> &work_tiles)
  {
  }

  void film_convert(DeviceTask &task,
                    device_ptr buffer,
                    device_ptr rgba_byte,
                    device_ptr rgba_half)
  {
  }

  void shader(DeviceTask &task)
  {
  }

  void pixels_alloc(device_memory &mem)
  {
  }

  void pixels_copy_from(device_memory &mem, int y, int w, int h)
  {
  }

  void pixels_free(device_memory &mem)
  {
  }

  void draw_pixels(device_memory &mem,
                   int y,
                   int w,
                   int h,
                   int width,
                   int height,
                   int dx,
                   int dy,
                   int dw,
                   int dh,
                   bool transparent,
                   const DeviceDrawParams &draw_params)
  {
    Device::draw_pixels(mem, y, w, h, width, height, dx, dy, dw, dh, transparent, draw_params);
  }

  void thread_run(DeviceTask *task)
  {
  }

  class MetalDeviceTask : public DeviceTask {
   public:
    MetalDeviceTask(MetalDevice *device, DeviceTask &task) : DeviceTask(task)
    {
      run = function_bind(&MetalDevice::thread_run, device, this);
    }
  };

  int get_split_task_count(DeviceTask & /*task*/)
  {
    return 1;
  }

  void task_add(DeviceTask &task)
  {
    if (task.type == DeviceTask::FILM_CONVERT) {
      /* must be done in main thread due to opengl access */
      film_convert(task, task.buffer, task.rgba_byte, task.rgba_half);
    }
    else {
      task_pool.push(new MetalDeviceTask(this, task));
    }
  }

  void task_wait()
  {
    task_pool.wait();
  }

  void task_cancel()
  {
    task_pool.cancel();
  }

  friend class MetalSplitKernelFunction;
  friend class MetalSplitKernel;
  friend class CUDAContextScope;
};

/* split kernel */

class MetalSplitKernelFunction : public SplitKernelFunction {
  MetalDevice *device;

 public:
  MetalSplitKernelFunction(MetalDevice *device) : device(device)
  {
  }

  /* enqueue the kernel, returns false if there is an error */
  bool enqueue(const KernelDimensions &dim, device_memory & /*kg*/, device_memory & /*data*/)
  {
    return false;
  }

  /* enqueue the kernel, returns false if there is an error */
  bool enqueue(const KernelDimensions &dim, void *args[])
  {
    return false;
  }
};

MetalSplitKernel::MetalSplitKernel(MetalDevice *device) : DeviceSplitKernel(device), device(device)
{
}

uint64_t MetalSplitKernel::state_buffer_size(device_memory & /*kg*/,
                                             device_memory & /*data*/,
                                             size_t num_threads)
{
  return 0;
}

bool MetalSplitKernel::enqueue_split_kernel_data_init(const KernelDimensions &dim,
                                                      RenderTile &rtile,
                                                      int num_global_elements,
                                                      device_memory & /*kernel_globals*/,
                                                      device_memory & /*kernel_data*/,
                                                      device_memory &split_data,
                                                      device_memory &ray_state,
                                                      device_memory &queue_index,
                                                      device_memory &use_queues_flag,
                                                      device_memory &work_pool_wgs)
{
  return false;
}

SplitKernelFunction *MetalSplitKernel::get_split_kernel_function(const string &kernel_name,
                                                                 const DeviceRequestedFeatures &)
{
  return new MetalSplitKernelFunction(NULL);
}

int2 MetalSplitKernel::split_kernel_local_size()
{
  return make_int2(32, 1);
}

int2 MetalSplitKernel::split_kernel_global_size(device_memory &kg,
                                                device_memory &data,
                                                DeviceTask * /*task*/)
{
  return make_int2(256, 256);
}

bool device_metal_init(void)
{
  return true;
}

Device *device_metal_create(DeviceInfo &info, Stats &stats, Profiler &profiler, bool background)
{
  return new MetalDevice(info, stats, profiler, background);
}

void device_metal_info(vector<DeviceInfo> &devices)
{
  NSArray<id<MTLDevice>> *mtl_devices = MTLCopyAllDevices();

  vector<DeviceInfo> display_devices;
  int num = 0;
  for (id<MTLDevice> mtl_device in mtl_devices) {
    DeviceInfo info;

    info.type = DEVICE_METAL;
    info.description = string(mtl_device.name.UTF8String);
    info.num = num++;

    info.id = string_printf("METAL_%s_%16llu", mtl_device.name.UTF8String, mtl_device.registryID);

    info.has_half_images = false;
    info.has_volume_decoupled = false;

    if (!mtl_device.headless) {
      VLOG(1) << "Device is recognized as display.";
      info.description += " (Display)";
      info.display_device = true;
      display_devices.push_back(info);
    }
    else {
      devices.push_back(info);
    }
    VLOG(1) << "Added device \"" << mtl_device.name.UTF8String << "\" with id \"" << info.id
            << "\".";
  }

  if (!display_devices.empty())
    devices.insert(devices.end(), display_devices.begin(), display_devices.end());
}

CCL_NAMESPACE_END

#endif /* WITH_METAL */
