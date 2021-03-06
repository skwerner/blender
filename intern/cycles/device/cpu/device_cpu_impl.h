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

/* So ImathMath is included before our kernel_cpu_compat. */
#ifdef WITH_OSL
/* So no context pollution happens from indirectly included windows.h */
#  include "util/util_windows.h"
#  include <OSL/oslexec.h>
#endif

#ifdef WITH_EMBREE
#  include <embree3/rtcore.h>
#endif

#include "device/cpu/kernel.h"
#include "device/device.h"
#include "device/device_memory.h"
#include "util/util_openimagedenoise.h"
#include "util/util_task.h"

// clang-format off
#include "kernel/kernel.h"
#include "kernel/kernel_compat_cpu.h"
#include "kernel/kernel_types.h"
#include "kernel/kernel_globals.h"

#include "kernel/osl/osl_shader.h"
#include "kernel/osl/osl_globals.h"
// clang-format on

CCL_NAMESPACE_BEGIN

class DenoisingTask;

class CPUDevice : public Device {
 public:
  TaskPool task_pool;
  KernelGlobals kernel_globals;

  device_vector<TextureInfo> texture_info;
  bool need_texture_info;

#ifdef WITH_OSL
  OSLGlobals osl_globals;
#endif
#ifdef WITH_OPENIMAGEDENOISE
  oidn::DeviceRef oidn_device;
  oidn::FilterRef oidn_filter;
#endif
  thread_spin_lock oidn_task_lock;
#ifdef WITH_EMBREE
  RTCScene embree_scene = NULL;
  RTCDevice embree_device;
#endif

  CPUKernels kernels;

  CPUDevice(DeviceInfo &info_, Stats &stats_, Profiler &profiler_, bool background_);
  ~CPUDevice();

  virtual bool show_samples() const override;

  virtual BVHLayoutMask get_bvh_layout_mask() const override;

  /* Returns true if the texture info was copied to the device (meaning, some more
   * re-initialization might be needed). */
  bool load_texture_info();

  virtual void mem_alloc(device_memory &mem) override;
  virtual void mem_copy_to(device_memory &mem) override;
  virtual void mem_copy_from(device_memory &mem, int y, int w, int h, int elem) override;
  virtual void mem_zero(device_memory &mem) override;
  virtual void mem_free(device_memory &mem) override;
  virtual device_ptr mem_alloc_sub_ptr(device_memory &mem, int offset, int /*size*/) override;

  virtual void const_copy_to(const char *name, void *host, size_t size) override;

  void global_alloc(device_memory &mem);
  void global_free(device_memory &mem);

  void tex_alloc(device_texture &mem);
  void tex_free(device_texture &mem);

  void build_bvh(BVH *bvh, Progress &progress, bool refit) override;

  void thread_run(DeviceTask &task);

  bool denoising_non_local_means(device_ptr image_ptr,
                                 device_ptr guide_ptr,
                                 device_ptr variance_ptr,
                                 device_ptr out_ptr,
                                 DenoisingTask *task);
  bool denoising_construct_transform(DenoisingTask *task);
  bool denoising_accumulate(device_ptr color_ptr,
                            device_ptr color_variance_ptr,
                            device_ptr scale_ptr,
                            int frame,
                            DenoisingTask *task);
  bool denoising_solve(device_ptr output_ptr, DenoisingTask *task);
  bool denoising_combine_halves(device_ptr a_ptr,
                                device_ptr b_ptr,
                                device_ptr mean_ptr,
                                device_ptr variance_ptr,
                                int r,
                                int4 rect,
                                DenoisingTask *task);
  bool denoising_divide_shadow(device_ptr a_ptr,
                               device_ptr b_ptr,
                               device_ptr sample_variance_ptr,
                               device_ptr sv_variance_ptr,
                               device_ptr buffer_variance_ptr,
                               DenoisingTask *task);
  bool denoising_get_feature(int mean_offset,
                             int variance_offset,
                             device_ptr mean_ptr,
                             device_ptr variance_ptr,
                             float scale,
                             DenoisingTask *task);
  bool denoising_write_feature(int out_offset,
                               device_ptr from_ptr,
                               device_ptr buffer_ptr,
                               DenoisingTask *task);
  bool denoising_detect_outliers(device_ptr image_ptr,
                                 device_ptr variance_ptr,
                                 device_ptr depth_ptr,
                                 device_ptr output_ptr,
                                 DenoisingTask *task);

  bool adaptive_sampling_filter(KernelGlobals *kg, RenderTile &tile, int sample);

  void adaptive_sampling_post(const RenderTile &tile, KernelGlobals *kg);

  void render(DeviceTask &task, RenderTile &tile, KernelGlobals *kg);

  void denoise_openimagedenoise_buffer(DeviceTask &task,
                                       float *buffer,
                                       const size_t offset,
                                       const size_t stride,
                                       const size_t x,
                                       const size_t y,
                                       const size_t w,
                                       const size_t h,
                                       const float scale);

  void denoise_openimagedenoise(DeviceTask &task, RenderTile &rtile);

  void denoise_nlm(DenoisingTask &denoising, RenderTile &tile);

  void thread_render(DeviceTask &task);
  void thread_denoise(DeviceTask &task);
  void thread_film_convert(DeviceTask &task);
  void thread_shader(DeviceTask &task);

  virtual int get_split_task_count(DeviceTask &task) override;

  virtual void task_add(DeviceTask &task) override;
  virtual void task_wait() override;
  virtual void task_cancel() override;

  virtual unique_ptr<DeviceQueue> queue_create() override;

  virtual const CPUKernels *get_cpu_kernels() const override;
  virtual const KernelGlobals *get_cpu_kernel_globals() override;
  virtual void *get_cpu_osl_memory() override;

 protected:
  virtual bool load_kernels(const DeviceRequestedFeatures & /*requested_features*/) override;
};

CCL_NAMESPACE_END
