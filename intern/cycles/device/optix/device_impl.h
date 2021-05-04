/*
 * Copyright 2019, NVIDIA Corporation.
 * Copyright 2019, Blender Foundation.
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

#ifdef WITH_OPTIX

#  include "device/cuda/device_impl.h"
#  include "device/optix/queue.h"
#  include "device/optix/util.h"

CCL_NAMESPACE_BEGIN

class BVHOptiX;
struct KernelParamsOptiX;

/* List of OptiX program groups. */
enum {
  PG_RGEN_INTERSECT_CLOSEST,
  PG_RGEN_INTERSECT_SHADOW,
  PG_RGEN_INTERSECT_SUBSURFACE,
  PG_MISS,
  PG_HITD, /* Default hit group. */
  PG_HITS, /* __SHADOW_RECORD_ALL__ hit group. */
  PG_HITL, /* __BVH_LOCAL__ hit group (only used for triangles). */
#  if OPTIX_ABI_VERSION >= 36
  PG_HITD_MOTION,
  PG_HITS_MOTION,
#  endif
  PG_CALL,
  NUM_PROGRAM_GROUPS = PG_CALL + 3
};

/* List of OptiX pipelines. */
enum { PIP_INTERSECT, NUM_PIPELINES };

/* A single shader binding table entry. */
struct SbtRecord {
  char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

class OptiXDevice : public CUDADevice {
 public:
  OptixDeviceContext context = NULL;

  OptixModule optix_module = NULL; /* All necessary OptiX kernels are in one module. */
  OptixModule builtin_modules[2] = {};
  OptixPipeline pipelines[NUM_PIPELINES] = {};

  bool motion_blur = false;
  device_vector<SbtRecord> sbt_data;
  device_only_memory<KernelParamsOptiX> launch_params;
  OptixTraversableHandle tlas_handle = 0;

  class Denoiser {
   public:
    explicit Denoiser(CUDADevice *device);
    ~Denoiser();

    CUDADevice *device;

    OptixDenoiser optix_denoiser = nullptr;

    /* Configuration size, as provided to `optixDenoiserSetup`.
     * If the `optixDenoiserSetup()` was never used on the current `optix_denoiser` the
     * `is_configured` will be false. */
    bool is_configured = false;
    int2 configured_size = make_int2(0, 0);

    /* OptiX denoiser state and scratch buffers, stored in a single memory buffer.
     * The memory layout goes as following: [denoiser state][scratch buffer]. */
    device_only_memory<unsigned char> state;
    size_t scratch_offset = 0;
    size_t scratch_size = 0;

    int input_passes = 0;
  };
  Denoiser denoiser_;

 public:
  OptiXDevice(const DeviceInfo &info, Stats &stats, Profiler &profiler);
  ~OptiXDevice();

 private:
  BVHLayoutMask get_bvh_layout_mask() const override;

  string compile_kernel_get_common_cflags(
      const DeviceRequestedFeatures &requested_features) override;

  bool load_kernels(const DeviceRequestedFeatures &requested_features) override;

  bool build_optix_bvh(BVHOptiX *bvh,
                       OptixBuildOperation operation,
                       const OptixBuildInput &build_input,
                       uint16_t num_motion_steps);

  void build_bvh(BVH *bvh, Progress &progress, bool refit) override;

  void const_copy_to(const char *name, void *host, size_t size) override;

  void update_launch_params(size_t offset, void *data, size_t data_size);

  virtual unique_ptr<DeviceQueue> gpu_queue_create() override;

  /* --------------------------------------------------------------------
   * Denoising.
   */

  virtual void denoise_buffer(const DeviceDenoiseTask &task) override;

  /* Run corresponding conversion kernels, preparing data for the denoiser or copying data from the
   * denoiser result to the render buffer. */
  bool denoise_filter_convert_to_rgb(OptiXDeviceQueue *queue,
                                     const DeviceDenoiseTask &task,
                                     const device_ptr d_input_rgb);
  bool denoise_filter_convert_from_rgb(OptiXDeviceQueue *queue,
                                       const DeviceDenoiseTask &task,
                                       const device_ptr d_input_rgb);

  /* Make sure the OptiX denoiser is created and configured for the given task. */
  bool denoise_ensure(const DeviceDenoiseTask &task);

  /* Create OptiX denoiser descriptor if needed.
   * Will do nothing if the current OptiX descriptor is usable for the given parameters.
   * If the OptiX denoiser descriptor did re-allocate here it is left unconfigured. */
  bool denoise_create_if_needed(const DenoiseParams &params);

  /* Configure existing OptiX denoiser descriptor for the use for the given task. */
  bool denoise_configure_if_needed(const DeviceDenoiseTask &task);

  /* Run configured denoiser on the given task. */
  bool denoise_run(OptiXDeviceQueue *queue,
                   const DeviceDenoiseTask &task,
                   const device_ptr d_input_rgb);
};

CCL_NAMESPACE_END

#endif /* WITH_OPTIX */
