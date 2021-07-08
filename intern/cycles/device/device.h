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

#ifndef __DEVICE_H__
#define __DEVICE_H__

#include <stdlib.h>

#include "bvh/bvh_params.h"

#include "device/device_denoise.h"
#include "device/device_memory.h"

#include "util/util_function.h"
#include "util/util_list.h"
#include "util/util_logging.h"
#include "util/util_stats.h"
#include "util/util_string.h"
#include "util/util_texture.h"
#include "util/util_thread.h"
#include "util/util_types.h"
#include "util/util_unique_ptr.h"
#include "util/util_vector.h"

CCL_NAMESPACE_BEGIN

class BVH;
class DeviceQueue;
class Progress;
class CPUKernels;
class CPUKernelThreadGlobals;

/* Device Types */

enum DeviceType {
  DEVICE_NONE = 0,
  DEVICE_CPU,
  DEVICE_OPENCL,
  DEVICE_CUDA,
  DEVICE_MULTI,
  DEVICE_OPTIX,
  DEVICE_DUMMY,
};

enum DeviceTypeMask {
  DEVICE_MASK_CPU = (1 << DEVICE_CPU),
  DEVICE_MASK_OPENCL = (1 << DEVICE_OPENCL),
  DEVICE_MASK_CUDA = (1 << DEVICE_CUDA),
  DEVICE_MASK_OPTIX = (1 << DEVICE_OPTIX),
  DEVICE_MASK_ALL = ~0
};

#define DEVICE_MASK(type) (DeviceTypeMask)(1 << type)

class DeviceInfo {
 public:
  DeviceType type;
  string description;
  string id; /* used for user preferences, should stay fixed with changing hardware config */
  int num;
  bool display_device;        /* GPU is used as a display device. */
  bool has_nanovdb;           /* Support NanoVDB volumes. */
  bool has_half_images;       /* Support half-float textures. */
  bool has_osl;               /* Support Open Shading Language. */
  bool has_profiling;         /* Supports runtime collection of profiling info. */
  bool has_peer_memory;       /* GPU has P2P access to memory of another GPU. */
  DenoiserTypeMask denoisers; /* Supported denoiser types. */
  int cpu_threads;
  vector<DeviceInfo> multi_devices;
  string error_msg;

  DeviceInfo()
  {
    type = DEVICE_CPU;
    id = "CPU";
    num = 0;
    cpu_threads = 0;
    display_device = false;
    has_half_images = false;
    has_nanovdb = false;
    has_osl = false;
    has_profiling = false;
    has_peer_memory = false;
    denoisers = DENOISER_NONE;
  }

  bool operator==(const DeviceInfo &info) const
  {
    /* Multiple Devices with the same ID would be very bad. */
    assert(id != info.id ||
           (type == info.type && num == info.num && description == info.description));
    return id == info.id;
  }
};

class DeviceRequestedFeatures {
 public:
  /* Use experimental feature set. */
  bool experimental;

  /* Selective nodes compilation. */

  /* Identifier of a node group up to which all the nodes needs to be
   * compiled in. Nodes from higher group indices will be ignores.
   */
  int max_nodes_group;

  /* Features bitfield indicating which features from the requested group
   * will be compiled in. Nodes which corresponds to features which are not
   * in this bitfield will be ignored even if they're in the requested group.
   */
  int nodes_features;

  /* BVH/sampling kernel features. */
  bool use_hair;
  bool use_hair_thick;
  bool use_object_motion;
  bool use_camera_motion;

  /* Denotes whether baking functionality is needed. */
  bool use_baking;

  /* Use subsurface scattering materials. */
  bool use_subsurface;

  /* Use volume materials. */
  bool use_volume;

  /* Use OpenSubdiv patch evaluation */
  bool use_patch_evaluation;

  /* Use Transparent shadows */
  bool use_transparent;

  /* Use shadow catcher. */
  bool use_shadow_catcher;

  /* Per-uber shader usage flags. */
  bool use_principled;

  /* Denoising features. */
  bool use_denoising;

  /* Use true displacement */
  bool use_true_displacement;

  /* Use background lights */
  bool use_background_light;

  /* Use path tracing kernels. */
  bool use_path_tracing;

  DeviceRequestedFeatures()
  {
    /* TODO(sergey): Find more meaningful defaults. */
    max_nodes_group = 0;
    nodes_features = 0;
    use_hair = false;
    use_hair_thick = false;
    use_object_motion = false;
    use_camera_motion = false;
    use_baking = false;
    use_subsurface = false;
    use_volume = false;
    use_patch_evaluation = false;
    use_transparent = false;
    use_shadow_catcher = false;
    use_principled = false;
    use_denoising = false;
    use_true_displacement = false;
    use_background_light = false;
    use_path_tracing = true;
  }

  bool modified(const DeviceRequestedFeatures &requested_features)
  {
    return !(max_nodes_group == requested_features.max_nodes_group &&
             nodes_features == requested_features.nodes_features &&
             use_hair == requested_features.use_hair &&
             use_hair_thick == requested_features.use_hair_thick &&
             use_object_motion == requested_features.use_object_motion &&
             use_camera_motion == requested_features.use_camera_motion &&
             use_baking == requested_features.use_baking &&
             use_subsurface == requested_features.use_subsurface &&
             use_volume == requested_features.use_volume &&
             use_patch_evaluation == requested_features.use_patch_evaluation &&
             use_transparent == requested_features.use_transparent &&
             use_shadow_catcher == requested_features.use_shadow_catcher &&
             use_principled == requested_features.use_principled &&
             use_denoising == requested_features.use_denoising &&
             use_true_displacement == requested_features.use_true_displacement &&
             use_background_light == requested_features.use_background_light);
  }

  /* Convert the requested features structure to a build options,
   * which could then be passed to compilers.
   */
  string get_build_options() const
  {
    string build_options = "";
    if (experimental) {
      build_options += "-D__KERNEL_EXPERIMENTAL__ ";
    }
    build_options += "-D__NODES_MAX_GROUP__=" + string_printf("%d", max_nodes_group);
    build_options += " -D__NODES_FEATURES__=" + string_printf("%d", nodes_features);
    if (!use_hair) {
      build_options += " -D__NO_HAIR__";
    }
    if (!use_object_motion) {
      build_options += " -D__NO_OBJECT_MOTION__";
    }
    if (!use_camera_motion) {
      build_options += " -D__NO_CAMERA_MOTION__";
    }
    if (!use_baking) {
      build_options += " -D__NO_BAKING__";
    }
    if (!use_volume) {
      build_options += " -D__NO_VOLUME__";
    }
    if (!use_subsurface) {
      build_options += " -D__NO_SUBSURFACE__";
    }
    if (!use_patch_evaluation) {
      build_options += " -D__NO_PATCH_EVAL__";
    }
    if (!use_transparent && !use_volume) {
      build_options += " -D__NO_TRANSPARENT__";
    }
    if (!use_shadow_catcher) {
      build_options += " -D__NO_SHADOW_CATCHER__";
    }
    if (!use_principled) {
      build_options += " -D__NO_PRINCIPLED__";
    }
    if (!use_denoising) {
      build_options += " -D__NO_DENOISING__";
    }
    return build_options;
  }
};

std::ostream &operator<<(std::ostream &os, const DeviceRequestedFeatures &requested_features);

/* Device */

class Device {
  friend class device_sub_ptr;

 protected:
  Device(const DeviceInfo &info_, Stats &stats_, Profiler &profiler_)
      : info(info_), stats(stats_), profiler(profiler_)
  {
  }

  string error_msg;

  virtual device_ptr mem_alloc_sub_ptr(device_memory & /*mem*/, int /*offset*/, int /*size*/)
  {
    /* Only required for devices that implement denoising. */
    assert(false);
    return (device_ptr)0;
  }
  virtual void mem_free_sub_ptr(device_ptr /*ptr*/){};

 public:
  /* noexcept needed to silence TBB warning. */
  virtual ~Device() noexcept(false);

  /* info */
  DeviceInfo info;
  virtual const string &error_message()
  {
    return error_msg;
  }
  bool have_error()
  {
    return !error_message().empty();
  }
  virtual void set_error(const string &error)
  {
    if (!have_error()) {
      error_msg = error;
    }
    fprintf(stderr, "%s\n", error.c_str());
    fflush(stderr);
  }
  virtual bool show_samples() const
  {
    return false;
  }
  virtual BVHLayoutMask get_bvh_layout_mask() const = 0;

  /* statistics */
  Stats &stats;
  Profiler &profiler;

  /* constant memory */
  virtual void const_copy_to(const char *name, void *host, size_t size) = 0;

  /* load/compile kernels, must be called before adding tasks */
  virtual bool load_kernels(const DeviceRequestedFeatures & /*requested_features*/)
  {
    return true;
  }

  /* GPU device only functions.
   * These may not be used on CPU or multi-devices. */

  /* Create new queue for executing kernels in. */
  virtual unique_ptr<DeviceQueue> gpu_queue_create();

  /* CPU device only functions.
   * These may not be used on GPU or multi-devices. */

  /* Get CPU kernel functions for native instruction set. */
  virtual const CPUKernels *get_cpu_kernels() const;
  /* Get kernel globals to pass to kernels. */
  virtual void get_cpu_kernel_thread_globals(
      vector<CPUKernelThreadGlobals> & /*kernel_thread_globals*/);
  /* Get OpenShadingLanguage memory buffer. */
  virtual void *get_cpu_osl_memory();

  /* acceleration structure building */
  virtual void build_bvh(BVH *bvh, Progress &progress, bool refit);

  /* OptiX specific destructor. */
  virtual void release_optix_bvh(BVH * /*bvh*/){};

  /* multi device */
  virtual int device_number(Device * /*sub_device*/)
  {
    return 0;
  }

  virtual bool is_resident(device_ptr /*key*/, Device *sub_device)
  {
    /* Memory is always resident if this is not a multi device, regardless of whether the pointer
     * is valid or not (since it may not have been allocated yet). */
    return sub_device == this;
  }
  virtual bool check_peer_access(Device * /*peer_device*/)
  {
    return false;
  }

  /* Graphics resources interoperability.
   *
   * The interoperability comes here by the meaning that the device is capable of computing result
   * directly into an OpenGL (or other graphics library) buffer. */

  /* Check display si to be updated using graphics interoperability.
   * The interoperability can not be used is it is not supported by the device. But the device
   * might also force disable the interoperability if it detects that it will be slower than
   * copying pixels from the render buffer. */
  virtual bool should_use_graphics_interop()
  {
    return false;
  }

  /* Buffer denoising. */

  /* TODO(sergey): Need to pass real parameters needed for denoising. */
  virtual void denoise_buffer(const DeviceDenoiseTask & /*task*/)
  {
    LOG(ERROR) << "Request buffer denoising from a device which does not support it.";
  }

  virtual DeviceQueue *get_denoise_queue()
  {
    LOG(ERROR) << "Request denoising queue from a device which does not support it.";
    return nullptr;
  }

  /* Sub-devices */

  /* Run given callback for every individual device which will be handling rendering.
   * For the single device the callback is called for the device itself. For the multi-device the
   * callback is only called for the sub-devices. */
  virtual void foreach_device(const function<void(Device *)> &callback)
  {
    callback(this);
  }

  /* static */
  static Device *create(const DeviceInfo &info, Stats &stats, Profiler &profiler);

  static DeviceType type_from_string(const char *name);
  static string string_from_type(DeviceType type);
  static vector<DeviceType> available_types();
  static vector<DeviceInfo> available_devices(uint device_type_mask = DEVICE_MASK_ALL);
  static DeviceInfo dummy_device(const string &error_msg = "");
  static string device_capabilities(uint device_type_mask = DEVICE_MASK_ALL);
  static DeviceInfo get_multi_device(const vector<DeviceInfo> &subdevices,
                                     int threads,
                                     bool background);

  /* Tag devices lists for update. */
  static void tag_update();

  static void free_memory();

 protected:
  /* Memory allocation, only accessed through device_memory. */
  friend class MultiDevice;
  friend class DeviceServer;
  friend class device_memory;

  virtual void mem_alloc(device_memory &mem) = 0;
  virtual void mem_copy_to(device_memory &mem) = 0;
  virtual void mem_copy_from(device_memory &mem, int y, int w, int h, int elem) = 0;
  virtual void mem_zero(device_memory &mem) = 0;
  virtual void mem_free(device_memory &mem) = 0;

 private:
  /* Indicted whether device types and devices lists were initialized. */
  static bool need_types_update, need_devices_update;
  static thread_mutex device_mutex;
  static vector<DeviceInfo> cuda_devices;
  static vector<DeviceInfo> optix_devices;
  static vector<DeviceInfo> opencl_devices;
  static vector<DeviceInfo> cpu_devices;
  static uint devices_initialized_mask;
};

CCL_NAMESPACE_END

#endif /* __DEVICE_H__ */
