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

#ifdef WITH_OPTIX

#  include "device/optix/device_impl.h"

#  include "bvh/bvh.h"
#  include "bvh/bvh_optix.h"
#  include "integrator/pass_accessor_gpu.h"
#  include "render/buffers.h"
#  include "render/hair.h"
#  include "render/mesh.h"
#  include "render/object.h"
#  include "render/scene.h"

#  include "util/util_debug.h"
#  include "util/util_logging.h"
#  include "util/util_md5.h"
#  include "util/util_path.h"
#  include "util/util_progress.h"
#  include "util/util_time.h"

#  undef __KERNEL_CPU__
#  define __KERNEL_OPTIX__
#  include "kernel/device/optix/globals.h"

CCL_NAMESPACE_BEGIN

OptiXDevice::Denoiser::Denoiser(CUDADevice *device)
    : device(device), state(device, "__denoiser_state")
{
}

OptiXDevice::Denoiser::~Denoiser()
{
  const CUDAContextScope scope(device);
  if (optix_denoiser != nullptr) {
    optixDenoiserDestroy(optix_denoiser);
  }
}

OptiXDevice::OptiXDevice(const DeviceInfo &info, Stats &stats, Profiler &profiler)
    : CUDADevice(info, stats, profiler),
      sbt_data(this, "__sbt", MEM_READ_ONLY),
      launch_params(this, "__params"),
      denoiser_(this)
{
  /* Make the CUDA context current. */
  if (!cuContext) {
    /* Do not initialize if CUDA context creation failed already. */
    return;
  }
  const CUDAContextScope scope(this);

  /* Create OptiX context for this device. */
  OptixDeviceContextOptions options = {};
#  ifdef WITH_CYCLES_LOGGING
  options.logCallbackLevel = 4; /* Fatal = 1, Error = 2, Warning = 3, Print = 4. */
  options.logCallbackFunction = [](unsigned int level, const char *, const char *message, void *) {
    switch (level) {
      case 1:
        LOG_IF(FATAL, VLOG_IS_ON(1)) << message;
        break;
      case 2:
        LOG_IF(ERROR, VLOG_IS_ON(1)) << message;
        break;
      case 3:
        LOG_IF(WARNING, VLOG_IS_ON(1)) << message;
        break;
      case 4:
        LOG_IF(INFO, VLOG_IS_ON(1)) << message;
        break;
    }
  };
#  endif
#  if OPTIX_ABI_VERSION >= 41
  if (DebugFlags().optix.use_debug) {
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
  }
#  endif
  optix_assert(optixDeviceContextCreate(cuContext, &options, &context));
#  ifdef WITH_CYCLES_LOGGING
  optix_assert(optixDeviceContextSetLogCallback(
      context, options.logCallbackFunction, options.logCallbackData, options.logCallbackLevel));
#  endif

  /* Fix weird compiler bug that assigns wrong size. */
  launch_params.data_elements = sizeof(KernelParamsOptiX);

  /* Allocate launch parameter buffer memory on device. */
  launch_params.alloc_to_device(1);
}

OptiXDevice::~OptiXDevice()
{
  /* Make CUDA context current. */
  const CUDAContextScope scope(this);

  free_bvh_memory_delayed();

  sbt_data.free();
  texture_info.free();
  launch_params.free();

  /* Unload modules. */
  if (optix_module != NULL) {
    optixModuleDestroy(optix_module);
  }
  for (unsigned int i = 0; i < 2; ++i) {
    if (builtin_modules[i] != NULL) {
      optixModuleDestroy(builtin_modules[i]);
    }
  }
  for (unsigned int i = 0; i < NUM_PIPELINES; ++i) {
    if (pipelines[i] != NULL) {
      optixPipelineDestroy(pipelines[i]);
    }
  }

  optixDeviceContextDestroy(context);
}

unique_ptr<DeviceQueue> OptiXDevice::gpu_queue_create()
{
  return make_unique<OptiXDeviceQueue>(this);
}

BVHLayoutMask OptiXDevice::get_bvh_layout_mask() const
{
  /* OptiX has its own internal acceleration structure format. */
  return BVH_LAYOUT_OPTIX;
}

string OptiXDevice::compile_kernel_get_common_cflags(
    const DeviceRequestedFeatures &requested_features)
{
  string common_cflags = CUDADevice::compile_kernel_get_common_cflags(requested_features);

  /* Add OptiX SDK include directory to include paths. */
  const char *optix_sdk_path = getenv("OPTIX_ROOT_DIR");
  if (optix_sdk_path) {
    common_cflags += string_printf(" -I\"%s/include\"", optix_sdk_path);
  }

  /* Specialization for shader raytracing. */
  if (requested_features.nodes_features & NODE_FEATURE_RAYTRACE) {
    common_cflags += " --keep-device-functions";
  }

  return common_cflags;
}

bool OptiXDevice::load_kernels(const DeviceRequestedFeatures &requested_features)
{
  if (have_error()) {
    /* Abort early if context creation failed already. */
    return false;
  }

  /* Load CUDA modules because we need some of the utility kernels. */
  if (!CUDADevice::load_kernels(requested_features)) {
    return false;
  }

  /* Skip creating OptiX module if only doing denoising. */
  if (!(requested_features.use_path_tracing || requested_features.use_baking)) {
    return true;
  }

  const CUDAContextScope scope(this);

  /* Unload existing OptiX module and pipelines first. */
  if (optix_module != NULL) {
    optixModuleDestroy(optix_module);
    optix_module = NULL;
  }
  for (unsigned int i = 0; i < 2; ++i) {
    if (builtin_modules[i] != NULL) {
      optixModuleDestroy(builtin_modules[i]);
      builtin_modules[i] = NULL;
    }
  }
  for (unsigned int i = 0; i < NUM_PIPELINES; ++i) {
    if (pipelines[i] != NULL) {
      optixPipelineDestroy(pipelines[i]);
      pipelines[i] = NULL;
    }
  }

  OptixModuleCompileOptions module_options = {};
  module_options.maxRegisterCount = 0; /* Do not set an explicit register limit. */

  if (DebugFlags().optix.use_debug) {
    module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
  }
  else {
    module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
    module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
  }

#  if OPTIX_ABI_VERSION >= 41
  module_options.boundValues = nullptr;
  module_options.numBoundValues = 0;
#  endif

  OptixPipelineCompileOptions pipeline_options = {};
  /* Default to no motion blur and two-level graph, since it is the fastest option. */
  pipeline_options.usesMotionBlur = false;
  pipeline_options.traversableGraphFlags =
      OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
  pipeline_options.numPayloadValues = 6;
  pipeline_options.numAttributeValues = 2; /* u, v */
  pipeline_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
  pipeline_options.pipelineLaunchParamsVariableName = "__params"; /* See globals.h */

#  if OPTIX_ABI_VERSION >= 36
  pipeline_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
  if (requested_features.use_hair) {
    if (DebugFlags().optix.use_curves_api && requested_features.use_hair_thick) {
      pipeline_options.usesPrimitiveTypeFlags |= OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE;
    }
    else {
      pipeline_options.usesPrimitiveTypeFlags |= OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;
    }
  }
#  endif

  /* Keep track of whether motion blur is enabled, so to enable/disable motion in BVH builds
   * This is necessary since objects may be reported to have motion if the Vector pass is
   * active, but may still need to be rendered without motion blur if that isn't active as well. */
  motion_blur = requested_features.use_object_motion;

  if (motion_blur) {
    pipeline_options.usesMotionBlur = true;
    /* Motion blur can insert motion transforms into the traversal graph.
     * It is no longer a two-level graph then, so need to set flags to allow any configuration. */
    pipeline_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
  }

  { /* Load and compile PTX module with OptiX kernels. */
    string ptx_data,
        ptx_filename = path_get((requested_features.nodes_features & NODE_FEATURE_RAYTRACE) ?
                                    "lib/kernel_optix_shader_raytrace.ptx" :
                                    "lib/kernel_optix.ptx");
    if (use_adaptive_compilation() || path_file_size(ptx_filename) == -1) {
      if (!getenv("OPTIX_ROOT_DIR")) {
        set_error(
            "Missing OPTIX_ROOT_DIR environment variable (which must be set with the path to "
            "the Optix SDK to be able to compile Optix kernels on demand).");
        return false;
      }
      ptx_filename = compile_kernel(requested_features,
                                    (requested_features.nodes_features & NODE_FEATURE_RAYTRACE) ?
                                        "kernel_shader_raytrace" :
                                        "kernel",
                                    "optix",
                                    true);
    }
    if (ptx_filename.empty() || !path_read_text(ptx_filename, ptx_data)) {
      set_error(string_printf("Failed to load OptiX kernel from '%s'", ptx_filename.c_str()));
      return false;
    }

    const OptixResult result = optixModuleCreateFromPTX(context,
                                                        &module_options,
                                                        &pipeline_options,
                                                        ptx_data.data(),
                                                        ptx_data.size(),
                                                        nullptr,
                                                        0,
                                                        &optix_module);
    if (result != OPTIX_SUCCESS) {
      set_error(string_printf("Failed to load OptiX kernel from '%s' (%s)",
                              ptx_filename.c_str(),
                              optixGetErrorName(result)));
      return false;
    }
  }

  /* Create program groups. */
  OptixProgramGroup groups[NUM_PROGRAM_GROUPS] = {};
  OptixProgramGroupDesc group_descs[NUM_PROGRAM_GROUPS] = {};
  OptixProgramGroupOptions group_options = {}; /* There are no options currently. */
  group_descs[PG_RGEN_INTERSECT_CLOSEST].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  group_descs[PG_RGEN_INTERSECT_CLOSEST].raygen.module = optix_module;
  group_descs[PG_RGEN_INTERSECT_CLOSEST].raygen.entryFunctionName =
      "__raygen__kernel_optix_integrator_intersect_closest";
  group_descs[PG_RGEN_INTERSECT_SHADOW].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  group_descs[PG_RGEN_INTERSECT_SHADOW].raygen.module = optix_module;
  group_descs[PG_RGEN_INTERSECT_SHADOW].raygen.entryFunctionName =
      "__raygen__kernel_optix_integrator_intersect_shadow";
  group_descs[PG_RGEN_INTERSECT_SUBSURFACE].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  group_descs[PG_RGEN_INTERSECT_SUBSURFACE].raygen.module = optix_module;
  group_descs[PG_RGEN_INTERSECT_SUBSURFACE].raygen.entryFunctionName =
      "__raygen__kernel_optix_integrator_intersect_subsurface";
  group_descs[PG_MISS].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
  group_descs[PG_MISS].miss.module = optix_module;
  group_descs[PG_MISS].miss.entryFunctionName = "__miss__kernel_optix_miss";
  group_descs[PG_HITD].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  group_descs[PG_HITD].hitgroup.moduleCH = optix_module;
  group_descs[PG_HITD].hitgroup.entryFunctionNameCH = "__closesthit__kernel_optix_hit";
  group_descs[PG_HITD].hitgroup.moduleAH = optix_module;
  group_descs[PG_HITD].hitgroup.entryFunctionNameAH = "__anyhit__kernel_optix_visibility_test";
  group_descs[PG_HITS].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  group_descs[PG_HITS].hitgroup.moduleAH = optix_module;
  group_descs[PG_HITS].hitgroup.entryFunctionNameAH = "__anyhit__kernel_optix_shadow_all_hit";

  if (requested_features.use_hair) {
    group_descs[PG_HITD].hitgroup.moduleIS = optix_module;
    group_descs[PG_HITS].hitgroup.moduleIS = optix_module;

    /* Add curve intersection programs. */
    if (requested_features.use_hair_thick) {
      /* Slower programs for thick hair since that also slows down ribbons.
       * Ideally this should not be needed. */
      group_descs[PG_HITD].hitgroup.entryFunctionNameIS = "__intersection__curve_all";
      group_descs[PG_HITS].hitgroup.entryFunctionNameIS = "__intersection__curve_all";
    }
    else {
      group_descs[PG_HITD].hitgroup.entryFunctionNameIS = "__intersection__curve_ribbon";
      group_descs[PG_HITS].hitgroup.entryFunctionNameIS = "__intersection__curve_ribbon";
    }

#  if OPTIX_ABI_VERSION >= 36
    if (DebugFlags().optix.use_curves_api && requested_features.use_hair_thick) {
      OptixBuiltinISOptions builtin_options = {};
      builtin_options.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE;
      builtin_options.usesMotionBlur = false;

      optix_assert(optixBuiltinISModuleGet(
          context, &module_options, &pipeline_options, &builtin_options, &builtin_modules[0]));

      group_descs[PG_HITD].hitgroup.moduleIS = builtin_modules[0];
      group_descs[PG_HITD].hitgroup.entryFunctionNameIS = nullptr;
      group_descs[PG_HITS].hitgroup.moduleIS = builtin_modules[0];
      group_descs[PG_HITS].hitgroup.entryFunctionNameIS = nullptr;

      if (motion_blur) {
        builtin_options.usesMotionBlur = true;

        optix_assert(optixBuiltinISModuleGet(
            context, &module_options, &pipeline_options, &builtin_options, &builtin_modules[1]));

        group_descs[PG_HITD_MOTION] = group_descs[PG_HITD];
        group_descs[PG_HITD_MOTION].hitgroup.moduleIS = builtin_modules[1];
        group_descs[PG_HITS_MOTION] = group_descs[PG_HITS];
        group_descs[PG_HITS_MOTION].hitgroup.moduleIS = builtin_modules[1];
      }
    }
#  endif
  }

  if (requested_features.use_subsurface ||
      (requested_features.nodes_features & NODE_FEATURE_RAYTRACE)) {
    /* Add hit group for local intersections. */
    group_descs[PG_HITL].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    group_descs[PG_HITL].hitgroup.moduleAH = optix_module;
    group_descs[PG_HITL].hitgroup.entryFunctionNameAH = "__anyhit__kernel_optix_local_hit";
  }

  /* Shader raytracing replaces some functions with direct callables. */
  if (requested_features.nodes_features & NODE_FEATURE_RAYTRACE) {
    group_descs[PG_RGEN_SHADE_SURFACE_RAYTRACE].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    group_descs[PG_RGEN_SHADE_SURFACE_RAYTRACE].raygen.module = optix_module;
    group_descs[PG_RGEN_SHADE_SURFACE_RAYTRACE].raygen.entryFunctionName =
        "__raygen__kernel_optix_integrator_shade_surface_raytrace";
    group_descs[PG_CALL_SVM_AO].kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    group_descs[PG_CALL_SVM_AO].callables.moduleDC = optix_module;
    group_descs[PG_CALL_SVM_AO].callables.entryFunctionNameDC = "__direct_callable__svm_node_ao";
    group_descs[PG_CALL_SVM_BEVEL].kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    group_descs[PG_CALL_SVM_BEVEL].callables.moduleDC = optix_module;
    group_descs[PG_CALL_SVM_BEVEL].callables.entryFunctionNameDC =
        "__direct_callable__svm_node_bevel";
  }

  optix_assert(optixProgramGroupCreate(
      context, group_descs, NUM_PROGRAM_GROUPS, &group_options, nullptr, 0, groups));

  /* Get program stack sizes. */
  OptixStackSizes stack_size[NUM_PROGRAM_GROUPS] = {};
  /* Set up SBT, which in this case is used only to select between different programs. */
  sbt_data.alloc(NUM_PROGRAM_GROUPS);
  memset(sbt_data.host_pointer, 0, sizeof(SbtRecord) * NUM_PROGRAM_GROUPS);
  for (unsigned int i = 0; i < NUM_PROGRAM_GROUPS; ++i) {
    optix_assert(optixSbtRecordPackHeader(groups[i], &sbt_data[i]));
    optix_assert(optixProgramGroupGetStackSize(groups[i], &stack_size[i]));
  }
  sbt_data.copy_to_device(); /* Upload SBT to device. */

  /* Calculate maximum trace continuation stack size. */
  unsigned int trace_css = stack_size[PG_HITD].cssCH;
  /* This is based on the maximum of closest-hit and any-hit/intersection programs. */
  trace_css = std::max(trace_css, stack_size[PG_HITD].cssIS + stack_size[PG_HITD].cssAH);
  trace_css = std::max(trace_css, stack_size[PG_HITS].cssIS + stack_size[PG_HITS].cssAH);
  trace_css = std::max(trace_css, stack_size[PG_HITL].cssIS + stack_size[PG_HITL].cssAH);
#  if OPTIX_ABI_VERSION >= 36
  trace_css = std::max(trace_css,
                       stack_size[PG_HITD_MOTION].cssIS + stack_size[PG_HITD_MOTION].cssAH);
  trace_css = std::max(trace_css,
                       stack_size[PG_HITS_MOTION].cssIS + stack_size[PG_HITS_MOTION].cssAH);
#  endif

  OptixPipelineLinkOptions link_options = {};
  link_options.maxTraceDepth = 1;

  if (DebugFlags().optix.use_debug) {
    link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
  }
  else {
    link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
  }

#  if OPTIX_ABI_VERSION < 24
  link_options.overrideUsesMotionBlur = motion_blur;
#  endif

  if (requested_features.nodes_features & NODE_FEATURE_RAYTRACE) {
    /* Create shader raytracing pipeline. */
    vector<OptixProgramGroup> pipeline_groups;
    pipeline_groups.reserve(NUM_PROGRAM_GROUPS);
    pipeline_groups.push_back(groups[PG_RGEN_SHADE_SURFACE_RAYTRACE]);
    pipeline_groups.push_back(groups[PG_MISS]);
    pipeline_groups.push_back(groups[PG_HITD]);
    pipeline_groups.push_back(groups[PG_HITS]);
    pipeline_groups.push_back(groups[PG_HITL]);
#  if OPTIX_ABI_VERSION >= 36
    if (motion_blur) {
      pipeline_groups.push_back(groups[PG_HITD_MOTION]);
      pipeline_groups.push_back(groups[PG_HITS_MOTION]);
    }
#  endif
    pipeline_groups.push_back(groups[PG_CALL_SVM_AO]);
    pipeline_groups.push_back(groups[PG_CALL_SVM_BEVEL]);

    optix_assert(optixPipelineCreate(context,
                                     &pipeline_options,
                                     &link_options,
                                     pipeline_groups.data(),
                                     pipeline_groups.size(),
                                     nullptr,
                                     0,
                                     &pipelines[PIP_SHADE_RAYTRACE]));

    /* Combine ray generation and trace continuation stack size. */
    const unsigned int css = stack_size[PG_RGEN_SHADE_SURFACE_RAYTRACE].cssRG +
                             link_options.maxTraceDepth * trace_css;
    const unsigned int dss = std::max(stack_size[PG_CALL_SVM_AO].dssDC,
                                      stack_size[PG_CALL_SVM_BEVEL].dssDC);

    /* Set stack size depending on pipeline options. */
    optix_assert(optixPipelineSetStackSize(
        pipelines[PIP_SHADE_RAYTRACE], 0, dss, css, motion_blur ? 3 : 2));
  }

  { /* Create intersection-only pipeline. */
    vector<OptixProgramGroup> pipeline_groups;
    pipeline_groups.reserve(NUM_PROGRAM_GROUPS);
    pipeline_groups.push_back(groups[PG_RGEN_INTERSECT_CLOSEST]);
    pipeline_groups.push_back(groups[PG_RGEN_INTERSECT_SHADOW]);
    pipeline_groups.push_back(groups[PG_RGEN_INTERSECT_SUBSURFACE]);
    pipeline_groups.push_back(groups[PG_MISS]);
    pipeline_groups.push_back(groups[PG_HITD]);
    pipeline_groups.push_back(groups[PG_HITS]);
    pipeline_groups.push_back(groups[PG_HITL]);
#  if OPTIX_ABI_VERSION >= 36
    if (motion_blur) {
      pipeline_groups.push_back(groups[PG_HITD_MOTION]);
      pipeline_groups.push_back(groups[PG_HITS_MOTION]);
    }
#  endif

    optix_assert(optixPipelineCreate(context,
                                     &pipeline_options,
                                     &link_options,
                                     pipeline_groups.data(),
                                     pipeline_groups.size(),
                                     nullptr,
                                     0,
                                     &pipelines[PIP_INTERSECT]));

    /* Calculate continuation stack size based on the maximum of all ray generation stack sizes. */
    const unsigned int css = std::max(stack_size[PG_RGEN_INTERSECT_CLOSEST].cssRG,
                                      std::max(stack_size[PG_RGEN_INTERSECT_SHADOW].cssRG,
                                               stack_size[PG_RGEN_INTERSECT_SUBSURFACE].cssRG)) +
                             link_options.maxTraceDepth * trace_css;

    optix_assert(
        optixPipelineSetStackSize(pipelines[PIP_INTERSECT], 0, 0, css, motion_blur ? 3 : 2));
  }

  /* Clean up program group objects. */
  for (unsigned int i = 0; i < NUM_PROGRAM_GROUPS; ++i) {
    optixProgramGroupDestroy(groups[i]);
  }

  return true;
}

/* --------------------------------------------------------------------
 * Buffer denoising.
 */

/* Calculate number of passes used by the denoiser. */
static int denoise_buffer_num_passes(const DenoiseParams &params)
{
  int num_passes = 1;

  if (params.use_pass_albedo) {
    num_passes += 1;

    if (params.use_pass_normal) {
      num_passes += 1;
    }
  }

  return num_passes;
}

/* Calculate number of floats per pixel for the input buffer used by the OptiX. */
static int denoise_buffer_pass_stride(const DenoiseParams &params)
{
  return denoise_buffer_num_passes(params) * 3;
}

class OptiXDevice::DenoiseContext {
 public:
  explicit DenoiseContext(OptiXDevice *device, const DeviceDenoiseTask &task)
      : queue(device),
        denoise_params(task.params),
        render_buffers(task.render_buffers),
        buffer_params(task.buffer_params),
        input_rgb(device, "denoiser input rgb"),
        num_samples(task.num_samples)
  {
    const int input_pass_stride = denoise_buffer_pass_stride(task.params);
    input_rgb.alloc_to_device(buffer_params.width * buffer_params.height * input_pass_stride);

    pass_sample_count = buffer_params.get_pass_offset(PASS_SAMPLE_COUNT);
  }

  OptiXDeviceQueue queue;

  const DenoiseParams &denoise_params;

  RenderBuffers *render_buffers;
  const BufferParams &buffer_params;

  /* Device-side storage of the input passes.
   * Start with the input color pass, followed with optional albedo and normal passes. */
  device_only_memory<float> input_rgb;

  int num_samples;
  int pass_sample_count;
};

class OptiXDevice::DenoisePass {
 public:
  PassType type;

  int noisy_offset;
  int denoised_offset;
};

void OptiXDevice::denoise_buffer(const DeviceDenoiseTask &task)
{
  const CUDAContextScope scope(this);

  if (!denoise_ensure(task)) {
    return;
  }

  DenoiseContext context(this, task);

  denoise_pass(context, PASS_COMBINED);
  denoise_pass(context, PASS_SHADOW_CATCHER);
  denoise_pass(context, PASS_SHADOW_CATCHER_MATTE);
}

void OptiXDevice::denoise_pass(DenoiseContext &context, PassType pass_type)
{
  const BufferParams &buffer_params = context.buffer_params;

  DenoisePass pass;
  pass.type = pass_type;
  pass.noisy_offset = buffer_params.get_pass_offset(pass_type, PassMode::NOISY);
  pass.denoised_offset = buffer_params.get_pass_offset(pass_type, PassMode::DENOISED);

  if (pass.noisy_offset == PASS_UNUSED) {
    return;
  }
  if (pass.denoised_offset == PASS_UNUSED) {
    LOG(DFATAL) << "Missing denoised pass " << pass_type_as_string(pass_type);
    return;
  }

  /* Read pixels from the noisy input pass, store them in the temporary buffer for further
   * clamping. */
  denoise_read_input_pixels(context, pass);

  /* Make sure input data is in [0 .. 10000] range by scaling the input buffer by the number of
   * samples in the buffer. Additionally, fill in the auxillary passes needed by the denoiser which
   * were not provided by the pass accessor. */
  if (!denoise_filter_convert_to_rgb(context, pass)) {
    LOG(ERROR) << "Error connverting denoising passes to RGB buffer.";
    return;
  }

  if (!denoise_run(context)) {
    LOG(ERROR) << "Error running OptiX denoiser.";
    return;
  }

  /* Store result in the combined pass of the render buffer.
   *
   * This will scale the denoiser result up to match the number of, possibly per-pixel, samples. */
  if (!denoise_filter_convert_from_rgb(context, pass)) {
    LOG(ERROR) << "Error copying denoiser result to the denoised pass.";
    return;
  }

  context.queue.synchronize();
}

void OptiXDevice::denoise_read_input_pixels(DenoiseContext &context, const DenoisePass &pass) const
{
  PassAccessor::PassAccessInfo pass_access_info;
  pass_access_info.type = pass.type;
  pass_access_info.mode = PassMode::NOISY;
  pass_access_info.offset = pass.noisy_offset;

  /* Denoiser operates on passes which are used to calculate the approximation, and is never used
   * on the approximation. The latter is not even possible because OptiX does not support
   * denoising of semi-transparent pixels. */
  pass_access_info.use_approximate_shadow_catcher = false;
  pass_access_info.show_active_pixels = false;

  /* TODO(sergey): Consider adding support of actual exposure, to avoid clamping in extreme cases.
   */
  const PassAccessorGPU pass_accessor(&context.queue, pass_access_info, 1.0f, context.num_samples);

  PassAccessor::Destination destination(pass_access_info.type);
  destination.d_pixels = context.input_rgb.device_pointer;
  destination.num_components = 3;

  pass_accessor.get_render_tile_pixels(context.render_buffers, context.buffer_params, destination);
}

bool OptiXDevice::denoise_filter_convert_to_rgb(DenoiseContext &context, const DenoisePass &pass)
{
  const BufferParams &buffer_params = context.buffer_params;

  const PassInfo pass_info = Pass::get_info(pass.type);

  const int work_size = buffer_params.width * buffer_params.height;

  const int pass_offset[3] = {pass.noisy_offset,
                              pass_info.use_denoising_albedo ?
                                  buffer_params.get_pass_offset(PASS_DENOISING_ALBEDO) :
                                  PASS_UNUSED,
                              buffer_params.get_pass_offset(PASS_DENOISING_NORMAL)};

  const int input_passes = denoise_buffer_num_passes(context.denoise_params);

  void *args[] = {const_cast<device_ptr *>(&context.input_rgb.device_pointer),
                  &context.render_buffers->buffer.device_pointer,
                  const_cast<int *>(&buffer_params.full_x),
                  const_cast<int *>(&buffer_params.full_y),
                  const_cast<int *>(&buffer_params.width),
                  const_cast<int *>(&buffer_params.height),
                  const_cast<int *>(&buffer_params.offset),
                  const_cast<int *>(&buffer_params.stride),
                  const_cast<int *>(&buffer_params.pass_stride),
                  const_cast<int *>(pass_offset),
                  const_cast<int *>(&input_passes),
                  const_cast<int *>(&context.num_samples),
                  const_cast<int *>(&context.pass_sample_count)};

  return context.queue.enqueue(DEVICE_KERNEL_FILTER_CONVERT_TO_RGB, work_size, args);
}

bool OptiXDevice::denoise_filter_convert_from_rgb(DenoiseContext &context, const DenoisePass &pass)
{
  const BufferParams &buffer_params = context.buffer_params;

  const int work_size = buffer_params.width * buffer_params.height;

  void *args[] = {const_cast<device_ptr *>(&context.input_rgb.device_pointer),
                  &context.render_buffers->buffer.device_pointer,
                  const_cast<int *>(&buffer_params.full_x),
                  const_cast<int *>(&buffer_params.full_y),
                  const_cast<int *>(&buffer_params.width),
                  const_cast<int *>(&buffer_params.height),
                  const_cast<int *>(&buffer_params.offset),
                  const_cast<int *>(&buffer_params.stride),
                  const_cast<int *>(&buffer_params.pass_stride),
                  const_cast<int *>(&context.num_samples),
                  const_cast<int *>(&pass.noisy_offset),
                  const_cast<int *>(&pass.denoised_offset),
                  const_cast<int *>(&context.pass_sample_count)};

  return context.queue.enqueue(DEVICE_KERNEL_FILTER_CONVERT_FROM_RGB, work_size, args);
}

bool OptiXDevice::denoise_ensure(const DeviceDenoiseTask &task)
{
  if (!denoise_create_if_needed(task.params)) {
    LOG(ERROR) << "OptiX denoiser creation has failed.";
    return false;
  }

  if (!denoise_configure_if_needed(task)) {
    LOG(ERROR) << "OptiX denoiser configuration has failed.";
    return false;
  }

  return true;
}

bool OptiXDevice::denoise_create_if_needed(const DenoiseParams &params)
{
  const int input_passes = denoise_buffer_num_passes(params);

  const bool recreate_denoiser = (denoiser_.optix_denoiser == nullptr) ||
                                 (input_passes != denoiser_.input_passes);
  if (!recreate_denoiser) {
    return true;
  }

  /* Destroy existing handle before creating new one. */
  if (denoiser_.optix_denoiser) {
    optixDenoiserDestroy(denoiser_.optix_denoiser);
  }

  /* Create OptiX denoiser handle on demand when it is first used. */
  OptixDenoiserOptions denoiser_options = {};
#  if OPTIX_ABI_VERSION >= 47
  denoiser_options.guideAlbedo = input_passes >= 2;
  denoiser_options.guideNormal = input_passes >= 3;
  const OptixResult result = optixDenoiserCreate(
      context, OPTIX_DENOISER_MODEL_KIND_HDR, &denoiser_options, &denoiser_.optix_denoiser);
#  else
  denoiser_options.inputKind = static_cast<OptixDenoiserInputKind>(OPTIX_DENOISER_INPUT_RGB +
                                                                   (input_passes - 1));
#    if OPTIX_ABI_VERSION < 28
  denoiser_options.pixelFormat = OPTIX_PIXEL_FORMAT_FLOAT3;
#    endif

  const OptixResult result = optixDenoiserCreate(
      context, &denoiser_options, &denoiser_.optix_denoiser);
#  endif

  if (result != OPTIX_SUCCESS) {
    set_error("Failed to create OptiX denoiser");
    return false;
  }

#  if OPTIX_ABI_VERSION < 47
  optix_assert(
      optixDenoiserSetModel(denoiser_.optix_denoiser, OPTIX_DENOISER_MODEL_KIND_HDR, nullptr, 0));
#  endif

  /* OptiX denoiser handle was created with the requested number of input passes. */
  denoiser_.input_passes = input_passes;

  /* OptiX denoiser has been created, but it needs configuration. */
  denoiser_.is_configured = false;

  return true;
}

bool OptiXDevice::denoise_configure_if_needed(const DeviceDenoiseTask &task)
{
  if (denoiser_.is_configured && (denoiser_.configured_size.x == task.buffer_params.width &&
                                  denoiser_.configured_size.y == task.buffer_params.height)) {
    return true;
  }

  OptixDenoiserSizes sizes = {};
  optix_assert(optixDenoiserComputeMemoryResources(
      denoiser_.optix_denoiser, task.buffer_params.width, task.buffer_params.height, &sizes));

#  if OPTIX_ABI_VERSION < 28
  denoiser_.scratch_size = sizes.recommendedScratchSizeInBytes;
#  else
  denoiser_.scratch_size = sizes.withOverlapScratchSizeInBytes;
#  endif
  denoiser_.scratch_offset = sizes.stateSizeInBytes;

  /* Allocate denoiser state if tile size has changed since last setup. */
  denoiser_.state.alloc_to_device(denoiser_.scratch_offset + denoiser_.scratch_size);

  /* Initialize denoiser state for the current tile size. */
  const OptixResult result = optixDenoiserSetup(denoiser_.optix_denoiser,
                                                0,
                                                task.buffer_params.width,
                                                task.buffer_params.height,
                                                denoiser_.state.device_pointer,
                                                denoiser_.scratch_offset,
                                                denoiser_.state.device_pointer +
                                                    denoiser_.scratch_offset,
                                                denoiser_.scratch_size);
  if (result != OPTIX_SUCCESS) {
    set_error("Failed to set up OptiX denoiser");
    return false;
  }

  denoiser_.is_configured = true;
  denoiser_.configured_size.x = task.buffer_params.width;
  denoiser_.configured_size.y = task.buffer_params.height;

  return true;
}

bool OptiXDevice::denoise_run(DenoiseContext &context)
{
  const BufferParams &buffer_params = context.buffer_params;
  const device_ptr d_input_rgb = context.input_rgb.device_pointer;
  const int pixel_stride = 3 * sizeof(float);
  const int input_stride = context.buffer_params.width * pixel_stride;

  /* Set up input and output layer information. */
  OptixImage2D input_layers[3] = {};
  OptixImage2D output_layers[1] = {};

  for (int i = 0; i < 3; ++i) {
    input_layers[i].data = d_input_rgb +
                           (buffer_params.width * buffer_params.height * pixel_stride * i);
    input_layers[i].width = buffer_params.width;
    input_layers[i].height = buffer_params.height;
    input_layers[i].rowStrideInBytes = input_stride;
    input_layers[i].pixelStrideInBytes = pixel_stride;
    input_layers[i].format = OPTIX_PIXEL_FORMAT_FLOAT3;
  }

  output_layers[0].data = d_input_rgb;
  output_layers[0].width = buffer_params.width;
  output_layers[0].height = buffer_params.height;
  output_layers[0].rowStrideInBytes = input_stride;
  output_layers[0].pixelStrideInBytes = pixel_stride;
  output_layers[0].format = OPTIX_PIXEL_FORMAT_FLOAT3;

  /* Finally run denonising. */
  OptixDenoiserParams params = {}; /* All parameters are disabled/zero. */
#  if OPTIX_ABI_VERSION >= 47
  OptixDenoiserLayer image_layers = {};
  image_layers.input = input_layers[0];
  image_layers.output = output_layers[0];

  OptixDenoiserGuideLayer guide_layers = {};
  guide_layers.albedo = input_layers[1];
  guide_layers.normal = input_layers[2];

  optix_assert(optixDenoiserInvoke(denoiser_.optix_denoiser,
                                   context.queue.stream(),
                                   &params,
                                   denoiser_.state.device_pointer,
                                   denoiser_.scratch_offset,
                                   &guide_layers,
                                   &image_layers,
                                   1,
                                   0,
                                   0,
                                   denoiser_.state.device_pointer + denoiser_.scratch_offset,
                                   denoiser_.scratch_size));
#  else
  const int input_passes = denoise_buffer_num_passes(context.denoise_params);

  optix_assert(optixDenoiserInvoke(denoiser_.optix_denoiser,
                                   context.queue.stream(),
                                   &params,
                                   denoiser_.state.device_pointer,
                                   denoiser_.scratch_offset,
                                   input_layers,
                                   input_passes,
                                   0,
                                   0,
                                   output_layers,
                                   denoiser_.state.device_pointer + denoiser_.scratch_offset,
                                   denoiser_.scratch_size));
#  endif

  return true;
}

bool OptiXDevice::build_optix_bvh(BVHOptiX *bvh,
                                  OptixBuildOperation operation,
                                  const OptixBuildInput &build_input,
                                  uint16_t num_motion_steps)
{
  const CUDAContextScope scope(this);

  const bool use_fast_trace_bvh = (bvh->params.bvh_type == BVH_TYPE_STATIC);

  /* Compute memory usage. */
  OptixAccelBufferSizes sizes = {};
  OptixAccelBuildOptions options = {};
  options.operation = operation;
  if (use_fast_trace_bvh) {
    VLOG(2) << "Using fast to trace OptiX BVH";
    options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
  }
  else {
    VLOG(2) << "Using fast to update OptiX BVH";
    options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_BUILD | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
  }

  options.motionOptions.numKeys = num_motion_steps;
  options.motionOptions.flags = OPTIX_MOTION_FLAG_START_VANISH | OPTIX_MOTION_FLAG_END_VANISH;
  options.motionOptions.timeBegin = 0.0f;
  options.motionOptions.timeEnd = 1.0f;

  optix_assert(optixAccelComputeMemoryUsage(context, &options, &build_input, 1, &sizes));

  /* Allocate required output buffers. */
  device_only_memory<char> temp_mem(this, "optix temp as build mem");
  temp_mem.alloc_to_device(align_up(sizes.tempSizeInBytes, 8) + 8);
  if (!temp_mem.device_pointer) {
    /* Make sure temporary memory allocation succeeded. */
    return false;
  }

  device_only_memory<char> &out_data = bvh->as_data;
  if (operation == OPTIX_BUILD_OPERATION_BUILD) {
    assert(out_data.device == this);
    out_data.alloc_to_device(sizes.outputSizeInBytes);
    if (!out_data.device_pointer) {
      return false;
    }
  }
  else {
    assert(out_data.device_pointer && out_data.device_size >= sizes.outputSizeInBytes);
  }

  /* Finally build the acceleration structure. */
  OptixAccelEmitDesc compacted_size_prop = {};
  compacted_size_prop.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
  /* A tiny space was allocated for this property at the end of the temporary buffer above.
   * Make sure this pointer is 8-byte aligned. */
  compacted_size_prop.result = align_up(temp_mem.device_pointer + sizes.tempSizeInBytes, 8);

  OptixTraversableHandle out_handle = 0;
  optix_assert(optixAccelBuild(context,
                               NULL,
                               &options,
                               &build_input,
                               1,
                               temp_mem.device_pointer,
                               sizes.tempSizeInBytes,
                               out_data.device_pointer,
                               sizes.outputSizeInBytes,
                               &out_handle,
                               use_fast_trace_bvh ? &compacted_size_prop : NULL,
                               use_fast_trace_bvh ? 1 : 0));
  bvh->traversable_handle = static_cast<uint64_t>(out_handle);

  /* Wait for all operations to finish. */
  cuda_assert(cuStreamSynchronize(NULL));

  /* Compact acceleration structure to save memory (do not do this in viewport for faster builds).
   */
  if (use_fast_trace_bvh) {
    uint64_t compacted_size = sizes.outputSizeInBytes;
    cuda_assert(cuMemcpyDtoH(&compacted_size, compacted_size_prop.result, sizeof(compacted_size)));

    /* Temporary memory is no longer needed, so free it now to make space. */
    temp_mem.free();

    /* There is no point compacting if the size does not change. */
    if (compacted_size < sizes.outputSizeInBytes) {
      device_only_memory<char> compacted_data(this, "optix compacted as");
      compacted_data.alloc_to_device(compacted_size);
      if (!compacted_data.device_pointer)
        /* Do not compact if memory allocation for compacted acceleration structure fails.
         * Can just use the uncompacted one then, so succeed here regardless. */
        return !have_error();

      optix_assert(optixAccelCompact(
          context, NULL, out_handle, compacted_data.device_pointer, compacted_size, &out_handle));
      bvh->traversable_handle = static_cast<uint64_t>(out_handle);

      /* Wait for compaction to finish. */
      cuda_assert(cuStreamSynchronize(NULL));

      std::swap(out_data.device_size, compacted_data.device_size);
      std::swap(out_data.device_pointer, compacted_data.device_pointer);
    }
  }

  return !have_error();
}

void OptiXDevice::build_bvh(BVH *bvh, Progress &progress, bool refit)
{
  const bool use_fast_trace_bvh = (bvh->params.bvh_type == BVH_TYPE_STATIC);

  free_bvh_memory_delayed();

  BVHOptiX *const bvh_optix = static_cast<BVHOptiX *>(bvh);

  progress.set_substatus("Building OptiX acceleration structure");

  if (!bvh->params.top_level) {
    assert(bvh->objects.size() == 1 && bvh->geometry.size() == 1);

    /* Refit is only possible in viewport for now (because AS is built with
     * OPTIX_BUILD_FLAG_ALLOW_UPDATE only there, see above). */
    OptixBuildOperation operation = OPTIX_BUILD_OPERATION_BUILD;
    if (refit && !use_fast_trace_bvh) {
      assert(bvh_optix->traversable_handle != 0);
      operation = OPTIX_BUILD_OPERATION_UPDATE;
    }
    else {
      bvh_optix->as_data.free();
      bvh_optix->traversable_handle = 0;
    }

    /* Build bottom level acceleration structures (BLAS). */
    Geometry *const geom = bvh->geometry[0];
    if (geom->geometry_type == Geometry::HAIR) {
      /* Build BLAS for curve primitives. */
      Hair *const hair = static_cast<Hair *const>(geom);
      if (hair->num_curves() == 0) {
        return;
      }

      const size_t num_segments = hair->num_segments();

      size_t num_motion_steps = 1;
      Attribute *motion_keys = hair->attributes.find(ATTR_STD_MOTION_VERTEX_POSITION);
      if (motion_blur && hair->get_use_motion_blur() && motion_keys) {
        num_motion_steps = hair->get_motion_steps();
      }

      device_vector<OptixAabb> aabb_data(this, "optix temp aabb data", MEM_READ_ONLY);
#  if OPTIX_ABI_VERSION >= 36
      device_vector<int> index_data(this, "optix temp index data", MEM_READ_ONLY);
      device_vector<float4> vertex_data(this, "optix temp vertex data", MEM_READ_ONLY);
      /* Four control points for each curve segment. */
      const size_t num_vertices = num_segments * 4;
      if (DebugFlags().optix.use_curves_api && hair->curve_shape == CURVE_THICK) {
        index_data.alloc(num_segments);
        vertex_data.alloc(num_vertices * num_motion_steps);
      }
      else
#  endif
        aabb_data.alloc(num_segments * num_motion_steps);

      /* Get AABBs for each motion step. */
      for (size_t step = 0; step < num_motion_steps; ++step) {
        /* The center step for motion vertices is not stored in the attribute. */
        const float3 *keys = hair->get_curve_keys().data();
        size_t center_step = (num_motion_steps - 1) / 2;
        if (step != center_step) {
          size_t attr_offset = (step > center_step) ? step - 1 : step;
          /* Technically this is a float4 array, but sizeof(float3) == sizeof(float4). */
          keys = motion_keys->data_float3() + attr_offset * hair->get_curve_keys().size();
        }

        for (size_t j = 0, i = 0; j < hair->num_curves(); ++j) {
          const Hair::Curve curve = hair->get_curve(j);
#  if OPTIX_ABI_VERSION >= 36
          const array<float> &curve_radius = hair->get_curve_radius();
#  endif

          for (int segment = 0; segment < curve.num_segments(); ++segment, ++i) {
#  if OPTIX_ABI_VERSION >= 36
            if (DebugFlags().optix.use_curves_api && hair->curve_shape == CURVE_THICK) {
              int k0 = curve.first_key + segment;
              int k1 = k0 + 1;
              int ka = max(k0 - 1, curve.first_key);
              int kb = min(k1 + 1, curve.first_key + curve.num_keys - 1);

              const float4 px = make_float4(keys[ka].x, keys[k0].x, keys[k1].x, keys[kb].x);
              const float4 py = make_float4(keys[ka].y, keys[k0].y, keys[k1].y, keys[kb].y);
              const float4 pz = make_float4(keys[ka].z, keys[k0].z, keys[k1].z, keys[kb].z);
              const float4 pw = make_float4(
                  curve_radius[ka], curve_radius[k0], curve_radius[k1], curve_radius[kb]);

              /* Convert Catmull-Rom data to Bezier spline. */
              static const float4 cr2bsp0 = make_float4(+7, -4, +5, -2) / 6.f;
              static const float4 cr2bsp1 = make_float4(-2, 11, -4, +1) / 6.f;
              static const float4 cr2bsp2 = make_float4(+1, -4, 11, -2) / 6.f;
              static const float4 cr2bsp3 = make_float4(-2, +5, -4, +7) / 6.f;

              index_data[i] = i * 4;
              float4 *const v = vertex_data.data() + step * num_vertices + index_data[i];
              v[0] = make_float4(
                  dot(cr2bsp0, px), dot(cr2bsp0, py), dot(cr2bsp0, pz), dot(cr2bsp0, pw));
              v[1] = make_float4(
                  dot(cr2bsp1, px), dot(cr2bsp1, py), dot(cr2bsp1, pz), dot(cr2bsp1, pw));
              v[2] = make_float4(
                  dot(cr2bsp2, px), dot(cr2bsp2, py), dot(cr2bsp2, pz), dot(cr2bsp2, pw));
              v[3] = make_float4(
                  dot(cr2bsp3, px), dot(cr2bsp3, py), dot(cr2bsp3, pz), dot(cr2bsp3, pw));
            }
            else
#  endif
            {
              BoundBox bounds = BoundBox::empty;
              curve.bounds_grow(segment, keys, hair->get_curve_radius().data(), bounds);

              const size_t index = step * num_segments + i;
              aabb_data[index].minX = bounds.min.x;
              aabb_data[index].minY = bounds.min.y;
              aabb_data[index].minZ = bounds.min.z;
              aabb_data[index].maxX = bounds.max.x;
              aabb_data[index].maxY = bounds.max.y;
              aabb_data[index].maxZ = bounds.max.z;
            }
          }
        }
      }

      /* Upload AABB data to GPU. */
      aabb_data.copy_to_device();
#  if OPTIX_ABI_VERSION >= 36
      index_data.copy_to_device();
      vertex_data.copy_to_device();
#  endif

      vector<device_ptr> aabb_ptrs;
      aabb_ptrs.reserve(num_motion_steps);
#  if OPTIX_ABI_VERSION >= 36
      vector<device_ptr> width_ptrs;
      vector<device_ptr> vertex_ptrs;
      width_ptrs.reserve(num_motion_steps);
      vertex_ptrs.reserve(num_motion_steps);
#  endif
      for (size_t step = 0; step < num_motion_steps; ++step) {
        aabb_ptrs.push_back(aabb_data.device_pointer + step * num_segments * sizeof(OptixAabb));
#  if OPTIX_ABI_VERSION >= 36
        const device_ptr base_ptr = vertex_data.device_pointer +
                                    step * num_vertices * sizeof(float4);
        width_ptrs.push_back(base_ptr + 3 * sizeof(float)); /* Offset by vertex size. */
        vertex_ptrs.push_back(base_ptr);
#  endif
      }

      /* Force a single any-hit call, so shadow record-all behavior works correctly. */
      unsigned int build_flags = OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL;
      OptixBuildInput build_input = {};
#  if OPTIX_ABI_VERSION >= 36
      if (DebugFlags().optix.use_curves_api && hair->curve_shape == CURVE_THICK) {
        build_input.type = OPTIX_BUILD_INPUT_TYPE_CURVES;
        build_input.curveArray.curveType = OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE;
        build_input.curveArray.numPrimitives = num_segments;
        build_input.curveArray.vertexBuffers = (CUdeviceptr *)vertex_ptrs.data();
        build_input.curveArray.numVertices = num_vertices;
        build_input.curveArray.vertexStrideInBytes = sizeof(float4);
        build_input.curveArray.widthBuffers = (CUdeviceptr *)width_ptrs.data();
        build_input.curveArray.widthStrideInBytes = sizeof(float4);
        build_input.curveArray.indexBuffer = (CUdeviceptr)index_data.device_pointer;
        build_input.curveArray.indexStrideInBytes = sizeof(int);
        build_input.curveArray.flag = build_flags;
        build_input.curveArray.primitiveIndexOffset = hair->optix_prim_offset;
      }
      else
#  endif
      {
        /* Disable visibility test any-hit program, since it is already checked during
         * intersection. Those trace calls that require anyhit can force it with a ray flag. */
        build_flags |= OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;

        build_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
#  if OPTIX_ABI_VERSION < 23
        build_input.aabbArray.aabbBuffers = (CUdeviceptr *)aabb_ptrs.data();
        build_input.aabbArray.numPrimitives = num_segments;
        build_input.aabbArray.strideInBytes = sizeof(OptixAabb);
        build_input.aabbArray.flags = &build_flags;
        build_input.aabbArray.numSbtRecords = 1;
        build_input.aabbArray.primitiveIndexOffset = hair->optix_prim_offset;
#  else
        build_input.customPrimitiveArray.aabbBuffers = (CUdeviceptr *)aabb_ptrs.data();
        build_input.customPrimitiveArray.numPrimitives = num_segments;
        build_input.customPrimitiveArray.strideInBytes = sizeof(OptixAabb);
        build_input.customPrimitiveArray.flags = &build_flags;
        build_input.customPrimitiveArray.numSbtRecords = 1;
        build_input.customPrimitiveArray.primitiveIndexOffset = hair->optix_prim_offset;
#  endif
      }

      if (!build_optix_bvh(bvh_optix, operation, build_input, num_motion_steps)) {
        progress.set_error("Failed to build OptiX acceleration structure");
      }
    }
    else if (geom->geometry_type == Geometry::MESH || geom->geometry_type == Geometry::VOLUME) {
      /* Build BLAS for triangle primitives. */
      Mesh *const mesh = static_cast<Mesh *const>(geom);
      if (mesh->num_triangles() == 0) {
        return;
      }

      const size_t num_verts = mesh->get_verts().size();

      size_t num_motion_steps = 1;
      Attribute *motion_keys = mesh->attributes.find(ATTR_STD_MOTION_VERTEX_POSITION);
      if (motion_blur && mesh->get_use_motion_blur() && motion_keys) {
        num_motion_steps = mesh->get_motion_steps();
      }

      device_vector<int> index_data(this, "optix temp index data", MEM_READ_ONLY);
      index_data.alloc(mesh->get_triangles().size());
      memcpy(index_data.data(),
             mesh->get_triangles().data(),
             mesh->get_triangles().size() * sizeof(int));
      device_vector<float3> vertex_data(this, "optix temp vertex data", MEM_READ_ONLY);
      vertex_data.alloc(num_verts * num_motion_steps);

      for (size_t step = 0; step < num_motion_steps; ++step) {
        const float3 *verts = mesh->get_verts().data();

        size_t center_step = (num_motion_steps - 1) / 2;
        /* The center step for motion vertices is not stored in the attribute. */
        if (step != center_step) {
          verts = motion_keys->data_float3() + (step > center_step ? step - 1 : step) * num_verts;
        }

        memcpy(vertex_data.data() + num_verts * step, verts, num_verts * sizeof(float3));
      }

      /* Upload triangle data to GPU. */
      index_data.copy_to_device();
      vertex_data.copy_to_device();

      vector<device_ptr> vertex_ptrs;
      vertex_ptrs.reserve(num_motion_steps);
      for (size_t step = 0; step < num_motion_steps; ++step) {
        vertex_ptrs.push_back(vertex_data.device_pointer + num_verts * step * sizeof(float3));
      }

      /* Force a single any-hit call, so shadow record-all behavior works correctly. */
      unsigned int build_flags = OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL;
      OptixBuildInput build_input = {};
      build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
      build_input.triangleArray.vertexBuffers = (CUdeviceptr *)vertex_ptrs.data();
      build_input.triangleArray.numVertices = num_verts;
      build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
      build_input.triangleArray.vertexStrideInBytes = sizeof(float3);
      build_input.triangleArray.indexBuffer = index_data.device_pointer;
      build_input.triangleArray.numIndexTriplets = mesh->num_triangles();
      build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
      build_input.triangleArray.indexStrideInBytes = 3 * sizeof(int);
      build_input.triangleArray.flags = &build_flags;
      /* The SBT does not store per primitive data since Cycles already allocates separate
       * buffers for that purpose. OptiX does not allow this to be zero though, so just pass in
       * one and rely on that having the same meaning in this case. */
      build_input.triangleArray.numSbtRecords = 1;
      build_input.triangleArray.primitiveIndexOffset = mesh->optix_prim_offset;

      if (!build_optix_bvh(bvh_optix, operation, build_input, num_motion_steps)) {
        progress.set_error("Failed to build OptiX acceleration structure");
      }
    }
  }
  else {
    unsigned int num_instances = 0;
    unsigned int max_num_instances = 0xFFFFFFFF;

    bvh_optix->as_data.free();
    bvh_optix->traversable_handle = 0;
    bvh_optix->motion_transform_data.free();

    optixDeviceContextGetProperty(context,
                                  OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID,
                                  &max_num_instances,
                                  sizeof(max_num_instances));
    /* Do not count first bit, which is used to distinguish instanced and non-instanced objects. */
    max_num_instances >>= 1;
    if (bvh->objects.size() > max_num_instances) {
      progress.set_error(
          "Failed to build OptiX acceleration structure because there are too many instances");
      return;
    }

    /* Fill instance descriptions. */
#  if OPTIX_ABI_VERSION < 41
    device_vector<OptixAabb> aabbs(this, "optix tlas aabbs", MEM_READ_ONLY);
    aabbs.alloc(bvh->objects.size());
#  endif
    device_vector<OptixInstance> instances(this, "optix tlas instances", MEM_READ_ONLY);
    instances.alloc(bvh->objects.size());

    /* Calculate total motion transform size and allocate memory for them. */
    size_t motion_transform_offset = 0;
    if (motion_blur) {
      size_t total_motion_transform_size = 0;
      for (Object *const ob : bvh->objects) {
        if (ob->is_traceable() && ob->use_motion()) {
          total_motion_transform_size = align_up(total_motion_transform_size,
                                                 OPTIX_TRANSFORM_BYTE_ALIGNMENT);
          const size_t motion_keys = max(ob->get_motion().size(), 2) - 2;
          total_motion_transform_size = total_motion_transform_size +
                                        sizeof(OptixSRTMotionTransform) +
                                        motion_keys * sizeof(OptixSRTData);
        }
      }

      assert(bvh_optix->motion_transform_data.device == this);
      bvh_optix->motion_transform_data.alloc_to_device(total_motion_transform_size);
    }

    for (Object *ob : bvh->objects) {
      /* Skip non-traceable objects. */
      if (!ob->is_traceable()) {
        continue;
      }

      BVHOptiX *const blas = static_cast<BVHOptiX *>(ob->get_geometry()->bvh);
      OptixTraversableHandle handle = blas->traversable_handle;

#  if OPTIX_ABI_VERSION < 41
      OptixAabb &aabb = aabbs[num_instances];
      aabb.minX = ob->bounds.min.x;
      aabb.minY = ob->bounds.min.y;
      aabb.minZ = ob->bounds.min.z;
      aabb.maxX = ob->bounds.max.x;
      aabb.maxY = ob->bounds.max.y;
      aabb.maxZ = ob->bounds.max.z;
#  endif

      OptixInstance &instance = instances[num_instances++];
      memset(&instance, 0, sizeof(instance));

      /* Clear transform to identity matrix. */
      instance.transform[0] = 1.0f;
      instance.transform[5] = 1.0f;
      instance.transform[10] = 1.0f;

      /* Set user instance ID to object index (but leave low bit blank). */
      instance.instanceId = ob->get_device_index() << 1;

      /* Have to have at least one bit in the mask, or else instance would always be culled. */
      instance.visibilityMask = 1;

      if (ob->get_geometry()->has_volume) {
        /* Volumes have a special bit set in the visibility mask so a trace can mask only volumes.
         */
        instance.visibilityMask |= 2;
      }

      if (ob->get_geometry()->geometry_type == Geometry::HAIR) {
        /* Same applies to curves (so they can be skipped in local trace calls). */
        instance.visibilityMask |= 4;

#  if OPTIX_ABI_VERSION >= 36
        if (motion_blur && ob->get_geometry()->has_motion_blur() &&
            DebugFlags().optix.use_curves_api &&
            static_cast<const Hair *>(ob->get_geometry())->curve_shape == CURVE_THICK) {
          /* Select between motion blur and non-motion blur built-in intersection module. */
          instance.sbtOffset = PG_HITD_MOTION - PG_HITD;
        }
#  endif
      }

      /* Insert motion traversable if object has motion. */
      if (motion_blur && ob->use_motion()) {
        size_t motion_keys = max(ob->get_motion().size(), 2) - 2;
        size_t motion_transform_size = sizeof(OptixSRTMotionTransform) +
                                       motion_keys * sizeof(OptixSRTData);

        const CUDAContextScope scope(this);

        motion_transform_offset = align_up(motion_transform_offset,
                                           OPTIX_TRANSFORM_BYTE_ALIGNMENT);
        CUdeviceptr motion_transform_gpu = bvh_optix->motion_transform_data.device_pointer +
                                           motion_transform_offset;
        motion_transform_offset += motion_transform_size;

        /* Allocate host side memory for motion transform and fill it with transform data. */
        OptixSRTMotionTransform &motion_transform = *reinterpret_cast<OptixSRTMotionTransform *>(
            new uint8_t[motion_transform_size]);
        motion_transform.child = handle;
        motion_transform.motionOptions.numKeys = ob->get_motion().size();
        motion_transform.motionOptions.flags = OPTIX_MOTION_FLAG_NONE;
        motion_transform.motionOptions.timeBegin = 0.0f;
        motion_transform.motionOptions.timeEnd = 1.0f;

        OptixSRTData *const srt_data = motion_transform.srtData;
        array<DecomposedTransform> decomp(ob->get_motion().size());
        transform_motion_decompose(
            decomp.data(), ob->get_motion().data(), ob->get_motion().size());

        for (size_t i = 0; i < ob->get_motion().size(); ++i) {
          /* Scale. */
          srt_data[i].sx = decomp[i].y.w; /* scale.x.x */
          srt_data[i].sy = decomp[i].z.w; /* scale.y.y */
          srt_data[i].sz = decomp[i].w.w; /* scale.z.z */

          /* Shear. */
          srt_data[i].a = decomp[i].z.x; /* scale.x.y */
          srt_data[i].b = decomp[i].z.y; /* scale.x.z */
          srt_data[i].c = decomp[i].w.x; /* scale.y.z */
          assert(decomp[i].z.z == 0.0f); /* scale.y.x */
          assert(decomp[i].w.y == 0.0f); /* scale.z.x */
          assert(decomp[i].w.z == 0.0f); /* scale.z.y */

          /* Pivot point. */
          srt_data[i].pvx = 0.0f;
          srt_data[i].pvy = 0.0f;
          srt_data[i].pvz = 0.0f;

          /* Rotation. */
          srt_data[i].qx = decomp[i].x.x;
          srt_data[i].qy = decomp[i].x.y;
          srt_data[i].qz = decomp[i].x.z;
          srt_data[i].qw = decomp[i].x.w;

          /* Translation. */
          srt_data[i].tx = decomp[i].y.x;
          srt_data[i].ty = decomp[i].y.y;
          srt_data[i].tz = decomp[i].y.z;
        }

        /* Upload motion transform to GPU. */
        cuMemcpyHtoD(motion_transform_gpu, &motion_transform, motion_transform_size);
        delete[] reinterpret_cast<uint8_t *>(&motion_transform);

        /* Disable instance transform if object uses motion transform already. */
        instance.flags = OPTIX_INSTANCE_FLAG_DISABLE_TRANSFORM;

        /* Get traversable handle to motion transform. */
        optixConvertPointerToTraversableHandle(context,
                                               motion_transform_gpu,
                                               OPTIX_TRAVERSABLE_TYPE_SRT_MOTION_TRANSFORM,
                                               &instance.traversableHandle);
      }
      else {
        instance.traversableHandle = handle;

        if (ob->get_geometry()->is_instanced()) {
          /* Set transform matrix. */
          memcpy(instance.transform, &ob->get_tfm(), sizeof(instance.transform));
        }
        else {
          /* Disable instance transform if geometry already has it applied to vertex data. */
          instance.flags = OPTIX_INSTANCE_FLAG_DISABLE_TRANSFORM;
          /* Non-instanced objects read ID from 'prim_object', so distinguish
           * them from instanced objects with the low bit set. */
          instance.instanceId |= 1;
        }
      }
    }

    /* Upload instance descriptions. */
#  if OPTIX_ABI_VERSION < 41
    aabbs.resize(num_instances);
    aabbs.copy_to_device();
#  endif
    instances.resize(num_instances);
    instances.copy_to_device();

    /* Build top-level acceleration structure (TLAS) */
    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
#  if OPTIX_ABI_VERSION < 41 /* Instance AABBs no longer need to be set since OptiX 7.2 */
    build_input.instanceArray.aabbs = aabbs.device_pointer;
    build_input.instanceArray.numAabbs = num_instances;
#  endif
    build_input.instanceArray.instances = instances.device_pointer;
    build_input.instanceArray.numInstances = num_instances;

    if (!build_optix_bvh(bvh_optix, OPTIX_BUILD_OPERATION_BUILD, build_input, 0)) {
      progress.set_error("Failed to build OptiX acceleration structure");
    }
    tlas_handle = bvh_optix->traversable_handle;
  }
}

void OptiXDevice::release_optix_bvh(BVH *bvh)
{
  thread_scoped_lock lock(delayed_free_bvh_mutex);
  /* Do delayed free of BVH memory, since geometry holding BVH might be deleted
   * while GPU is still rendering. */
  BVHOptiX *const bvh_optix = static_cast<BVHOptiX *>(bvh);

  delayed_free_bvh_memory.emplace_back(std::move(bvh_optix->as_data));
  delayed_free_bvh_memory.emplace_back(std::move(bvh_optix->motion_transform_data));
  bvh_optix->traversable_handle = 0;
}

void OptiXDevice::free_bvh_memory_delayed()
{
  thread_scoped_lock lock(delayed_free_bvh_mutex);
  delayed_free_bvh_memory.free_memory();
}

void OptiXDevice::const_copy_to(const char *name, void *host, size_t size)
{
  /* Set constant memory for CUDA module. */
  CUDADevice::const_copy_to(name, host, size);

  if (strcmp(name, "__data") == 0) {
    assert(size <= sizeof(KernelData));

    /* Update traversable handle (since it is different for each device on multi devices). */
    KernelData *const data = (KernelData *)host;
    *(OptixTraversableHandle *)&data->bvh.scene = tlas_handle;

    update_launch_params(offsetof(KernelParamsOptiX, data), host, size);
    return;
  }

  /* Update data storage pointers in launch parameters. */
#  define KERNEL_TEX(data_type, tex_name) \
    if (strcmp(name, #tex_name) == 0) { \
      update_launch_params(offsetof(KernelParamsOptiX, tex_name), host, size); \
      return; \
    }
  KERNEL_TEX(IntegratorState, __integrator_state)
#  include "kernel/kernel_textures.h"
#  undef KERNEL_TEX
}

void OptiXDevice::update_launch_params(size_t offset, void *data, size_t data_size)
{
  const CUDAContextScope scope(this);

  cuda_assert(cuMemcpyHtoD(launch_params.device_pointer + offset, data, data_size));
}

CCL_NAMESPACE_END

#endif /* WITH_OPTIX */
