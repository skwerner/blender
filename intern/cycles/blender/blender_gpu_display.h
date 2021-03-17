/*
 * Copyright 2021 Blender Foundation
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

#include "MEM_guardedalloc.h"

#include "RNA_blender_cpp.h"

#include "render/gpu_display.h"
#include "util/util_unique_ptr.h"

/* TODO(sergey): Only for until BlenderGPUDisplay can have its own OpenGL context. */
#include "util/util_array.h"
#include "util/util_half.h"
#include "util/util_types.h"

CCL_NAMESPACE_BEGIN

/* Base class of shader used for GPU display rendering. */
class BlenderDisplayShader {
 public:
  static constexpr const char *position_attribute_name = "pos";
  static constexpr const char *tex_coord_attribute_name = "texCoord";

  /* Create shader implementation suitable for the given render engine and scene configuration. */
  static unique_ptr<BlenderDisplayShader> create(BL::RenderEngine &b_engine, BL::Scene &b_scene);

  BlenderDisplayShader() = default;
  virtual ~BlenderDisplayShader() = default;

  virtual void bind(int width, int height) = 0;
  virtual void unbind() = 0;

  /* Get attribute location for position and texture coordinate respectively.
   * NOTE: The shader needs to be bound to have access to those. */
  virtual int get_position_attrib_location();
  virtual int get_tex_coord_attrib_location();

 protected:
  /* Get program of this display shader.
   * NOTE: The shader needs to be bound to have access to this. */
  virtual uint get_shader_program() = 0;

  /* Cached values of various OpenGL resources. */
  int position_attribute_location_ = -1;
  int tex_coord_attribute_location_ = -1;
};

/* Implementation of display rendering shader used in the case when render engine does not support
 * display space shader. */
class BlenderFallbackDisplayShader : public BlenderDisplayShader {
 public:
  virtual void bind(int width, int height) override;
  virtual void unbind() override;

 protected:
  virtual uint get_shader_program() override;

  void create_shader_if_needed();
  void destroy_shader();

  uint shader_program_ = 0;
  int image_texture_location_ = -1;
  int fullscreen_location_ = -1;

  /* Shader compilation attempted. Which means, that if the shader program is 0 then compilation or
   * linking has failed. Do not attempt to re-compile the shader. */
  bool shader_compile_attempted_ = false;
};

class BlenderDisplaySpaceShader : public BlenderDisplayShader {
 public:
  BlenderDisplaySpaceShader(BL::RenderEngine &b_engine, BL::Scene &b_scene);

  virtual void bind(int width, int height) override;
  virtual void unbind() override;

 protected:
  virtual uint get_shader_program() override;

  BL::RenderEngine b_engine_;
  BL::Scene &b_scene_;

  /* Cached values of various OpenGL resources. */
  uint shader_program_ = 0;
};

/* GPU display implementation which is specific for Blender viewport integration. */
class BlenderGPUDisplay : public GPUDisplay {
 public:
  BlenderGPUDisplay(BL::RenderEngine &b_engine, BL::Scene &b_scene);
  ~BlenderGPUDisplay();

  virtual void reset(BufferParams &buffer_params) override;

  virtual void copy_pixels_to_texture(const half4 *rgba_pixels, int width, int height) override;

  virtual void get_cuda_buffer() override;

  virtual bool draw() override;

 protected:
  /* Helper function which allocates new GPU context, without affecting the current
   * active GPU context. */
  void gpu_context_create();

  /* Ensure all runtime GPU resources are allocated.
   * Returns true if all resources needed for drawing are available. */
  bool gpu_resources_ensure();

  /* Destroy all GPU resources which are being used by this object. */
  void gpu_resources_destroy();

  /* Create and perform initial configuration of texture on the GPU side.
   *
   * NOTE: Must be called from a proper active GPU context. */
  bool create_texture();

  /* Update vetrex buffer with new coordinates of vertex positions and texture coordinates.
   * This buffer is used to render texture in the viewport.
   *
   * NOTE: The buffer needs to be bound. */
  void update_vertex_buffer();

  unique_ptr<BlenderDisplayShader> display_shader_;

  /* Special track of whether GPU resources were attempted to be created, to avoid attempts of
   * their re-creation on failure on every redraw. */
  bool gpu_resource_creation_attempted_ = false;
  bool gpu_resources_created_ = false;

  /* Texture which contains pixels of the render result. */
  uint texture_id_ = 0;

  /* Vertex buffer which hold vertrices of a triangle fan which is textures with the texture
   * holding the render result.  */
  uint vertex_buffer_ = 0;

  /* Temporary CPU-side code, which is here only until this GPU display have own OpenGL context. */

  /* Storage of pixels which are to be uploaded to the GPU texture. */
  array<half4> rgba_pixels_;

  /* Dimension of the GPU side texture. Could be different from the viewport resolution when there
   * is a non-unit resolution divider. Should match number of pixels in the rgba_ storage. */
  int2 texture_size_ = make_int2(0, 0);

  /* There is a new data in the rgba_ buffer which is to be uploaded to the GPU texture. */
  bool need_update_texture_ = false;

  /* Reset happenned but there is no new data for the buffer yet.
   * Is used to return false from the draw() function, so that the redraw does happen while the new
   * sample is being rendered, but without considering it a "final" or "up-to-date" draw. This
   * makes it so there is no flickering in the viewport and the session reset happens after an
   * actual up-to-date sample rendered. */
  /* TODO(sergey): This is something annoying to have: ideally, it will be very easy to subclass
   * the GPUDisplay without any tricky logic in it. Currently this is needed for an integration
   * with the render Session. We should either avoid such tricky logic all together, or to move it
   * to a base class. */
  bool texture_outdated_ = true;
};

CCL_NAMESPACE_END
