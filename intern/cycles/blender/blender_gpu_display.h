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

 protected:
  virtual bool do_update_begin(int texture_width, int texture_height) override;
  virtual void do_update_end() override;

  virtual void do_copy_pixels_to_texture(const half4 *rgba_pixels,
                                         int texture_x,
                                         int texture_y,
                                         int pixels_width,
                                         int pixels_height) override;
  virtual void do_draw() override;

  virtual half4 *do_map_texture_buffer() override;
  virtual void do_unmap_texture_buffer() override;

  virtual DeviceGraphicsInteropDestination do_graphics_interop_get() override;

  /* Helper function which allocates new GPU context, without affecting the current
   * active GPU context. */
  void gl_context_create();

  /* Make sure texture is allocated and its initial configuration is performed. */
  bool gl_texture_resources_ensure();

  /* Ensure all runtime GPU resources needefd for drawing are allocated.
   * Returns true if all resources needed for drawing are available. */
  bool gl_draw_resources_ensure();

  /* Destroy all GPU resources which are being used by this object. */
  void gl_resources_destroy();

  /* Update GPU texture dimensions and content if needed (new pixel data was provided).
   *
   * NOTE: The texture needs to be bound. */
  void texture_update_if_needed();

  /* Update vetrex buffer with new coordinates of vertex positions and texture coordinates.
   * This buffer is used to render texture in the viewport.
   *
   * NOTE: The buffer needs to be bound. */
  void vertex_buffer_update();

  /* OpenGL context created by Blender's Window Manager.
   * This context is used to perform texture update from the render thread, asynchronously from the
   * main thread which draws the viewport. */
  void *gl_context_ = nullptr;

  /* Texture which contains pixels of the render result. */
  struct {
    /* Indicates whether texture creation was attempted and succeeded.
     * Used to avoid multiple attempts of texture creation on GPU issues or GPU context
     * misconfiguration. */
    bool creation_attempted = false;
    bool is_created = false;

    /* OpenGL resource IDs of the texture itself and Pixel Buffer Object (PBO) used to write
     * pixels to it.
     *
     * NOTE: Allocated on the `gl_context_` context. */
    uint gl_id_ = 0;
    uint gl_pbo_id_ = 0;

    /* Is true when new data was written to the PBO, meaning, the texture might need to be resized
     * and new data is to be uploaded to the GPU. */
    bool need_update = false;

    /* Dimensions of the texture in pixels. */
    int width = 0;
    int height = 0;

    /* Dimensions of the underlying PBO. */
    int buffer_width = 0;
    int buffer_height = 0;
  } texture_;

  unique_ptr<BlenderDisplayShader> display_shader_;

  /* Special track of whether GPU resources were attempted to be created, to avoid attempts of
   * their re-creation on failure on every redraw. */
  bool gl_draw_resource_creation_attempted_ = false;
  bool gl_draw_resources_created_ = false;

  /* Vertex buffer which hold vertrices of a triangle fan which is textures with the texture
   * holding the render result. */
  uint vertex_buffer_ = 0;

  void *gl_render_sync_ = nullptr;
  void *gl_upload_sync_ = nullptr;
};

CCL_NAMESPACE_END
