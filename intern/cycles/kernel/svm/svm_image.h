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

#ifdef __OIIO__
#  include "kernel/kernel_oiio_globals.h"
#  define NEAREST_LOOKUP_PATHS \
    (PATH_RAY_DIFFUSE | PATH_RAY_SHADOW | PATH_RAY_DIFFUSE_ANCESTOR | PATH_RAY_VOLUME_SCATTER | \
     PATH_RAY_GLOSSY | PATH_RAY_EMISSION)
#  define DIFFUSE_BLUR_PATHS (PATH_RAY_DIFFUSE | PATH_RAY_DIFFUSE_ANCESTOR)
#endif

CCL_NAMESPACE_BEGIN

#ifdef __TEXTURES__
ccl_device float4 svm_image_texture(KernelGlobals *kg,
                                    int id,
                                    float x,
                                    float y,
                                    differential ds,
                                    differential dt,
                                    uint flags,
                                    int path_flag)
{
  float4 r;
#ifdef __OIIO__
  if (kg->oiio && kg->oiio->textures.size() > id && kg->oiio->textures[id].handle) {
    OIIO::TextureOpt options;
    options.swrap = options.twrap = kg->oiio->textures[id].extension;
    options.anisotropic = 8;
    float missingcolor[4] = {
        TEX_IMAGE_MISSING_R, TEX_IMAGE_MISSING_G, TEX_IMAGE_MISSING_B, TEX_IMAGE_MISSING_A};
    options.missingcolor = missingcolor;

    if (path_flag & NEAREST_LOOKUP_PATHS && !(path_flag & PATH_RAY_SINGULAR)) {
      options.interpmode = OIIO::TextureOpt::InterpClosest;
      options.mipmode = OIIO::TextureOpt::MipModeOneLevel;
    }
    else {
      options.interpmode = kg->oiio->textures[id].interpolation;
      options.mipmode = OIIO::TextureOpt::MipModeAniso;
    }

    if (path_flag & DIFFUSE_BLUR_PATHS) {
      options.sblur = options.tblur = kg->oiio->diffuse_blur;
    }
    else if (path_flag & PATH_RAY_GLOSSY) {
      options.sblur = options.tblur = kg->oiio->glossy_blur;
    }
    else {
      options.sblur = options.tblur = 0.0f;
    }

    bool success = kg->oiio->tex_sys->texture(kg->oiio->textures[id].handle,
                                              (OIIO::TextureSystem::Perthread *)kg->oiio_tdata,
                                              options,
                                              x,
                                              y,
                                              ds.dx,
                                              ds.dy,
                                              dt.dx,
                                              dt.dy,
                                              4,
                                              (float *)&r);
    if (!success) {
      (void)kg->oiio->tex_sys->geterror();
    }
    else {
      /* Mip maps are always linear. */
      if (kg->oiio->textures[id].is_linear) {
        flags &= ~NODE_IMAGE_COMPRESS_AS_SRGB;
      }
    }
  }
  else
#endif
    r = kernel_tex_image_interp(kg, id, x, y);
  const float alpha = r.w;

  if ((flags & NODE_IMAGE_ALPHA_UNASSOCIATE) && alpha != 1.0f && alpha != 0.0f) {
    r /= alpha;
    const int texture_type = kernel_tex_type(id);
    if (texture_type == IMAGE_DATA_TYPE_BYTE4 || texture_type == IMAGE_DATA_TYPE_BYTE) {
      r = min(r, make_float4(1.0f, 1.0f, 1.0f, 1.0f));
    }
    r.w = alpha;
  }

  if (flags & NODE_IMAGE_COMPRESS_AS_SRGB) {
    r = color_srgb_to_linear_v4(r);
  }

  return r;
}

/* Remap coordnate from 0..1 box to -1..-1 */
ccl_device_inline float3 texco_remap_square(float3 co)
{
  return (co - make_float3(0.5f, 0.5f, 0.5f)) * 2.0f;
}

ccl_device void svm_node_tex_image(
    KernelGlobals *kg, ShaderData *sd, int path_flag, float *stack, uint4 node)
{
  uint id = node.y;

  uint co_offset, out_offset, alpha_offset, flags;
  uint projection, dx_offset, dy_offset, unused;

  svm_unpack_node_uchar4(node.z, &co_offset, &out_offset, &alpha_offset, &flags);
  svm_unpack_node_uchar4(node.w, &projection, &dx_offset, &dy_offset, &unused);

  float3 co = stack_load_float3(stack, co_offset);
  float2 tex_co;
  if (projection == NODE_IMAGE_PROJ_SPHERE) {
    co = texco_remap_square(co);
    tex_co = map_to_sphere(co);
  }
  else if (projection == NODE_IMAGE_PROJ_TUBE) {
    co = texco_remap_square(co);
    tex_co = map_to_tube(co);
  }
  else {
    tex_co = make_float2(co.x, co.y);
  }

  differential ds, dt;
#ifdef __KERNEL_CPU__
  if (stack_valid(dx_offset) && stack_valid(dy_offset)) {
    float3 dx = stack_load_float3(stack, dx_offset);
    float3 dy = stack_load_float3(stack, dy_offset);
    float2 tex_co_dx, tex_co_dy;
    if (projection == NODE_IMAGE_PROJ_SPHERE) {
      dx = texco_remap_square(dx);
      tex_co_dx = map_to_sphere(dx);
      dy = texco_remap_square(dy);
      tex_co_dy = map_to_sphere(dy);
    }
    else if (projection == NODE_IMAGE_PROJ_TUBE) {
      dx = texco_remap_square(dx);
      tex_co_dx = map_to_tube(dx);
      dy = texco_remap_square(dy);
      tex_co_dy = map_to_tube(dy);
    }
    else {
      tex_co_dx = make_float2(dx.x, dx.y);
      tex_co_dy = make_float2(dy.x, dy.y);
    }
    ds.dx = tex_co_dx.x - tex_co.x;
    ds.dy = tex_co_dy.x - tex_co.x;
    dt.dx = tex_co_dx.y - tex_co.y;
    dt.dy = tex_co_dy.y - tex_co.y;
  }
  else
#endif
  {
    ds = differential_zero();
    dt = differential_zero();
  }
  float4 f = svm_image_texture(kg, id, tex_co.x, tex_co.y, ds, dt, flags, path_flag);

  if (stack_valid(out_offset))
    stack_store_float3(stack, out_offset, make_float3(f.x, f.y, f.z));
  if (stack_valid(alpha_offset))
    stack_store_float(stack, alpha_offset, f.w);
}

ccl_device void svm_node_tex_image_box(KernelGlobals *kg, ShaderData *sd, int path_flag, float *stack, uint4 node)
{
  /* get object space normal */
  float3 N = sd->N;

  N = sd->N;
  object_inverse_normal_transform(kg, sd, &N);

  /* project from direction vector to barycentric coordinates in triangles */
  float3 signed_N = N;

  N.x = fabsf(N.x);
  N.y = fabsf(N.y);
  N.z = fabsf(N.z);

  N /= (N.x + N.y + N.z);

  /* basic idea is to think of this as a triangle, each corner representing
   * one of the 3 faces of the cube. in the corners we have single textures,
   * in between we blend between two textures, and in the middle we a blend
   * between three textures.
   *
   * the Nxyz values are the barycentric coordinates in an equilateral
   * triangle, which in case of blending, in the middle has a smaller
   * equilateral triangle where 3 textures blend. this divides things into
   * 7 zones, with an if() test for each zone */

  float3 weight = make_float3(0.0f, 0.0f, 0.0f);
  uint blend_hi, blend_lo, dx_offset, dy_offset;
  svm_unpack_node_uchar4(node.w, &blend_hi, &blend_lo, &dx_offset, &dy_offset);
  float blend = ((blend_hi << 8) + blend_lo) / 65536.0f;
  float limit = 0.5f * (1.0f + blend);

  /* first test for corners with single texture */
  if (N.x > limit * (N.x + N.y) && N.x > limit * (N.x + N.z)) {
    weight.x = 1.0f;
  }
  else if (N.y > limit * (N.x + N.y) && N.y > limit * (N.y + N.z)) {
    weight.y = 1.0f;
  }
  else if (N.z > limit * (N.x + N.z) && N.z > limit * (N.y + N.z)) {
    weight.z = 1.0f;
  }
  else if (blend > 0.0f) {
    /* in case of blending, test for mixes between two textures */
    if (N.z < (1.0f - limit) * (N.y + N.x)) {
      weight.x = N.x / (N.x + N.y);
      weight.x = saturate((weight.x - 0.5f * (1.0f - blend)) / blend);
      weight.y = 1.0f - weight.x;
    }
    else if (N.x < (1.0f - limit) * (N.y + N.z)) {
      weight.y = N.y / (N.y + N.z);
      weight.y = saturate((weight.y - 0.5f * (1.0f - blend)) / blend);
      weight.z = 1.0f - weight.y;
    }
    else if (N.y < (1.0f - limit) * (N.x + N.z)) {
      weight.x = N.x / (N.x + N.z);
      weight.x = saturate((weight.x - 0.5f * (1.0f - blend)) / blend);
      weight.z = 1.0f - weight.x;
    }
    else {
      /* last case, we have a mix between three */
      weight.x = ((2.0f - limit) * N.x + (limit - 1.0f)) / (2.0f * limit - 1.0f);
      weight.y = ((2.0f - limit) * N.y + (limit - 1.0f)) / (2.0f * limit - 1.0f);
      weight.z = ((2.0f - limit) * N.z + (limit - 1.0f)) / (2.0f * limit - 1.0f);
    }
  }
  else {
    /* Desperate mode, no valid choice anyway, fallback to one side.*/
    weight.x = 1.0f;
  }

  /* now fetch textures */
  uint co_offset, out_offset, alpha_offset, flags;
  svm_unpack_node_uchar4(node.z, &co_offset, &out_offset, &alpha_offset, &flags);

  float3 co = stack_load_float3(stack, co_offset);
  uint id = node.y;

  float4 f = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

  float3 co_dx = make_float3(0.0f, 0.0f, 0.0f);
  float3 co_dy = make_float3(0.0f, 0.0f, 0.0f);
  differential ds, dt;
#ifdef __KERNEL_CPU__
  if (stack_valid(dx_offset) && stack_valid(dy_offset)) {
    co_dx = co - stack_load_float3(stack, dx_offset);
    co_dy = co - stack_load_float3(stack, dy_offset);
  }
#endif

  /* Map so that no textures are flipped, rotation is somewhat arbitrary. */
  if (weight.x > 0.0f) {
    float2 uv = make_float2((signed_N.x < 0.0f) ? 1.0f - co.y : co.y, co.z);
    ds.dx = co_dx.y;
    ds.dy = co_dy.y;
    dt.dx = co_dx.z;
    dt.dy = co_dy.z;
    f += weight.x * svm_image_texture(kg, id, uv.x, uv.y, ds, dt, flags, path_flag);
  }
  if (weight.y > 0.0f) {
    float2 uv = make_float2((signed_N.y < 0.0f) ? 1.0f - co.x : co.x, co.z);
    ds.dx = co_dx.x;
    ds.dy = co_dy.x;
    dt.dx = co_dx.z;
    dt.dy = co_dy.z;
    f += weight.y * svm_image_texture(kg, id, uv.x, uv.y, ds, dt, flags, path_flag);
  }
  if (weight.z > 0.0f) {
    float2 uv = make_float2((signed_N.z < 0.0f) ? 1.0f - co.y : co.y, co.x);
    ds.dx = co_dx.y;
    ds.dy = co_dy.y;
    dt.dx = co_dx.x;
    dt.dy = co_dy.x;
    f += weight.z * svm_image_texture(kg, id, uv.x, uv.y, ds, dt, flags, path_flag);
  }
  if (stack_valid(out_offset))
    stack_store_float3(stack, out_offset, make_float3(f.x, f.y, f.z));
  if (stack_valid(alpha_offset))
    stack_store_float(stack, alpha_offset, f.w);
}

ccl_device void svm_node_tex_environment(
    KernelGlobals *kg, ShaderData *sd, int path_flag, float *stack, uint4 node)
{
  uint id = node.y;
  uint co_offset, out_offset, alpha_offset, flags;
  uint projection, dx_offset, dy_offset, unused;

  svm_unpack_node_uchar4(node.z, &co_offset, &out_offset, &alpha_offset, &flags);
  svm_unpack_node_uchar4(node.w, &projection, &dx_offset, &dy_offset, &unused);

  uint use_alpha = stack_valid(alpha_offset);
  float4 f;

  float3 co = stack_load_float3(stack, co_offset);
  float2 uv;

  co = safe_normalize(co);
#ifdef __OIIO__
  float3 dRdx, dRdy;
  if (stack_valid(dx_offset) && stack_valid(dy_offset)) {
    dRdx = co - normalize(stack_load_float3(stack, dx_offset));
    dRdy = co - normalize(stack_load_float3(stack, dy_offset));
  }
  else {
    dRdx = make_float3(0.0f, 0.0f, 0.0f);
    dRdy = make_float3(0.0f, 0.0f, 0.0f);
  }
  if (kg->oiio && kg->oiio->textures.size() > id && kg->oiio->textures[id].handle &&
      projection == 0) {
    OIIO::TextureOpt options;
    options.swrap = options.twrap = kg->oiio->textures[id].extension;
    options.anisotropic = 8;

    if (path_flag & NEAREST_LOOKUP_PATHS && !(path_flag & PATH_RAY_SINGULAR)) {
      options.interpmode = OIIO::TextureOpt::InterpClosest;
      options.mipmode = OIIO::TextureOpt::MipModeOneLevel;
    }
    else {
      options.interpmode = kg->oiio->textures[id].interpolation;
      options.mipmode = OIIO::TextureOpt::MipModeTrilinear;
    }

    if (path_flag & DIFFUSE_BLUR_PATHS) {
      options.sblur = options.tblur = kg->oiio->diffuse_blur;
    }
    else if (path_flag & PATH_RAY_GLOSSY) {
      options.sblur = options.tblur = kg->oiio->glossy_blur;
    }
    else {
      options.sblur = options.tblur = 0.0f;
    }

    bool success = kg->oiio->tex_sys->environment(kg->oiio->textures[id].handle,
                                                  (OIIO::TextureSystem::Perthread *)kg->oiio_tdata,
                                                  options,
                                                  Imath::V3f(co.x, -co.y, co.z),
                                                  Imath::V3f(dRdx.x, -dRdx.y, dRdx.z),
                                                  Imath::V3f(dRdy.x, -dRdy.y, dRdy.z),
                                                  use_alpha ? 4 : 3,
                                                  (float *)&f);

    if (!success) {
      (void)kg->oiio->tex_sys->geterror();
    }
  }
  else {
#endif

    if (projection == 0)
      uv = direction_to_equirectangular(co);
    else
      uv = direction_to_mirrorball(co);

    f = svm_image_texture(
        kg, id, uv.x, uv.y, differential_zero(), differential_zero(), flags, path_flag);
#ifdef __OIIO__
  }
#endif

  if (stack_valid(out_offset))
    stack_store_float3(stack, out_offset, make_float3(f.x, f.y, f.z));
  if (stack_valid(alpha_offset))
    stack_store_float(stack, alpha_offset, f.w);
}

#endif /* __TEXTURES__ */

CCL_NAMESPACE_END
