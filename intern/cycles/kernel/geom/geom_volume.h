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

/* Volume Primitive
 *
 * Volumes are just regions inside meshes with the mesh surface as boundaries.
 * There isn't as much data to access as for surfaces, there is only a position
 * to do lookups in 3D voxel or procedural textures.
 *
 * 3D voxel textures can be assigned as attributes per mesh, which means the
 * same shader can be used for volume objects with different densities, etc. */

CCL_NAMESPACE_BEGIN

#ifdef __VOLUME__

/* Return position normalized to 0..1 in mesh bounds */

ccl_device_inline float3 volume_normalized_position(KernelGlobals *kg,
                                                    const ShaderData *sd,
                                                    float3 P)
{
  /* todo: optimize this so it's just a single matrix multiplication when
   * possible (not motion blur), or perhaps even just translation + scale */
  const AttributeDescriptor desc = find_attribute(kg, sd, ATTR_STD_GENERATED_TRANSFORM);

  object_inverse_position_transform(kg, sd, &P);

  if (desc.offset != ATTR_STD_NOT_FOUND) {
    Transform tfm = primitive_attribute_matrix(kg, sd, desc);
    P = transform_point(&tfm, P);
  }

  return P;
}

/* Returns normalized P.
 * If motion blur is enabled, returns normalized and advected P. */

ccl_device_inline float3 volume_get_position(KernelGlobals *kg, const ShaderData *sd)
{
  float3 P = volume_normalized_position(kg, sd, sd->P);

#  ifdef __OBJECT_MOTION__
  /* Eulerian motion blur. */
  if (kernel_data.cam.shuttertime != -1.0f) {
    AttributeDescriptor v_desc = find_attribute(kg, sd, ATTR_STD_VOLUME_VELOCITY);

    if (v_desc.offset != ATTR_STD_NOT_FOUND) {
      InterpolationType interp = (sd->flag & SD_VOLUME_CUBIC) ? INTERPOLATION_CUBIC :
                                                                INTERPOLATION_NONE;

      /* Find velocity. */
      float3 velocity = float4_to_float3(
          kernel_tex_image_interp_3d(kg, v_desc.offset, P.x, P.y, P.z, interp));

      /* Find advected velocity. */
      P = volume_normalized_position(kg, sd, sd->P + velocity * sd->time);
      velocity = float4_to_float3(
          kernel_tex_image_interp_3d(kg, v_desc.offset, P.x, P.y, P.z, interp));

      /* Find advected P. */
      P = volume_normalized_position(kg, sd, sd->P + velocity * sd->time);
    }
  }
#  endif

  return P;
}

ccl_device float volume_attribute_float(KernelGlobals *kg,
                                        const ShaderData *sd,
                                        const AttributeDescriptor desc)
{
  float3 P = volume_normalized_position(kg, sd, sd->P);
  InterpolationType interp = (sd->flag & SD_VOLUME_CUBIC) ? INTERPOLATION_CUBIC :
                                                            INTERPOLATION_NONE;
  float4 r = kernel_tex_image_interp_3d(kg, desc.offset, P.x, P.y, P.z, interp);
  return average(float4_to_float3(r));
}

ccl_device float3 volume_attribute_float3(KernelGlobals *kg,
                                          const ShaderData *sd,
                                          const AttributeDescriptor desc)
{
  InterpolationType interp = (sd->flag & SD_VOLUME_CUBIC) ? INTERPOLATION_CUBIC :
                                                            INTERPOLATION_NONE;
  float4 r = kernel_tex_image_interp_3d(kg, desc.offset, sd->P_v.x, sd->P_v.y, sd->P_v.z, interp);

  if (r.w > 1e-6f && r.w != 1.0f) {
    /* For RGBA colors, unpremultiply after interpolation. */
    return float4_to_float3(r) / r.w;
  }
  else {
    return float4_to_float3(r);
  }
}

#endif

CCL_NAMESPACE_END
