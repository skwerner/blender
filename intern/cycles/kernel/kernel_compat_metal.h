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

#ifndef __KERNEL_COMPAT_METAL_H__
#define __KERNEL_COMPAT_METAL_H__

#define __KERNEL_GPU__
#define __KERNEL_METAL__

/* TODO */
class KernelGlobals_Metal {
public:
  device float **buffers;
};
#define KernelGlobals thread KernelGlobals_Metal

#define kernel_assert(x) (void)

#define CCL_NAMESPACE_BEGIN
#define CCL_NAMESPACE_END

#define ccl_device
#define ccl_device_inline ccl_device
#define ccl_device_forceinline ccl_device
#define ccl_device_noinline ccl_device ccl_noinline
#define ccl_device_noinline_cpu ccl_device
#define ccl_may_alias
#define ccl_static_constant static __constant
#define ccl_constant __constant
#define ccl_global device
#define ccl_local __local
#define ccl_local_param __local
#define ccl_private __private
#define ccl_restrict restrict
#define ccl_ref
#define ccl_align(n) __attribute__((aligned(n)))
#define ccl_optional_struct_init

#define ccl_pointer thread

#define NULL 0

#define make_float2(x, y) (float2(x, y))
#define make_float3(x, y, z) (float3(x, y, z))
#define make_float4(x, y, z, w) (float4(x, y, z, w))
#define make_int2(x, y) (int2(x, y))
#define make_int3(x, y, z) (int3(x, y, z))
#define make_int4(x, y, z, w) (int4(x, y, z, w))
#define make_uchar4(x, y, z, w) (uchar4(x, y, z, w))


#define __uint_as_float(x) as_float(x)
//#define __float_as_uint(x) as_uint(x)
#define __int_as_float(x) as_float(x)
#define __float_as_int(x) as_int(x)
#define powf(x, y) pow(((float)(x)), ((float)(y)))
#define fabsf(x) fabs(((float)(x)))
#define copysignf(x, y) copysign(((float)(x)), ((float)(y)))
#define asinf(x) asin(((float)(x)))
#define acosf(x) acos(((float)(x)))
#define atanf(x) atan(((float)(x)))
#define floorf(x) floor(((float)(x)))
#define ceilf(x) ceil(((float)(x)))
#define hypotf(x, y) hypot(((float)(x)), ((float)(y)))
#define atan2f(x, y) atan2(((float)(x)), ((float)(y)))
#define fmaxf(x, y) fmax(((float)(x)), ((float)(y)))
#define fminf(x, y) fmin(((float)(x)), ((float)(y)))
#define fmodf(x, y) fmod((float)(x), (float)(y))
#define sinhf(x) sinh(((float)(x)))

#define lgamma(x) (x)

#  define sinf(x) sin(((float)(x)))
#  define cosf(x) cos(((float)(x)))
#  define tanf(x) tan(((float)(x)))
#  define expf(x) exp(((float)(x)))
#  define sqrtf(x) sqrt(((float)(x)))
#  define logf(x) log(((float)(x)))
#  define rcp(x) recip(x)
#endif /* __KERNEL_COMPAT_METAL_H__ */
