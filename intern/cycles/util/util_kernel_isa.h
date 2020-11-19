/*
* Copyright 2020 Blender Foundation
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

#ifndef __UTIL_KERNEL_ISA_H__
#define __UTIL_KERNEL_ISA_H__

/* On x86-64, we can assume SSE2, so avoid the extra kernel and compile this
* one with SSE2 intrinsics.
*/

#if !defined(__KERNEL_SSE2__) && (defined(__x86_64__) || defined(_M_X64))
#  define __KERNEL_SSE2__
#endif

/* When building kernel for native machine detect kernel features from the flags
 * set by compiler.
 */
#ifdef WITH_KERNEL_NATIVE
#  ifdef __SSE2__
#    ifndef __KERNEL_SSE2__
#      define __KERNEL_SSE2__
#    endif
#  endif
#  ifdef __SSE3__
#    define __KERNEL_SSE3__
#  endif
#  ifdef __SSSE3__
#    define __KERNEL_SSSE3__
#  endif
#  ifdef __SSE4_1__
#    define __KERNEL_SSE41__
#  endif
#  ifdef __AVX__
#    define __KERNEL_SSE__
#    define __KERNEL_AVX__
#  endif
#  ifdef __AVX2__
#    define __KERNEL_SSE__
#    define __KERNEL_AVX2__
#  endif
#endif


#if  defined(WITH_KERNEL_NEON) && (defined(__aarch64__) \
        || defined (__ARM_NEON__) || defined(_M_ARM64) )
#define __KERNEL_NEON__
#define __KERNEL_FMA_X4__

#endif



#if defined(__KERNEL_SSE2__) || defined(__KERNEL_NEON__)
#define __KERNEL_SSE2_OR_NEON__
#endif

#if defined(__KERNEL_SSE__) || defined(__KERNEL_NEON__)
#define __KERNEL_SSE_OR_NEON__
#endif



#if defined(__KERNEL_SSE__) || defined(__KERNEL_NEON__)
#define __KERNEL_SSE3_OR_NEON__
#endif


#if defined(__KERNEL_SSSE2__) || defined(__KERNEL_NEON__)
#define __KERNEL_SSSE3_OR_NEON__
#endif

#if defined(__KERNEL_SSE41__) || defined(__KERNEL_NEON__)
#define __KERNEL_SSE41_OR_NEON__
#endif


#if defined(__KERNEL_AVX2__) || defined(__KERNEL_NEON__)
#define __KERNEL_FMA_X4__
#endif

/* quiet unused define warnings */
#if defined(__KERNEL_SSE2__)
/* do nothing */
#endif


#endif //__UTIL_KERNEL_ISA_H__

