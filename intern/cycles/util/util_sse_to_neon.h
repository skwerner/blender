#ifndef __UTIL_SSE_TO_NEON__
#define __UTIL_SSE_TO_NEON__

#if defined(__ARM_NEON__)

#include <arm_neon.h>
#include <math.h>

/*
 We have internal version of sse to neon, which is using lexical replacement (i.e. with defines were possible) and is easier for compiler
 with the internal version we have better performance - using it by default
 when not using it, we add some more functionality 9mainly fma) to the public sse2neon header file
 */
#define USE_INTERNAL_SSE_TO_NEON
#ifdef USE_INTERNAL_SSE_TO_NEON

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#endif

#include "util_sse_to_neon_internal.h"

#ifdef __clang__
#pragma clang diagnostic pop
#endif


#else //USE_INTERNAL_SSE_TO_NEON

#include <sse2neon.h>

/* these are addition and fixes to sse2neon.h project*/

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#endif


#undef _mm_srai_epi32
#undef _mm_srli_epi32
#undef _mm_slli_epi32

__attribute__((always_inline))
 static int32x4_t _mm_srai_epi32(const int32x4_t& _a, const int _b)
{
    const int32x4_t b = vdupq_n_s32(-_b);
    return vshlq_s32(_a,b);
}

__attribute__((always_inline))
static  int32x4_t _mm_srli_epi32(const int32x4_t& _a, const int _b)
{
    const int32x4_t b = vdupq_n_s32(-_b);
    return vshlq_u32(_a,b);
}

 __attribute__((always_inline))
static  int32x4_t _mm_slli_epi32(const int32x4_t& _a, const int _b)
{
    const int32x4_t b = vdupq_n_s32(_b);
    return vshlq_s32(_a,b);
}




#define _mm_fmadd_ps(a,b,c) vfmaq_f32(c,a,b)
#define _mm_fmsub_ps(a,b,c) vfmaq_f32(vnegq_f32(c),a,b)
#define _mm_fnmadd_ps(a,b,c) vfmsq_f32(c,a,b)
#define _mm_fnmsub_ps(a,b,c) vfmsq_f32(vnegq_f32(c),a,b)
#define _mm_ceil_ps  vrndpq_f32
#define _mm_floor_ps vrndnq_f32
#define _mm_castps_pd(a) (a)
#define _mm_cmpnle_ps _mm_cmpgt_ps
#define _mm_cmpnlt_ps _mm_cmpge_ps
#define _mm_stream_ps _mm_store_ps



template<int code>
static float32x4_t dpps_neon(const float32x4_t& a,const float32x4_t& b)
{
    float v;
    v = 0;
    v += (code & 0x10) ? a[0]*b[0] : 0;
    v += (code & 0x20) ? a[1]*b[1] : 0;
    v += (code & 0x40) ? a[2]*b[2] : 0;
    v += (code & 0x80) ? a[3]*b[3] : 0;
    float32x4_t res;
    res[0] = (code & 0x1) ? v : 0;
    res[1] = (code & 0x2) ? v : 0;
    res[2] = (code & 0x4) ? v : 0;
    res[3] = (code & 0x8) ? v : 0;
    return res;
}

template<>
float32x4_t dpps_neon<0x7f>(const float32x4_t& a,const float32x4_t& b)
{
    float v;
    float32x4_t m = _mm_mul_ps(a,b);
    m[3] = 0;
    v = vaddvq_f32(m);
    return _mm_set1_ps(v);
}

template<>
float32x4_t dpps_neon<0xff>(const float32x4_t& a,const float32x4_t& b)
{
    float v;
    float32x4_t m = _mm_mul_ps(a,b);
    v = vaddvq_f32(m);
    return _mm_set1_ps(v);
}

#define _mm_dp_ps(a,b,c) dpps_neon<c>((a),(b))



template<class type, int i0, int i1, int i2, int i3>
 type shuffle_neon(const type& a)
{
    if (i0 == i1 && i0 == i2 && i0 == i3)
    {
        return vdupq_laneq_s32(a,i0);
    }
        static const uint8_t tbl[16] = {
            (i0*4) + 0,(i0*4) + 1,(i0*4) + 2,(i0*4) + 3,
            (i1*4) + 0,(i1*4) + 1,(i1*4) + 2,(i1*4) + 3,
            (i2*4) + 0,(i2*4) + 1,(i2*4) + 2,(i2*4) + 3,
            (i3*4) + 0,(i3*4) + 1,(i3*4) + 2,(i3*4) + 3
        };
        
        return vqtbl1q_s8(int8x16_t(a),*(int8x16_t *)tbl);
    
}


template<class type, int i0, int i1, int i2, int i3>
 type shuffle_neon(const type& a, const type& b)
{
    if (&a == &b)
    {
        static const uint8_t tbl[16] = {
            (i0*4) + 0,(i0*4) + 1,(i0*4) + 2,(i0*4) + 3,
            (i1*4) + 0,(i1*4) + 1,(i1*4) + 2,(i1*4) + 3,
            (i2*4) + 0,(i2*4) + 1,(i2*4) + 2,(i2*4) + 3,
            (i3*4) + 0,(i3*4) + 1,(i3*4) + 2,(i3*4) + 3
        };
        
        return vqtbl1q_s8(int8x16_t(b),*(int8x16_t *)tbl);
        
    }
    else
    {
        
        static const uint8_t tbl[16] = {
            (i0*4) + 0,(i0*4) + 1,(i0*4) + 2,(i0*4) + 3,
            (i1*4) + 0,(i1*4) + 1,(i1*4) + 2,(i1*4) + 3,
            (i2*4) + 0 + 16,(i2*4) + 1 + 16,(i2*4) + 2 + 16,(i2*4) + 3 + 16,
            (i3*4) + 0 + 16,(i3*4) + 1 + 16,(i3*4) + 2 + 16,(i3*4) + 3 + 16
        };
        
        return vqtbl2q_s8((int8x16x2_t){a,b},*(int8x16_t *)tbl);
    }

   
}

 
#ifdef __clang__
#pragma clang diagnostic pop
#endif


#endif //USE_INTERNAL_SSE_TO_NEON


#endif

#endif

