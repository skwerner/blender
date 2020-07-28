#ifndef __UTIL_SSE_TO_NEON_INTERNAL_
#define __UTIL_SSE_TO_NEON_INTERNAL_

#if defined(__ARM_NEON__)

#include <arm_neon.h>
#include <math.h>


typedef float32x4_t __m128;
typedef float64x2_t __m128d;
typedef int32x4_t __m128i;
typedef uint64_t __m64;


 static constexpr int32_t _all_zero = 0;
 static constexpr int32_t _all_one = -1;
 static constexpr int32x4_t all_one_mask = {-1,-1,-1,-1};


#define _mm_and_ps vandq_s32
#define _mm_xor_ps veorq_s32
#define _mm_or_ps vorrq_s32
#define _mm_sub_epi32 vsubq_s32
#define _mm_add_epi32 vaddq_s32
#define _mm_mullo_epi32 vmulq_s32
#define _mm_min_epi32 vminq_s32
#define _mm_max_epi32 vmaxq_s32

#define _mm_cmplt_epi32 vcltq_s32
#define _mm_cmple_epi32 vcleq_s32
#define _mm_cmpgt_epi32 vcgtq_s32
#define _mm_cmpge_epi32 vcgeq_s32
#define _mm_cmpeq_epi32 vceqq_s32



#define _mm_add_ps vaddq_f32
#define _mm_div_ps vdivq_f32
#define _mm_sub_ps vsubq_f32
#define _mm_mul_ps vmulq_f32
#define _mm_min_ps vminq_f32
#define _mm_max_ps vmaxq_f32
#define _mm_cmpeq_ps vceqq_f32
#define _mm_cmplt_ps vcltq_f32
#define _mm_cmpnle_ps vcgtq_f32
#define _mm_cmple_ps vcleq_f32
#define _mm_cmpnlt_ps vcgeq_f32
#define _mm_cmpgt_ps vcgtq_f32

#define _mm_fmadd_ps(a,b,c) vfmaq_f32(c,a,b)
#define _mm_fmsub_ps(a,b,c) vfmaq_f32(vnegq_f32(c),a,b)
#define _mm_fnmadd_ps(a,b,c) vfmsq_f32(c,a,b)
#define _mm_fnmsub_ps(a,b,c) vfmsq_f32(vnegq_f32(c),a,b)




#define _mm_rcp_ps vrecpeq_f32
#define _mm_sqrt_ps vsqrtq_f32
#define _mm_rsqrt_ps vrsqrteq_f32
#define _mm_floor_ps  vrndnq_f32
#define _mm_ceil_ps  vrndpq_f32
#define _mm_abs_epi32 vabsq_s32





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




#define _mm_andnot_ps(_a,_b) vbicq_s32((_b),(_a))
/*
__attribute__((always_inline))
static  int32x4_t _mm_andnot_ps(const int32x4_t& a, const int32x4_t& b)
{
    int32x4_t res = vandq_s32(veorq_s32(a,all_one_mask),b);
    return res;
}
*/

__attribute__((always_inline))
static  float32x4_t _mm_cmpneq_ps(const float32x4_t& a, const float32x4_t& b)
{
  return vornq_s32(vdupq_n_s32(0),vceqq_f32(a,b));
//    return veorq_s32(vceqq_f32(a,b),all_one_mask);
}










#define _mm_setzero_ps() vdupq_n_f32(0.0f)
#define _mm_setzero_si128 _mm_setzero_ps




#define _mm_hadd_ps vpaddq_f32


__attribute__((always_inline))
static  __m128 _mm_sqrt_ss(const __m128& a)
{
    __m128 res;
    res[0] = sqrtf(a[0]);
    return res;
}


#define _mm_castsi128_ps(a) ((__m128)(a))
#define _mm_castps_si128(a) ((__m128i)(a))
#define _mm_castps_pd(a) ((__m128d)(a))
#define _mm_cvtps_epi32 vcvtnq_s32_f32
#define _mm_cvttps_epi32 vcvtq_s32_f32
#define  _mm_cvtepi32_ps vcvtq_f32_s32



__attribute__((always_inline))
static  float _mm_cvtss_f32(const __m128& x)
{
    return x[0];
}


__attribute__((always_inline))
static  __m128i _mm_load_si128(const __m128i *x)
{
    return *x;
}

__attribute__((always_inline))
static  __m128 _mm_load_ps(const float *x)
{
    return *(__m128 *)x;
}


__attribute__((always_inline))
static  void _mm_store_si128(__m128i *p, const __m128i& a)
{
    *p = a;
}

__attribute__((always_inline))
static  void _mm_storel_pi(__m64 *p, uint64x2_t x)
{
    *p = x[0];
}

__attribute__((always_inline))
static  void _mm_store_ps(float *p, __m128 a)
{
    *(__m128 *)p = a;
}

__attribute__((always_inline))
static  void _mm_stream_ps(float *p, __m128 a)
{
    *(__m128 *)p = a;
}







#define  _mm_set1_epi32 vdupq_n_u32

__attribute__((always_inline))
static  __m128 _mm_set_ss(const float x)
{
    float32x4_t res = {x,0,0,0};
    return res;
}


__attribute__((always_inline))
static  __m128i _mm_setr_epi32(const int a,const int b,const int c,const int d)
{
    int32x4_t t = {a,b,c,d};
    return t;
}

__attribute__((always_inline))
static  __m128 _mm_set_ps(const float d,const float c,const float b,const float a)
{
    float32x4_t t = {a,b,c,d};
    return t;
}

__attribute__((always_inline))
static  __m128i _mm_set_epi32(const int d,const int c,const int b,const int a)
{
    int32x4_t t = {a,b,c,d};
    return t;
}


__attribute__((always_inline))
static  __m128 _mm_setr_ps(const float a,const float b,const float c,const float d)
{
    float32x4_t t = {a,b,c,d};
    return t;
}

__attribute__((always_inline))
static  __m128 _mm_set1_ps(const float x)
{
    return vdupq_n_f32(x);
   
}

__attribute__((always_inline))
static  int32x4_t _mm_movehl_ps(const int32x4_t& a, const int32x4_t& b)
{
    return vzip2q_u64(a,b);
    int32x4_t res = a;
    res[0] = b[2];
    res[1] = b[3];
    return res;
}

__attribute__((always_inline))
static  int32x4_t _mm_movelh_ps(const int32x4_t& a, const int32x4_t& b)
{
    return vzip1q_u64(a,b);
    int32x4_t res = a;
    res[2] = b[0];
    res[3] = b[1];
    return res;
}

__attribute__((always_inline))
static  int32x4_t _mm_packs_epi32(const int32x4_t& a, const int32x4_t& b)
{
    int16x8_t res;
    for (int i=0;i<4;i++)
    {
        res[i] = a[i];
        res[i+4] = b[i];
    }
    return res;
}

__attribute__((always_inline))
static  int32x4_t _mm_unpacklo_ps(const int32x4_t& a, const int32x4_t& b)
{
    return vzip1q_s32(a,b);
}

__attribute__((always_inline))
static  int32x4_t _mm_unpackhi_ps(const int32x4_t& a, const int32x4_t& b)
{
    return vzip2q_s32(a,b);
}


__attribute__((always_inline))
static  int _mm_movemask_ps(const int32x4_t& a)
{
    static const int32x4_t shift = {0,1,2,3};
    uint32x4_t tmp = vshrq_n_u32(a,31);
    tmp = vshlq_u32(tmp,shift);
    return vaddvq_u32(tmp);
}

__attribute__((always_inline))
static  void _mm_store_ss(float *x, const __m128& i)
{
    *x = i[0];
}


__attribute__((always_inline))
static  int32x4_t _mm_blendv_ps(const int32x4_t& f, const int32x4_t& t, const int32x4_t m)
{

    return vbslq_s32(m,t,f);
}


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

 __attribute__((always_inline))
static __m128i _mm_shuffle_epi8(const __m128i& a, const __m128i& _b)
{
    int8x16_t res;
    int8x16_t b = _b;
    for (int i=0;i<16;i++)
    {
        res[i] = a[b[i]];
    }
    return res;
}

__attribute__((always_inline))
static __m128i _mm_set_epi8(const uint8_t _15,const uint8_t _14,const uint8_t _13,const uint8_t _12,
                      const uint8_t _11,const uint8_t _10,const uint8_t _9,const uint8_t _8,
                      const uint8_t _7,const uint8_t _6,const uint8_t _5,const uint8_t _4,
                      const uint8_t _3,const uint8_t _2,const uint8_t _1,const uint8_t _0
                      )
{
    uint8x16_t res = {_0,_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,_11,_12,_13,_14,_15};
    return res;
}


#define _mm_unpacklo_epi32 _mm_unpacklo_ps
#define _mm_unpackhi_epi32 _mm_unpackhi_ps
#define _mm_storeu_si128 _mm_store_si128
#define _mm_stream_load_si128 _mm_load_si128
#define _mm_storeu_ps _mm_store_ps
#define _mm_loadu_ps _mm_load_ps
#define _mm_loadu_si128 _mm_load_si128
#define _mm_set_ps1 _mm_set1_ps
#define _mm_cmpge_ps _mm_cmpnlt_ps

#define _mm_or_si128 _mm_or_ps
#define _mm_xor_si128 _mm_xor_ps
#define _mm_and_si128 _mm_and_ps
#define _mm_andnot_si128 _mm_andnot_ps


__attribute__((always_inline))
static
void _MM_TRANSPOSE4_PS(__m128& a0,__m128& a1, __m128& a2, __m128& a3)
{
    __m128 b0 = vtrn1q_f64(a0,a2);
    __m128 b1 = vtrn1q_f64(a1,a3);
    __m128 b2 = vtrn2q_f64(a0,a2);
    __m128 b3 = vtrn2q_f64(a1,a3);
    
    a0 = vtrn1q_f32(b0,b1);
    a1 = vtrn2q_f32(b0,b1);
    a2 = vtrn1q_f32(b2,b3);
    a3 = vtrn2q_f32(b2,b3);

}

#endif //aarch64

#endif

