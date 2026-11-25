/* 8-float SIMD vectors for the CPU kernels, one source for two instruction sets (P2.2.5).
 * The kernel files are compiled twice: with ROPE_ISA_AVX2 (AVX2 + FMA + F16C: vf8 is __m256, every vf8_* is the
 * intrinsic itself) and with ROPE_ISA_SSE41 (any x64 CPU since 2008: vf8 is two __m128; FMA is a multiply and an
 * add, F16 conversion is scalar). Public kernel names get the ISA suffix (rope_matmul_avx2, rope_matmul_sse41);
 * dispatch.c exports rope_matmul and picks one variant per process (cpuid, or ROPE_CPU=sse41|avx2). */
#pragma once
#include <stdint.h>

#if defined(ROPE_ISA_AVX2)
#include <immintrin.h>
#define ROPE_ISA_SUFFIX _avx2
typedef __m256 vf8;
#define vf8_setzero()       _mm256_setzero_ps()
#define vf8_set1(f)         _mm256_set1_ps(f)
#define vf8_loadu(p)        _mm256_loadu_ps(p)
#define vf8_load(p)         _mm256_load_ps(p)
#define vf8_storeu(p, v)    _mm256_storeu_ps(p, v)
#define vf8_broadcast(p)    _mm256_broadcast_ss(p)
#define vf8_add(a, b)       _mm256_add_ps(a, b)
#define vf8_sub(a, b)       _mm256_sub_ps(a, b)
#define vf8_mul(a, b)       _mm256_mul_ps(a, b)
#define vf8_min(a, b)       _mm256_min_ps(a, b)
#define vf8_max(a, b)       _mm256_max_ps(a, b)
#define vf8_fmadd(a, b, c)  _mm256_fmadd_ps(a, b, c)
#define vf8_round(a)        _mm256_round_ps(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)
#define vf8_lo(v)           _mm256_castps256_ps128(v)
#define vf8_hi(v)           _mm256_extractf128_ps(v, 1)
#define vf8_from_f16(p)     _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(p)))   /* 8 halves at p */

#elif defined(ROPE_ISA_SSE41)
#include <smmintrin.h>
#define ROPE_ISA_SUFFIX _sse41
typedef struct { __m128 lo, hi; } vf8;
static inline vf8 vf8_mk(__m128 lo, __m128 hi) { vf8 r; r.lo = lo; r.hi = hi; return r; }
#define vf8_setzero()       vf8_mk(_mm_setzero_ps(), _mm_setzero_ps())
#define vf8_set1(f)         vf8_mk(_mm_set1_ps(f), _mm_set1_ps(f))
#define vf8_loadu(p)        vf8_mk(_mm_loadu_ps(p), _mm_loadu_ps((p) + 4))
#define vf8_load(p)         vf8_mk(_mm_load_ps(p), _mm_load_ps((p) + 4))
static inline void vf8_storeu(float *p, vf8 v) { _mm_storeu_ps(p, v.lo); _mm_storeu_ps(p + 4, v.hi); }
#define vf8_broadcast(p)    vf8_set1(*(p))
static inline vf8 vf8_add(vf8 a, vf8 b) { return vf8_mk(_mm_add_ps(a.lo, b.lo), _mm_add_ps(a.hi, b.hi)); }
static inline vf8 vf8_sub(vf8 a, vf8 b) { return vf8_mk(_mm_sub_ps(a.lo, b.lo), _mm_sub_ps(a.hi, b.hi)); }
static inline vf8 vf8_mul(vf8 a, vf8 b) { return vf8_mk(_mm_mul_ps(a.lo, b.lo), _mm_mul_ps(a.hi, b.hi)); }
static inline vf8 vf8_min(vf8 a, vf8 b) { return vf8_mk(_mm_min_ps(a.lo, b.lo), _mm_min_ps(a.hi, b.hi)); }
static inline vf8 vf8_max(vf8 a, vf8 b) { return vf8_mk(_mm_max_ps(a.lo, b.lo), _mm_max_ps(a.hi, b.hi)); }
/* no FMA: a*b + c rounds twice (results differ from the AVX2 path in the last bits) */
static inline vf8 vf8_fmadd(vf8 a, vf8 b, vf8 c) { return vf8_add(vf8_mul(a, b), c); }
static inline vf8 vf8_round(vf8 a) {
    return vf8_mk(_mm_round_ps(a.lo, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), _mm_round_ps(a.hi, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}
#define vf8_lo(v)           ((v).lo)
#define vf8_hi(v)           ((v).hi)
float rope_h2f_sse41(uint16_t h);
static inline vf8 vf8_from_f16(const void *p) {
    const uint16_t *h = (const uint16_t *)p;
    float f[8];
    for (int i = 0; i < 8; i++) f[i] = rope_h2f_sse41(h[i]);
    return vf8_loadu(f);
}

#else
#error "compile the kernels with ROPE_ISA_AVX2 or ROPE_ISA_SSE41"
#endif

/* Public kernel names of this compilation get the ISA suffix (kernels.h is included after this header). */
#define ROPE_CAT2(a, b) a##b
#define ROPE_CAT(a, b) ROPE_CAT2(a, b)
#define rope_h2f             ROPE_CAT(rope_h2f, ROPE_ISA_SUFFIX)
#define rope_f2h             ROPE_CAT(rope_f2h, ROPE_ISA_SUFFIX)
#define rope_bf2f            ROPE_CAT(rope_bf2f, ROPE_ISA_SUFFIX)
#define rope_wtype_supported ROPE_CAT(rope_wtype_supported, ROPE_ISA_SUFFIX)
#define rope_wtype_block     ROPE_CAT(rope_wtype_block, ROPE_ISA_SUFFIX)
#define rope_row_bytes       ROPE_CAT(rope_row_bytes, ROPE_ISA_SUFFIX)
#define rope_dequant_row     ROPE_CAT(rope_dequant_row, ROPE_ISA_SUFFIX)
#define rope_set_threads     ROPE_CAT(rope_set_threads, ROPE_ISA_SUFFIX)
#define rope_get_threads     ROPE_CAT(rope_get_threads, ROPE_ISA_SUFFIX)
#define rope_matmul          ROPE_CAT(rope_matmul, ROPE_ISA_SUFFIX)
#define rope_int8_act        ROPE_CAT(rope_int8_act, ROPE_ISA_SUFFIX)
#define rope_int8_mode       ROPE_CAT(rope_int8_mode, ROPE_ISA_SUFFIX)
#define rope_conv2d          ROPE_CAT(rope_conv2d, ROPE_ISA_SUFFIX)
#define rope_attention       ROPE_CAT(rope_attention, ROPE_ISA_SUFFIX)
#define rope_attention_ex    ROPE_CAT(rope_attention_ex, ROPE_ISA_SUFFIX)
#define rope_softmax_rows    ROPE_CAT(rope_softmax_rows, ROPE_ISA_SUFFIX)
#define rope_layernorm       ROPE_CAT(rope_layernorm, ROPE_ISA_SUFFIX)
#define rope_rmsnorm         ROPE_CAT(rope_rmsnorm, ROPE_ISA_SUFFIX)
#define rope_groupnorm       ROPE_CAT(rope_groupnorm, ROPE_ISA_SUFFIX)
#define rope_gelu_erf        ROPE_CAT(rope_gelu_erf, ROPE_ISA_SUFFIX)
#define rope_gelu_tanh       ROPE_CAT(rope_gelu_tanh, ROPE_ISA_SUFFIX)
#define rope_quantize_rows   ROPE_CAT(rope_quantize_rows, ROPE_ISA_SUFFIX)
#define rope_add_rows        ROPE_CAT(rope_add_rows, ROPE_ISA_SUFFIX)
#define rope_silu            ROPE_CAT(rope_silu, ROPE_ISA_SUFFIX)
#define rope_rotary_tables   ROPE_CAT(rope_rotary_tables, ROPE_ISA_SUFFIX)
#define rope_rotary          ROPE_CAT(rope_rotary, ROPE_ISA_SUFFIX)
