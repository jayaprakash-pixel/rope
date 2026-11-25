/* Runtime CPU dispatch (P2.2.5): kernels_cpu.c and quant.c are compiled twice (vf8.h), as *_avx2 (AVX2 + FMA + F16C)
 * and *_sse41 (any x64 CPU since 2008). This file, compiled for baseline x64, exports the public names and sends
 * every call to the variant chosen once per process: the best one this CPU and OS support, or ROPE_CPU=sse41|avx2. */
#include "kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _MSC_VER
#include <intrin.h>
#define rope_cpuid(r, leaf, sub) __cpuidex(r, leaf, sub)
#define rope_xgetbv0() _xgetbv(0)
#else
#include <cpuid.h>
#define rope_cpuid(r, leaf, sub) __cpuid_count(leaf, sub, (r)[0], (r)[1], (r)[2], (r)[3])
static unsigned long long rope_xgetbv0(void) { unsigned lo, hi; __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0)); return ((unsigned long long)hi << 32) | lo; }
#endif
#include <immintrin.h>

/* every public kernel: X(return type, name, (parameters), (arguments)) */
#define ROPE_KERNELS(X) \
    X(float, h2f, (uint16_t h), (h)) \
    X(uint16_t, f2h, (float f), (f)) \
    X(float, bf2f, (uint16_t h), (h)) \
    X(int, wtype_supported, (int wtype), (wtype)) \
    X(int, wtype_block, (int wtype), (wtype)) \
    X(size_t, row_bytes, (int wtype, int64_t k), (wtype, k)) \
    X(void, dequant_row, (int wtype, const void *src, float *dst, int64_t k), (wtype, src, dst, k)) \
    X(void, set_threads, (int n), (n)) \
    X(int, get_threads, (void), ()) \
    X(void, matmul, (const float *x, int64_t M, int64_t K, const void *w, int wtype, int64_t N, float *y), (x, M, K, w, wtype, N, y)) \
    X(int, int8_act, (int wtype, int64_t M, const float *x, int64_t k, float *xr), (wtype, M, x, k, xr)) \
    X(int, int8_mode, (int mode), (mode)) \
    X(void, conv2d, (const float *in, int C, int H, int W, const void *w, int wtype, const float *bias, int O, int kh, int kw, int stride, int pad, float *out), \
                    (in, C, H, W, w, wtype, bias, O, kh, kw, stride, pad, out)) \
    X(void, attention, (const float *q, const float *k, const float *v, float *o, int64_t nq, int64_t nk, int d, int64_t qs, int64_t ks, int64_t vs, int64_t os, float scale), \
                       (q, k, v, o, nq, nk, d, qs, ks, vs, os, scale)) \
    X(void, attention_ex, (const float *q, const float *k, const float *v, float *o, int64_t nq, int64_t nk, int d, int64_t qs, int64_t ks, int64_t vs, int64_t os, float scale, int causal, const float *bias), \
                          (q, k, v, o, nq, nk, d, qs, ks, vs, os, scale, causal, bias)) \
    X(void, softmax_rows, (float *x, int64_t m, int64_t n), (x, m, n)) \
    X(void, layernorm, (float *x, int64_t m, int64_t n, const float *g, const float *b, float eps), (x, m, n, g, b, eps)) \
    X(void, rmsnorm, (float *x, int64_t m, int64_t n, const float *g, float eps), (x, m, n, g, eps)) \
    X(void, groupnorm, (float *x, int64_t c, int64_t hw, int groups, const float *g, const float *b, float eps), (x, c, hw, groups, g, b, eps)) \
    X(void, gelu_erf, (float *x, int64_t n), (x, n)) \
    X(void, gelu_tanh, (float *x, int64_t n), (x, n)) \
    X(int, quantize_rows, (int t, const float *x, void *dst, int64_t rows, int64_t k), (t, x, dst, rows, k)) \
    X(void, add_rows, (float *y, int64_t m, int64_t n, const float *b), (y, m, n, b)) \
    X(void, silu, (float *x, int64_t n), (x, n)) \
    X(void, rotary_tables, (const float *pos, int64_t n_tok, int d_head, float theta, float *cos_t, float *sin_t), (pos, n_tok, d_head, theta, cos_t, sin_t)) \
    X(void, rotary, (float *x, int64_t n_tok, int n_head, int d_head, int64_t tok_stride, const float *cos_t, const float *sin_t, int neox), \
                    (x, n_tok, n_head, d_head, tok_stride, cos_t, sin_t, neox))

#define DECLARE(ret, name, params, args) ret rope_##name##_avx2 params; ret rope_##name##_sse41 params;
ROPE_KERNELS(DECLARE)

/* What this CPU and OS can run: 2 = AVX2 + FMA + F16C (with the OS saving the YMM registers), 1 = SSE4.1, 0 = neither. */
static int cpu_level(void) {
    int r[4];
    rope_cpuid(r, 0, 0);
    if (r[0] < 1) return 0;
    int max = r[0];
    rope_cpuid(r, 1, 0);
    int ecx = r[2];
    int sse41 = (ecx >> 19) & 1, fma = (ecx >> 12) & 1, osxsave = (ecx >> 27) & 1, avx = (ecx >> 28) & 1, f16c = (ecx >> 29) & 1;
    int avx2 = 0;
    if (max >= 7) { rope_cpuid(r, 7, 0); avx2 = (r[1] >> 5) & 1; }
    int ymm_saved = osxsave && (rope_xgetbv0() & 6) == 6;
    if (avx && avx2 && fma && f16c && ymm_saved) return 2;
    return sse41 ? 1 : 0;
}

static int g_isa = -1;   /* 2 = avx2, 1 = sse41 */

/* The variant this process uses (chosen on first use). A forced ROPE_CPU the CPU cannot run is refused. */
int rope_cpu_isa(void) {
    if (g_isa >= 0) return g_isa;
    int have = cpu_level();
    const char *f = getenv("ROPE_CPU");
    int want = have;
    if (f && !strcmp(f, "sse41")) want = 1;
    else if (f && !strcmp(f, "avx2")) want = 2;
    if (want > have || want == 0) {
        fprintf(stderr, "rope: this CPU has no %s (SSE4.1 is the minimum)\n", want == 2 ? "AVX2 + FMA + F16C" : "SSE4.1");
        exit(3);
    }
    g_isa = want;
    return g_isa;
}
const char *rope_cpu_isa_name(void) { return rope_cpu_isa() == 2 ? "avx2" : "sse41"; }

#define RET_float return
#define RET_uint16_t return
#define RET_int return
#define RET_size_t return
#define RET_void
#define WRAP(ret, name, params, args) ret rope_##name params { RET_##ret (rope_cpu_isa() == 2 ? rope_##name##_avx2 : rope_##name##_sse41) args; }
ROPE_KERNELS(WRAP)
