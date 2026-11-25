/* rope-engine CPU kernels: AVX2 + FMA, OpenMP threads. Written from scratch. */
#include "vf8.h"      /* before kernels.h: SIMD layer + ISA-suffixed public names */
#include "kernels.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <stdio.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

/* ---------------- threads ---------------- */
static int g_threads;

static int physical_cores(void) {
#ifdef _WIN32
    DWORD len = 0; GetLogicalProcessorInformation(NULL, &len);
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION *buf = malloc(len); int n = 0;
    if (buf && GetLogicalProcessorInformation(buf, &len))
        for (DWORD i = 0; i < len / sizeof *buf; i++) n += buf[i].Relationship == RelationProcessorCore;
    free(buf);
    return n > 0 ? n : omp_get_num_procs();
#else
    /* Linux: distinct (physical id, core id) pairs of /proc/cpuinfo; the OpenMP processor count if that fails */
    FILE *f = fopen("/proc/cpuinfo", "r"); char l[256]; int P[512], C[512], np = 0, pp = 0, cc = -1, procs = 0, n = 0;
    while (f && fgets(l, sizeof l, f)) {
        if (!strncmp(l, "processor", 9)) { if (procs && cc >= 0 && np < 512) { P[np] = pp; C[np++] = cc; } procs++; pp = 0; cc = -1; }
        else if (!strncmp(l, "physical id", 11)) { const char *v = strchr(l, ':'); if (v) pp = atoi(v + 1); }
        else if (!strncmp(l, "core id", 7)) { const char *v = strchr(l, ':'); if (v) cc = atoi(v + 1); }
    }
    if (procs && cc >= 0 && np < 512) { P[np] = pp; C[np++] = cc; }
    if (f) fclose(f);
    for (int i = 0; i < np; i++) { int d = 0; for (int j = 0; j < i; j++) if (P[j] == P[i] && C[j] == C[i]) { d = 1; break; } n += !d; }
    return n > 0 ? n : omp_get_num_procs();
#endif
}

void rope_set_threads(int n) { g_threads = n > 0 ? n : physical_cores(); }
int  rope_get_threads(void) { if (!g_threads) rope_set_threads(0); return g_threads; }

#ifdef _WIN32
static void *amalloc(size_t n) { return _aligned_malloc(n ? n : 32, 32); }
static void  afree(void *p) { _aligned_free(p); }
#else
static void *amalloc(size_t n) { void *p = NULL; return posix_memalign(&p, 32, n ? n : 32) ? NULL : p; }
static void  afree(void *p) { free(p); }
#endif

static float hsum(vf8 v) {
    __m128 a = _mm_add_ps(vf8_lo(v), vf8_hi(v));
    a = _mm_add_ps(a, _mm_movehl_ps(a, a));
    a = _mm_add_ss(a, _mm_movehdup_ps(a));
    return _mm_cvtss_f32(a);
}

static float dot(const float *a, const float *b, int64_t n) {
    vf8 s0 = vf8_setzero(), s1 = vf8_setzero(), s2 = vf8_setzero(), s3 = vf8_setzero();
    int64_t i = 0;
    for (; i + 32 <= n; i += 32) {
        s0 = vf8_fmadd(vf8_loadu(a + i), vf8_loadu(b + i), s0);
        s1 = vf8_fmadd(vf8_loadu(a + i + 8), vf8_loadu(b + i + 8), s1);
        s2 = vf8_fmadd(vf8_loadu(a + i + 16), vf8_loadu(b + i + 16), s2);
        s3 = vf8_fmadd(vf8_loadu(a + i + 24), vf8_loadu(b + i + 24), s3);
    }
    for (; i + 8 <= n; i += 8) s0 = vf8_fmadd(vf8_loadu(a + i), vf8_loadu(b + i), s0);
    float s = hsum(vf8_add(vf8_add(s0, s1), vf8_add(s2, s3)));
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

/* ---------------- matmul ----------------
 * Large M (diffusion: thousands of tokens): weights are processed in panels of NR=16 rows. Each panel is
 * dequantized once and packed k-major ([K][16]), then reused for all M rows by a 6x16 register micro-kernel
 * over K-blocks of KC. Small M (LLM decode): each weight row is dequantized and dotted with the few x rows. */
#define MR 6
#define NR 16
#define KC 256
#define MC 96          /* rows of x per L2 block (96 x KC floats = 96 KB) */
#define GROUP 8        /* panels sharing one x block (8 x 16 = 128 weight rows) */
#define X_CACHE_BYTES (16ll << 20) /* x larger than the shared L3 (16 MB on the dev CPU) does not stay cached: group panels */

static void store_tile(vf8 c[MR][2], int mr, int nr, float *y, int64_t ldy, int acc) {
    for (int r = 0; r < mr; r++) {
        float *yr = y + r * ldy;
        if (nr == NR) {
            if (acc) { c[r][0] = vf8_add(c[r][0], vf8_loadu(yr)); c[r][1] = vf8_add(c[r][1], vf8_loadu(yr + 8)); }
            vf8_storeu(yr, c[r][0]); vf8_storeu(yr + 8, c[r][1]);
        } else {
            float t[NR]; vf8_storeu(t, c[r][0]); vf8_storeu(t + 8, c[r][1]);
            for (int j = 0; j < nr; j++) yr[j] = acc ? yr[j] + t[j] : t[j];
        }
    }
}

/* 6x16 register tile: 12 accumulators, 2 weight loads + 6 broadcasts per 12 FMAs. */
static void ukernel6(const float *x, int64_t ldx, const float *p, int64_t kc, float *y, int64_t ldy, int nr, int acc) {
    vf8 c00 = vf8_setzero(), c01 = c00, c10 = c00, c11 = c00, c20 = c00, c21 = c00,
           c30 = c00, c31 = c00, c40 = c00, c41 = c00, c50 = c00, c51 = c00;
    const float *x0 = x, *x1 = x + ldx, *x2 = x + 2 * ldx, *x3 = x + 3 * ldx, *x4 = x + 4 * ldx, *x5 = x + 5 * ldx;
    for (int64_t k = 0; k < kc; k++, p += NR) {
        vf8 w0 = vf8_load(p), w1 = vf8_load(p + 8), b;
        b = vf8_broadcast(x0 + k); c00 = vf8_fmadd(b, w0, c00); c01 = vf8_fmadd(b, w1, c01);
        b = vf8_broadcast(x1 + k); c10 = vf8_fmadd(b, w0, c10); c11 = vf8_fmadd(b, w1, c11);
        b = vf8_broadcast(x2 + k); c20 = vf8_fmadd(b, w0, c20); c21 = vf8_fmadd(b, w1, c21);
        b = vf8_broadcast(x3 + k); c30 = vf8_fmadd(b, w0, c30); c31 = vf8_fmadd(b, w1, c31);
        b = vf8_broadcast(x4 + k); c40 = vf8_fmadd(b, w0, c40); c41 = vf8_fmadd(b, w1, c41);
        b = vf8_broadcast(x5 + k); c50 = vf8_fmadd(b, w0, c50); c51 = vf8_fmadd(b, w1, c51);
    }
    vf8 c[MR][2] = {{c00, c01}, {c10, c11}, {c20, c21}, {c30, c31}, {c40, c41}, {c50, c51}};
    store_tile(c, MR, nr, y, ldy, acc);
}

/* Fewer than 6 rows left (edge of M): same math, generic loop. */
static void ukernel_small(const float *x, int64_t ldx, const float *p, int64_t kc, float *y, int64_t ldy, int mr, int nr, int acc) {
    vf8 c[MR][2];
    for (int r = 0; r < MR; r++) c[r][0] = c[r][1] = vf8_setzero();
    for (int64_t k = 0; k < kc; k++, p += NR) {
        vf8 w0 = vf8_load(p), w1 = vf8_load(p + 8);
        for (int r = 0; r < mr; r++) {
            vf8 b = vf8_broadcast(x + r * ldx + k);
            c[r][0] = vf8_fmadd(b, w0, c[r][0]); c[r][1] = vf8_fmadd(b, w1, c[r][1]);
        }
    }
    store_tile(c, mr, nr, y, ldy, acc);
}

#if defined(ROPE_ISA_AVX2)
#include "int8_dot.h"
#else
int rope_int8_act(int wt, int64_t M, const float *x, int64_t k, float *xr) { (void)wt; (void)M; (void)x; (void)k; (void)xr; return 0; }
int rope_int8_mode(int mode) { (void)mode; return 0; }
#endif

static void matmul_impl(const float *x, int64_t M, int64_t K, const void *w, int wt, int64_t N, float *y, int parallel) {
    const uint8_t *wb = w; size_t rowb = rope_row_bytes(wt, K);
    int nth = parallel ? rope_get_threads() : 1;
#if defined(ROPE_ISA_AVX2)
    if (int8_used(wt, M)) { matmul_int8(x, M, K, w, wt, N, y, nth); return; }   /* P2.2.3: integer dot products */
#endif
    if (M < 4) {                                   /* decode path */
        int n_rows = (int)N;
        #pragma omp parallel num_threads(nth)
        {
            float *rb = amalloc((size_t)K * 4);
            int n;
            #pragma omp for schedule(static)
            for (n = 0; n < n_rows; n++) {
                rope_dequant_row(wt, wb + (size_t)n * rowb, rb, K);
                for (int64_t m = 0; m < M; m++) y[m * N + n] = dot(x + m * K, rb, K);
            }
            afree(rb);
        }
        return;
    }
    /* Large M: one task per group of G panels. Inside a task each MC x KC block of x stays in L2 and is reused
     * by all G packed panels, so x is read N/(16G) times instead of N/16 times. Grouping only pays when x is
     * too big for the shared cache (measured: +2x at 4096x3072; -6% when x fits; splitting rows into chunks
     * halved small-N conv shapes), so small x keeps G = 1, which is the plain one-panel-per-task loop. */
    int n_panels = (int)((N + NR - 1) / NR);
    int G = M * K * 4 > X_CACHE_BYTES ? n_panels / (3 * nth) : 1;
    G = G < 1 ? 1 : G > GROUP ? GROUP : G;
    int n_tasks = (n_panels + G - 1) / G;
    #pragma omp parallel num_threads(nth)
    {
        float *rb = amalloc((size_t)K * 4), *pk = amalloc((size_t)K * NR * G * 4);
        int ti;
        #pragma omp for schedule(dynamic, 1)
        for (ti = 0; ti < n_tasks; ti++) {
            int gi = ti;
            int64_t m_lo = 0, m_hi = M;
            int p0 = gi * G, np = n_panels - p0 < G ? n_panels - p0 : G;
            for (int g = 0; g < np; g++) {               /* dequantize + pack each panel k-major, zero-pad rows */
                int64_t n0 = (int64_t)(p0 + g) * NR; int nr = (int)(N - n0 < NR ? N - n0 : NR);
                float *pg = pk + (size_t)g * K * NR;
                for (int j = 0; j < NR; j++) {
                    if (j < nr) rope_dequant_row(wt, wb + (size_t)(n0 + j) * rowb, rb, K);
                    for (int64_t k = 0; k < K; k++) pg[k * NR + j] = j < nr ? rb[k] : 0.0f;
                }
            }
            for (int64_t k0 = 0; k0 < K; k0 += KC) {
                int64_t kc = K - k0 < KC ? K - k0 : KC;
                for (int64_t mb = m_lo; mb < m_hi; mb += MC) {
                    int64_t me = mb + MC < m_hi ? mb + MC : m_hi;
                    for (int g = 0; g < np; g++) {
                        int64_t n0 = (int64_t)(p0 + g) * NR; int nr = (int)(N - n0 < NR ? N - n0 : NR);
                        const float *pg = pk + (size_t)g * K * NR + k0 * NR;
                        for (int64_t m0 = mb; m0 < me; m0 += MR) {
                            int mr = (int)(me - m0 < MR ? me - m0 : MR);
                            if (mr == MR) ukernel6(x + m0 * K + k0, K, pg, kc, y + m0 * N + n0, N, nr, k0 > 0);
                            else ukernel_small(x + m0 * K + k0, K, pg, kc, y + m0 * N + n0, N, mr, nr, k0 > 0);
                        }
                    }
                }
            }
        }
        afree(rb); afree(pk);
    }
}

void rope_matmul(const float *x, int64_t M, int64_t K, const void *w, int wtype, int64_t N, float *y) {
    matmul_impl(x, M, K, w, wtype, N, y, 1);
}

/* ---------------- conv2d: implicit GEMM ----------------
 * out[O, P] = W[O, CK] . col[CK, P] without building col: W rows are the streamed operand (6 rows per micro-kernel
 * call) and each task gathers its 64 output positions' input values straight from the image into a k-major packed
 * buffer, one KC block at a time (64 KB, stays in L2). A 6 x KC weight strip (6 KB, L1) is reused by four 16-wide
 * sub-panels. Results land in the [O, P] output layout directly: no im2col buffer, no transpose pass. */
#define CONV_PW 64
void rope_conv2d(const float *in, int C, int H, int W, const void *w, int wtype, const float *bias,
                 int O, int kh, int kw, int stride, int pad, float *out) {
    int Ho = (H + 2 * pad - kh) / stride + 1, Wo = (W + 2 * pad - kw) / stride + 1;
    int64_t P = (int64_t)Ho * Wo, CK = (int64_t)C * kh * kw;
    /* weights as F32 rows [O][CK]: used in place when stored as F32, else dequantized once per call */
    const float *wf = (const float *)w; float *wbuf = NULL;
    if (wtype != RT_F32) {
        size_t rowb = rope_row_bytes(wtype, CK);
        wbuf = amalloc((size_t)O * CK * 4);
        int o;
        #pragma omp parallel for num_threads(rope_get_threads()) schedule(static)
        for (o = 0; o < O; o++) rope_dequant_row(wtype, (const uint8_t *)w + (size_t)o * rowb, wbuf + (int64_t)o * CK, CK);
        wf = wbuf;
    }
    /* k -> (channel offset, ky, kx), so the gather does no division */
    int *kc_c = malloc((size_t)CK * sizeof(int)), *kc_y = malloc((size_t)CK * sizeof(int)), *kc_x = malloc((size_t)CK * sizeof(int));
    for (int64_t k = 0; k < CK; k++) { int r = (int)(k % (kh * kw)); kc_c[k] = (int)(k / (kh * kw)); kc_y[k] = r / kw; kc_x[k] = r % kw; }
    /* tasks = (64-position block) x (block of output rows). Small images (UNet's 8x8 level: one position block)
       split the output rows too, so every thread has work; each task gathers its own input panel. */
    int nth = rope_get_threads();
    int n_ptasks = (int)((P + CONV_PW - 1) / CONV_PW), n_oblk = 1;
    int o_strips = (O + MR - 1) / MR;
    if (n_ptasks < 3 * nth) { n_oblk = (3 * nth + n_ptasks - 1) / n_ptasks; if (n_oblk > o_strips) n_oblk = o_strips; }
    int ob = ((o_strips + n_oblk - 1) / n_oblk) * MR;           /* rows per block, a multiple of MR */
    n_oblk = (O + ob - 1) / ob;
    int n_tasks = n_ptasks * n_oblk;
    #pragma omp parallel num_threads(nth)
    {
        float *pk = amalloc((size_t)KC * CONV_PW * 4);           /* [CONV_PW/NR sub-panels][KC][NR] */
        int ti;
        #pragma omp for schedule(dynamic, 4)
        for (ti = 0; ti < n_tasks; ti++) {
            int64_t p0 = (int64_t)(ti / n_oblk) * CONV_PW;
            int64_t o_lo = (int64_t)(ti % n_oblk) * ob, o_hi = o_lo + ob < O ? o_lo + ob : O;
            int np = (int)(P - p0 < CONV_PW ? P - p0 : CONV_PW), nsub = (np + NR - 1) / NR;
            int iy0[CONV_PW], ix0[CONV_PW];
            for (int j = 0; j < CONV_PW; j++) {
                int64_t p = p0 + (j < np ? j : 0);
                iy0[j] = (int)(p / Wo) * stride - pad; ix0[j] = (int)(p % Wo) * stride - pad;
            }
            for (int64_t k0 = 0; k0 < CK; k0 += KC) {
                int64_t kc = CK - k0 < KC ? CK - k0 : KC;
                for (int64_t k = 0; k < kc; k++) {              /* gather: on-the-fly im2col of this k block */
                    int c = kc_c[k0 + k], ky = kc_y[k0 + k], kx = kc_x[k0 + k];
                    const float *plane = in + (int64_t)c * H * W;
                    for (int j = 0; j < nsub * NR; j++) {
                        int iy = iy0[j] + ky, ix = ix0[j] + kx;
                        pk[((int64_t)(j / NR) * KC + k) * NR + j % NR] =
                            (j < np && iy >= 0 && iy < H && ix >= 0 && ix < W) ? plane[(int64_t)iy * W + ix] : 0.0f;
                    }
                }
                for (int64_t m0 = o_lo; m0 < o_hi; m0 += MR) {
                    int mr = (int)(o_hi - m0 < MR ? o_hi - m0 : MR);
                    for (int s = 0; s < nsub; s++) {
                        int nr = np - s * NR < NR ? np - s * NR : NR;
                        float *yo = out + m0 * P + p0 + s * NR;
                        const float *ps = pk + (int64_t)s * KC * NR;
                        if (mr == MR) ukernel6(wf + m0 * CK + k0, CK, ps, kc, yo, P, nr, k0 > 0);
                        else ukernel_small(wf + m0 * CK + k0, CK, ps, kc, yo, P, mr, nr, k0 > 0);
                    }
                }
            }
            if (bias) for (int64_t o = o_lo; o < o_hi; o++) { float *r = out + (int64_t)o * P + p0; for (int j = 0; j < np; j++) r[j] += bias[o]; }
        }
        afree(pk);
    }
    free(kc_c); free(kc_y); free(kc_x);
    if (wbuf) afree(wbuf);
}

/* ---------------- attention: per query block, S = QK^T (matmul), softmax, O = S V (matmul with V^T) ---------------- */
void rope_softmax_rows(float *x, int64_t m, int64_t n) {
    int mi = (int)m;
    int r;
    #pragma omp parallel for num_threads(rope_get_threads()) schedule(static) if (m * n > 32768)
    for (r = 0; r < mi; r++) {
        float *v = x + (int64_t)r * n, mx = -INFINITY, s = 0.0f;
        for (int64_t i = 0; i < n; i++) mx = v[i] > mx ? v[i] : mx;
        for (int64_t i = 0; i < n; i++) { v[i] = expf(v[i] - mx); s += v[i]; }
        float inv = 1.0f / s;
        for (int64_t i = 0; i < n; i++) v[i] *= inv;
    }
}

void rope_attention(const float *q, const float *k, const float *v, float *o, int64_t nq, int64_t nk, int d,
                    int64_t qs, int64_t ks, int64_t vs, int64_t os, float scale) {
    rope_attention_ex(q, k, v, o, nq, nk, d, qs, ks, vs, os, scale, 0, NULL);
}

/* causal != 0: query i sees keys 0..i only (text encoders). */
void rope_attention_ex(const float *q, const float *k, const float *v, float *o, int64_t nq, int64_t nk, int d,
                       int64_t qs, int64_t ks, int64_t vs, int64_t os, float scale, int causal, const float *bias) {
    float *kc = amalloc((size_t)(nk * d) * 4), *vt = amalloc((size_t)(nk * d) * 4);
    for (int64_t j = 0; j < nk; j++)
        for (int c = 0; c < d; c++) { kc[j * d + c] = k[j * ks + c]; vt[(int64_t)c * nk + j] = v[j * vs + c]; }
    int64_t qb = (2ll << 20) / (4 * nk);                     /* ~2 MB of scores per thread */
    qb = qb < 1 ? 1 : qb > 64 ? 64 : qb;
    int n_blocks = (int)((nq + qb - 1) / qb);
    #pragma omp parallel num_threads(rope_get_threads())
    {
        float *qc = amalloc((size_t)(qb * d) * 4), *s = amalloc((size_t)(qb * nk) * 4), *ob = amalloc((size_t)(qb * d) * 4);
        int bi;
        #pragma omp for schedule(dynamic, 1)
        for (bi = 0; bi < n_blocks; bi++) {
            int64_t q0 = (int64_t)bi * qb, n = nq - q0 < qb ? nq - q0 : qb;
            for (int64_t i = 0; i < n; i++)
                for (int c = 0; c < d; c++) qc[i * d + c] = q[(q0 + i) * qs + c] * scale;
            matmul_impl(qc, n, d, kc, RT_F32, nk, s, 0);
            for (int64_t i = 0; i < n; i++) {                /* row softmax, serial inside the thread */
                float *r = s + i * nk, mx = -INFINITY, sum = 0.0f;
                if (bias) { const float *br = bias + (q0 + i) * nk; for (int64_t j = 0; j < nk; j++) r[j] += br[j]; }
                if (causal) for (int64_t j = q0 + i + 1; j < nk; j++) r[j] = -INFINITY;
                for (int64_t j = 0; j < nk; j++) mx = r[j] > mx ? r[j] : mx;
                for (int64_t j = 0; j < nk; j++) { r[j] = expf(r[j] - mx); sum += r[j]; }
                float inv = 1.0f / sum;
                for (int64_t j = 0; j < nk; j++) r[j] *= inv;
            }
            matmul_impl(s, n, nk, vt, RT_F32, d, ob, 0);
            for (int64_t i = 0; i < n; i++) memcpy(o + (q0 + i) * os, ob + i * d, (size_t)d * 4);
        }
        afree(qc); afree(s); afree(ob);
    }
    afree(kc); afree(vt);
}

/* ---------------- norms and activations (statistics accumulated in double) ---------------- */
void rope_layernorm(float *x, int64_t m, int64_t n, const float *g, const float *b, float eps) {
    int mi = (int)m;
    int r;
    #pragma omp parallel for num_threads(rope_get_threads()) schedule(static) if (m * n > 32768)
    for (r = 0; r < mi; r++) {
        float *v = x + (int64_t)r * n; double s = 0, s2 = 0;
        for (int64_t i = 0; i < n; i++) s += v[i];
        double mean = s / n;
        for (int64_t i = 0; i < n; i++) { double t = v[i] - mean; s2 += t * t; }
        float inv = (float)(1.0 / sqrt(s2 / n + eps)), mf = (float)mean;
        for (int64_t i = 0; i < n; i++) v[i] = (v[i] - mf) * inv * (g ? g[i] : 1.0f) + (b ? b[i] : 0.0f);
    }
}

void rope_rmsnorm(float *x, int64_t m, int64_t n, const float *g, float eps) {
    int mi = (int)m;
    int r;
    #pragma omp parallel for num_threads(rope_get_threads()) schedule(static) if (m * n > 32768)
    for (r = 0; r < mi; r++) {
        float *v = x + (int64_t)r * n; double s2 = 0;
        for (int64_t i = 0; i < n; i++) s2 += (double)v[i] * v[i];
        float inv = (float)(1.0 / sqrt(s2 / n + eps));
        for (int64_t i = 0; i < n; i++) v[i] = v[i] * inv * (g ? g[i] : 1.0f);
    }
}

void rope_groupnorm(float *x, int64_t c, int64_t hw, int groups, const float *g, const float *b, float eps) {
    int64_t cpg = c / groups;
    int gi;
    #pragma omp parallel for num_threads(rope_get_threads()) schedule(static)
    for (gi = 0; gi < groups; gi++) {
        float *v = x + (int64_t)gi * cpg * hw; int64_t n = cpg * hw; double s = 0, s2 = 0;
        for (int64_t i = 0; i < n; i++) s += v[i];
        double mean = s / n;
        for (int64_t i = 0; i < n; i++) { double t = v[i] - mean; s2 += t * t; }
        float inv = (float)(1.0 / sqrt(s2 / n + eps)), mf = (float)mean;
        for (int64_t ch = 0; ch < cpg; ch++) {
            int64_t cc = gi * cpg + ch; float gg = g ? g[cc] : 1.0f, bb = b ? b[cc] : 0.0f;
            float *row = v + ch * hw;
            for (int64_t i = 0; i < hw; i++) row[i] = (row[i] - mf) * inv * gg + bb;
        }
    }
}

void rope_gelu_erf(float *x, int64_t n) {
    int64_t i;
    #pragma omp parallel for num_threads(rope_get_threads()) schedule(static) if (n > 65536)
    for (i = 0; i < n; i++) { float v = x[i]; x[i] = 0.5f * v * (1.0f + erff(v * 0.70710678118654752f)); }
}

void rope_gelu_tanh(float *x, int64_t n) {
    int64_t i;
    #pragma omp parallel for num_threads(rope_get_threads()) schedule(static) if (n > 65536)
    for (i = 0; i < n; i++) { float v = x[i]; x[i] = 0.5f * v * (1.0f + tanhf(0.79788456080286536f * (v + 0.044715f * v * v * v))); }
}

void rope_add_rows(float *y, int64_t m, int64_t n, const float *b) {
    int64_t i;
    #pragma omp parallel for num_threads(rope_get_threads()) schedule(static) if (m * n > 65536)
    for (i = 0; i < m; i++) { float *r = y + i * n; for (int64_t j = 0; j < n; j++) r[j] += b[j]; }
}

void rope_silu(float *x, int64_t n) {
    int64_t i;
    #pragma omp parallel for num_threads(rope_get_threads()) schedule(static) if (n > 65536)
    for (i = 0; i < n; i++) { float v = x[i]; x[i] = v / (1.0f + expf(-v)); }
}

/* ---------------- rotary ---------------- */
void rope_rotary_tables(const float *pos, int64_t n_tok, int d_head, float theta, float *cos_t, float *sin_t) {
    int half = d_head / 2;
    for (int64_t t = 0; t < n_tok; t++)
        for (int i = 0; i < half; i++) {
            double a = pos[t] * pow((double)theta, -2.0 * i / d_head);
            cos_t[t * half + i] = (float)cos(a); sin_t[t * half + i] = (float)sin(a);
        }
}

void rope_rotary(float *x, int64_t n_tok, int n_head, int d_head, int64_t tok_stride,
                 const float *cos_t, const float *sin_t, int neox) {
    int half = d_head / 2, nt = (int)n_tok;
    int t;
    #pragma omp parallel for num_threads(rope_get_threads()) schedule(static) if (n_tok * n_head * d_head > 65536)
    for (t = 0; t < nt; t++)
        for (int h = 0; h < n_head; h++) {
            float *v = x + (int64_t)t * tok_stride + (int64_t)h * d_head;
            const float *c = cos_t + (int64_t)t * half, *s = sin_t + (int64_t)t * half;
            for (int i = 0; i < half; i++) {
                int a = neox ? i : 2 * i, b = neox ? i + half : 2 * i + 1;
                float x0 = v[a], x1 = v[b];
                v[a] = x0 * c[i] - x1 * s[i];
                v[b] = x0 * s[i] + x1 * c[i];
            }
        }
}
