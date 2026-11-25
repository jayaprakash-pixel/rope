// rope-engine CUDA kernels. Written from scratch for GPUs without tensor cores (Pascal sm_61, Turing sm_75).
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
extern "C" {
#include "../kernels.h"
#include "../cuda_api.h"
}

#define NEG_INF __int_as_float(0xff800000)   /* -inf as bits: MSVC INFINITY is an overflow expression nvcc warns about */
static char g_err[256];
static bool ck(cudaError_t e, const char *what) {
    if (e == cudaSuccess) return true;
    snprintf(g_err, sizeof g_err, "%s: %s", what, cudaGetErrorString(e));
    return false;
}
#define CK(x) ck((x), #x)
#define LAUNCH_CHECK(what) ck(cudaGetLastError(), what)

extern "C" int rope_cuda_device_count(void) { int n = 0; return cudaGetDeviceCount(&n) == cudaSuccess ? n : 0; }
extern "C" const char *rope_cuda_last_error(void) {
    if (!g_err[0]) return NULL;
    static char out[256]; snprintf(out, sizeof out, "%s", g_err); g_err[0] = 0; return out;
}
extern "C" void *rope_cuda_alloc(size_t n) { void *p = NULL; return CK(cudaMalloc(&p, n ? n : 16)) ? p : NULL; }
extern "C" void rope_cuda_free(void *d) { if (d) CK(cudaFree(d)); }
extern "C" void rope_cuda_upload(void *d, const void *h, size_t n) { CK(cudaMemcpy(d, h, n, cudaMemcpyHostToDevice)); }
extern "C" void rope_cuda_download(void *h, const void *d, size_t n) { CK(cudaMemcpy(h, d, n, cudaMemcpyDeviceToHost)); }
extern "C" void rope_cuda_sync(void) { CK(cudaDeviceSynchronize()); }

static unsigned blocks(int64_t n, int t) { return (unsigned)((n + t - 1) / t); }

// ---------------------------------------------------------------- weight decoding (per element, from the spec)
__device__ __forceinline__ float h2f(const uint8_t *p) { return __half2float(*(const __half *)p); }

// Value i (0..255) of a Q4_K block as  s * q + m  (q = 4-bit code), split so dp4a can use (q, s, m).
__device__ __forceinline__ void q4k_parts(const uint8_t *blk, int i, int *q, float *s, float *m) {
    const uint8_t *sc = blk + 4, *qs = blk + 16;
    int j = i / 32, l = i % 32, sv, mv;
    if (j < 4) { sv = sc[j] & 63; mv = sc[j + 4] & 63; }
    else { sv = (sc[j + 4] & 15) | ((sc[j - 4] >> 6) << 4); mv = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4); }
    int byte = qs[(j / 2) * 32 + l];
    *q = (j & 1) ? byte >> 4 : byte & 15;
    *s = h2f(blk) * sv; *m = -h2f(blk + 2) * mv;
}
__device__ __forceinline__ void q6k_parts(const uint8_t *blk, int i, int *q, float *s) {
    const uint8_t *ql = blk, *qh = blk + 128; const int8_t *sc = (const int8_t *)(blk + 192);
    int h = i / 128, r = i % 128, g = r / 32, l = r % 32;
    int lo = (g & 1) ? ql[h * 64 + 32 + l] : ql[h * 64 + l];
    lo = (g < 2) ? (lo & 15) : (lo >> 4);
    int hi = (qh[h * 32 + l] >> (2 * g)) & 3;
    *q = (lo | (hi << 4)) - 32;
    *s = h2f(blk + 208) * sc[h * 8 + g * 2 + l / 16];
}

__device__ float deq(const uint8_t *row, int t, int64_t i) {
    switch (t) {
    case RT_F32: return ((const float *)row)[i];
    case RT_F16: return __half2float(((const __half *)row)[i]);
    case RT_BF16: { uint32_t b = (uint32_t)((const uint16_t *)row)[i] << 16; return __uint_as_float(b); }
    case RT_Q8_0: { const uint8_t *b = row + (i / 32) * 34; return h2f(b) * (int8_t)b[2 + i % 32]; }
    case RT_Q4_0: { const uint8_t *b = row + (i / 32) * 18; int j = i % 32;
                    int nib = j < 16 ? (b[2 + j] & 15) : (b[2 + j - 16] >> 4); return h2f(b) * (nib - 8); }
    case RT_Q4_K: { int q; float s, m; q4k_parts(row + (i / 256) * 144, (int)(i % 256), &q, &s, &m); return s * q + m; }
    case RT_Q6_K: { int q; float s; q6k_parts(row + (i / 256) * 210, (int)(i % 256), &q, &s); return s * q; }
    }
    return 0.0f;
}

__global__ void k_dequant_f32(const uint8_t *w, int t, size_t rowb, int64_t N, int64_t K, float *out) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * K) return;
    int64_t n = idx / K, k = idx % K;
    out[idx] = deq(w + n * rowb, t, k);
}
__global__ void k_dequant_f16(const uint8_t *w, int t, size_t rowb, int64_t N, int64_t K, __half *out) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * K) return;
    int64_t n = idx / K, k = idx % K;
    out[idx] = __float2half(deq(w + n * rowb, t, k));
}
// One thread per group of 16 weights: int8 codes + one (scale, offset) pair, w = s*q + m.
__global__ void k_prepare_dp4a(const uint8_t *w, int t, size_t rowb, int64_t N, int64_t K, int8_t *q, float *s, float *m) {
    int64_t g = (int64_t)blockIdx.x * blockDim.x + threadIdx.x, ng = K / 16;
    if (g >= N * ng) return;
    int64_t n = g / ng, k0 = (g % ng) * 16;
    const uint8_t *row = w + n * rowb;
    float sv = 0, mv = 0;
    for (int i = 0; i < 16; i++) {
        int64_t k = k0 + i; int qv = 0;
        if (t == RT_Q8_0) { const uint8_t *b = row + (k / 32) * 34; qv = (int8_t)b[2 + k % 32]; sv = h2f(b); }
        else if (t == RT_Q4_0) { const uint8_t *b = row + (k / 32) * 18; int j = (int)(k % 32);
                                 qv = (j < 16 ? (b[2 + j] & 15) : (b[2 + j - 16] >> 4)) - 8; sv = h2f(b); }
        else if (t == RT_Q4_K) q4k_parts(row + (k / 256) * 144, (int)(k % 256), &qv, &sv, &mv);
        else if (t == RT_Q6_K) q6k_parts(row + (k / 256) * 210, (int)(k % 256), &qv, &sv);
        q[n * K + k] = (int8_t)qv;
    }
    s[g] = sv; m[g] = mv;   // constant within a group of 16 for all four formats
}

struct rope_cuda_w { int mode; int64_t N, K; void *w; float *s, *m; };

extern "C" rope_cuda_w *rope_cuda_prepare(const void *w_host, int wtype, int64_t N, int64_t K, int mode) {
    if (!rope_wtype_supported(wtype) || K % rope_wtype_block(wtype)) { snprintf(g_err, sizeof g_err, "prepare: bad type/K"); return NULL; }
    bool quant = wtype == RT_Q8_0 || wtype == RT_Q4_0 || wtype == RT_Q4_K || wtype == RT_Q6_K;
    if (mode == RC_DP4A && (!quant || K % 16)) mode = RC_F16;
    size_t rowb = rope_row_bytes(wtype, K);
    uint8_t *raw = (uint8_t *)rope_cuda_alloc(rowb * N);
    if (!raw) return NULL;
    rope_cuda_upload(raw, w_host, rowb * N);
    rope_cuda_w *p = (rope_cuda_w *)calloc(1, sizeof *p);
    p->mode = mode; p->N = N; p->K = K;
    if (mode == RC_F32) { p->w = rope_cuda_alloc((size_t)(N * K) * 4); k_dequant_f32<<<blocks(N * K, 256), 256>>>(raw, wtype, rowb, N, K, (float *)p->w); }
    else if (mode == RC_F16) { p->w = rope_cuda_alloc((size_t)(N * K) * 2); k_dequant_f16<<<blocks(N * K, 256), 256>>>(raw, wtype, rowb, N, K, (__half *)p->w); }
    else {
        p->w = rope_cuda_alloc((size_t)(N * K)); p->s = (float *)rope_cuda_alloc((size_t)(N * K / 16) * 4); p->m = (float *)rope_cuda_alloc((size_t)(N * K / 16) * 4);
        k_prepare_dp4a<<<blocks(N * K / 16, 256), 256>>>(raw, wtype, rowb, N, K, (int8_t *)p->w, p->s, p->m);
    }
    LAUNCH_CHECK("prepare");
    rope_cuda_sync();
    rope_cuda_free(raw);
    return p;
}
extern "C" void rope_cuda_release(rope_cuda_w *w) { if (!w) return; rope_cuda_free(w->w); rope_cuda_free(w->s); rope_cuda_free(w->m); free(w); }
extern "C" int rope_cuda_mode(const rope_cuda_w *w) { return w->mode; }

// ---------------------------------------------------------------- GEMM: Y[M,N] = X[M,K] W[N,K]^T, 64x64 tiles, 256 threads x 4x4
#define BM 64
#define BN 64
#define BK 16

__global__ void __launch_bounds__(256) k_sgemm(const float *__restrict__ X, const float *__restrict__ Wt, float *__restrict__ Y, int M, int N, int K) {
    __shared__ __align__(16) float As[BK][BM], Bs[BK][BN];
    int tid = threadIdx.x, tx = tid % 16, ty = tid / 16, m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    float acc[4][4] = {};
    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int i = tid; i < BM * BK; i += 256) {
            int r = i / BK, c = i % BK, gk = k0 + c;
            As[c][r] = (m0 + r < M && gk < K) ? X[(int64_t)(m0 + r) * K + gk] : 0.0f;
            Bs[c][r] = (n0 + r < N && gk < K) ? Wt[(int64_t)(n0 + r) * K + gk] : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            float4 a = *(const float4 *)&As[k][ty * 4], b = *(const float4 *)&Bs[k][tx * 4];
            float av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w};
            #pragma unroll
            for (int i = 0; i < 4; i++)
                #pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
        }
        __syncthreads();
    }
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) {
        int m = m0 + ty * 4 + i, n = n0 + tx * 4 + j;
        if (m < M && n < N) Y[(int64_t)m * N + n] = acc[i][j];
    }
}

// FP16: tiles hold k-pairs as half2; each k-tile's 8 HFMA2 partial sums are added into FP32 accumulators.
__global__ void __launch_bounds__(256) k_hgemm(const float *__restrict__ X, const __half *__restrict__ Wt, float *__restrict__ Y, int M, int N, int K) {
    __shared__ __align__(16) __half2 As[BK / 2][BM], Bs[BK / 2][BN];
    int tid = threadIdx.x, tx = tid % 16, ty = tid / 16, m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    float acc[4][4] = {};
    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int i = tid; i < BM * (BK / 2); i += 256) {
            int r = i / (BK / 2), c = i % (BK / 2), gk = k0 + 2 * c;
            float x0 = 0, x1 = 0; __half w0 = __float2half(0.0f), w1 = w0;
            if (m0 + r < M) { if (gk < K) x0 = X[(int64_t)(m0 + r) * K + gk]; if (gk + 1 < K) x1 = X[(int64_t)(m0 + r) * K + gk + 1]; }
            if (n0 + r < N) { if (gk < K) w0 = Wt[(int64_t)(n0 + r) * K + gk]; if (gk + 1 < K) w1 = Wt[(int64_t)(n0 + r) * K + gk + 1]; }
            As[c][r] = __floats2half2_rn(x0, x1); Bs[c][r] = __halves2half2(w0, w1);
        }
        __syncthreads();
        __half2 h[4][4];
        #pragma unroll
        for (int i = 0; i < 4; i++)
            #pragma unroll
            for (int j = 0; j < 4; j++) h[i][j] = __float2half2_rn(0.0f);
        #pragma unroll
        for (int k = 0; k < BK / 2; k++) {
            #pragma unroll
            for (int i = 0; i < 4; i++)
                #pragma unroll
                for (int j = 0; j < 4; j++) h[i][j] = __hfma2(As[k][ty * 4 + i], Bs[k][tx * 4 + j], h[i][j]);
        }
        #pragma unroll
        for (int i = 0; i < 4; i++)
            #pragma unroll
            for (int j = 0; j < 4; j++) { float2 f = __half22float2(h[i][j]); acc[i][j] += f.x + f.y; }
        __syncthreads();
    }
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) {
        int m = m0 + ty * 4 + i, n = n0 + tx * 4 + j;
        if (m < M && n < N) Y[(int64_t)m * N + n] = acc[i][j];
    }
}

// Activations -> int8 per group of 16: xq, scale xs = amax/127, and the code sum xsum (for the weight offset term).
__global__ void k_quant_x(const float *X, int64_t M, int64_t K, int8_t *xq, float *xs, int *xsum) {
    int64_t g = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= M * (K / 16)) return;
    const float *x = X + g * 16;
    float amax = 0; for (int i = 0; i < 16; i++) amax = fmaxf(amax, fabsf(x[i]));
    float s = amax / 127.0f, inv = s > 0 ? 1.0f / s : 0.0f; int sum = 0;
    for (int i = 0; i < 16; i++) { int q = __float2int_rn(x[i] * inv); xq[g * 16 + i] = (int8_t)q; sum += q; }
    xs[g] = s; xsum[g] = sum;
}

// dp4a GEMM: k-tile = one group of 16 (4 int32 words). y += ws*xs*dot(q_w, q_x) + wm*xs*sum(q_x).
__global__ void __launch_bounds__(256) k_gemm_dp4a(const int8_t *__restrict__ xq, const float *__restrict__ xs, const int *__restrict__ xsum,
                                                   const int8_t *__restrict__ wq, const float *__restrict__ ws, const float *__restrict__ wm,
                                                   float *__restrict__ Y, int M, int N, int K) {
    __shared__ int Aq[BM][4], Bq[BN][4];
    __shared__ float Axs[BM], Bws[BN], Bwm[BN]; __shared__ int Axsum[BM];
    int tid = threadIdx.x, tx = tid % 16, ty = tid / 16, m0 = blockIdx.y * BM, n0 = blockIdx.x * BN, G = K / 16;
    float acc[4][4] = {};
    for (int g = 0; g < G; g++) {
        { int r = tid / 4, w = tid % 4;   // 256 threads load 64 rows x 4 words for both tiles
          Aq[r][w] = m0 + r < M ? ((const int *)(xq + ((int64_t)(m0 + r) * K + g * 16)))[w] : 0;
          Bq[r][w] = n0 + r < N ? ((const int *)(wq + ((int64_t)(n0 + r) * K + g * 16)))[w] : 0; }
        if (tid < BM) { int m = m0 + tid; Axs[tid] = m < M ? xs[(int64_t)m * G + g] : 0.0f; Axsum[tid] = m < M ? xsum[(int64_t)m * G + g] : 0; }
        else if (tid < BM + BN) { int c = tid - BM, n = n0 + c; Bws[c] = n < N ? ws[(int64_t)n * G + g] : 0.0f; Bwm[c] = n < N ? wm[(int64_t)n * G + g] : 0.0f; }
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int r = ty * 4 + i; float sx = Axs[r]; float sumx = (float)Axsum[r];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                int c = tx * 4 + j, d = 0;
                d = __dp4a(Aq[r][0], Bq[c][0], d); d = __dp4a(Aq[r][1], Bq[c][1], d);
                d = __dp4a(Aq[r][2], Bq[c][2], d); d = __dp4a(Aq[r][3], Bq[c][3], d);
                acc[i][j] += sx * (Bws[c] * (float)d + Bwm[c] * sumx);
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) {
        int m = m0 + ty * 4 + i, n = n0 + tx * 4 + j;
        if (m < M && n < N) Y[(int64_t)m * N + n] = acc[i][j];
    }
}

static void sgemm_dev(const float *x, int64_t M, int64_t K, const float *w, int64_t N, float *y) {
    dim3 grid(blocks(N, BN), blocks(M, BM));
    k_sgemm<<<grid, 256>>>(x, w, y, (int)M, (int)N, (int)K);
    LAUNCH_CHECK("sgemm");
}

extern "C" void rope_cuda_matmul(const float *dx, int64_t M, const rope_cuda_w *w, float *dy) {
    int64_t N = w->N, K = w->K;
    dim3 grid(blocks(N, BN), blocks(M, BM));
    if (w->mode == RC_F32) { k_sgemm<<<grid, 256>>>(dx, (const float *)w->w, dy, (int)M, (int)N, (int)K); LAUNCH_CHECK("sgemm"); return; }
    if (w->mode == RC_F16) { k_hgemm<<<grid, 256>>>(dx, (const __half *)w->w, dy, (int)M, (int)N, (int)K); LAUNCH_CHECK("hgemm"); return; }
    int64_t G = K / 16;
    int8_t *xq = (int8_t *)rope_cuda_alloc((size_t)(M * K)); float *xs = (float *)rope_cuda_alloc((size_t)(M * G) * 4);
    int *xsum = (int *)rope_cuda_alloc((size_t)(M * G) * 4);
    k_quant_x<<<blocks(M * G, 256), 256>>>(dx, M, K, xq, xs, xsum);
    k_gemm_dp4a<<<grid, 256>>>(xq, xs, xsum, (const int8_t *)w->w, w->s, w->m, dy, (int)M, (int)N, (int)K);
    LAUNCH_CHECK("dp4a");
    rope_cuda_free(xq); rope_cuda_free(xs); rope_cuda_free(xsum);
}

// ---------------------------------------------------------------- conv2d: chunked im2col on the GPU + GEMM
__global__ void k_im2col(const float *in, int C, int H, int W, int kh, int kw, int stride, int pad, int Wo,
                         int64_t p0, int64_t np, float *col) {
    int64_t CK = (int64_t)C * kh * kw, idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= np * CK) return;
    int64_t i = idx / CK, ck = idx % CK, p = p0 + i;
    int oy = (int)(p / Wo), ox = (int)(p % Wo), c = (int)(ck / (kh * kw)), r = (int)(ck % (kh * kw)), ky = r / kw, kx = r % kw;
    int iy = oy * stride - pad + ky, ix = ox * stride - pad + kx;
    col[idx] = (iy >= 0 && iy < H && ix >= 0 && ix < W) ? in[((int64_t)c * H + iy) * W + ix] : 0.0f;
}
__global__ void k_scatter_bias(const float *yc, int64_t np, int O, const float *bias, int64_t P, int64_t p0, float *out) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= np * O) return;
    int64_t i = idx / O; int o = (int)(idx % O);
    out[(int64_t)o * P + p0 + i] = yc[idx] + (bias ? bias[o] : 0.0f);
}

extern "C" void rope_cuda_conv2d(const float *din, int C, int H, int W, const rope_cuda_w *w, const float *dbias,
                                 int kh, int kw, int stride, int pad, float *dout) {
    int O = (int)w->N, Ho = (H + 2 * pad - kh) / stride + 1, Wo = (W + 2 * pad - kw) / stride + 1;
    int64_t P = (int64_t)Ho * Wo, CK = w->K;
    int64_t chunk = (256ll << 20) / (4 * (CK + O)); if (chunk < 1) chunk = 1; if (chunk > P) chunk = P;
    /* im2col + output scratch grown once and kept: a cudaMalloc per conv call costs time and spikes VRAM */
    static float *g_col = NULL; static size_t g_cap = 0;
    size_t need = (size_t)(chunk * CK + chunk * O) * 4;
    if (need > g_cap) { rope_cuda_free(g_col); g_col = (float *)rope_cuda_alloc(need); g_cap = g_col ? need : 0; }
    if (!g_col) return;
    float *col = g_col, *yc = g_col + chunk * CK;
    for (int64_t p0 = 0; p0 < P; p0 += chunk) {
        int64_t np = P - p0 < chunk ? P - p0 : chunk;
        k_im2col<<<blocks(np * CK, 256), 256>>>(din, C, H, W, kh, kw, stride, pad, Wo, p0, np, col);
        rope_cuda_matmul(col, np, w, yc);
        k_scatter_bias<<<blocks(np * O, 256), 256>>>(yc, np, O, dbias, P, p0, dout);
    }
    LAUNCH_CHECK("conv2d");
}

// ---------------------------------------------------------------- softmax / norms (one block per row or group)
__device__ float block_sum(float v, float *sh) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    int w = threadIdx.x / 32, l = threadIdx.x % 32;
    __syncthreads();
    if (l == 0) sh[w] = v;
    __syncthreads();
    v = threadIdx.x < blockDim.x / 32 ? sh[threadIdx.x] : 0.0f;
    if (w == 0) for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    return sh[0];
}
__device__ float block_max(float v, float *sh) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    int w = threadIdx.x / 32, l = threadIdx.x % 32;
    __syncthreads();
    if (l == 0) sh[w] = v;
    __syncthreads();
    v = threadIdx.x < blockDim.x / 32 ? sh[threadIdx.x] : NEG_INF;
    if (w == 0) for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    return sh[0];
}

__global__ void k_softmax(float *x, int64_t n) {
    __shared__ float sh[32];
    float *r = x + (int64_t)blockIdx.x * n, mx = NEG_INF, s = 0;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) mx = fmaxf(mx, r[i]);
    mx = block_max(mx, sh);
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) { float e = __expf(r[i] - mx); r[i] = e; s += e; }
    float inv = 1.0f / block_sum(s, sh);
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) r[i] *= inv;
}
__global__ void k_layernorm(float *x, int64_t n, const float *g, const float *b, float eps, int rms) {
    __shared__ float sh[32];
    float *r = x + (int64_t)blockIdx.x * n, s = 0;
    float mean = 0;
    if (!rms) { for (int64_t i = threadIdx.x; i < n; i += blockDim.x) s += r[i]; mean = block_sum(s, sh) / n; }
    float s2 = 0;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) { float t = r[i] - mean; s2 += t * t; }
    float inv = rsqrtf(block_sum(s2, sh) / n + eps);
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) r[i] = (r[i] - mean) * inv * (g ? g[i] : 1.0f) + (b ? b[i] : 0.0f);
}
__global__ void k_groupnorm(float *x, int64_t cpg, int64_t hw, const float *g, const float *b, float eps) {
    __shared__ float sh[32];
    int64_t n = cpg * hw; float *v = x + (int64_t)blockIdx.x * n, s = 0, s2 = 0;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) s += v[i];
    float mean = block_sum(s, sh) / n;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) { float t = v[i] - mean; s2 += t * t; }
    float inv = rsqrtf(block_sum(s2, sh) / n + eps);
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        int64_t c = blockIdx.x * cpg + i / hw;
        v[i] = (v[i] - mean) * inv * (g ? g[c] : 1.0f) + (b ? b[c] : 0.0f);
    }
}

extern "C" void rope_cuda_softmax_rows(float *dx, int64_t m, int64_t n) { k_softmax<<<(unsigned)m, 256>>>(dx, n); LAUNCH_CHECK("softmax"); }
extern "C" void rope_cuda_layernorm(float *dx, int64_t m, int64_t n, const float *dg, const float *db, float eps) { k_layernorm<<<(unsigned)m, 256>>>(dx, n, dg, db, eps, 0); LAUNCH_CHECK("layernorm"); }
extern "C" void rope_cuda_rmsnorm(float *dx, int64_t m, int64_t n, const float *dg, float eps) { k_layernorm<<<(unsigned)m, 256>>>(dx, n, dg, NULL, eps, 1); LAUNCH_CHECK("rmsnorm"); }
extern "C" void rope_cuda_groupnorm(float *dx, int64_t c, int64_t hw, int groups, const float *dg, const float *db, float eps) {
    k_groupnorm<<<groups, 1024>>>(dx, c / groups, hw, dg, db, eps); LAUNCH_CHECK("groupnorm");
}

// ---------------------------------------------------------------- activations, rotary
__global__ void k_act(float *x, int64_t n, int kind) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        float v = x[i];
        x[i] = kind == 0 ? 0.5f * v * (1.0f + erff(v * 0.70710678118654752f))
             : kind == 1 ? 0.5f * v * (1.0f + tanhf(0.79788456080286536f * (v + 0.044715f * v * v * v)))
             : v / (1.0f + __expf(-v));
    }
}
static unsigned act_grid(int64_t n) { int64_t b = blocks(n, 256); return (unsigned)(b > 65535 ? 65535 : b); }
extern "C" void rope_cuda_gelu_erf(float *dx, int64_t n) { k_act<<<act_grid(n), 256>>>(dx, n, 0); LAUNCH_CHECK("gelu_erf"); }
extern "C" void rope_cuda_gelu_tanh(float *dx, int64_t n) { k_act<<<act_grid(n), 256>>>(dx, n, 1); LAUNCH_CHECK("gelu_tanh"); }
extern "C" void rope_cuda_silu(float *dx, int64_t n) { k_act<<<act_grid(n), 256>>>(dx, n, 2); LAUNCH_CHECK("silu"); }

__global__ void k_rotary(float *x, int64_t n_tok, int n_head, int d_head, int64_t st, const float *c, const float *s, int neox) {
    int half = d_head / 2;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_tok * n_head * half) return;
    int i = (int)(idx % half); int64_t th = idx / half; int h = (int)(th % n_head); int64_t t = th / n_head;
    float *v = x + t * st + (int64_t)h * d_head;
    int a = neox ? i : 2 * i, b = neox ? i + half : 2 * i + 1;
    float x0 = v[a], x1 = v[b], cc = c[t * half + i], ss = s[t * half + i];
    v[a] = x0 * cc - x1 * ss; v[b] = x0 * ss + x1 * cc;
}
extern "C" void rope_cuda_rotary(float *dx, int64_t n_tok, int n_head, int d_head, int64_t tok_stride,
                                 const float *dcos, const float *dsin, int neox) {
    k_rotary<<<blocks(n_tok * n_head * (d_head / 2), 256), 256>>>(dx, n_tok, n_head, d_head, tok_stride, dcos, dsin, neox);
    LAUNCH_CHECK("rotary");
}

// ---------------------------------------------------------------- attention: S = (Q*scale) K^T, softmax, O = S V
__global__ void k_gather(const float *src, int64_t rows, int d, int64_t stride, float scale, float *dst) {   // dst[r*d+c]
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * d) return;
    dst[idx] = src[(idx / d) * stride + idx % d] * scale;
}
__global__ void k_gather_t(const float *src, int64_t rows, int d, int64_t stride, float *dst) {             // dst[c*rows+r]
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * d) return;
    int64_t r = idx / d; int c = (int)(idx % d);
    dst[(int64_t)c * rows + r] = src[r * stride + c];
}
__global__ void k_scatter(const float *src, int64_t rows, int d, int64_t stride, float *dst) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * d) return;
    dst[(idx / d) * stride + idx % d] = src[idx];
}

extern "C" void rope_cuda_attention(const float *dq, const float *dk, const float *dv, float *dout, int64_t nq, int64_t nk, int d,
                                    int64_t qs, int64_t ks, int64_t vs, int64_t os, float scale) {
    /* scratch grown once and kept: this is called once per head (thousands of calls per image) */
    int64_t qb = (256ll << 20) / (4 * nk); if (qb < 1) qb = 1; if (qb > nq) qb = nq;    // scores <= 256 MB
    static float *g_a = NULL; static size_t g_cap = 0;
    size_t need = (size_t)(nq * d + 2 * nk * d + qb * nk + qb * d) * 4;
    if (need > g_cap) { rope_cuda_free(g_a); g_a = (float *)rope_cuda_alloc(need); g_cap = g_a ? need : 0; }
    if (!g_a) return;
    float *qc = g_a, *kc = qc + nq * d, *vt = kc + nk * d, *s = vt + nk * d, *ob = s + qb * nk;
    k_gather<<<blocks(nq * d, 256), 256>>>(dq, nq, d, qs, scale, qc);
    k_gather<<<blocks(nk * d, 256), 256>>>(dk, nk, d, ks, 1.0f, kc);
    k_gather_t<<<blocks(nk * d, 256), 256>>>(dv, nk, d, vs, vt);
    for (int64_t q0 = 0; q0 < nq; q0 += qb) {
        int64_t n = nq - q0 < qb ? nq - q0 : qb;
        sgemm_dev(qc + q0 * d, n, d, kc, nk, s);
        k_softmax<<<(unsigned)n, 256>>>(s, nk);
        sgemm_dev(s, n, nk, vt, d, ob);
        k_scatter<<<blocks(n * d, 256), 256>>>(ob, n, d, os, dout + q0 * os);
    }
    LAUNCH_CHECK("attention");
}

// ---------------------------------------------------------------- caller-owned buffers (VRAM manager, P2.1.2)
// The Rust VRAM manager owns every large device buffer (weights kept resident, persistent scratch); these entry
// points never allocate per call except the conv im2col scratch, which is grown once and kept.
extern "C" void *rope_cuda_host_alloc(size_t n) { void *p = NULL; return CK(cudaHostAlloc(&p, n ? n : 16, cudaHostAllocDefault)) ? p : NULL; }
extern "C" void  rope_cuda_host_free(void *h) { if (h) CK(cudaFreeHost(h)); }
extern "C" size_t rope_cuda_free_bytes(void) { size_t f = 0, t = 0; return CK(cudaMemGetInfo(&f, &t)) ? f : 0; }
extern "C" void rope_cuda_dequant(const void *d_raw, int wtype, int64_t N, int64_t K, int mode, void *d_out) {
    size_t rowb = rope_row_bytes(wtype, K);
    if (mode == RC_F16) k_dequant_f16<<<blocks(N * K, 256), 256>>>((const uint8_t *)d_raw, wtype, rowb, N, K, (__half *)d_out);
    else k_dequant_f32<<<blocks(N * K, 256), 256>>>((const uint8_t *)d_raw, wtype, rowb, N, K, (float *)d_out);
    LAUNCH_CHECK("dequant");
}
extern "C" void rope_cuda_gemm(const float *dx, int64_t M, int64_t N, int64_t K, int mode, const void *dw, float *dy) {
    rope_cuda_w w = { mode == RC_F16 ? RC_F16 : RC_F32, N, K, (void *)dw, NULL, NULL };
    rope_cuda_matmul(dx, M, &w, dy);
}
extern "C" void rope_cuda_conv2d_raw(const float *din, int C, int H, int W, int O, int mode, const void *dw, const float *dbias,
                                     int kh, int kw, int stride, int pad, float *dout) {
    rope_cuda_w w = { mode == RC_F16 ? RC_F16 : RC_F32, O, (int64_t)C * kh * kw, (void *)dw, NULL, NULL };
    rope_cuda_conv2d(din, C, H, W, &w, dbias, kh, kw, stride, pad, dout);
}

// ---------------------------------------------------------------- light ops for device-resident activations (G3)
// y = (1 + scale) * LN(x) + shift, LN without affine (eps); one block per row.
__global__ void k_modulate(const float *x, float *y, int64_t n, const float *shift, const float *scale, float eps) {
    __shared__ float sh[32];
    const float *xr = x + (int64_t)blockIdx.x * n; float *yr = y + (int64_t)blockIdx.x * n;
    float s = 0;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) s += xr[i];
    float mean = block_sum(s, sh) / n;
    float v = 0;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) { float d = xr[i] - mean; v += d * d; }
    float inv = rsqrtf(block_sum(v, sh) / n + eps);
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) yr[i] = (xr[i] - mean) * inv * (1.0f + scale[i]) + shift[i];
}
__global__ void k_add_gated(float *x, const float *gate, const float *y, int64_t m, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m * n) x[i] += gate[i % n] * y[i];
}
__global__ void k_add_rows(float *y, const float *b, int64_t m, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m * n) y[i] += b[i % n];
}
// Per row, per head, for q (cols [0, D)) and k (cols [D, 2D)) of a fused buffer with row stride rs: RMSNorm over the
// head (eps 1e-6) times the learned scale, then interleaved-pair RoPE with the row's cos/sin (hd/2 pairs).
// Grid (rows, heads, 2); hd threads.
__global__ void k_qk_norm_rope(float *buf, int64_t rs, int D, int hd, const float *qsc, const float *ksc,
                               const float *cosv, const float *sinv, int64_t row0) {
    __shared__ float sh[32];
    __shared__ float v[256];
    int64_t r = blockIdx.x; int h = blockIdx.y, isk = blockIdx.z, t = threadIdx.x;
    float *p = buf + r * rs + isk * D + h * hd;
    const float *sc = isk ? ksc : qsc;
    float x = p[t];
    float inv = rsqrtf(block_sum(x * x, sh) / hd + 1e-6f);
    v[t] = x * inv * sc[t];
    __syncthreads();
    if (t < hd / 2) {
        float c = cosv[(row0 + r) * (hd / 2) + t], s = sinv[(row0 + r) * (hd / 2) + t];
        float a = v[2 * t], b = v[2 * t + 1];
        p[2 * t] = c * a - s * b; p[2 * t + 1] = s * a + c * b;
    }
}
extern "C" void rope_cuda_modulate(const float *dx, float *dy, int64_t m, int64_t n, const float *dshift, const float *dscale, float eps) {
    k_modulate<<<(unsigned)m, 256>>>(dx, dy, n, dshift, dscale, eps); LAUNCH_CHECK("modulate");
}
extern "C" void rope_cuda_add_gated(float *dx, const float *dgate, const float *dy, int64_t m, int64_t n) {
    k_add_gated<<<blocks(m * n, 256), 256>>>(dx, dgate, dy, m, n); LAUNCH_CHECK("add_gated");
}
extern "C" void rope_cuda_add_rows(float *dy, const float *db, int64_t m, int64_t n) {
    k_add_rows<<<blocks(m * n, 256), 256>>>(dy, db, m, n); LAUNCH_CHECK("add_rows");
}
extern "C" void rope_cuda_qk_norm_rope(float *dbuf, int64_t rows, int64_t rs, int D, int heads, int hd, const float *dqs, const float *dks,
                                       const float *dcos, const float *dsin, int64_t row0) {
    if (hd > 256 || hd % 32) { snprintf(g_err, sizeof g_err, "qk_norm_rope: head size %d unsupported", hd); return; }
    dim3 grid((unsigned)rows, (unsigned)heads, 2);
    k_qk_norm_rope<<<grid, hd>>>(dbuf, rs, D, hd, dqs, dks, dcos, dsin, row0); LAUNCH_CHECK("qk_norm_rope");
}
/* rows x width_bytes copy between strided device buffers (row slicing, column concatenation) */
extern "C" void rope_cuda_copy2d(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t rows) {
    CK(cudaMemcpy2D(dst, dpitch, src, spitch, width, rows, cudaMemcpyDeviceToDevice));
}

// ---------------------------------------------------------------- more light ops (SD UNet / VAE on the device, P2.6.4)
__global__ void k_transpose(const float *x, float *y, int64_t r, int64_t c) {        // y[c, r] = x[r, c]
    __shared__ float t[32][33];
    int64_t bx = (int64_t)blockIdx.x * 32, by = (int64_t)blockIdx.y * 32;
    for (int j = threadIdx.y; j < 32; j += 8) {
        int64_t rr = by + j, cc = bx + threadIdx.x;
        if (rr < r && cc < c) t[j][threadIdx.x] = x[rr * c + cc];
    }
    __syncthreads();
    for (int j = threadIdx.y; j < 32; j += 8) {
        int64_t cc = bx + j, rr = by + threadIdx.x;
        if (rr < r && cc < c) y[cc * r + rr] = t[threadIdx.x][j];
    }
}
__global__ void k_add(float *x, const float *y, int64_t n) { int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) x[i] += y[i]; }
__global__ void k_mul(float *x, const float *y, int64_t n) { int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) x[i] *= y[i]; }
__global__ void k_add_row_scalars(float *x, const float *s, int64_t r, int64_t c) {   // x[i, :] += s[i]
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < r * c) x[i] += s[i / c];
}
__global__ void k_upsample2x(const float *x, float *y, int64_t ch, int64_t h, int64_t w) {  // nearest, [ch, h, w] -> [ch, 2h, 2w]
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x, h2 = 2 * h, w2 = 2 * w;
    if (i >= ch * h2 * w2) return;
    int64_t c = i / (h2 * w2), yy = (i / w2) % h2, xx = i % w2;
    y[i] = x[(c * h + yy / 2) * w + xx / 2];
}
extern "C" void rope_cuda_transpose(const float *dx, float *dy, int64_t r, int64_t c) {
    dim3 g(blocks(c, 32), blocks(r, 32)), b(32, 8); k_transpose<<<g, b>>>(dx, dy, r, c); LAUNCH_CHECK("transpose");
}
extern "C" void rope_cuda_add(float *dx, const float *dy, int64_t n) { k_add<<<blocks(n, 256), 256>>>(dx, dy, n); LAUNCH_CHECK("add"); }
extern "C" void rope_cuda_mul(float *dx, const float *dy, int64_t n) { k_mul<<<blocks(n, 256), 256>>>(dx, dy, n); LAUNCH_CHECK("mul"); }
extern "C" void rope_cuda_add_row_scalars(float *dx, const float *ds, int64_t r, int64_t c) { k_add_row_scalars<<<blocks(r * c, 256), 256>>>(dx, ds, r, c); LAUNCH_CHECK("add_row_scalars"); }
extern "C" void rope_cuda_upsample2x(const float *dx, float *dy, int64_t ch, int64_t h, int64_t w) { k_upsample2x<<<blocks(ch * h * w * 4, 256), 256>>>(dx, dy, ch, h, w); LAUNCH_CHECK("upsample2x"); }

// ---------------------------------------------------------------- async uploads (G2b): copy stream + events
// The copy stream is non-blocking, so its transfers overlap with kernels on the default (compute) stream.
static cudaStream_t g_copy = NULL;
extern "C" void *rope_cuda_copy_stream(void) {
    if (!g_copy && !CK(cudaStreamCreateWithFlags(&g_copy, cudaStreamNonBlocking))) return NULL;
    return (void *)g_copy;
}
extern "C" void rope_cuda_upload_async(void *d, const void *h, size_t n, void *stream) {
    CK(cudaMemcpyAsync(d, h, n, cudaMemcpyHostToDevice, (cudaStream_t)stream));
}
extern "C" void *rope_cuda_event_new(void) { cudaEvent_t e = NULL; return CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming)) ? (void *)e : NULL; }
extern "C" void rope_cuda_event_record(void *e, void *stream) { CK(cudaEventRecord((cudaEvent_t)e, (cudaStream_t)stream)); }
extern "C" void rope_cuda_event_sync(void *e) { CK(cudaEventSynchronize((cudaEvent_t)e)); }
/* the compute (default) stream waits for e before its next kernels */
extern "C" void rope_cuda_compute_wait(void *e) { CK(cudaStreamWaitEvent(0, (cudaEvent_t)e, 0)); }
/* the copy stream waits for all compute work queued so far (a buffer may be reused only after its readers ran) */
extern "C" void rope_cuda_copy_after_compute(void *e, void *stream) {
    CK(cudaEventRecord((cudaEvent_t)e, 0));
    CK(cudaStreamWaitEvent((cudaStream_t)stream, (cudaEvent_t)e, 0));
}
/* a further non-blocking stream (block-ahead staging uses its own, so the main thread's upload fence never delays it) */
extern "C" void *rope_cuda_new_stream(void) { cudaStream_t s = NULL; return CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)) ? (void *)s : NULL; }
