/* CUDA kernel tests (P1.5.4): every CUDA kernel against the float64-verified CPU kernel, max rel err <= 1e-2. */
#include "test.h"
#include "kernels.h"
#include "cuda_api.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rng = 0x2545F4914F6CDD1Dull;
static uint32_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)(rng >> 32); }
static float rndf(void) { return (float)rnd() / 4294967296.0f * 2.0f - 1.0f; }
static void fill(float *x, int64_t n, float s) { for (int64_t i = 0; i < n; i++) x[i] = rndf() * s; }

static double rel(const float *a, const float *b, int64_t n) {   /* a = CUDA, b = CPU reference */
    double e = 0, m = 0;
    for (int64_t i = 0; i < n; i++) { double d = fabs((double)a[i] - b[i]); e = d > e ? d : e; m = fabs(b[i]) > m ? fabs(b[i]) : m; }
    return m > 0 ? e / m : e;
}
#define CTOL 1e-2
#define CHECK_CUDA_REL(a, b, n, what) do { double e_ = rel(a, b, n); const char *ce_ = rope_cuda_last_error(); \
    if (ce_) fprintf(stderr, "  %s: CUDA error %s\n", what, ce_); \
    if (!(e_ <= CTOL) || getenv("ROPE_TEST_VERBOSE")) fprintf(stderr, "  %s: rel err %.3g\n", what, e_); CHECK(!ce_ && e_ <= CTOL); } while (0)

static float *up(const float *h, int64_t n) { float *d = rope_cuda_alloc((size_t)n * 4); rope_cuda_upload(d, h, (size_t)n * 4); return d; }
static void down(float *h, const float *d, int64_t n) { rope_cuda_sync(); rope_cuda_download(h, d, (size_t)n * 4); }

static uint8_t *make_w(int t, int64_t n, int64_t k) {
    size_t rb = rope_row_bytes(t, k); uint8_t *w = malloc(rb * n);
    for (int64_t r = 0; r < n; r++) {
        uint8_t *p = w + r * rb;
        if (t == RT_F32) for (int64_t i = 0; i < k; i++) { float f = rndf(); memcpy(p + 4 * i, &f, 4); }
        else if (t == RT_F16) for (int64_t i = 0; i < k; i++) { uint16_t h = rope_f2h(rndf()); memcpy(p + 2 * i, &h, 2); }
        else {
            for (size_t i = 0; i < rb; i++) p[i] = (uint8_t)rnd();
            int bs = t == RT_Q4_K ? 144 : t == RT_Q6_K ? 210 : t == RT_Q8_0 ? 34 : 18;
            for (size_t b = 0; b < rb; b += bs) {
                uint16_t d = rope_f2h(0.002f + 0.02f * (rnd() / 4294967296.0f)); memcpy(p + b + (t == RT_Q6_K ? 208 : 0), &d, 2);
                if (t == RT_Q4_K) { uint16_t m = rope_f2h(0.001f + 0.01f * (rnd() / 4294967296.0f)); memcpy(p + b + 2, &m, 2); }
            }
        }
    }
    return w;
}

static void check_mm(int t, int mode, int64_t M, int64_t K, int64_t N) {
    float *x = malloc(M * K * 4), *ref = malloc(M * N * 4), *got = malloc(M * N * 4);
    fill(x, M * K, 1.0f);
    uint8_t *w = make_w(t, N, K);
    rope_matmul(x, M, K, w, t, N, ref);
    rope_cuda_w *cw = rope_cuda_prepare(w, t, N, K, mode);
    float *dx = up(x, M * K), *dy = rope_cuda_alloc((size_t)(M * N) * 4);
    rope_cuda_matmul(dx, M, cw, dy); down(got, dy, M * N);
    char what[96]; snprintf(what, sizeof what, "cuda matmul t=%d mode=%d(%d) %lldx%lldx%lld", t, mode, rope_cuda_mode(cw), (long long)M, (long long)K, (long long)N);
    CHECK_CUDA_REL(got, ref, M * N, what);
    rope_cuda_release(cw); rope_cuda_free(dx); rope_cuda_free(dy); free(x); free(ref); free(got); free(w);
}

static void test_cuda_matmul(void) {
    const int types[] = {RT_F32, RT_F16, RT_Q8_0, RT_Q4_0, RT_Q4_K, RT_Q6_K};
    const int64_t shapes[][3] = {{1, 256, 33}, {37, 768, 129}, {64, 1024, 96}, {130, 512, 200}};
    for (int ti = 0; ti < 6; ti++)
        for (int mode = 0; mode < 3; mode++) {
            if (mode == RC_DP4A && types[ti] <= RT_F16) continue;
            for (int s = 0; s < 4; s++) { check_mm(types[ti], mode, shapes[s][0], shapes[s][1], shapes[s][2]); if (rope_test_failures) return; }
        }
}

static void test_cuda_conv(void) {
    const int C = 6, H = 12, W = 10, O = 9, k = 3, stride = 2, pad = 1;
    int Ho = (H + 2 * pad - k) / stride + 1, Wo = (W + 2 * pad - k) / stride + 1;
    float *in = malloc(C * H * W * 4), *w = malloc(O * C * k * k * 4), *b = malloc(O * 4), *ref = malloc(O * Ho * Wo * 4), *got = malloc(O * Ho * Wo * 4);
    fill(in, C * H * W, 1); fill(w, O * C * k * k, 0.5f); fill(b, O, 0.5f);
    rope_conv2d(in, C, H, W, w, RT_F32, b, O, k, k, stride, pad, ref);
    for (int mode = 0; mode < 2; mode++) {
        rope_cuda_w *cw = rope_cuda_prepare(w, RT_F32, O, C * k * k, mode);
        float *din = up(in, C * H * W), *db = up(b, O), *dout = rope_cuda_alloc((size_t)O * Ho * Wo * 4);
        rope_cuda_conv2d(din, C, H, W, cw, db, k, k, stride, pad, dout); down(got, dout, O * Ho * Wo);
        CHECK_CUDA_REL(got, ref, O * Ho * Wo, mode ? "cuda conv2d f16" : "cuda conv2d f32");
        rope_cuda_release(cw); rope_cuda_free(din); rope_cuda_free(db); rope_cuda_free(dout);
    }
    free(in); free(w); free(b); free(ref); free(got);
}

static void test_cuda_attention(void) {
    const int nh = 2, d = 24; const int64_t nq = 37, nk = 53, st = nh * d;
    float *q = malloc(nq * st * 4), *k = malloc(nk * st * 4), *v = malloc(nk * st * 4), *ref = malloc(nq * st * 4), *got = malloc(nq * st * 4);
    fill(q, nq * st, 2); fill(k, nk * st, 2); fill(v, nk * st, 1);
    float *dq = up(q, nq * st), *dk = up(k, nk * st), *dv = up(v, nk * st), *dout = rope_cuda_alloc((size_t)(nq * st) * 4);
    for (int h = 0; h < nh; h++) {
        rope_attention(q + h * d, k + h * d, v + h * d, ref + h * d, nq, nk, d, st, st, st, st, 1.0f / sqrtf((float)d));
        rope_cuda_attention(dq + h * d, dk + h * d, dv + h * d, dout + h * d, nq, nk, d, st, st, st, st, 1.0f / sqrtf((float)d));
    }
    down(got, dout, nq * st);
    CHECK_CUDA_REL(got, ref, nq * st, "cuda attention");
    rope_cuda_free(dq); rope_cuda_free(dk); rope_cuda_free(dv); rope_cuda_free(dout); free(q); free(k); free(v); free(ref); free(got);
}

static void test_cuda_elementwise(void) {
    const int64_t m = 13, n = 3000, N = m * n; const int C = 32, HW = 75, G = 8;
    float *x = malloc(N * 4), *ref = malloc(N * 4), *got = malloc(N * 4), *g = malloc(n * 4), *b = malloc(n * 4);
    fill(g, n, 1); fill(b, n, 1);
    float *dg = up(g, n), *db = up(b, n);
#define RUN(cpu_call, cuda_call, count, what) do { fill(x, count, 4); memcpy(ref, x, (count) * 4); cpu_call; \
        float *dx = up(x, count); cuda_call; down(got, dx, count); rope_cuda_free(dx); CHECK_CUDA_REL(got, ref, count, what); } while (0)
    RUN(rope_softmax_rows(ref, m, n), rope_cuda_softmax_rows(dx, m, n), N, "cuda softmax");
    RUN(rope_layernorm(ref, m, n, g, b, 1e-5f), rope_cuda_layernorm(dx, m, n, dg, db, 1e-5f), N, "cuda layernorm");
    RUN(rope_rmsnorm(ref, m, n, g, 1e-6f), rope_cuda_rmsnorm(dx, m, n, dg, 1e-6f), N, "cuda rmsnorm");
    RUN(rope_groupnorm(ref, C, HW, G, g, b, 1e-6f), rope_cuda_groupnorm(dx, C, HW, G, dg, db, 1e-6f), C * HW, "cuda groupnorm");
    RUN(rope_gelu_erf(ref, N), rope_cuda_gelu_erf(dx, N), N, "cuda gelu_erf");
    RUN(rope_gelu_tanh(ref, N), rope_cuda_gelu_tanh(dx, N), N, "cuda gelu_tanh");
    RUN(rope_silu(ref, N), rope_cuda_silu(dx, N), N, "cuda silu");
    /* rotary, both modes */
    const int64_t nt = 11; const int nh = 3, d = 16; const int64_t st = nh * d + 5;
    float pos[11], *ct = malloc(nt * d / 2 * 4), *sn = malloc(nt * d / 2 * 4);
    for (int t = 0; t < nt; t++) pos[t] = (float)(t * 3 + 1);
    rope_rotary_tables(pos, nt, d, 10000.0f, ct, sn);
    float *dct = up(ct, nt * d / 2), *dsn = up(sn, nt * d / 2);
    for (int neox = 0; neox < 2; neox++)
        RUN(rope_rotary(ref, nt, nh, d, st, ct, sn, neox), rope_cuda_rotary(dx, nt, nh, d, st, dct, dsn, neox), nt * st, neox ? "cuda rotary neox" : "cuda rotary");
#undef RUN
    rope_cuda_free(dg); rope_cuda_free(db); rope_cuda_free(dct); rope_cuda_free(dsn);
    free(x); free(ref); free(got); free(g); free(b); free(ct); free(sn);
}

void rope_register_cuda_tests(void (*reg)(const char *, void (*)(void))) {
    reg("cuda_matmul", test_cuda_matmul);
    reg("cuda_conv2d", test_cuda_conv);
    reg("cuda_attention", test_cuda_attention);
    reg("cuda_elementwise", test_cuda_elementwise);
}
