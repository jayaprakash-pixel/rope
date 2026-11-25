/* Kernel tests: every kernel against an independent float64 reference on random inputs.
 * Pass rule (P1.5.3): max|got - ref| / max|ref| <= 1e-3. Quantized weights are compared with the exact
 * float64 value of the same quantized weights, using a dequantizer written separately here from the spec. */
#include "test.h"
#include "kernels.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint32_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)(rng >> 32); }
static float rndf(void) { return (float)rnd() / 4294967296.0f * 2.0f - 1.0f; }   /* [-1, 1) */
static void fill(float *x, int64_t n, float s) { for (int64_t i = 0; i < n; i++) x[i] = rndf() * s; }

static double rel_err(const float *got, const double *ref, int64_t n) {
    double e = 0, m = 0;
    for (int64_t i = 0; i < n; i++) { double d = fabs(got[i] - ref[i]); e = d > e ? d : e; m = fabs(ref[i]) > m ? fabs(ref[i]) : m; }
    return m > 0 ? e / m : e;
}
#define TOL 1e-3
#define CHECK_REL(got, ref, n, what) do { double e_ = rel_err(got, ref, n); \
    if (!(e_ <= TOL)) fprintf(stderr, "  %s: rel err %.3g\n", what, e_); CHECK(e_ <= TOL); } while (0)

/* ---- independent float64 dequantizer (spec re-implementation) ---- */
static double h2d(const uint8_t *p) {
    unsigned h = p[0] | (p[1] << 8), e = (h >> 10) & 31, m = h & 1023;
    double v = e == 0 ? ldexp(m, -24) : ldexp(1024 + m, (int)e - 25);
    return (h & 0x8000) ? -v : v;
}
static void ref_dequant(int t, const uint8_t *p, double *y, int64_t k) {
    if (t == RT_F32) { for (int64_t i = 0; i < k; i++) { float f; memcpy(&f, p + 4 * i, 4); y[i] = f; } return; }
    if (t == RT_F16) { for (int64_t i = 0; i < k; i++) y[i] = h2d(p + 2 * i); return; }
    for (int64_t b = 0; b < k / rope_wtype_block(t); b++) {
        if (t == RT_Q8_0) { const uint8_t *q = p + b * 34; for (int i = 0; i < 32; i++) y[b * 32 + i] = h2d(q) * (int8_t)q[2 + i]; }
        if (t == RT_Q4_0) { const uint8_t *q = p + b * 18;
            for (int i = 0; i < 32; i++) y[b * 32 + i] = h2d(q) * (((i < 16 ? q[2 + i] : q[2 + i - 16] >> 4) & 15) - 8); }
        if (t == RT_Q4_K) {
            const uint8_t *q = p + b * 144, *sc = q + 4, *qs = q + 16;
            for (int i = 0; i < 256; i++) {
                int j = i / 32, l = i % 32, s, m;          /* sub-block j, element l */
                if (j < 4) { s = sc[j] & 63; m = sc[j + 4] & 63; }
                else { s = (sc[j + 4] & 15) | ((sc[j - 4] >> 6) << 4); m = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4); }
                int byte = qs[(j / 2) * 32 + l], nib = (j % 2) ? byte >> 4 : byte & 15;
                y[b * 256 + i] = h2d(q) * s * nib - h2d(q + 2) * m;
            }
        }
        if (t == RT_Q6_K) {   /* element i of 256: half h=i/128, r=i%128, quarter g=r/32, l=r%32 */
            const uint8_t *q = p + b * 210, *ql = q, *qh = q + 128; const int8_t *sc = (const int8_t *)(q + 192);
            for (int i = 0; i < 256; i++) {
                int h = i / 128, r = i % 128, g = r / 32, l = r % 32;
                int lo = (g & 1) ? ql[h * 64 + 32 + l] : ql[h * 64 + l];
                lo = (g < 2) ? lo & 15 : lo >> 4;
                int hi = (qh[h * 32 + l] >> (2 * g)) & 3;
                y[b * 256 + i] = h2d(q + 208) * sc[h * 8 + g * 2 + l / 16] * ((lo | (hi << 4)) - 32);
            }
        }
    }
}

/* random weights in format t: n rows x k */
static uint8_t *make_weights(int t, int64_t n, int64_t k) {
    size_t rb = rope_row_bytes(t, k); uint8_t *w = malloc(rb * n);
    for (int64_t r = 0; r < n; r++) {
        uint8_t *p = w + r * rb;
        if (t == RT_F32) for (int64_t i = 0; i < k; i++) { float f = rndf(); memcpy(p + 4 * i, &f, 4); }
        else if (t == RT_F16) for (int64_t i = 0; i < k; i++) { uint16_t h = rope_f2h(rndf()); memcpy(p + 2 * i, &h, 2); }
        else {
            for (size_t i = 0; i < rb; i++) p[i] = (uint8_t)rnd();
            int bs = t == RT_Q4_K ? 144 : t == RT_Q6_K ? 210 : t == RT_Q8_0 ? 34 : 18;
            for (size_t b = 0; b < rb; b += bs) {       /* sane positive fp16 scales (Q6_K keeps d at the end) */
                uint16_t d = rope_f2h(0.002f + 0.02f * (rnd() / 4294967296.0f)); memcpy(p + b + (t == RT_Q6_K ? 208 : 0), &d, 2);
                if (t == RT_Q4_K) { uint16_t m = rope_f2h(0.001f + 0.01f * (rnd() / 4294967296.0f)); memcpy(p + b + 2, &m, 2); }
            }
        }
    }
    return w;
}

static void check_matmul(int t, int64_t M, int64_t K, int64_t N) {
    float *x = malloc(M * K * 4), *y = malloc(M * N * 4); double *ref = malloc(M * N * 8), *wd = malloc(K * 8);
    fill(x, M * K, 1.0f);
    uint8_t *w = make_weights(t, N, K); size_t rb = rope_row_bytes(t, K);
    rope_matmul(x, M, K, w, t, N, y);
    /* int8 path: the reference takes the activations as that path rounds them (the kernel's own error stays 1e-3) */
    for (int64_t m = 0; m < M; m++) rope_int8_act(t, M, x + m * K, K, x + m * K);
    for (int64_t n = 0; n < N; n++) {
        ref_dequant(t, w + n * rb, wd, K);
        for (int64_t m = 0; m < M; m++) { double s = 0; for (int64_t k = 0; k < K; k++) s += x[m * K + k] * wd[k]; ref[m * N + n] = s; }
    }
    char what[64]; snprintf(what, sizeof what, "matmul t=%d %lldx%lldx%lld", t, (long long)M, (long long)K, (long long)N);
    double e = rel_err(y, ref, M * N);
    if (!(e <= TOL)) fprintf(stderr, "  %s: rel err %.3g\n", what, e);
    free(x); free(y); free(ref); free(wd); free(w);
    CHECK(e <= TOL);
}

static void test_matmul(void) {
    const int types[] = {RT_F32, RT_F16, RT_Q8_0, RT_Q4_0, RT_Q4_K, RT_Q6_K};
    int old = rope_int8_mode(-1);
    for (int mode = 0; mode <= 1; mode++) {                 /* float path, then the int8 path (P2.2.3) */
    rope_int8_mode(mode);
    /* the last two: K above one 4096-wide K-slice of the int8 path, M not a multiple of its 8-row tile */
    const int64_t shapes[][3] = {{1, 256, 33}, {3, 512, 70}, {7, 512, 50}, {37, 768, 129}, {64, 1024, 96}, {13, 9216, 20}, {2, 12288, 9}};
    for (int ti = 0; ti < 6; ti++)
        for (int s = 0; s < 7; s++) { check_matmul(types[ti], shapes[s][0], shapes[s][1], shapes[s][2]); if (rope_test_failures) { rope_int8_mode(old); return; } }
    }
    rope_int8_mode(old);
}

static void check_conv(int C, int H, int W, int O, int k, int stride, int pad) {
    int Ho = (H + 2 * pad - k) / stride + 1, Wo = (W + 2 * pad - k) / stride + 1;
    float *in = malloc(C * H * W * 4), *w = malloc(O * C * k * k * 4), *b = malloc(O * 4), *out = malloc(O * Ho * Wo * 4);
    double *ref = malloc(O * Ho * Wo * 8);
    fill(in, C * H * W, 1.0f); fill(w, O * C * k * k, 0.5f); fill(b, O, 0.5f);
    rope_conv2d(in, C, H, W, w, RT_F32, b, O, k, k, stride, pad, out);
    for (int o = 0; o < O; o++) for (int y = 0; y < Ho; y++) for (int xx = 0; xx < Wo; xx++) {
        double s = b[o];
        for (int c = 0; c < C; c++) for (int ky = 0; ky < k; ky++) for (int kx = 0; kx < k; kx++) {
            int iy = y * stride - pad + ky, ix = xx * stride - pad + kx;
            if (iy >= 0 && iy < H && ix >= 0 && ix < W) s += (double)in[(c * H + iy) * W + ix] * w[((o * C + c) * k + ky) * k + kx];
        }
        ref[(o * Ho + y) * Wo + xx] = s;
    }
    double e = rel_err(out, ref, O * Ho * Wo);
    if (!(e <= TOL)) fprintf(stderr, "  conv %d,%d,%d,%d k%d s%d p%d: rel err %.3g\n", C, H, W, O, k, stride, pad, e);
    free(in); free(w); free(b); free(out); free(ref);
    CHECK(e <= TOL);
}

static void test_conv2d(void) { check_conv(5, 9, 11, 7, 3, 1, 1); check_conv(6, 12, 10, 9, 3, 2, 0); check_conv(4, 8, 8, 3, 1, 1, 0); }

static void test_attention(void) {
    const int nh = 2, d = 24; const int64_t nq = 37, nk = 53, st = nh * d;   /* heads interleaved per token */
    float *q = malloc(nq * st * 4), *k = malloc(nk * st * 4), *v = malloc(nk * st * 4), *o = malloc(nq * st * 4);
    double *ref = malloc(nq * st * 8), *p = malloc(nk * 8);
    fill(q, nq * st, 2.0f); fill(k, nk * st, 2.0f); fill(v, nk * st, 1.0f);
    float scale = 1.0f / sqrtf((float)d);
    for (int h = 0; h < nh; h++) {
        rope_attention(q + h * d, k + h * d, v + h * d, o + h * d, nq, nk, d, st, st, st, st, scale);
        for (int64_t i = 0; i < nq; i++) {
            double mx = -1e300, sum = 0;
            for (int64_t j = 0; j < nk; j++) { double s = 0; for (int c = 0; c < d; c++) s += (double)q[i * st + h * d + c] * k[j * st + h * d + c]; p[j] = s * scale; mx = p[j] > mx ? p[j] : mx; }
            for (int64_t j = 0; j < nk; j++) { p[j] = exp(p[j] - mx); sum += p[j]; }
            for (int c = 0; c < d; c++) { double s = 0; for (int64_t j = 0; j < nk; j++) s += p[j] / sum * v[j * st + h * d + c]; ref[i * st + h * d + c] = s; }
        }
    }
    CHECK_REL(o, ref, nq * st, "attention");
    free(q); free(k); free(v); free(o); free(ref); free(p);
}

static void test_attention_causal(void) {
    const int d = 16; const int64_t n = 29;                 /* nq == nk, query i sees keys 0..i */
    float *q = malloc(n * d * 4), *k = malloc(n * d * 4), *v = malloc(n * d * 4), *o = malloc(n * d * 4);
    double *ref = malloc(n * d * 8), *p = malloc(n * 8);
    fill(q, n * d, 2.0f); fill(k, n * d, 2.0f); fill(v, n * d, 1.0f);
    rope_attention_ex(q, k, v, o, n, n, d, d, d, d, d, 0.25f, 1, NULL);
    for (int64_t i = 0; i < n; i++) {
        double mx = -1e300, sum = 0;
        for (int64_t j = 0; j <= i; j++) { double s = 0; for (int c = 0; c < d; c++) s += (double)q[i * d + c] * k[j * d + c]; p[j] = s * 0.25; mx = p[j] > mx ? p[j] : mx; }
        for (int64_t j = 0; j <= i; j++) { p[j] = exp(p[j] - mx); sum += p[j]; }
        for (int c = 0; c < d; c++) { double s = 0; for (int64_t j = 0; j <= i; j++) s += p[j] / sum * v[j * d + c]; ref[i * d + c] = s; }
    }
    CHECK_REL(o, ref, n * d, "attention causal");
    CHECK(fabs(o[0] - v[0]) < 1e-6);                     /* first query sees only the first key */
    free(q); free(k); free(v); free(o); free(ref); free(p);
}

static void test_norms_activations(void) {
    const int64_t m = 13, n = 300, N = m * n; const int C = 32, HW = 75, G = 8;
    float *x = malloc(N * 4), *g = malloc(n * 4), *b = malloc(n * 4); double *ref = malloc(N * 8);
    fill(g, n, 1.0f); fill(b, n, 1.0f);
    /* softmax */
    fill(x, N, 4.0f);
    for (int64_t r = 0; r < m; r++) { double mx = -1e300, s = 0;
        for (int64_t i = 0; i < n; i++) mx = x[r * n + i] > mx ? x[r * n + i] : mx;
        for (int64_t i = 0; i < n; i++) s += exp(x[r * n + i] - mx);
        for (int64_t i = 0; i < n; i++) ref[r * n + i] = exp(x[r * n + i] - mx) / s; }
    rope_softmax_rows(x, m, n); CHECK_REL(x, ref, N, "softmax");
    /* layernorm */
    fill(x, N, 3.0f);
    for (int64_t r = 0; r < m; r++) { double s = 0, s2 = 0;
        for (int64_t i = 0; i < n; i++) s += x[r * n + i];
        double mu = s / n; for (int64_t i = 0; i < n; i++) s2 += (x[r * n + i] - mu) * (x[r * n + i] - mu);
        double inv = 1 / sqrt(s2 / n + 1e-5);
        for (int64_t i = 0; i < n; i++) ref[r * n + i] = (x[r * n + i] - mu) * inv * g[i] + b[i]; }
    rope_layernorm(x, m, n, g, b, 1e-5f); CHECK_REL(x, ref, N, "layernorm");
    /* rmsnorm */
    fill(x, N, 3.0f);
    for (int64_t r = 0; r < m; r++) { double s2 = 0; for (int64_t i = 0; i < n; i++) s2 += (double)x[r * n + i] * x[r * n + i];
        double inv = 1 / sqrt(s2 / n + 1e-6); for (int64_t i = 0; i < n; i++) ref[r * n + i] = x[r * n + i] * inv * g[i]; }
    rope_rmsnorm(x, m, n, g, 1e-6f); CHECK_REL(x, ref, N, "rmsnorm");
    /* groupnorm over [C, HW] */
    fill(x, C * HW, 3.0f);
    for (int gi = 0; gi < G; gi++) { int cpg = C / G; double s = 0, s2 = 0; int64_t base = (int64_t)gi * cpg * HW, cnt = (int64_t)cpg * HW;
        for (int64_t i = 0; i < cnt; i++) s += x[base + i];
        double mu = s / cnt; for (int64_t i = 0; i < cnt; i++) s2 += (x[base + i] - mu) * (x[base + i] - mu);
        double inv = 1 / sqrt(s2 / cnt + 1e-6);
        for (int64_t i = 0; i < cnt; i++) { int c = gi * cpg + (int)(i / HW); ref[base + i] = (x[base + i] - mu) * inv * g[c] + b[c]; } }
    rope_groupnorm(x, C, HW, G, g, b, 1e-6f); CHECK_REL(x, ref, C * HW, "groupnorm");
    /* activations */
    fill(x, N, 5.0f); for (int64_t i = 0; i < N; i++) ref[i] = 0.5 * x[i] * (1 + erf(x[i] / sqrt(2.0)));
    rope_gelu_erf(x, N); CHECK_REL(x, ref, N, "gelu_erf");
    fill(x, N, 5.0f); for (int64_t i = 0; i < N; i++) { double v = x[i]; ref[i] = 0.5 * v * (1 + tanh(sqrt(2 / 3.14159265358979323846) * (v + 0.044715 * v * v * v))); }
    rope_gelu_tanh(x, N); CHECK_REL(x, ref, N, "gelu_tanh");
    fill(x, N, 5.0f); for (int64_t i = 0; i < N; i++) ref[i] = x[i] / (1 + exp(-(double)x[i]));
    rope_silu(x, N); CHECK_REL(x, ref, N, "silu");
    free(x); free(g); free(b); free(ref);
}

static void test_rotary(void) {
    const int64_t nt = 11; const int nh = 3, d = 16; const int64_t st = nh * d + 5;   /* padded token stride */
    float pos[11], *x = malloc(nt * st * 4), *ct = malloc(nt * d / 2 * 4), *sn = malloc(nt * d / 2 * 4);
    double *ref = malloc(nt * st * 8);
    for (int t = 0; t < nt; t++) pos[t] = (float)(t * 3 + 1);
    rope_rotary_tables(pos, nt, d, 10000.0f, ct, sn);
    for (int neox = 0; neox < 2; neox++) {
        fill(x, nt * st, 2.0f);
        for (int64_t i = 0; i < nt * st; i++) ref[i] = x[i];
        for (int64_t t = 0; t < nt; t++) for (int h = 0; h < nh; h++) for (int i = 0; i < d / 2; i++) {
            double a = pos[t] * pow(10000.0, -2.0 * i / d); int64_t base = t * st + h * d;
            int ia = neox ? i : 2 * i, ib = neox ? i + d / 2 : 2 * i + 1;
            double x0 = x[base + ia], x1 = x[base + ib];
            ref[base + ia] = x0 * cos(a) - x1 * sin(a); ref[base + ib] = x0 * sin(a) + x1 * cos(a);
        }
        rope_rotary(x, nt, nh, d, st, ct, sn, neox);
        CHECK_REL(x, ref, nt * st, neox ? "rotary neox" : "rotary interleaved");
    }
    free(x); free(ct); free(sn); free(ref);
}

void rope_register_kernel_tests(void (*reg)(const char *, void (*)(void))) {
    reg("kernel_matmul", test_matmul);
    reg("kernel_conv2d", test_conv2d);
    reg("kernel_attention", test_attention);
    reg("kernel_attention_causal", test_attention_causal);
    reg("kernel_norms_activations", test_norms_activations);
    reg("kernel_rotary", test_rotary);
}
