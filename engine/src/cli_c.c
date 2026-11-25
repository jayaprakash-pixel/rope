/* C entry points called from the Rust CLI (engine/rust/main.rs): kernel benchmarks and the hardware probe. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "kernels.h"
#include "probe.h"
#ifdef ROPE_CUDA
#include "cuda_api.h"
#endif
#include <omp.h>

int rope_c_has_cuda(void) {
#ifdef ROPE_CUDA
    return 1;
#else
    return 0;
#endif
}

int rope_c_probe(const char *dir, int disk_mb) {
    rope_probe p;
    if (rope_probe_run(&p, dir, disk_mb)) { fprintf(stderr, "probe failed\n"); return 1; }
    rope_probe_print_json(&p);
    return 0;
}

/* rope bench M K N type: best-of-5 GFLOPS of y = x W^T (2*M*K*N flops). */
/* Device speed for the planner: FP32 GEMM GFLOPS on a 512x2048x2048 problem (4.3 GFLOP), 1 warm-up + best of 4 (~0.1 s).
   Large enough that timer noise and thread wake-up do not dominate. */
double rope_c_gemm_gflops(void) {
    const int64_t M = 512, K = 2048, N = 2048;
    float *x = malloc((size_t)(M * K) * 4), *w = malloc((size_t)(N * K) * 4), *y = malloc((size_t)(M * N) * 4);
    if (!x || !w || !y) { free(x); free(w); free(y); return 0; }
    for (int64_t i = 0; i < M * K; i++) x[i] = (float)(i % 13) * 0.01f;
    for (int64_t i = 0; i < N * K; i++) w[i] = (float)(i % 7) * 0.01f;
    double best = 1e30;
    for (int it = 0; it < 5; it++) { double t0 = omp_get_wtime(); rope_matmul(x, M, K, w, RT_F32, N, y); double dt = omp_get_wtime() - t0; if (it && dt < best) best = dt; }
    free(x); free(w); free(y);
    return 2.0 * M * K * N / best / 1e9;
}

int rope_c_bench(int64_t M, int64_t K, int64_t N, int t) {
    if (!rope_wtype_supported(t) || K % rope_wtype_block(t)) { fprintf(stderr, "bad type/K\n"); return 2; }
    size_t rb = rope_row_bytes(t, K);
    float *x = malloc((size_t)(M * K) * 4), *y = malloc((size_t)(M * N) * 4); unsigned char *w = malloc(rb * N);
    for (int64_t i = 0; i < M * K; i++) x[i] = (float)((i * 2654435761u) % 1000) / 1000.0f - 0.5f;
    for (size_t i = 0; i < rb * N; i++) w[i] = (unsigned char)(i * 131);
    if (t == RT_F32) for (int64_t i = 0; i < N * K; i++) ((float *)w)[i] = 0.001f * (float)(i % 97);
    if (t != RT_F32) for (int64_t r = 0; r < N; r++) for (size_t b = 0; b < rb; b += rope_row_bytes(t, rope_wtype_block(t))) {
        uint16_t d = rope_f2h(0.01f); memcpy(w + r * rb + b, &d, 2); if (t == RT_Q4_K) memcpy(w + r * rb + b + 2, &d, 2); }
    double best = 1e30;
    for (int it = 0; it < 6; it++) { double t0 = omp_get_wtime(); rope_matmul(x, M, K, w, t, N, y); double dt = omp_get_wtime() - t0; if (it && dt < best) best = dt; }
    printf("matmul M=%lld K=%lld N=%lld type=%d threads=%d: %.2f ms, %.1f GFLOPS\n", (long long)M, (long long)K, (long long)N, t,
           rope_get_threads(), best * 1e3, 2.0 * M * K * N / best / 1e9);
    free(x); free(y); free(w);
    return 0;
}

#ifdef ROPE_CUDA
/* rope bench-cuda M K N type mode: weights prepared once (not timed), best of 5 matmuls incl. activation quantization. */
int rope_c_bench_cuda(int64_t M, int64_t K, int64_t N, int t, int mode) {
    if (!rope_wtype_supported(t) || K % rope_wtype_block(t)) { fprintf(stderr, "bad type/K\n"); return 2; }
    size_t rb = rope_row_bytes(t, K);
    float *x = malloc((size_t)(M * K) * 4); unsigned char *w = malloc(rb * N);
    for (int64_t i = 0; i < M * K; i++) x[i] = (float)((i * 2654435761u) % 1000) / 1000.0f - 0.5f;
    for (size_t i = 0; i < rb * N; i++) w[i] = (unsigned char)(i * 131);
    if (t == RT_F32) for (int64_t i = 0; i < N * K; i++) ((float *)w)[i] = 0.001f * (float)(i % 97);
    if (t == RT_F16) for (int64_t i = 0; i < N * K; i++) ((uint16_t *)w)[i] = rope_f2h(0.001f * (float)(i % 97));
    if (t != RT_F32 && t != RT_F16) for (int64_t r = 0; r < N; r++) for (size_t b = 0; b < rb; b += rope_row_bytes(t, rope_wtype_block(t))) {
        uint16_t d = rope_f2h(0.01f); memcpy(w + r * rb + b + (t == RT_Q6_K ? 208 : 0), &d, 2); if (t == RT_Q4_K) memcpy(w + r * rb + b + 2, &d, 2); }
    rope_cuda_w *cw = rope_cuda_prepare(w, t, N, K, mode);
    float *dx = rope_cuda_alloc((size_t)(M * K) * 4), *dy = rope_cuda_alloc((size_t)(M * N) * 4);
    if (!cw || !dx || !dy) { fprintf(stderr, "cuda: %s\n", rope_cuda_last_error()); return 1; }
    rope_cuda_upload(dx, x, (size_t)(M * K) * 4);
    double best = 1e30;
    for (int it = 0; it < 6; it++) {
        rope_cuda_sync(); double t0 = omp_get_wtime();
        rope_cuda_matmul(dx, M, cw, dy); rope_cuda_sync();
        double dt = omp_get_wtime() - t0; if (it && dt < best) best = dt;
    }
    const char *e = rope_cuda_last_error();
    static const char *MODES[] = {"f32", "f16", "dp4a"};
    printf("cuda matmul M=%lld K=%lld N=%lld type=%d mode=%s: %.2f ms, %.1f GFLOPS%s%s\n", (long long)M, (long long)K, (long long)N, t,
           MODES[rope_cuda_mode(cw)], best * 1e3, 2.0 * M * K * N / best / 1e9, e ? "  ERROR " : "", e ? e : "");
    rope_cuda_release(cw); rope_cuda_free(dx); rope_cuda_free(dy); free(x); free(w);
    return e ? 1 : 0;
}
#endif

