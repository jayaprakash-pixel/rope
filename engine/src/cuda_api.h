/* rope-engine CUDA kernels (sm_61 / sm_75, no tensor cores needed). All pointers named d* are device memory.
 * Weights are prepared once (uploaded + converted to the mode's native layout), then used by matmul/conv2d. */
#pragma once
#include <stddef.h>
#include <stdint.h>

enum { RC_F32 = 0,      /* weights dequantized to float32, FP32 FMA tiles            */
       RC_F16 = 1,      /* weights in float16, half2 FMA (HFMA2) with FP32 summation  */
       RC_DP4A = 2 };   /* quantized weights as int8 + per-16 scale/offset, dp4a      */

int   rope_cuda_device_count(void);
const char *rope_cuda_last_error(void);     /* NULL if no error since the last call */
void *rope_cuda_alloc(size_t n);
void  rope_cuda_free(void *d);
void  rope_cuda_upload(void *d, const void *h, size_t n);
void  rope_cuda_download(void *h, const void *d, size_t n);
void  rope_cuda_sync(void);

typedef struct rope_cuda_w rope_cuda_w;
/* w_host: N rows of K elements in GGUF type wtype. RC_DP4A needs a quantized type (else falls back to RC_F16). */
rope_cuda_w *rope_cuda_prepare(const void *w_host, int wtype, int64_t N, int64_t K, int mode);
void rope_cuda_release(rope_cuda_w *w);
int  rope_cuda_mode(const rope_cuda_w *w);

void rope_cuda_matmul(const float *dx, int64_t M, const rope_cuda_w *w, float *dy);   /* dy[M,N] = dx[M,K] W^T */
/* w prepared with N = O output channels and K = C*kh*kw */
void rope_cuda_conv2d(const float *din, int C, int H, int W, const rope_cuda_w *w, const float *dbias,
                      int kh, int kw, int stride, int pad, float *dout);
void rope_cuda_attention(const float *dq, const float *dk, const float *dv, float *dout, int64_t nq, int64_t nk, int d,
                         int64_t qs, int64_t ks, int64_t vs, int64_t os, float scale);
void rope_cuda_softmax_rows(float *dx, int64_t m, int64_t n);
void rope_cuda_layernorm(float *dx, int64_t m, int64_t n, const float *dg, const float *db, float eps);
void rope_cuda_rmsnorm(float *dx, int64_t m, int64_t n, const float *dg, float eps);
void rope_cuda_groupnorm(float *dx, int64_t c, int64_t hw, int groups, const float *dg, const float *db, float eps);
void rope_cuda_gelu_erf(float *dx, int64_t n);
void rope_cuda_gelu_tanh(float *dx, int64_t n);
void rope_cuda_silu(float *dx, int64_t n);
void rope_cuda_rotary(float *dx, int64_t n_tok, int n_head, int d_head, int64_t tok_stride,
                      const float *dcos, const float *dsin, int neox);

/* caller-owned buffers (VRAM manager): pinned host memory, free VRAM, dequantize into a given buffer,
   GEMM / conv2d on raw device weight pointers (mode RC_F32 or RC_F16 layout) */
void  *rope_cuda_host_alloc(size_t n);
void   rope_cuda_host_free(void *h);
size_t rope_cuda_free_bytes(void);
void   rope_cuda_dequant(const void *d_raw, int wtype, int64_t N, int64_t K, int mode, void *d_out);
void   rope_cuda_gemm(const float *dx, int64_t M, int64_t N, int64_t K, int mode, const void *dw, float *dy);
void   rope_cuda_conv2d_raw(const float *din, int C, int H, int W, int O, int mode, const void *dw, const float *dbias,
                            int kh, int kw, int stride, int pad, float *dout);

/* light ops on device-resident activations (G3) */
void rope_cuda_modulate(const float *dx, float *dy, int64_t m, int64_t n, const float *dshift, const float *dscale, float eps);
void rope_cuda_add_gated(float *dx, const float *dgate, const float *dy, int64_t m, int64_t n);
void rope_cuda_add_rows(float *dy, const float *db, int64_t m, int64_t n);
void rope_cuda_qk_norm_rope(float *dbuf, int64_t rows, int64_t rs, int D, int heads, int hd, const float *dqs, const float *dks,
                            const float *dcos, const float *dsin, int64_t row0);
void rope_cuda_copy2d(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t rows);
void rope_cuda_transpose(const float *dx, float *dy, int64_t r, int64_t c);      /* dy[c, r] = dx[r, c] */
void rope_cuda_add(float *dx, const float *dy, int64_t n);                      /* dx += dy */
void rope_cuda_mul(float *dx, const float *dy, int64_t n);                      /* dx *= dy */
void rope_cuda_add_row_scalars(float *dx, const float *ds, int64_t r, int64_t c); /* dx[i, :] += ds[i] */
void rope_cuda_upsample2x(const float *dx, float *dy, int64_t ch, int64_t h, int64_t w);
/* async uploads (G2b): non-blocking copy stream, events */
void *rope_cuda_copy_stream(void);
void  rope_cuda_upload_async(void *d, const void *h, size_t n, void *stream);
void *rope_cuda_event_new(void);
void  rope_cuda_event_record(void *e, void *stream);
void  rope_cuda_event_sync(void *e);
void  rope_cuda_compute_wait(void *e);
void  rope_cuda_copy_after_compute(void *e, void *stream);
void *rope_cuda_new_stream(void);
