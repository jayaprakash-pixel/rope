/* rope-engine CPU kernels (OpenMP; AVX2 + FMA or SSE4.1, chosen at run time: vf8.h, dispatch.c). Activations are float32.
 * Weight matrices are stored as N rows of K elements (GGUF / PyTorch [out, in] layout), in one of the
 * weight types below (ids are the GGUF ggml type ids, so file tensors can be used directly). */
#pragma once
#include <stddef.h>
#include <stdint.h>

enum { RT_F32 = 0, RT_F16 = 1, RT_Q4_0 = 2, RT_Q8_0 = 8, RT_Q4_K = 12, RT_Q6_K = 14, RT_BF16 = 30 };

/* CPU path of this process: 2 = AVX2 + FMA + F16C, 1 = SSE4.1 (ROPE_CPU=sse41|avx2 forces one) */
int         rope_cpu_isa(void);
const char *rope_cpu_isa_name(void);

float    rope_h2f(uint16_t h);
uint16_t rope_f2h(float f);
float    rope_bf2f(uint16_t h);

int    rope_wtype_supported(int wtype);
int    rope_wtype_block(int wtype);                   /* elements per block */
size_t rope_row_bytes(int wtype, int64_t k);          /* bytes of one row of k elements */
void   rope_dequant_row(int wtype, const void *src, float *dst, int64_t k);

void rope_set_threads(int n);                         /* 0 = physical cores */
int  rope_get_threads(void);

/* y[M,N] = x[M,K] . W[N,K]^T   (x, y row-major, contiguous) */
void rope_matmul(const float *x, int64_t M, int64_t K, const void *w, int wtype, int64_t N, float *y);
/* Integer-dot path for quantized weights (Q4_K, Q6_K; Q4_0, Q8_0 when M < 4) with int8 activations, AVX2 only.
   Off by default (image models lose accuracy, see int8_dot.h); rope_int8_mode(m) sets 0 off / 1 all / 2 decode /
   3 batch and returns the previous mode (m < 0: query); ROPE_INT8 sets the start mode. rope_int8_act writes one
   activation row as that path rounds it (for exact tests) and returns 1, else returns 0. */
int  rope_int8_act(int wtype, int64_t M, const float *x, int64_t k, float *xr);
int  rope_int8_mode(int mode);

/* out[O,Ho,Wo] = conv2d(in[C,H,W], w[O, C*kh*kw]) + bias[O] (bias may be NULL) */
void rope_conv2d(const float *in, int C, int H, int W, const void *w, int wtype, const float *bias,
                 int O, int kh, int kw, int stride, int pad, float *out);

/* One attention head: o = softmax(scale * q k^T) v.  Row i of q is q[i*qs .. i*qs+d), likewise k, v, o. */
void rope_attention(const float *q, const float *k, const float *v, float *o, int64_t nq, int64_t nk, int d,
                    int64_t qs, int64_t ks, int64_t vs, int64_t os, float scale);

/* Same, with an optional causal mask (query i attends to keys 0..i; requires nq <= nk, aligned at 0) and an
   optional additive bias [nq, nk] added to the scaled scores before the softmax (T5 relative position bias). */
void rope_attention_ex(const float *q, const float *k, const float *v, float *o, int64_t nq, int64_t nk, int d,
                       int64_t qs, int64_t ks, int64_t vs, int64_t os, float scale, int causal, const float *bias);

void rope_softmax_rows(float *x, int64_t m, int64_t n);
void rope_layernorm(float *x, int64_t m, int64_t n, const float *g, const float *b, float eps);   /* g, b may be NULL */
void rope_rmsnorm(float *x, int64_t m, int64_t n, const float *g, float eps);
void rope_groupnorm(float *x, int64_t c, int64_t hw, int groups, const float *g, const float *b, float eps);
void rope_gelu_erf(float *x, int64_t n);
void rope_gelu_tanh(float *x, int64_t n);
/* Quantize rows [rows, k] of float32 into GGUF type t (RT_Q8_0 or RT_Q4_K); returns 0, or -1 if unsupported. */
int rope_quantize_rows(int t, const float *x, void *dst, int64_t rows, int64_t k);
/* y[i, :] += b for every row i of y [m, n] (linear-layer bias). */
void rope_add_rows(float *y, int64_t m, int64_t n, const float *b);
void rope_silu(float *x, int64_t n);

/* Rotary embedding. x[t*tok_stride + h*d_head + i]; cos/sin tables are [n_tok][d_head/2].
 * neox=0: rotate interleaved pairs (2i, 2i+1); neox=1: rotate halves (i, i + d_head/2). */
void rope_rotary_tables(const float *pos, int64_t n_tok, int d_head, float theta, float *cos_t, float *sin_t);
void rope_rotary(float *x, int64_t n_tok, int n_head, int d_head, int64_t tok_stride,
                 const float *cos_t, const float *sin_t, int neox);
