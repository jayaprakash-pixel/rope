/* Integer-dot matmul for quantized weights (P2.2.3), AVX2 build only; included by kernels_cpu.c.
 * Activations are quantized once per call to int8: per 256 values for Q4_K / Q6_K weights (one scale per 256, int16
 * sums per 16 for the weights' min / offset terms, like ggml's Q8_K), per 32 for Q4_0 / Q8_0 (like Q8_0). Weight
 * blocks are decoded once into byte vectors and used by up to 8 activation rows: maddubs (u8 x s8 -> s16 pairs),
 * madd with the 6-bit / int8 sub-block scales (-> s32), integer sums over the block, one float multiply per block.
 * Results equal the float64 dot of the dequantized weights with the int8-rounded activations (rope_int8_act). */
#pragma once
#ifndef _MSC_VER
#define __forceinline inline __attribute__((always_inline))
#endif

#define I8_MR 8          /* activation rows per weight-block decode: 8 int accumulators + 4 work registers fit 16 YMM */

static inline float h2f_fast(int h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h))); }   /* F16C */

typedef struct { int8_t *q; float *d; int16_t *s; int64_t bq, bs; } actq;   /* bq: values per scale, bs: values per sum */

/* Off by default (measured 2026-11-16): image models' activations have outlier channels that int8 per 256 values
 * cannot hold. FLUX one call 0.49 max rel error vs 1.3e-4 in float, image SSIM 0.71, for 4% speed (264 vs 275 s).
 * Kept for text models (P4), where int8 activations are the norm and Q4_K decode is 4.3x faster (19 -> 82 GFLOPS).
 * ROPE_INT8 / rope_int8_mode: 0 off, 1 all, 2 decode only (M < 4), 3 batch only. */
static int g_i8mode = -1;
int rope_int8_mode(int mode) {
    if (g_i8mode < 0) { const char *e = getenv("ROPE_INT8"); g_i8mode = !e ? 0 : e[0] == '0' ? 0 : !strcmp(e, "decode") ? 2 : !strcmp(e, "batch") ? 3 : 1; }
    int old = g_i8mode;
    if (mode >= 0) g_i8mode = mode;
    return old;
}
static int int8_used(int wt, int64_t M) {
    int mode = rope_int8_mode(-1);
    if (mode == 0 || (mode == 2 && M >= 4) || (mode == 3 && M < 4)) return 0;
    return wt == RT_Q4_K || wt == RT_Q6_K || ((wt == RT_Q4_0 || wt == RT_Q8_0) && M < 4);
}

/* one activation row of k values -> int8 (round to nearest), scale per bq values, int16 sum per bs values */
static void quant_act_row(const float *x, int64_t k, int64_t bq, int64_t bs, int8_t *q, float *d, int16_t *s) {
    for (int64_t b = 0; b < k / bq; b++) {
        const float *xb = x + b * bq;
        float amax = 0;
        for (int64_t i = 0; i < bq; i++) amax = fmaxf(amax, fabsf(xb[i]));
        float sc = amax / 127.0f, id = amax > 0 ? 127.0f / amax : 0.0f;
        d[b] = sc;
        for (int64_t i = 0; i < bq; i++) q[b * bq + i] = (int8_t)lrintf(xb[i] * id);
    }
    for (int64_t g = 0; g < k / bs; g++) { int t = 0; for (int64_t i = 0; i < bs; i++) t += q[g * bs + i]; s[g] = (int16_t)t; }
}

static int64_t act_block(int wt) { return wt == RT_Q4_K || wt == RT_Q6_K ? 256 : 32; }
static int64_t act_sum(int wt) { return wt == RT_Q6_K ? 16 : 32; }

static inline float hsum8(__m256 v) {
    __m128 a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    a = _mm_add_ps(a, _mm_movehl_ps(a, a));
    a = _mm_add_ss(a, _mm_movehdup_ps(a));
    return _mm_cvtss_f32(a);
}

/* out[r] += dot(activation rows r = 0..R-1, weight row w) over weight blocks [b0, b1) */
static __forceinline void dot_q4k(const uint8_t *w, int64_t b0, int64_t b1, const actq *a, const int64_t *rows, const int R, float *out) {
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    __m256 acc[I8_MR]; float accm[I8_MR];
    for (int r = 0; r < R; r++) { acc[r] = _mm256_setzero_ps(); accm[r] = 0; }
    for (int64_t b = b0; b < b1; b++) {
        const uint8_t *wb = w + b * 144, *scb = wb + 4, *qs = wb + 16;
        float d = h2f_fast(wb[0] | wb[1] << 8), dmin = h2f_fast(wb[2] | wb[3] << 8);
        uint8_t sc[8], mn[8];
        for (int j = 0; j < 8; j++) {
            if (j < 4) { sc[j] = scb[j] & 63; mn[j] = scb[j + 4] & 63; }
            else { sc[j] = (uint8_t)((scb[j + 4] & 0xF) | ((scb[j - 4] >> 6) << 4)); mn[j] = (uint8_t)((scb[j + 4] >> 4) | ((scb[j] >> 6) << 4)); }
        }
        const __m128i mn16 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)mn));
        __m256i si[I8_MR];
        for (int r = 0; r < R; r++) si[r] = _mm256_setzero_si256();
        for (int jp = 0; jp < 4; jp++) {
            __m256i q4 = _mm256_loadu_si256((const __m256i *)(qs + 32 * jp));
            __m256i lo = _mm256_and_si256(q4, m4), hi = _mm256_and_si256(_mm256_srli_epi16(q4, 4), m4);
            __m256i sl = _mm256_set1_epi16(sc[2 * jp]), sh = _mm256_set1_epi16(sc[2 * jp + 1]);
            for (int r = 0; r < R; r++) {
                const int8_t *xq = a->q + rows[r] + b * 256 + 64 * jp;
                __m256i pl = _mm256_madd_epi16(_mm256_maddubs_epi16(lo, _mm256_loadu_si256((const __m256i *)xq)), sl);
                __m256i ph = _mm256_madd_epi16(_mm256_maddubs_epi16(hi, _mm256_loadu_si256((const __m256i *)(xq + 32))), sh);
                si[r] = _mm256_add_epi32(si[r], _mm256_add_epi32(pl, ph));
            }
        }
        for (int r = 0; r < R; r++) {
            int64_t xb = rows[r] / 256 + b;                          /* activation block of row r */
            float dx = a->d[xb];
            __m128i pm = _mm_madd_epi16(mn16, _mm_loadu_si128((const __m128i *)(a->s + rows[r] / 32 + b * 8)));
            pm = _mm_add_epi32(pm, _mm_shuffle_epi32(pm, 0x4E)); pm = _mm_add_epi32(pm, _mm_shuffle_epi32(pm, 0xB1));
            int sm = _mm_cvtsi128_si32(pm);                          /* sum of min[j] * activation sums */
            acc[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(si[r]), _mm256_set1_ps(d * dx), acc[r]);
            accm[r] -= dmin * dx * (float)sm;
        }
    }
    for (int r = 0; r < R; r++) out[r] += hsum8(acc[r]) + accm[r];
}

static __forceinline void dot_q6k(const uint8_t *w, int64_t b0, int64_t b1, const actq *a, const int64_t *rows, const int R, float *out) {
    const __m256i m4 = _mm256_set1_epi8(0x0F), m2 = _mm256_set1_epi8(0x03);
    __m256 acc[I8_MR];
    for (int r = 0; r < R; r++) acc[r] = _mm256_setzero_ps();
    for (int64_t b = b0; b < b1; b++) {
        const uint8_t *wb = w + b * 210;
        const int8_t *sc = (const int8_t *)(wb + 192);
        float d = h2f_fast(wb[208] | wb[209] << 8);
        __m256i si[I8_MR];
        for (int r = 0; r < R; r++) si[r] = _mm256_setzero_si256();
        for (int h = 0; h < 2; h++) {
            const uint8_t *ql = wb + 64 * h, *qh = wb + 128 + 32 * h;
            __m256i qa = _mm256_loadu_si256((const __m256i *)ql), qb = _mm256_loadu_si256((const __m256i *)(ql + 32));
            __m256i hv = _mm256_loadu_si256((const __m256i *)qh);
            __m256i q[4];
            q[0] = _mm256_or_si256(_mm256_and_si256(qa, m4), _mm256_slli_epi16(_mm256_and_si256(hv, m2), 4));
            q[1] = _mm256_or_si256(_mm256_and_si256(qb, m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hv, 2), m2), 4));
            q[2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qa, 4), m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hv, 4), m2), 4));
            q[3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qb, 4), m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hv, 6), m2), 4));
            for (int i = 0; i < 4; i++) {
                const int8_t *s2 = sc + 8 * h + 2 * i;                  /* scales of values 0..15 and 16..31 */
                __m256i sv = _mm256_set_m128i(_mm_set1_epi16(s2[1]), _mm_set1_epi16(s2[0]));
                for (int r = 0; r < R; r++) {
                    const int8_t *xq = a->q + rows[r] + b * 256 + 128 * h + 32 * i;
                    __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(q[i], _mm256_loadu_si256((const __m256i *)xq)), sv);
                    si[r] = _mm256_add_epi32(si[r], p);
                }
            }
        }
        for (int r = 0; r < R; r++) {
            float dx = a->d[rows[r] / 256 + b];
            const int16_t *bs = a->s + rows[r] / 16 + b * 16;
            int off = 0;
            for (int g = 0; g < 16; g++) off += sc[g] * bs[g];               /* the -32 offset: 32 * sum(sc * x) */
            __m256i t = _mm256_sub_epi32(si[r], _mm256_setr_epi32(32 * off, 0, 0, 0, 0, 0, 0, 0));
            acc[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(t), _mm256_set1_ps(d * dx), acc[r]);
        }
    }
    for (int r = 0; r < R; r++) out[r] += hsum8(acc[r]);
}

/* Q8_0 / Q4_0 weights with per-32 activations (decode) */
static __forceinline void dot_q8_0(const uint8_t *w, int64_t b0, int64_t b1, const actq *a, const int64_t *rows, const int R, float *out) {
    __m256 acc[I8_MR];
    for (int r = 0; r < R; r++) acc[r] = _mm256_setzero_ps();
    const __m256i one = _mm256_set1_epi16(1);
    for (int64_t b = b0; b < b1; b++) {
        const uint8_t *wb = w + b * 34;
        float d = h2f_fast(wb[0] | wb[1] << 8);
        __m256i wq = _mm256_loadu_si256((const __m256i *)(wb + 2)), aw = _mm256_sign_epi8(wq, wq);
        for (int r = 0; r < R; r++) {
            __m256i xv = _mm256_loadu_si256((const __m256i *)(a->q + rows[r] + b * 32));
            __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, _mm256_sign_epi8(xv, wq)), one);
            acc[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(d * a->d[rows[r] / 32 + b]), acc[r]);
        }
    }
    for (int r = 0; r < R; r++) out[r] += hsum8(acc[r]);
}

static __forceinline void dot_q4_0(const uint8_t *w, int64_t b0, int64_t b1, const actq *a, const int64_t *rows, const int R, float *out) {
    __m256 acc[I8_MR]; float acco[I8_MR];
    for (int r = 0; r < R; r++) { acc[r] = _mm256_setzero_ps(); acco[r] = 0; }
    const __m256i one = _mm256_set1_epi16(1), m4 = _mm256_set1_epi8(0x0F);
    for (int64_t b = b0; b < b1; b++) {
        const uint8_t *wb = w + b * 18;
        float d = h2f_fast(wb[0] | wb[1] << 8);
        __m128i q = _mm_loadu_si128((const __m128i *)(wb + 2));
        __m256i qq = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(q, 4), q), m4);   /* values 0..15 | 16..31 */
        for (int r = 0; r < R; r++) {
            __m256i xv = _mm256_loadu_si256((const __m256i *)(a->q + rows[r] + b * 32));
            __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(qq, xv), one);
            float dd = d * a->d[rows[r] / 32 + b];
            acc[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(dd), acc[r]);
            acco[r] -= 8.0f * dd * (float)a->s[rows[r] / 32 + b];               /* the -8 offset */
        }
    }
    for (int r = 0; r < R; r++) out[r] += hsum8(acc[r]) + acco[r];
}

/* R (<= I8_MR) rows of activations (offsets rows[r] = m * K into the quantized buffers) against one weight row */
static __forceinline void dot_rows(int wt, const uint8_t *w, int64_t b0, int64_t b1, const actq *a, const int64_t *rows, const int R, float *out) {
    switch (wt) {
    case RT_Q4_K: dot_q4k(w, b0, b1, a, rows, R, out); break;
    case RT_Q6_K: dot_q6k(w, b0, b1, a, rows, R, out); break;
    case RT_Q8_0: dot_q8_0(w, b0, b1, a, rows, R, out); break;
    default:      dot_q4_0(w, b0, b1, a, rows, R, out); break;
    }
}

#define I8_NT 16        /* weight rows per task: their K-slice stays in L1 while all activation rows pass */
#define I8_KC 4096      /* K values per slice */

static void matmul_int8(const float *x, int64_t M, int64_t K, const void *w, int wt, int64_t N, float *y, int nth) {
    int64_t bq = act_block(wt), bsz = act_sum(wt), wblk = rope_wtype_block(wt);
    actq a = { amalloc((size_t)(M * K)), amalloc((size_t)(M * K / bq) * 4), amalloc((size_t)(M * K / bsz) * 2), bq, bsz };
    int64_t m;
    #pragma omp parallel for num_threads(nth) schedule(static)
    for (m = 0; m < M; m++) quant_act_row(x + m * K, K, bq, bsz, a.q + m * K, a.d + m * K / bq, a.s + m * K / bsz);
    size_t rowb = rope_row_bytes(wt, K);
    int n_tasks = (int)((N + I8_NT - 1) / I8_NT), ti;
    #pragma omp parallel for num_threads(nth) schedule(dynamic, 1)
    for (ti = 0; ti < n_tasks; ti++) {
        int64_t n0 = (int64_t)ti * I8_NT, n1 = n0 + I8_NT < N ? n0 + I8_NT : N;
        for (int64_t mm = 0; mm < M; mm++) for (int64_t n = n0; n < n1; n++) y[mm * N + n] = 0;
        for (int64_t k0 = 0; k0 < K; k0 += I8_KC) {
            int64_t b0 = k0 / wblk, b1 = (k0 + I8_KC < K ? k0 + I8_KC : K) / wblk;
            for (int64_t m0 = 0; m0 < M; m0 += I8_MR) {
                int R = (int)(M - m0 < I8_MR ? M - m0 : I8_MR);
                int64_t rows[I8_MR];
                for (int r = 0; r < R; r++) rows[r] = (m0 + r) * K;
                for (int64_t n = n0; n < n1; n++) {
                    float o[I8_MR] = {0};
                    const uint8_t *wr = (const uint8_t *)w + n * rowb;
                    switch (R) {                                 /* constant row counts: accumulators stay in registers */
                    case 8: dot_rows(wt, wr, b0, b1, &a, rows, 8, o); break;
                    case 7: dot_rows(wt, wr, b0, b1, &a, rows, 7, o); break;
                    case 6: dot_rows(wt, wr, b0, b1, &a, rows, 6, o); break;
                    case 5: dot_rows(wt, wr, b0, b1, &a, rows, 5, o); break;
                    case 4: dot_rows(wt, wr, b0, b1, &a, rows, 4, o); break;
                    case 3: dot_rows(wt, wr, b0, b1, &a, rows, 3, o); break;
                    case 2: dot_rows(wt, wr, b0, b1, &a, rows, 2, o); break;
                    default: dot_rows(wt, wr, b0, b1, &a, rows, 1, o); break;
                    }
                    for (int r = 0; r < R; r++) y[(m0 + r) * N + n] += o[r];
                }
            }
        }
    }
    afree(a.q); afree(a.d); afree(a.s);
}

/* The activations exactly as the integer path rounds them (for tests); 0 when this type / M does not use it. */
int rope_int8_act(int wt, int64_t M, const float *x, int64_t k, float *xr) {
    if (!int8_used(wt, M)) return 0;
    int64_t bq = act_block(wt), bsz = act_sum(wt);
    int8_t *q = malloc((size_t)k); float *d = malloc((size_t)(k / bq) * 4); int16_t *s = malloc((size_t)(k / bsz) * 2);
    quant_act_row(x, k, bq, bsz, q, d, s);
    for (int64_t i = 0; i < k; i++) xr[i] = d[i / bq] * q[i];
    free(q); free(d); free(s);
    return 1;
}
