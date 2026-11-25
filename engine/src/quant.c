/* Weight formats: fp16/bf16 conversion and dequantization of GGUF block formats (from the format spec). */
#include "vf8.h"      /* before kernels.h: SIMD layer + ISA-suffixed public names */
#include "kernels.h"
#include <string.h>
#include <math.h>

float rope_h2f(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000) << 16, e = (h >> 10) & 0x1F, m = h & 0x3FF, bits;
    if (e == 0) {
        if (m == 0) bits = s;
        else { int k = -1; do { k++; m <<= 1; } while (!(m & 0x400)); bits = s | ((uint32_t)(112 - k) << 23) | ((m & 0x3FF) << 13); }
    } else if (e == 31) bits = s | 0x7F800000u | (m << 13);
    else bits = s | ((e + 112) << 23) | (m << 13);
    float f; memcpy(&f, &bits, 4); return f;
}

#if defined(ROPE_ISA_AVX2)
uint16_t rope_f2h(float f) { return (uint16_t)_mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(f), _MM_FROUND_TO_NEAREST_INT), 0); }
#else
/* float -> half, round to nearest even (what F16C's cvtps_ph does), for CPUs without F16C */
uint16_t rope_f2h(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t s = (x >> 16) & 0x8000, e = (x >> 23) & 0xFF, m = x & 0x7FFFFF;
    if (e == 0xFF) return (uint16_t)(s | 0x7C00 | (m ? 0x200 | (m >> 13) : 0));       /* inf / nan */
    int ex = (int)e - 127 + 15;
    if (ex >= 31) return (uint16_t)(s | 0x7C00);                                      /* overflow -> inf */
    if (ex <= 0) {                                                                   /* subnormal or zero */
        if (ex < -10) return (uint16_t)s;
        m |= 0x800000;
        int sh = 14 - ex;
        uint32_t r = m >> sh, rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (r & 1))) r++;
        return (uint16_t)(s | r);
    }
    uint32_t r = ((uint32_t)ex << 10) | (m >> 13), rem = m & 0x1FFF;
    if (rem > 0x1000 || (rem == 0x1000 && (r & 1))) r++;                             /* may carry into the exponent: correct */
    return (uint16_t)(s | r);
}
#endif

float rope_bf2f(uint16_t h) { uint32_t b = (uint32_t)h << 16; float f; memcpy(&f, &b, 4); return f; }

int rope_wtype_supported(int t) { return t == RT_F32 || t == RT_F16 || t == RT_BF16 || t == RT_Q4_0 || t == RT_Q8_0 || t == RT_Q4_K || t == RT_Q6_K; }

int rope_wtype_block(int t) { return t == RT_Q4_0 || t == RT_Q8_0 ? 32 : (t == RT_Q4_K || t == RT_Q6_K) ? 256 : 1; }

static size_t block_bytes(int t) {
    switch (t) { case RT_F32: return 4; case RT_F16: case RT_BF16: return 2;
                 case RT_Q4_0: return 18; case RT_Q8_0: return 34; case RT_Q4_K: return 144; case RT_Q6_K: return 210; }
    return 0;
}

size_t rope_row_bytes(int t, int64_t k) { return (size_t)(k / rope_wtype_block(t)) * block_bytes(t); }

static uint16_t rd16(const uint8_t *p) { uint16_t v; memcpy(&v, p, 2); return v; }

/* Q4_K: 256 weights = 8 sub-blocks of 32. Per block: d, dmin (fp16), 12 bytes of packed 6-bit scales/mins,
 * 128 bytes of 4-bit quants. Sub-block j: w = d*sc[j]*q - dmin*m[j]. */
static void q4k_scale_min(int j, const uint8_t *q, uint8_t *sc, uint8_t *m) {
    if (j < 4) { *sc = q[j] & 63; *m = q[j + 4] & 63; }
    else { *sc = (uint8_t)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)); *m = (uint8_t)((q[j + 4] >> 4) | ((q[j] >> 6) << 4)); }
}

void rope_dequant_row(int t, const void *src, float *y, int64_t k) {
    const uint8_t *p = src;
    switch (t) {
    case RT_F32: memcpy(y, p, (size_t)k * 4); return;
    case RT_F16: {
        int64_t i = 0;
        for (; i + 8 <= k; i += 8) vf8_storeu(y + i, vf8_from_f16(p + 2 * i));
        for (; i < k; i++) y[i] = rope_h2f(rd16(p + 2 * i));
        return;
    }
    case RT_BF16: for (int64_t i = 0; i < k; i++) y[i] = rope_bf2f(rd16(p + 2 * i)); return;
    case RT_Q8_0:
        for (int64_t b = 0; b < k / 32; b++, p += 34, y += 32) {
            float d = rope_h2f(rd16(p));
            for (int i = 0; i < 32; i++) y[i] = d * (float)(int8_t)p[2 + i];
        }
        return;
    case RT_Q4_0:
        for (int64_t b = 0; b < k / 32; b++, p += 18, y += 32) {
            float d = rope_h2f(rd16(p));
            for (int i = 0; i < 16; i++) { y[i] = d * (float)((p[2 + i] & 0xF) - 8); y[i + 16] = d * (float)((p[2 + i] >> 4) - 8); }
        }
        return;
    case RT_Q4_K:
        for (int64_t b = 0; b < k / 256; b++, p += 144) {
            float d = rope_h2f(rd16(p)), dmin = rope_h2f(rd16(p + 2));
            const uint8_t *sc = p + 4, *q = p + 16;
            for (int j = 0; j < 8; j += 2, q += 32) {
                uint8_t s1, m1, s2, m2;
                q4k_scale_min(j, sc, &s1, &m1); q4k_scale_min(j + 1, sc, &s2, &m2);
                float d1 = d * s1, mm1 = dmin * m1, d2 = d * s2, mm2 = dmin * m2;
                for (int l = 0; l < 32; l++) { y[l] = d1 * (q[l] & 0xF) - mm1; y[l + 32] = d2 * (q[l] >> 4) - mm2; }
                y += 64;
            }
        }
        return;
    case RT_Q6_K:   /* 256 weights: ql[128] low 4 bits, qh[64] high 2 bits, int8 scales[16] (one per 16), fp16 d */
        for (int64_t b = 0; b < k / 256; b++, p += 210) {
            const uint8_t *ql = p, *qh = p + 128; const int8_t *sc = (const int8_t *)(p + 192);
            float d = rope_h2f(rd16(p + 208));
            for (int n = 0; n < 2; n++, y += 128, ql += 64, qh += 32, sc += 8)
                for (int l = 0; l < 32; l++) {
                    int is = l / 16;
                    y[l]      = d * sc[is]     * (float)((int)((ql[l] & 0xF)      | (((qh[l] >> 0) & 3) << 4)) - 32);
                    y[l + 32] = d * sc[is + 2] * (float)((int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
                    y[l + 64] = d * sc[is + 4] * (float)((int)((ql[l] >> 4)       | (((qh[l] >> 4) & 3) << 4)) - 32);
                    y[l + 96] = d * sc[is + 6] * (float)((int)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32);
                }
        }
        return;
    }
}

/* ---------------- quantization (local, one-time): Q8_0 and Q4_K encoders matching the decoders above ---------------- */
static float roundf_away(float v) { return v < 0 ? -floorf(-v + 0.5f) : floorf(v + 0.5f); }

static void quant_q8_0(const float *x, uint8_t *p, int64_t k) {
    for (int64_t b = 0; b < k / 32; b++, x += 32, p += 34) {
        float amax = 0;
        for (int i = 0; i < 32; i++) amax = fmaxf(amax, fabsf(x[i]));
        float d = amax / 127.0f, id = d ? 1.0f / d : 0.0f;
        uint16_t h = rope_f2h(d); memcpy(p, &h, 2);
        for (int i = 0; i < 32; i++) p[2 + i] = (uint8_t)(int8_t)roundf_away(x[i] * id);
    }
}

/* Best (scale, min) for n values quantized to 0..nmax with weights w: x ~= scale * L + min, min <= 0.
 * Weighted least squares over candidate inverse scales around nmax / (max - min). Returns scale; *mn = -min >= 0. */
static float fit_scale_min(int n, int nmax, const float *x, const float *w, uint8_t *L, float *mn) {
    float lo = x[0], hi = x[0], sw = 0, sx = 0, sx2 = 0;
    for (int i = 0; i < n; i++) { lo = fminf(lo, x[i]); hi = fmaxf(hi, x[i]); sw += w[i]; sx += w[i] * x[i]; sx2 += w[i] * x[i] * x[i]; }
    if (lo > 0) lo = 0;
    if (hi == lo) { for (int i = 0; i < n; i++) L[i] = 0; *mn = -lo; return 0; }
    float iscale = nmax / (hi - lo), scale = 1 / iscale, best = 0;
    for (int i = 0; i < n; i++) {
        int l = (int)lrintf(iscale * (x[i] - lo)); l = l < 0 ? 0 : l > nmax ? nmax : l; L[i] = (uint8_t)l;
        float e = scale * l + lo - x[i]; best += w[i] * e * e;
    }
    uint8_t La[256];
    for (int is = 0; is <= 20; is++) {
        float isc = (-1.0f + 0.1f * is + nmax) / (hi - lo), sl = 0, sl2 = 0, sxl = 0;
        if (n == 32) {                             /* Q4_K sub-block: four AVX2 vectors (round-to-nearest-even, as lrintf) */
            vf8 vs = vf8_set1(isc), vlo = vf8_set1(lo), vmax = vf8_set1((float)nmax), z = vf8_setzero();
            vf8 a1 = z, a2 = z, a3 = z;
            for (int i = 0; i < 32; i += 8) {
                vf8 xv = vf8_loadu(x + i), wv = vf8_loadu(w + i);
                vf8 l = vf8_round(vf8_mul(vs, vf8_sub(xv, vlo)));
                l = vf8_min(vf8_max(l, z), vmax);
                vf8 wl = vf8_mul(wv, l);
                a1 = vf8_add(a1, wl); a2 = vf8_fmadd(wl, l, a2); a3 = vf8_fmadd(wl, xv, a3);
                float lf[8]; vf8_storeu(lf, l);
                for (int j = 0; j < 8; j++) La[i + j] = (uint8_t)lf[j];
            }
            float t[8];
            vf8_storeu(t, a1); for (int j = 0; j < 8; j++) sl += t[j];
            vf8_storeu(t, a2); for (int j = 0; j < 8; j++) sl2 += t[j];
            vf8_storeu(t, a3); for (int j = 0; j < 8; j++) sxl += t[j];
        } else for (int i = 0; i < n; i++) {
            int l = (int)lrintf(isc * (x[i] - lo)); l = l < 0 ? 0 : l > nmax ? nmax : l; La[i] = (uint8_t)l;
            sl += w[i] * l; sl2 += w[i] * l * l; sxl += w[i] * l * x[i];
        }
        float D = sw * sl2 - sl * sl;
        if (D <= 0) continue;
        float s = (sw * sxl - sx * sl) / D, m = (sl2 * sx - sl * sxl) / D;
        if (m > 0) { m = 0; s = sxl / sl2; }
        /* weighted squared error from the sums already at hand (no second pass over the values) */
        float err = s * s * sl2 + 2 * s * m * sl + m * m * sw - 2 * s * sxl - 2 * m * sx + sx2;
        if (err < best) { best = err; scale = s; lo = m; memcpy(L, La, (size_t)n); }
    }
    *mn = -lo;
    return scale;
}

static void quant_q4_k(const float *x, uint8_t *p, int64_t k) {
    for (int64_t b = 0; b < k / 256; b++, x += 256, p += 144) {
        float sc[8], mn[8], w[32]; uint8_t L[256];
        for (int j = 0; j < 8; j++) {
            const float *xs = x + 32 * j; float s2 = 0;
            for (int i = 0; i < 32; i++) s2 += xs[i] * xs[i];
            float av = sqrtf(s2 / 32);
            for (int i = 0; i < 32; i++) w[i] = av + fabsf(xs[i]);
            sc[j] = fit_scale_min(32, 15, xs, w, L + 32 * j, &mn[j]);
        }
        float msc = 0, mmn = 0;
        for (int j = 0; j < 8; j++) { msc = fmaxf(msc, sc[j]); mmn = fmaxf(mmn, mn[j]); }
        float isc = msc > 0 ? 63 / msc : 0, imn = mmn > 0 ? 63 / mmn : 0;
        uint8_t *q = p + 4;
        memset(q, 0, 12);
        for (int j = 0; j < 8; j++) {
            int ls = (int)lrintf(isc * sc[j]), lm = (int)lrintf(imn * mn[j]);
            ls = ls > 63 ? 63 : ls; lm = lm > 63 ? 63 : lm;
            if (j < 4) { q[j] = (uint8_t)ls; q[j + 4] = (uint8_t)lm; }
            else { q[j + 4] = (uint8_t)((ls & 0xF) | ((lm & 0xF) << 4)); q[j - 4] |= (uint8_t)((ls >> 4) << 6); q[j] |= (uint8_t)((lm >> 4) << 6); }
        }
        uint16_t hd = rope_f2h(msc / 63), hm = rope_f2h(mmn / 63);
        memcpy(p, &hd, 2); memcpy(p + 2, &hm, 2);
        float d = rope_h2f(hd), dmin = rope_h2f(hm);
        for (int j = 0; j < 8; j++) {            /* re-quantize against the stored (rounded) scales */
            uint8_t s, m; q4k_scale_min(j, q, &s, &m);
            float d1 = d * s, dm = dmin * m;
            for (int i = 0; i < 32; i++) {
                int l = d1 ? (int)lrintf((x[32 * j + i] + dm) / d1) : 0;
                L[32 * j + i] = (uint8_t)(l < 0 ? 0 : l > 15 ? 15 : l);
            }
        }
        uint8_t *qs = p + 16;
        for (int j = 0; j < 8; j += 2, qs += 32)
            for (int l = 0; l < 32; l++) qs[l] = (uint8_t)(L[32 * j + l] | (L[32 * (j + 1) + l] << 4));
    }
}

int rope_quantize_rows(int t, const float *x, void *dst, int64_t rows, int64_t k) {
    if ((t != RT_Q8_0 && t != RT_Q4_K) || k % rope_wtype_block(t)) return -1;
    size_t rb = rope_row_bytes(t, k);
    int64_t r;
    #pragma omp parallel for num_threads(rope_get_threads()) schedule(dynamic, 4)
    for (r = 0; r < rows; r++) {
        uint8_t *o = (uint8_t *)dst + (size_t)r * rb;
        if (t == RT_Q8_0) quant_q8_0(x + r * k, o, k); else quant_q4_k(x + r * k, o, k);
    }
    return 0;
}
