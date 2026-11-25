//! T5 v1.1 encoder (FLUX's T5-XXL text encoder) on rope kernels, from llama.cpp-style GGUF names
//! (`enc.blk.N.*`, `token_embd`, `enc.output_norm`). Weights stay quantized and memory-mapped.
//! T5 specifics: RMSNorm without bias, no 1/sqrt(d) attention scaling, one relative-position bias table
//! (block 0) shared by every block, gated-GELU (tanh form) feed-forward. No attention mask: FLUX encodes the
//! padded 256 tokens without one.
#![forbid(unsafe_code)]

use crate::ffi;
use crate::weights::Weights;

pub struct T5Encoder<'w> { w: &'w Weights, d: usize, heads: usize, layers: usize }

const BUCKETS: usize = 32;
const MAX_DISTANCE: f64 = 128.0;

/// Hugging Face `_relative_position_bucket`, bidirectional: rel = key position - query position.
pub fn bucket(rel: i64) -> usize {
    let half = BUCKETS / 2;
    let mut b = if rel > 0 { half } else { 0 };
    let r = rel.unsigned_abs() as usize;
    let exact = half / 2;
    b += if r < exact { r } else {
        let large = exact + ((r as f64 / exact as f64).ln() / (MAX_DISTANCE / exact as f64).ln() * (half - exact) as f64) as usize;
        large.min(half - 1)
    };
    b
}

fn rms_norm(x: &mut [f32], d: usize, g: &[f32]) {
    for row in x.chunks_exact_mut(d) {
        let ms = row.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / d as f64;
        let inv = (1.0 / (ms + 1e-6).sqrt()) as f32;
        for (v, gi) in row.iter_mut().zip(g) { *v *= inv * gi; }
    }
}

impl<'w> T5Encoder<'w> {
    pub fn new(w: &'w Weights) -> Result<Self, String> {
        let t = w.tensor("token_embd.weight").map_err(|_| "no T5 encoder in this file (token_embd.weight missing)".to_string())?;
        let d = t.shape[0] as usize;
        let heads = w.tensor("enc.blk.0.attn_rel_b.weight")?.shape[0] as usize;
        let layers = (0..).take_while(|l| w.tensor(&format!("enc.blk.{l}.attn_q.weight")).is_ok()).count();
        Ok(T5Encoder { w, d, heads, layers })
    }

    /// y[m, n] = x[m, k] W^T for a GGUF weight with ne = [k, n].
    fn lin(&self, x: &[f32], m: usize, name: &str) -> Result<Vec<f32>, String> {
        let t = self.w.tensor(name)?;
        let (k, n) = (t.shape[0] as usize, t.shape[1] as usize);
        let (wb, ty) = self.w.raw(name)?;
        let mut y = vec![0f32; m * n];
        ffi::matmul(x, m, k, wb, ty, n, &mut y);
        Ok(y)
    }

    /// ids (any length, usually 256) -> hidden states [len, 4096].
    pub fn encode(&self, ids: &[u32]) -> Result<Vec<f32>, String> {
        let (n, d, hh) = (ids.len(), self.d, self.heads);
        let hd = d / hh;
        let mut x = self.w.f32_rows("token_embd.weight", ids, d)?;
        // position bias [head][q][k] from the shared table [bucket][head]
        let table = self.w.f32("enc.blk.0.attn_rel_b.weight")?;
        let mut bias = vec![0f32; hh * n * n];
        for q in 0..n { for k in 0..n {
            let b = bucket(k as i64 - q as i64);
            for h in 0..hh { bias[(h * n + q) * n + k] = table[b * hh + h]; }
        } }
        self.w.prefetch_async("enc.blk.0.");
        for l in 0..self.layers {
            self.w.prefetch_async(&format!("enc.blk.{}.", l + 1));
            let p = format!("enc.blk.{l}.");
            let mut h = x.clone();
            rms_norm(&mut h, d, &self.w.f32(&format!("{p}attn_norm.weight"))?);
            let (q, k, v) = (self.lin(&h, n, &format!("{p}attn_q.weight"))?, self.lin(&h, n, &format!("{p}attn_k.weight"))?, self.lin(&h, n, &format!("{p}attn_v.weight"))?);
            let mut o = vec![0f32; n * d];
            for i in 0..hh {
                ffi::attention_bias(&q[i * hd..], &k[i * hd..], &v[i * hd..], &mut o[i * hd..], n, n, hd, d, d, d, d, 1.0, false,
                                    Some(&bias[i * n * n..(i + 1) * n * n]));
            }
            let a = self.lin(&o, n, &format!("{p}attn_o.weight"))?;
            for (xi, ai) in x.iter_mut().zip(&a) { *xi += ai; }
            let mut h = x.clone();
            rms_norm(&mut h, d, &self.w.f32(&format!("{p}ffn_norm.weight"))?);
            let mut g = self.lin(&h, n, &format!("{p}ffn_gate.weight"))?;
            ffi::gelu_tanh(&mut g);
            let u = self.lin(&h, n, &format!("{p}ffn_up.weight"))?;
            for (gi, ui) in g.iter_mut().zip(&u) { *gi *= ui; }
            let f = self.lin(&g, n, &format!("{p}ffn_down.weight"))?;
            for (xi, fi) in x.iter_mut().zip(&f) { *xi += fi; }
        }
        rms_norm(&mut x, d, &self.w.f32("enc.output_norm.weight")?);
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn buckets_match_hugging_face() {
        // values from transformers T5Attention._relative_position_bucket(bidirectional=True, 32, 128)
        let cases = [(0, 0), (1, 17), (-1, 1), (7, 23), (8, 24), (-8, 8), (12, 25), (-20, 10), (100, 31), (-127, 15), (300, 31), (-300, 15)];
        for (rel, want) in cases { assert_eq!(super::bucket(rel), want, "rel {rel}"); }
    }
}
