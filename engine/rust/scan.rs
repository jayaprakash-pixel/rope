//! Outlier-injection scan (defence against "Widening the Gap", analysis B): run on full-precision weights BEFORE
//! quantizing them locally. The attack plants one huge value per quantization group (e.g. 32 weights) so the other
//! weights of the group round to zero after quantization. Rule per group of 32 consecutive weights in a row:
//! ratio = max|w| / mean of the other 31 |w|. Normal weights: ~3-10 (strong LLM outlier channels stay < ~100);
//! an injected 4096x value: ~5000. A tensor is flagged when >= 16 groups and >= 1% of its groups exceed 256
//! (attack ~100%; worst natural case measured 0.094%, so 10x margin both ways).
#![forbid(unsafe_code)]

use crate::weights::Weights;

pub const GROUP: usize = 32;
pub const RATIO_LIMIT: f32 = 256.0;
const MIN_GROUPS: u64 = 16;
const MIN_FRACTION: f64 = 0.01;   // attack: ~100% of a matrix's groups; worst natural seen: 0.094% (Wan VAE time_conv)

#[derive(Debug, Clone, Default)]
pub struct GroupStats { pub groups: u64, pub spiky: u64, pub max_ratio: f32 }

impl GroupStats {
    pub fn flagged(&self) -> bool { self.spiky >= MIN_GROUPS && self.spiky as f64 >= MIN_FRACTION * self.groups as f64 }

    /// Feed one row; trailing elements that do not fill a group are ignored.
    pub fn add_row(&mut self, row: &[f32]) {
        for g in row.chunks_exact(GROUP) {
            let (mut sum, mut max) = (0f64, 0f32);
            for &v in g { let a = v.abs(); sum += a as f64; if a > max { max = a; } }
            let rest = ((sum - max as f64) / (GROUP - 1) as f64) as f32;
            let ratio = if rest > 0.0 { max / rest } else if max > 0.0 { f32::INFINITY } else { 0.0 };
            self.groups += 1;
            if ratio > RATIO_LIMIT { self.spiky += 1; }
            if ratio.is_finite() && ratio > self.max_ratio { self.max_ratio = ratio; }
            if ratio.is_infinite() { self.max_ratio = f32::INFINITY; }
        }
    }
}

pub struct TensorReport { pub name: String, pub stats: GroupStats }

/// Scan every float weight matrix (2+ dims, rows of >= 32) in a model file.
pub fn scan(w: &Weights) -> Result<Vec<TensorReport>, String> {
    let mut out = Vec::new();
    for t in &w.mf.tensors {
        if !matches!(t.dtype.as_str(), "F32" | "F16" | "BF16") || t.shape.len() < 2 { continue; }
        // Quantization rows: GGUF ne[0] (innermost); safetensors = everything after the first (output) dim,
        // so a conv weight [O, C, kh, kw] is scanned as O rows of C*kh*kw, as it would be quantized.
        let row = if t.ggml_type.is_some() { t.shape[0] } else { t.shape[1..].iter().product::<u64>() } as usize;
        if row < GROUP { continue; }
        let (bytes, ty) = w.raw(&t.name)?;
        let esize = if ty == 0 { 4 } else { 2 };
        let mut st = GroupStats::default();
        for r in bytes.chunks_exact(row * esize) {
            let vals = if ty == 0 { r.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() } else { crate::ffi::dequant_row(ty, r, row) };
            st.add_row(&vals);
        }
        out.push(TensorReport { name: t.name.clone(), stats: st });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn normal(n: usize, seed: u64) -> Vec<f32> { crate::sd::gaussian_noise(n, seed).iter().map(|v| v * 0.02).collect() }

    #[test]
    fn injected_tensor_is_flagged_clean_is_not() {
        let (rows, cols) = (64, 1024);
        let clean = normal(rows * cols, 1);
        let mut s = GroupStats::default();
        for r in clean.chunks_exact(cols) { s.add_row(r); }
        assert!(!s.flagged(), "clean gaussian weights flagged: {s:?}");
        let mut bad = clean.clone();
        for g in 0..rows * cols / GROUP { bad[g * GROUP + 7] = 0.02 * 4096.0; }   // one 4096x value per group of 32
        let mut s = GroupStats::default();
        for r in bad.chunks_exact(cols) { s.add_row(r); }
        assert!(s.flagged() && s.spiky == s.groups, "injection not caught: {s:?}");
        // a realistic outlier channel (one column 50x larger in every row) must NOT be flagged
        let mut chan = clean;
        for r in 0..rows { chan[r * cols + 13] = 0.02 * 50.0; }
        let mut s = GroupStats::default();
        for r in chan.chunks_exact(cols) { s.add_row(r); }
        assert!(!s.flagged(), "natural outlier channel flagged: {s:?}");
    }
}
