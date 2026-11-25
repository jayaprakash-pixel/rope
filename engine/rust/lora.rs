//! LoRA adapters for SD 1.x / 2.x / SDXL UNets (P2.4.4).
//!
//! A LoRA file holds, per adapted linear layer, a down matrix [r, k], an up matrix [n, r] and a scalar `alpha`
//! (kohya format: `lora_unet_<layer name with underscores>.lora_down.weight` / `.lora_up.weight` / `.alpha`; the layer names
//! are the UNet's own `ldm` names, `input_blocks.4.1.proj_in` -> `input_blocks_4_1_proj_in`).
//! The layer computes y = x W^T + b + strength * (alpha / r) * (x down^T) up^T. The base weights are never changed or
//! copied (no merged copy: streaming from disk and the GPU placement stay as they are); the low-rank part reads the
//! adapter's BF16 / F16 / F32 tensors in place from the memory map. The adapter lives for the whole process (one image).
//! Only 2-D (linear) layers of the UNet are supported; text-encoder and convolution entries are counted and reported, not applied.
#![forbid(unsafe_code)]

use crate::ffi;
use crate::weights::Weights;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Mutex, RwLock};

pub struct Module { down: &'static [u8], down_ty: u32, up: &'static [u8], up_ty: u32, pub rank: usize, pub k: usize, pub n: usize, scale: f32 }

/// The active adapter (None = no LoRA). A library user runs many jobs in one process, so it can change; files and layer maps are
/// cached by path and strength (the memory maps live for the process, one per file used).
static REG: RwLock<Option<&'static HashMap<String, Module>>> = RwLock::new(None);
type Cache = HashMap<(String, u32), &'static HashMap<String, Module>>;
static CACHE: Mutex<Option<Cache>> = Mutex::new(None);

/// No adapter (the next job runs the plain model).
pub fn clear() { if let Ok(mut g) = REG.write() { *g = None; } }

pub struct Info { pub modules: usize, pub rank_min: usize, pub rank_max: usize, pub skipped: usize }

#[cfg(test)]
/// y += scale * (x down^T) up^T with float matrices: x [m, k], down [r, k], up [n, r], y [m, n].
pub fn lowrank_add(y: &mut [f32], x: &[f32], m: usize, k: usize, n: usize, r: usize, down: &[f32], up: &[f32], scale: f32) {
    let mut t = vec![0f32; m * r];
    ffi::matmul(x, m, k, &bytes_of(down), 0, r, &mut t);                 // t = x down^T
    let mut u = vec![0f32; m * n];
    ffi::matmul(&t, m, r, &bytes_of(up), 0, n, &mut u);                  // u = t up^T
    for (a, b) in y.iter_mut().zip(&u) { *a += scale * b; }
}

#[cfg(test)]
/// f32 slice as little-endian bytes (the matmul takes raw rows); only for tests and small float adapters.
fn bytes_of(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

/// Open `path` as a LoRA and make it the active adapter, with `strength` (1.0 = as trained). Returns what it holds.
pub fn load(path: &Path, strength: f32) -> Result<Info, String> {
    let key = (path.to_string_lossy().into_owned(), strength.to_bits());
    if let Some(m) = CACHE.lock().map_err(|e| e.to_string())?.as_ref().and_then(|c| c.get(&key)) {
        let ranks: Vec<usize> = m.values().map(|x| x.rank).collect();
        *REG.write().map_err(|e| e.to_string())? = Some(m);
        return Ok(Info { modules: m.len(), rank_min: *ranks.iter().min().unwrap_or(&0), rank_max: *ranks.iter().max().unwrap_or(&0), skipped: 0 });
    }
    let w: &'static Weights = Box::leak(Box::new(Weights::open(path)?));
    // same outlier-injection scan as for models: an adapter is untrusted input too
    if let Some(bad) = crate::scan::scan(w)?.iter().find(|r| r.stats.flagged()) { return Err(format!("{}: SUSPICIOUS add-on file, refused ({} has an outlier injection pattern)", path.display(), bad.name)); }
    let mut map = HashMap::new();
    let (mut ranks, mut skipped) = (Vec::new(), 0usize);
    for t in &w.mf.tensors {
        let Some(base) = t.name.strip_suffix(".lora_down.weight") else { continue };
        let Some(key) = base.strip_prefix("lora_unet_") else { skipped += 1; continue };          // text-encoder entries: not applied
        let dd = t.dims();
        let up_name = format!("{base}.lora_up.weight");
        let Ok(ut) = w.tensor(&up_name) else { return Err(format!("{}: no matching {up_name}", path.display())) };
        let ud = ut.dims();
        if dd.len() != 2 || ud.len() != 2 { skipped += 1; continue; }                              // convolution LoRA: not applied
        let (r, k, n) = (dd[0] as usize, dd[1] as usize, ud[0] as usize);
        if ud[1] as usize != r { return Err(format!("{base}: down rank {r} and up rank {} differ", ud[1])); }
        let alpha = match w.tensor(&format!("{base}.alpha")) { Ok(_) => w.f32(&format!("{base}.alpha"))?.first().copied().unwrap_or(r as f32), Err(_) => r as f32 };
        let (down, down_ty) = w.raw(&t.name)?;
        let (up, up_ty) = w.raw(&up_name)?;
        ranks.push(r);
        map.insert(key.to_string(), Module { down, down_ty, up, up_ty, rank: r, k, n, scale: strength * alpha / r as f32 });
    }
    if map.is_empty() { return Err(format!("{}: no UNet LoRA layers found (lora_unet_*.lora_down.weight)", path.display())); }
    let info = Info { modules: map.len(), rank_min: *ranks.iter().min().unwrap(), rank_max: *ranks.iter().max().unwrap(), skipped };
    let map: &'static HashMap<String, Module> = Box::leak(Box::new(map));
    CACHE.lock().map_err(|e| e.to_string())?.get_or_insert_with(HashMap::new).insert(key, map);
    *REG.write().map_err(|e| e.to_string())? = Some(map);
    Ok(info)
}

/// Every adapter layer must exist in the base UNet with the same shape: a LoRA made for another model is refused,
/// not half applied. `base` = the checkpoint with `model.diffusion_model.*` names.
pub fn verify(base: &Weights) -> Result<(), String> {
    let Some(reg) = *REG.read().map_err(|e| e.to_string())? else { return Ok(()) };
    let (mut bad, mut first) = (0usize, String::new());
    for (key, m) in reg.iter() {
        let mut found = false;
        // the key with underscores: try the layer names that normalize to it
        for t in base.mf.tensors.iter().filter(|t| t.name.starts_with("model.diffusion_model.") && t.name.ends_with(".weight")) {
            if normalize(&t.name) == *key {
                let d = t.dims();
                found = d.len() >= 2 && d[0] as usize == m.n && d[1..].iter().product::<u64>() as usize == m.k;   // 1x1 convs count as linear
                break;
            }
        }
        if !found { bad += 1; if first.is_empty() { first = key.clone(); } }
    }
    if bad > 0 { return Err(format!("this LoRA does not fit this model: {bad} of {} layers have no matching layer of the same shape in it (first: {first})", reg.len())); }
    Ok(())
}

/// `model.diffusion_model.input_blocks.4.1.proj_in.weight` -> `input_blocks_4_1_proj_in`
fn normalize(name: &str) -> String {
    name.strip_prefix("model.diffusion_model.").unwrap_or(name).trim_end_matches(".weight").replace('.', "_")
}

/// The adapter layer for a UNet layer (`model.diffusion_model.input_blocks.4.1.proj_in`), if any.
pub fn module(layer: &str) -> Option<&'static Module> {
    let reg = (*REG.read().ok()?)?;
    reg.get(&normalize(layer))
}

impl Module {
    /// y [m, n] += scale * (x [m, k] down^T) up^T, reading the adapter in place.
    pub fn apply(&self, x: &[f32], m: usize, y: &mut [f32]) {
        let mut t = vec![0f32; m * self.rank];
        ffi::matmul(x, m, self.k, self.down, self.down_ty, self.rank, &mut t);
        let mut u = vec![0f32; m * self.n];
        ffi::matmul(&t, m, self.rank, self.up, self.up_ty, self.n, &mut u);
        for (a, b) in y.iter_mut().zip(&u) { *a += self.scale * b; }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The low-rank add equals the merged-weight product x (W + s up down)^T (float64 reference).
    #[test]
    fn low_rank_add_equals_merged_weights() {
        let (m, k, n, r, s) = (7usize, 40usize, 24usize, 4usize, 0.75f32);
        let g = |i: usize, salt: u32| (((i as u32).wrapping_mul(2654435761).wrapping_add(salt) >> 8) % 2001) as f32 / 1000.0 - 1.0;
        let x: Vec<f32> = (0..m * k).map(|i| g(i, 1)).collect();
        let wmat: Vec<f32> = (0..n * k).map(|i| g(i, 2)).collect();
        let down: Vec<f32> = (0..r * k).map(|i| g(i, 3)).collect();
        let up: Vec<f32> = (0..n * r).map(|i| g(i, 4)).collect();
        let mut y = vec![0f32; m * n];
        ffi::matmul(&x, m, k, &bytes_of(&wmat), 0, n, &mut y);
        lowrank_add(&mut y, &x, m, k, n, r, &down, &up, s);
        for i in 0..m { for j in 0..n {
            let mut acc = 0f64;
            for c in 0..k {
                let mut w = wmat[j * k + c] as f64;
                for q in 0..r { w += s as f64 * up[j * r + q] as f64 * down[q * k + c] as f64; }
                acc += x[i * k + c] as f64 * w;
            }
            assert!((acc - y[i * n + j] as f64).abs() < 1e-3 * (1.0 + acc.abs()), "[{i},{j}] {acc} vs {}", y[i * n + j]);
        } }
    }

    #[test]
    fn layer_names_normalize_like_kohya() {
        assert_eq!(normalize("model.diffusion_model.input_blocks.4.1.proj_in.weight"), "input_blocks_4_1_proj_in");
        assert_eq!(normalize("model.diffusion_model.output_blocks.5.1.transformer_blocks.1.ff.net.2"), "output_blocks_5_1_transformer_blocks_1_ff_net_2");
    }
}
