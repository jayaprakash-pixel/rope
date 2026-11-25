//! FLUX.1 diffusion transformer (BFL reference layout: `double_blocks.N`, `single_blocks.N`, `img_in`, ...) on rope
//! kernels. Weights stay quantized (Q4_K) and memory-mapped; one block's activations live at a time.
//! Shapes: hidden 3072 = 24 heads x 128; RoPE axes (16, 56, 56) over (0, row, col) ids, theta 10000, interleaved pairs.
#![forbid(unsafe_code)]

use crate::ffi;
use crate::ops::{self, T};
use crate::sd::timestep_embedding;
use crate::weights::Weights;

const D: usize = 3072;
const H: usize = 24;
const HD: usize = 128;
const AXES: [usize; 3] = [16, 56, 56];

pub struct Flux<'w> { w: &'w Weights, double: usize, single: usize }

/// Patchify latents [16, h, w] -> tokens [(h/2)(w/2), 64], feature = c*4 + ph*2 + pw.
pub fn pack(lat: &[f32], h: usize, w: usize) -> Vec<f32> {
    let (h2, w2) = (h / 2, w / 2);
    let mut t = vec![0f32; h2 * w2 * 64];
    for c in 0..16 { for y in 0..h { for x in 0..w {
        t[((y / 2) * w2 + x / 2) * 64 + c * 4 + (y % 2) * 2 + x % 2] = lat[(c * h + y) * w + x];
    } } }
    t
}

/// Inverse of `pack`.
pub fn unpack(t: &[f32], h: usize, w: usize) -> Vec<f32> {
    let w2 = w / 2;
    let mut lat = vec![0f32; 16 * h * w];
    for c in 0..16 { for y in 0..h { for x in 0..w {
        lat[(c * h + y) * w + x] = t[((y / 2) * w2 + x / 2) * 64 + c * 4 + (y % 2) * 2 + x % 2];
    } } }
    lat
}

/// cos/sin tables [L, 64 pairs] for txt tokens (ids 0) followed by image tokens (0, row, col).
fn rope_tables(n_txt: usize, h2: usize, w2: usize) -> (Vec<f32>, Vec<f32>) {
    let n = n_txt + h2 * w2;
    let (mut cos, mut sin) = (vec![1f32; n * HD / 2], vec![0f32; n * HD / 2]);
    for i in 0..h2 * w2 {
        let pos = [0.0f64, (i / w2) as f64, (i % w2) as f64];
        let mut j = 0;
        for (a, &dim) in AXES.iter().enumerate() {
            for p in 0..dim / 2 {
                let ang = pos[a] / 10000f64.powf(2.0 * p as f64 / dim as f64);
                cos[(n_txt + i) * HD / 2 + j] = ang.cos() as f32;
                sin[(n_txt + i) * HD / 2 + j] = ang.sin() as f32;
                j += 1;
            }
        }
    }
    (cos, sin)
}

impl<'w> Flux<'w> {
    pub fn new(w: &'w Weights) -> Result<Self, String> {
        w.tensor("double_blocks.0.img_attn.qkv.weight").map_err(|_| "no FLUX transformer in this file".to_string())?;
        let count = |p: &str| (0..).take_while(|i| w.tensor(&format!("{p}.{i}.norm.key_norm.scale")).is_ok()
                                                 || w.tensor(&format!("{p}.{i}.img_attn.norm.key_norm.scale")).is_ok()).count();
        Ok(Flux { w, double: count("double_blocks"), single: count("single_blocks") })
    }

    fn lin(&self, x: &T, name: &str) -> Result<T, String> { ops::lin(self.w, x, name, true) }

    /// silu(vec) -> lin -> chunks of D (one row: stays on the host).
    fn modulation(&self, vec: &[f32], name: &str) -> Result<Vec<Vec<f32>>, String> {
        let mut v = vec.to_vec();
        ffi::silu(&mut v);
        let n = v.len();
        Ok(self.lin(&T::host(v, 1, n), name)?.into_host()?.chunks_exact(D).map(<[f32]>::to_vec).collect())
    }

    fn qk_norm_rope(&self, buf: &mut T, prefix: &str, rope: &ops::Rope, row0: usize) -> Result<(), String> {
        let (qs, ks) = (self.w.f32(&format!("{prefix}.query_norm.scale"))?, self.w.f32(&format!("{prefix}.key_norm.scale"))?);
        ops::qk_norm_rope(buf, D, H, HD, &qs, &ks, rope, row0)
    }

    fn double_block(&self, i: usize, img: &mut T, txt: &mut T, vec: &[f32], rope: &ops::Rope) -> Result<(), String> {
        let p = format!("double_blocks.{i}");
        let (ni, nt) = (img.rows, txt.rows);
        let im = self.modulation(vec, &format!("{p}.img_mod.lin"))?;
        let tm = self.modulation(vec, &format!("{p}.txt_mod.lin"))?;
        let mut tq = self.lin(&ops::modulate(txt, &tm[0], &tm[1])?, &format!("{p}.txt_attn.qkv"))?;
        let mut iq = self.lin(&ops::modulate(img, &im[0], &im[1])?, &format!("{p}.img_attn.qkv"))?;
        self.qk_norm_rope(&mut tq, &format!("{p}.txt_attn.norm"), rope, 0)?;
        self.qk_norm_rope(&mut iq, &format!("{p}.img_attn.norm"), rope, nt)?;
        let q = ops::cat_rows(tq, &iq)?;                               // joint sequence: txt rows, then img rows
        drop(iq);
        let a = ops::attention(&q, 0, D, 2 * D, H, HD, 1.0 / (HD as f32).sqrt())?;
        drop(q);
        ops::add_gated(txt, &tm[2], &self.lin(&ops::rows(&a, 0, nt)?, &format!("{p}.txt_attn.proj"))?)?;
        ops::add_gated(img, &im[2], &self.lin(&ops::rows(&a, nt, nt + ni)?, &format!("{p}.img_attn.proj"))?)?;
        for (x, m, s) in [(img, &im, "img"), (txt, &tm, "txt")] {
            let mut f = self.lin(&ops::modulate(x, &m[3], &m[4])?, &format!("{p}.{s}_mlp.0"))?;
            ops::gelu_tanh(&mut f)?;
            ops::add_gated(x, &m[5], &self.lin(&f, &format!("{p}.{s}_mlp.2"))?)?;
        }
        Ok(())
    }

    fn single_block(&self, i: usize, x: &mut T, vec: &[f32], rope: &ops::Rope) -> Result<(), String> {
        let p = format!("single_blocks.{i}");
        let m = self.modulation(vec, &format!("{p}.modulation.lin"))?;
        let mut l1 = self.lin(&ops::modulate(x, &m[0], &m[1])?, &format!("{p}.linear1"))?;   // [n, 3D + 4D]
        self.qk_norm_rope(&mut l1, &format!("{p}.norm"), rope, 0)?;
        let a = ops::attention(&l1, 0, D, 2 * D, H, HD, 1.0 / (HD as f32).sqrt())?;
        // [attention | gelu(mlp)]: GELU runs once over the whole MLP part
        let mut g = ops::cols(&l1, 3 * D, l1.cols)?;
        drop(l1);
        ops::gelu_tanh(&mut g)?;
        let cat = ops::cat_cols(&a, &g)?;
        ops::add_gated(x, &m[2], &self.lin(&cat, &format!("{p}.linear2"))?)?;
        Ok(())
    }

    /// One call: latents [16, h, w], t in [0, 1], T5 hidden [n_txt, 4096], CLIP pooled [768] -> velocity [16, h, w].
    pub fn forward(&self, lat: &[f32], h: usize, w: usize, t: f32, txt: &[f32], pooled: &[f32]) -> Result<Vec<f32>, String> {
        let (h2, w2, nt) = (h / 2, w / 2, txt.len() / 4096);
        let row = |v: Vec<f32>| { let n = v.len(); T::host(v, 1, n) };
        let mut e = self.lin(&row(timestep_embedding(t * 1000.0, 256)), "time_in.in_layer")?.into_host()?;
        ffi::silu(&mut e);
        let mut vec = self.lin(&row(e), "time_in.out_layer")?.into_host()?;
        let mut y = self.lin(&row(pooled.to_vec()), "vector_in.in_layer")?.into_host()?;
        ffi::silu(&mut y);
        for (a, b) in vec.iter_mut().zip(self.lin(&row(y), "vector_in.out_layer")?.into_host()?) { *a += b; }
        let mut img = self.lin(&T::host(pack(lat, h, w), h2 * w2, 64), "img_in")?;
        let mut tx = self.lin(&T::host(txt.to_vec(), nt, 4096), "txt_in")?;
        let (cos, sin) = rope_tables(nt, h2, w2);
        let rope = ops::Rope::new(cos, sin);
        // stream: block i+1's weights are read from disk while block i computes
        let next = |i: usize| if i < self.double { format!("double_blocks.{i}.") } else { format!("single_blocks.{}.", i - self.double) };
        self.w.prefetch_async(&next(0));
        for i in 0..self.double { self.w.prefetch_async(&next(i + 1)); self.double_block(i, &mut img, &mut tx, &vec, &rope)?; }
        let mut tx = ops::cat_rows(tx, &img)?;                          // single stream: txt rows, then img rows
        drop(img);
        for i in 0..self.single { self.w.prefetch_async(&next(self.double + i + 1)); self.single_block(i, &mut tx, &vec, &rope)?; }
        let img = ops::rows(&tx, nt, tx.rows)?;
        drop(tx);
        let mut v = vec.clone();
        ffi::silu(&mut v);
        let sc = self.lin(&row(v), "final_layer.adaLN_modulation.1")?.into_host()?;    // BFL order: shift, scale
        let out = self.lin(&ops::modulate(&img, &sc[..D], &sc[D..])?, "final_layer.linear")?.into_host()?;
        Ok(unpack(&out, h, w))
    }
}

/// FLUX-schnell sampling: flow matching, Euler, sigmas = linspace(1, 1/n, n) then 0 (no shift for schnell).
/// The model predicts the velocity v; x <- x + (sigma_next - sigma) v. `noise` is the start latent [16, h, w].
/// `steer` (P5.1.7): asked at the start of every step; a new (text, pooled) conditioning replaces the old from this step on.
pub fn sample(m: &Flux, noise: Vec<f32>, h: usize, w: usize, txt: &[f32], pooled: &[f32], steps: usize,
              mut on_x0: Option<&mut dyn FnMut(usize, &crate::sd::Act) -> Result<(), String>>,
              mut steer: Option<&mut dyn FnMut(usize) -> Result<crate::sd::Steer<(Vec<f32>, Vec<f32>)>, String>>) -> Result<Vec<f32>, String> {
    use std::borrow::Cow;
    let sig: Vec<f32> = (0..=steps).map(|i| if i == steps { 0.0 } else { 1.0 - i as f32 / steps as f32 }).collect();
    let mut x = noise;
    let (mut txt_cur, mut pooled_cur): (Cow<[f32]>, Cow<[f32]>) = (Cow::Borrowed(txt), Cow::Borrowed(pooled));
    for i in 0..steps {
        crate::sd::check_cancel()?;
        if let Some(st) = steer.as_mut() {
            if let Some((t, pl)) = st(i)?.cond { txt_cur = Cow::Owned(t); pooled_cur = Cow::Owned(pl); }
        }
        crate::sd::progress("denoise", i, steps);
        let v = m.forward(&x, h, w, sig[i], &txt_cur, &pooled_cur)?;
        if let (Some(cb), true) = (on_x0.as_mut(), i + 1 < steps) {          // denoised estimate x0 = x - sigma * v (x_t = (1-s) x0 + s noise, v = noise - x0)
            let est: Vec<f32> = x.iter().zip(&v).map(|(xv, vv)| xv - sig[i] * vv).collect();
            cb(i, &crate::sd::Act { c: 16, h, w, d: est })?;
        }
        for (a, b) in x.iter_mut().zip(&v) { *a += (sig[i + 1] - sig[i]) * b; }
    }
    crate::sd::progress("denoise", steps, steps);
    Ok(x)
}

#[cfg(test)]
mod tests {
    #[test]
    fn pack_roundtrip() {
        let (h, w) = (4, 6);
        let lat: Vec<f32> = (0..16 * h * w).map(|i| i as f32).collect();
        assert_eq!(super::unpack(&super::pack(&lat, h, w), h, w), lat);
        // token 0, feature c*4 + ph*2 + pw: c=1, ph=1, pw=0 -> lat[1][1][0]
        assert_eq!(super::pack(&lat, h, w)[4 + 2], lat[(h + 1) * w]);
    }
}
