//! Stable Diffusion 1.x / 2.x / SD-Turbo model code on rope kernels. Weights are read in place from the mapped file.
#![forbid(unsafe_code)]

use crate::ffi;
use crate::ops::{self, T};
use crate::weights::Weights;

/// CLIP text encoders of SD checkpoints, one code path, two weight layouts:
/// - OpenCLIP (SD 2.x ViT-H, SDXL bigG): fused in_proj, gelu, ln_final.
/// - OpenAI CLIP ViT-L/14 (SD 1.x, SDXL first encoder; Hugging Face names): separate q/k/v, quick_gelu, final_layer_norm.
/// `layers`: blocks `encode` runs (SD 2.x: all but the last, then the final norm; SD 1.x: all).
pub struct TextEncoder<'w> { w: &'w Weights, prefix: String, hf: bool, d: usize, heads: usize, layers: usize, depth: usize }

const T: usize = 77;

impl<'w> TextEncoder<'w> {
    pub fn new(w: &'w Weights) -> Result<Self, String> {
        let find = |cands: &[&str], probe: &str| cands.iter().find(|p| w.tensor(&format!("{p}{probe}")).is_ok()).map(|p| p.to_string());
        if let Some(prefix) = find(&["conditioner.embedders.0.model.", "cond_stage_model.model."], "token_embedding.weight") {
            let mut te = Self::at(w, &prefix)?;
            te.layers = te.depth - 1;                              // SD 2.x: penultimate block, then ln_final
            return Ok(te);
        }
        let prefix = find(&["cond_stage_model.transformer.text_model.", "cond_stage_model.transformer.", "text_model."], "embeddings.token_embedding.weight")
            .ok_or("no CLIP text encoder in this file")?;
        Self::at(w, &prefix)
    }

    /// The encoder whose tensors start with `prefix` (layout and size read from the file).
    pub fn at(w: &'w Weights, prefix: &str) -> Result<Self, String> {
        let hf = w.tensor(&format!("{prefix}embeddings.token_embedding.weight")).is_ok();
        let tok = if hf { "embeddings.token_embedding.weight" } else { "token_embedding.weight" };
        let t = w.tensor(&format!("{prefix}{tok}")).map_err(|_| format!("no CLIP text encoder at {prefix}"))?;
        let d = *t.dims().last().unwrap() as usize;
        let blk = |l: usize| if hf { format!("{prefix}encoder.layers.{l}.layer_norm1.weight") } else { format!("{prefix}transformer.resblocks.{l}.ln_1.weight") };
        let depth = (0..).take_while(|&l| w.tensor(&blk(l)).is_ok()).count();
        Ok(TextEncoder { w, prefix: prefix.to_string(), hf, d, heads: d / 64, layers: depth, depth })
    }

    fn lin(&self, x: &[f32], m: usize, k: usize, name: &str, n: usize) -> Result<Vec<f32>, String> {
        let (wb, ty) = self.w.raw(&format!("{}{name}.weight", self.prefix))
            .or_else(|_| self.w.raw(&format!("{}{name}_weight", self.prefix)))?;
        let mut y = vec![0f32; m * n];
        ffi::matmul(x, m, k, wb, ty, n, &mut y);
        let bias = self.w.f32(&format!("{}{name}.bias", self.prefix)).or_else(|_| self.w.f32(&format!("{}{name}_bias", self.prefix)))?;
        ffi::add_rows(&mut y, &bias[..n]);
        Ok(y)
    }

    fn norm(&self, x: &mut [f32], name: &str) -> Result<(), String> {
        let g = self.w.f32(&format!("{}{name}.weight", self.prefix))?;
        let b = self.w.f32(&format!("{}{name}.bias", self.prefix))?;
        ffi::layernorm(x, T, self.d, Some(&g), Some(&b), 1e-5);
        Ok(())
    }

    /// Pooled output (what FLUX uses from CLIP-L): the hidden state at the first EOS token, after the final norm.
    pub fn pooled(hidden: &[f32], ids: &[u32; 77], eos: u32) -> Vec<f32> {
        let d = hidden.len() / T;
        let i = ids.iter().position(|&t| t == eos).unwrap_or(T - 1);
        hidden[i * d..(i + 1) * d].to_vec()
    }

    /// ids: 77 token ids -> hidden states [77, width] (`layers` blocks, then the final norm).
    pub fn encode(&self, ids: &[u32; 77]) -> Result<Vec<f32>, String> {
        let mut x = self.embed(ids)?;
        for l in 0..self.layers { self.layer(&mut x, l)?; }
        self.final_norm(&mut x)?;
        Ok(x)
    }

    /// Token + position embeddings [77, width].
    pub fn embed(&self, ids: &[u32; 77]) -> Result<Vec<f32>, String> {
        let (p, d, hf) = (&self.prefix, self.d, self.hf);
        let (tok, pos) = if hf { ("embeddings.token_embedding.weight", "embeddings.position_embedding.weight") } else { ("token_embedding.weight", "positional_embedding") };
        let mut x = self.w.f32_rows(&format!("{p}{tok}"), ids, d)?;
        let pos = self.w.f32(&format!("{p}{pos}"))?;
        for (a, b) in x.iter_mut().zip(&pos) { *a += b; }
        self.w.prefetch_async(&format!("{p}{}", self.block(0)));
        Ok(x)
    }

    fn block(&self, l: usize) -> String { if self.hf { format!("encoder.layers.{l}.") } else { format!("transformer.resblocks.{l}.") } }

    /// One transformer block (causal self-attention + MLP) on x [77, width].
    pub fn layer(&self, x: &mut [f32], l: usize) -> Result<(), String> {
        let (d, hf) = (self.d, self.hf);
        let ff = 4 * d;
        self.w.prefetch_async(&format!("{}{}", self.prefix, self.block(l + 1)));     // next block loads while this one runs
        let b = self.block(l);
        let n = |hf_name: &str, oc_name: &str| format!("{b}{}", if hf { hf_name } else { oc_name });
        // attention: x += out_proj(MHA(ln_1(x))) with a causal mask
        let mut h = x.to_vec();
        self.norm(&mut h, &n("layer_norm1", "ln_1"))?;
        let qkv = if hf {                                     // [T, 3d] like the fused projection
            let (q, k, v) = (self.lin(&h, T, d, &n("self_attn.q_proj", ""), d)?, self.lin(&h, T, d, &n("self_attn.k_proj", ""), d)?, self.lin(&h, T, d, &n("self_attn.v_proj", ""), d)?);
            (0..T).flat_map(|t| [&q, &k, &v].into_iter().flat_map(move |m| m[t * d..(t + 1) * d].iter().copied())).collect()
        } else { self.lin(&h, T, d, &n("", "attn.in_proj"), 3 * d)? };
        let mut o = vec![0f32; T * d];
        let hd = d / self.heads;
        for hh in 0..self.heads {
            let (q, k, v) = (&qkv[hh * hd..], &qkv[d + hh * hd..], &qkv[2 * d + hh * hd..]);
            ffi::attention(q, k, v, &mut o[hh * hd..], T, T, hd, 3 * d, 3 * d, 3 * d, d, 1.0 / (hd as f32).sqrt(), true);
        }
        let a = self.lin(&o, T, d, &n("self_attn.out_proj", "attn.out_proj"), d)?;
        for (xi, ai) in x.iter_mut().zip(&a) { *xi += ai; }
        // MLP: x += fc2(act(fc1(ln_2(x))))
        let mut h = x.to_vec();
        self.norm(&mut h, &n("layer_norm2", "ln_2"))?;
        let mut f = self.lin(&h, T, d, &n("mlp.fc1", "mlp.c_fc"), ff)?;
        if hf { for v in &mut f { *v *= 1.0 / (1.0 + (-1.702 * *v).exp()); } }   // quick_gelu
        else { ffi::gelu_erf(&mut f); }
        let f2 = self.lin(&f, T, ff, &n("mlp.fc2", "mlp.c_proj"), d)?;
        for (xi, fi) in x.iter_mut().zip(&f2) { *xi += fi; }
        Ok(())
    }

    pub fn final_norm(&self, x: &mut [f32]) -> Result<(), String> { self.norm(x, if self.hf { "final_layer_norm" } else { "ln_final" }) }
}

/// Machine-readable progress for `rope serve` (stderr line `progress: <stage> <i>/<n>`), only when ROPE_PROGRESS=1.
pub fn progress(stage: &str, i: usize, n: usize) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *ON.get_or_init(|| std::env::var("ROPE_PROGRESS").as_deref() == Ok("1")) { eprintln!("progress: {stage} {i}/{n}"); }
    if let Ok(h) = HOOK.read() { if let Some(f) = h.as_ref() { f(stage, i, n); } }
}

/// A callback for the library (librope): called at every progress point, from the thread that runs the job.
pub type ProgressHook = Box<dyn Fn(&str, usize, usize) + Send + Sync>;
static HOOK: std::sync::RwLock<Option<ProgressHook>> = std::sync::RwLock::new(None);
pub fn set_progress_hook(h: Option<ProgressHook>) { if let Ok(mut g) = HOOK.write() { *g = h; } }

static CANCEL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Ask the running job to stop at its next step (denoising step, decoder tile, upscaler tile).
pub fn request_cancel(on: bool) { CANCEL.store(on, std::sync::atomic::Ordering::SeqCst); }
pub fn check_cancel() -> Result<(), String> {
    if CANCEL.load(std::sync::atomic::Ordering::SeqCst) { Err("cancelled".into()) } else { Ok(()) }
}

/// SDXL VAE latent scale.
pub const SDXL_VAE: (f32, f32) = (0.13025, 0.0);

/// SDXL conditioning for one prompt (None = the empty-prompt "uncond", all zeros as in SDXL's zero-for-empty rule):
/// [77 x 2048] context (CLIP-L and CLIP-G penultimate hidden states, no final norm, side by side) followed by the
/// 2816-wide vector [pooled CLIP-G (ln_final at EOS, x text_projection), sinusoid(256) of h, w, 0, 0, h, w].
/// The UNet takes the vector off the end of the context (`UNet::forward`).
pub fn sdxl_condition(w: &Weights, ids: Option<(&[u32; 77], &[u32; 77], u32)>, size: usize) -> Result<Vec<f32>, String> {
    let (gw, lw) = (1280usize, 768usize);
    let mut ctx = vec![0f32; T * (lw + gw)];
    let mut pooled = vec![0f32; gw];
    if let Some((il, ig, eos)) = ids {
        let l = TextEncoder::at(w, "conditioner.embedders.0.transformer.text_model.")?;
        let g = TextEncoder::at(w, "conditioner.embedders.1.model.")?;
        let mut xl = l.embed(il)?;
        for i in 0..l.depth - 1 { l.layer(&mut xl, i)?; }
        let mut xg = g.embed(ig)?;
        for i in 0..g.depth - 1 { g.layer(&mut xg, i)?; }
        for t in 0..T {
            ctx[t * (lw + gw)..t * (lw + gw) + lw].copy_from_slice(&xl[t * lw..(t + 1) * lw]);
            ctx[t * (lw + gw) + lw..(t + 1) * (lw + gw)].copy_from_slice(&xg[t * gw..(t + 1) * gw]);
        }
        g.layer(&mut xg, g.depth - 1)?;
        g.final_norm(&mut xg)?;
        let e = TextEncoder::pooled(&xg, ig, eos);
        // OpenCLIP: pooled = x @ text_projection, the matrix stored [in, out] (not a Linear weight)
        let p = w.f32("conditioner.embedders.1.model.text_projection")?;
        for (j, o) in pooled.iter_mut().enumerate() { *o = (0..gw).map(|i| e[i] * p[i * gw + j]).sum(); }
    }
    ctx.extend(pooled);
    for v in [size, size, 0, 0, size, size] { ctx.extend(timestep_embedding(v as f32, 256)); }
    Ok(ctx)
}

// ======================================================================================================
// SD 1.x / 2.x UNet (ldm layout). SD 1.x: 1x1-conv proj_in/out, 8 heads, 768-wide context.
// SD 2.x: linear proj_in/out, 64 channels per head, 1024-wide context. A 1x1 conv on [c, hw] equals a linear
// layer on the [hw, c] tokens with the same weights, so one code path serves both.
// ======================================================================================================

/// Activation in channel-major layout [c, h, w].
#[derive(Clone)]
pub struct Act { pub c: usize, pub h: usize, pub w: usize, pub d: Vec<f32> }

/// An image activation on the op interface: T [channels, h*w].
struct Im { t: T, h: usize, w: usize }

/// `adm`: width of the extra conditioning vector (SDXL label_emb input, 0 = none); `n_in`/`n_out`: block counts.
pub struct UNet<'w> { w: &'w Weights, ctx_w: usize, fixed_heads: Option<usize>, adm: usize, n_in: usize, n_out: usize }

const UP: &str = "model.diffusion_model.";

/// ldm timestep embedding: [cos(t*f), sin(t*f)], f_i = 10000^(-i/half).
pub fn timestep_embedding(t: f32, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let f = |i: usize| (-(10000f64.ln()) * i as f64 / half as f64).exp();
    let mut e: Vec<f32> = (0..half).map(|i| (t as f64 * f(i)).cos() as f32).collect();
    e.extend((0..half).map(|i| (t as f64 * f(i)).sin() as f32));
    e
}

impl<'w> UNet<'w> {
    pub fn new(w: &'w Weights) -> Result<Self, String> {
        w.tensor(&format!("{UP}input_blocks.0.0.weight")).map_err(|_| "no SD UNet in this file".to_string())?;
        // context width from the first cross-attention (SDXL has none in its first level)
        let k = (1..12).find_map(|i| w.tensor(&format!("{UP}input_blocks.{i}.1.transformer_blocks.0.attn2.to_k.weight")).ok())
            .ok_or("no cross-attention in this UNet")?;
        let ctx_w = k.dims()[1] as usize;
        let adm = w.tensor(&format!("{UP}label_emb.0.0.weight")).map_or(0, |t| t.dims()[1] as usize);
        let count = |part: &str| (0..).take_while(|i| !w.ranges(&format!("{UP}{part}.{i}.")).is_empty()).count();
        // ldm configs: SD 1.x num_heads = 8 (head width grows with channels); SD 2.x and SDXL num_head_channels = 64.
        Ok(UNet { w, ctx_w, fixed_heads: (ctx_w == 768).then_some(8), adm, n_in: count("input_blocks"), n_out: count("output_blocks") })
    }

    fn has(&self, name: &str) -> bool { self.w.tensor(&format!("{UP}{name}")).is_ok() }

    // All UNet math goes through the op interface (ops.rs): each op runs where its weights are placed (GPU or CPU);
    // the CPU side is the exact code used before, so CPU results are unchanged. Images are T [channels, h*w].

    fn lin(&self, x: &T, name: &str) -> Result<T, String> { ops::lin_opt(self.w, x, &format!("{UP}{name}")) }

    fn conv(&self, x: &Im, name: &str, stride: usize, pad: usize) -> Result<Im, String> {
        let (t, h, w) = ops::conv(self.w, &x.t, x.h, x.w, &format!("{UP}{name}"), stride, pad)?;
        Ok(Im { t, h, w })
    }

    fn gn(&self, x: &mut T, name: &str, eps: f32) -> Result<(), String> {
        let g = self.w.f32(&format!("{UP}{name}.weight"))?;
        let b = self.w.f32(&format!("{UP}{name}.bias"))?;
        ops::groupnorm(x, 32, &g, &b, eps)
    }

    fn ln(&self, x: &mut T, name: &str) -> Result<(), String> {
        let g = self.w.f32(&format!("{UP}{name}.weight"))?;
        let b = self.w.f32(&format!("{UP}{name}.bias"))?;
        ops::layernorm(x, Some(&g), Some(&b), 1e-5)
    }

    fn resblock(&self, x: &Im, emb: &[f32], name: &str) -> Result<Im, String> {
        let mut h = Im { t: ops::clone(&x.t)?, h: x.h, w: x.w };
        self.gn(&mut h.t, &format!("{name}.in_layers.0"), 1e-5)?;
        ops::silu(&mut h.t)?;
        let mut h = self.conv(&h, &format!("{name}.in_layers.2"), 1, 1)?;
        let mut e = emb.to_vec();
        ffi::silu(&mut e);
        let n = e.len();
        let e = self.lin(&T::host(e, 1, n), &format!("{name}.emb_layers.1"))?.into_host()?;
        ops::add_row_scalars(&mut h.t, &e)?;
        self.gn(&mut h.t, &format!("{name}.out_layers.0"), 1e-5)?;
        ops::silu(&mut h.t)?;
        let mut h = self.conv(&h, &format!("{name}.out_layers.3"), 1, 1)?;
        if self.has(&format!("{name}.skip_connection.weight")) {
            let skip = self.conv(x, &format!("{name}.skip_connection"), 1, 0)?;
            ops::add(&mut h.t, &skip.t)?;
        } else { ops::add(&mut h.t, &x.t)?; }
        Ok(h)
    }

    fn attention(&self, x: &T, ctx: Option<&T>, name: &str) -> Result<T, String> {
        let c = x.cols;
        let q = self.lin(x, &format!("{name}.to_q"))?;
        let kv = ctx.unwrap_or(x);
        let k = self.lin(kv, &format!("{name}.to_k"))?;
        let v = self.lin(kv, &format!("{name}.to_v"))?;
        let hd = self.fixed_heads.map_or(64, |h| c / h);
        let o = ops::attention3(&q, &k, &v, c / hd, hd, 1.0 / (hd as f32).sqrt())?;
        self.lin(&o, &format!("{name}.to_out.0"))
    }

    fn transformer(&self, x: &Im, ctx: &T, name: &str) -> Result<Im, String> {
        let mut h = ops::clone(&x.t)?;
        self.gn(&mut h, &format!("{name}.norm"), 1e-6)?;
        let mut t = self.lin(&ops::transpose(&h)?, &format!("{name}.proj_in"))?;     // tokens [hw, c]
        drop(h);
        // SD 1.x / 2.x: one transformer block; SDXL: 2 or 10 (as many as the file has)
        for j in (0..).take_while(|j| self.has(&format!("{name}.transformer_blocks.{j}.norm1.weight"))) {
            let b = format!("{name}.transformer_blocks.{j}");
            let mut a = ops::clone(&t)?;
            self.ln(&mut a, &format!("{b}.norm1"))?;
            ops::add(&mut t, &self.attention(&a, None, &format!("{b}.attn1"))?)?;
            let mut a = ops::clone(&t)?;
            self.ln(&mut a, &format!("{b}.norm2"))?;
            ops::add(&mut t, &self.attention(&a, Some(ctx), &format!("{b}.attn2"))?)?;
            let mut a = ops::clone(&t)?;
            self.ln(&mut a, &format!("{b}.norm3"))?;
            let g = self.lin(&a, &format!("{b}.ff.net.0.proj"))?;                         // GEGLU: hidden * gelu(gate)
            let inner = g.cols / 2;
            let mut gate = ops::cols(&g, inner, g.cols)?;
            ops::gelu_erf(&mut gate)?;
            let mut hid = ops::cols(&g, 0, inner)?;
            drop(g);
            ops::mul(&mut hid, &gate)?;
            ops::add(&mut t, &self.lin(&hid, &format!("{b}.ff.net.2"))?)?;
        }
        let t = self.lin(&t, &format!("{name}.proj_out"))?;
        let mut out = ops::clone(&x.t)?;
        ops::add(&mut out, &ops::transpose(&t)?)?;
        Ok(Im { t: out, h: x.h, w: x.w })
    }

    fn upsample(x: &Act) -> Act {
        let (h2, w2) = (x.h * 2, x.w * 2);
        let mut d = vec![0f32; x.c * h2 * w2];
        for c in 0..x.c { for y in 0..h2 { for xx in 0..w2 { d[(c * h2 + y) * w2 + xx] = x.d[(c * x.h + y / 2) * x.w + xx / 2]; } } }
        Act { c: x.c, h: h2, w: w2, d }
    }

    /// One UNet call: latent [4, h, w], timestep t, context [77, ctx width] -> noise prediction [4, h, w].
    /// SDXL: the context is followed by its extra conditioning vector (`adm` values, see `sdxl_condition`).
    pub fn forward(&self, latent: &Act, t: f32, ctx: &[f32]) -> Result<Act, String> {
        let row = |v: Vec<f32>| { let n = v.len(); T::host(v, 1, n) };
        let (ctx, y) = ctx.split_at(ctx.len() - self.adm);
        let mut e = self.lin(&row(timestep_embedding(t, 320)), "time_embed.0")?.into_host()?;
        ffi::silu(&mut e);
        let mut emb = self.lin(&row(e), "time_embed.2")?.into_host()?;
        if self.adm > 0 {                                          // SDXL: emb += label_emb(y)
            let mut a = self.lin(&row(y.to_vec()), "label_emb.0.0")?.into_host()?;
            ffi::silu(&mut a);
            let a = self.lin(&row(a), "label_emb.0.2")?.into_host()?;
            for (x, v) in emb.iter_mut().zip(&a) { *x += v; }
        }
        let ctx = T::host(ctx.to_vec(), ctx.len() / self.ctx_w, self.ctx_w);
        // weights of the next block are prefetched (read + mapped) while the current block computes
        let order: Vec<String> = (0..self.n_in).map(|i| format!("input_blocks.{i}.")).chain(["middle_block.".to_string()])
            .chain((0..self.n_out).map(|i| format!("output_blocks.{i}."))).chain(["out.".to_string()]).collect();
        let ahead = |i: usize| if let Some(n) = order.get(i) { self.w.prefetch_async(&format!("{UP}{n}")) };
        ahead(0); ahead(1);
        let x = Im { t: T::host(latent.d.clone(), latent.c, latent.h * latent.w), h: latent.h, w: latent.w };
        let mut hs: Vec<Im> = Vec::new();
        let mut h = self.conv(&x, "input_blocks.0.0", 1, 1)?;
        hs.push(Im { t: ops::clone(&h.t)?, h: h.h, w: h.w });
        for i in 1..self.n_in {
            ahead(i + 1);
            let b = format!("input_blocks.{i}");
            if self.has(&format!("{b}.0.op.weight")) { h = self.conv(&h, &format!("{b}.0.op"), 2, 1)?; } else {
                h = self.resblock(&h, &emb, &format!("{b}.0"))?;
                if self.has(&format!("{b}.1.norm.weight")) { h = self.transformer(&h, &ctx, &format!("{b}.1"))?; }
            }
            hs.push(Im { t: ops::clone(&h.t)?, h: h.h, w: h.w });
        }
        ahead(self.n_in + 1);
        h = self.resblock(&h, &emb, "middle_block.0")?;
        h = self.transformer(&h, &ctx, "middle_block.1")?;
        h = self.resblock(&h, &emb, "middle_block.2")?;
        for i in 0..self.n_out {
            ahead(self.n_in + 2 + i);
            let b = format!("output_blocks.{i}");
            let skip = hs.pop().ok_or("skip stack empty")?;
            let cat = Im { t: ops::cat_rows(h.t, &skip.t)?, h: h.h, w: h.w };        // channel concat [h; skip]
            h = self.resblock(&cat, &emb, &format!("{b}.0"))?;
            let mut j = 1;
            if self.has(&format!("{b}.1.norm.weight")) { h = self.transformer(&h, &ctx, &format!("{b}.1"))?; j = 2; }
            if self.has(&format!("{b}.{j}.conv.weight")) {
                let up = Im { t: ops::upsample2x(&h.t, h.h, h.w)?, h: h.h * 2, w: h.w * 2 };
                h = self.conv(&up, &format!("{b}.{j}.conv"), 1, 1)?;
            }
        }
        self.gn(&mut h.t, "out.0", 1e-5)?;
        ops::silu(&mut h.t)?;
        let o = self.conv(&h, "out.2", 1, 1)?;
        Ok(Act { c: o.t.rows, h: o.h, w: o.w, d: o.t.into_host()? })
    }
}

/// conv with an optional bias, for decoders whose convs may have none.
fn conv_any(w: &Weights, x: &Act, name: &str, stride: usize, pad: usize) -> Result<Act, String> {
    let t = w.tensor(&format!("{name}.weight"))?;
    let s = t.dims();
    let (o, c, kh, kw) = (s[0] as usize, s[1] as usize, s[2] as usize, s[3] as usize);
    if c != x.c { return Err(format!("{name}: expects {c} channels, got {}", x.c)); }
    let (wb, ty) = w.raw(&format!("{name}.weight"))?;
    let b = w.f32(&format!("{name}.bias")).ok();
    let mut out = Vec::new();
    let (h, ww) = ffi::conv2d(&x.d, c, x.h, x.w, wb, ty, b.as_deref(), o, kh, kw, stride, pad, &mut out);
    Ok(Act { c: o, h, w: ww, d: out })
}

fn relu(x: &mut Act) { for v in &mut x.d { *v = v.max(0.0); } }

/// TAESD decoder (madebyollin/taesd, diffusers DecoderTiny). Input: diffusion latents [4,h,w];
/// output: image [3, 8h, 8w] in [-1, 1] (diffusers convention).
pub struct Taesd<'w> { w: &'w Weights }

impl<'w> Taesd<'w> {
    pub fn new(w: &'w Weights) -> Result<Self, String> {
        w.tensor("decoder.layers.0.weight").map_err(|_| "not a TAESD file (decoder.layers.0 missing)".to_string())?;
        Ok(Taesd { w })
    }

    fn block(&self, x: &Act, i: usize) -> Result<Act, String> {
        let p = format!("decoder.layers.{i}.conv");
        let mut h = conv_any(self.w, x, &format!("{p}.0"), 1, 1)?;
        relu(&mut h);
        let mut h = conv_any(self.w, &h, &format!("{p}.2"), 1, 1)?;
        relu(&mut h);
        let mut h = conv_any(self.w, &h, &format!("{p}.4"), 1, 1)?;
        let skip = if self.w.tensor(&format!("decoder.layers.{i}.skip.weight")).is_ok() { conv_any(self.w, x, &format!("decoder.layers.{i}.skip"), 1, 0)? } else { x.clone() };
        for (a, b) in h.d.iter_mut().zip(&skip.d) { *a = (*a + b).max(0.0); }
        Ok(h)
    }

    pub fn decode(&self, latent: &Act) -> Result<Act, String> {
        let mut x = latent.clone();
        for v in &mut x.d { *v = (*v / 3.0).tanh() * 3.0; }
        let mut h = conv_any(self.w, &x, "decoder.layers.0", 1, 1)?;
        relu(&mut h);
        for (blocks, up_conv) in [([2, 3, 4], Some(6)), ([7, 8, 9], Some(11)), ([12, 13, 14], Some(16))] {
            for b in blocks { h = self.block(&h, b)?; }
            if let Some(c) = up_conv { h = conv_any(self.w, &UNet::upsample(&h), &format!("decoder.layers.{c}"), 1, 1)?; }
        }
        h = self.block(&h, 17)?;
        let mut out = conv_any(self.w, &h, "decoder.layers.18", 1, 1)?;
        for v in &mut out.d { *v = *v * 2.0 - 1.0; }
        Ok(out)
    }
}

// ======================================================================================================
// SD-Turbo text-to-image: Euler (epsilon prediction, "trailing" spacing) for 1..n steps
// ======================================================================================================

/// sigma(t) for the scaled-linear schedule (beta 0.00085..0.012, 1000 steps), as diffusers computes it.
pub fn sigma_at(t: usize) -> f32 {
    let (b0, b1) = (0.00085f32.sqrt(), 0.012f32.sqrt());
    let mut ac = 1.0f32;
    for i in 0..=t {
        let beta = (b0 + (b1 - b0) * i as f32 / 999.0).powi(2);
        ac *= 1.0 - beta;
    }
    ((1.0 - ac) / ac).sqrt()
}

/// Timesteps for "trailing" spacing: round(arange(1000, 0, -1000/n)) - 1.
pub fn trailing_timesteps(n: usize) -> Vec<usize> {
    (0..n).map(|i| (1000.0 - i as f64 * 1000.0 / n as f64).round() as usize - 1).collect()
}

/// Standard normal noise from a seeded xorshift64* + Box-Muller (rope's own, reproducible on every platform).
pub fn gaussian_noise(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
    let mut u = || { s ^= s >> 12; s ^= s << 25; s ^= s >> 27; ((s.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 + 0.5) / (1u64 << 53) as f64 };
    let mut out = Vec::with_capacity(n + 1);
    while out.len() < n {
        let (a, b) = (u(), u());
        let r = (-2.0 * a.ln()).sqrt();
        out.push((r * (2.0 * std::f64::consts::PI * b).cos()) as f32);
        out.push((r * (2.0 * std::f64::consts::PI * b).sin()) as f32);
    }
    out.truncate(n);
    out
}

/// Full ldm VAE decoder: latents [z,h,w] -> image [3,8h,8w] in [-1,1]. SD 1.x/2.x: inside the checkpoint under
/// `first_stage_model.`, 4 channels, with post_quant_conv. FLUX `ae.safetensors`: bare names, 16 channels, none.
pub struct VaeDecoder<'w> { w: &'w Weights, p: &'static str, pub z: usize, scale: f32, shift: f32 }

/// Latent scaling of the SD 1.x/2.x VAE.
pub const SD_VAE: (f32, f32) = (0.18215, 0.0);
/// Latent scaling of the FLUX VAE.
pub const FLUX_VAE: (f32, f32) = (0.3611, 0.1159);

impl<'w> VaeDecoder<'w> {
    /// `(scale, shift)` belong to the model family (SD_VAE, FLUX_VAE): decode(x) = D(x / scale + shift).
    pub fn new(w: &'w Weights, (scale, shift): (f32, f32)) -> Result<Self, String> {
        let p = ["first_stage_model.", ""].into_iter().find(|p| w.tensor(&format!("{p}decoder.conv_in.weight")).is_ok())
            .ok_or("no VAE decoder in this file")?;
        let z = w.tensor(&format!("{p}decoder.conv_in.weight"))?.dims()[1] as usize;
        Ok(VaeDecoder { w, p, z, scale, shift })
    }

    // On the op interface like the UNet: each conv runs where its weights are placed; CPU results unchanged.
    fn gn_silu(&self, x: &T, name: &str, silu: bool) -> Result<T, String> {
        let mut h = ops::clone(x)?;
        let g = self.w.f32(&format!("{}{name}.weight", self.p))?;
        let b = self.w.f32(&format!("{}{name}.bias", self.p))?;
        ops::groupnorm(&mut h, 32, &g, &b, 1e-6)?;
        if silu { ops::silu(&mut h)?; }
        Ok(h)
    }

    fn conv(&self, x: &Im, name: &str, pad: usize) -> Result<Im, String> {
        let (t, h, w) = ops::conv(self.w, &x.t, x.h, x.w, &format!("{}{name}", self.p), 1, pad)?;
        Ok(Im { t, h, w })
    }

    fn resnet(&self, x: &Im, name: &str) -> Result<Im, String> {
        let h = Im { t: self.gn_silu(&x.t, &format!("{name}.norm1"), true)?, h: x.h, w: x.w };
        let h = self.conv(&h, &format!("{name}.conv1"), 1)?;
        let h = Im { t: self.gn_silu(&h.t, &format!("{name}.norm2"), true)?, h: h.h, w: h.w };
        let mut h = self.conv(&h, &format!("{name}.conv2"), 1)?;
        if self.w.tensor(&format!("{}{name}.nin_shortcut.weight", self.p)).is_ok() {
            let skip = self.conv(x, &format!("{name}.nin_shortcut"), 0)?;
            ops::add(&mut h.t, &skip.t)?;
        } else { ops::add(&mut h.t, &x.t)?; }
        Ok(h)
    }

    /// Single-head self-attention over all pixels (q, k, v, proj_out are 1x1 convs).
    fn attn(&self, x: &Im, name: &str) -> Result<Im, String> {
        let h = Im { t: self.gn_silu(&x.t, &format!("{name}.norm"), false)?, h: x.h, w: x.w };
        let c = h.t.rows;
        let q = ops::transpose(&self.conv(&h, &format!("{name}.q"), 0)?.t)?;
        let k = ops::transpose(&self.conv(&h, &format!("{name}.k"), 0)?.t)?;
        let v = ops::transpose(&self.conv(&h, &format!("{name}.v"), 0)?.t)?;
        let o = ops::attention3(&q, &k, &v, 1, c, 1.0 / (c as f32).sqrt())?;
        let oa = Im { t: ops::transpose(&o)?, h: x.h, w: x.w };
        let mut out = self.conv(&oa, &format!("{name}.proj_out"), 0)?;
        ops::add(&mut out.t, &x.t)?;
        Ok(out)
    }

    pub fn decode(&self, latent: &Act) -> Result<Act, String> {
        self.w.prefetch_async(&format!("{}decoder.", self.p));          // 0.3 GB, reused by every tile
        let z: Vec<f32> = latent.d.iter().map(|v| *v / self.scale + self.shift).collect();
        let z = Im { t: T::host(z, latent.c, latent.h * latent.w), h: latent.h, w: latent.w };
        let z = if self.w.tensor(&format!("{}post_quant_conv.weight", self.p)).is_ok() { self.conv(&z, "post_quant_conv", 0)? } else { z };
        let mut h = self.conv(&z, "decoder.conv_in", 1)?;
        h = self.resnet(&h, "decoder.mid.block_1")?;
        h = self.attn(&h, "decoder.mid.attn_1")?;
        h = self.resnet(&h, "decoder.mid.block_2")?;
        for lvl in (0..4).rev() {
            for b in 0..3 { h = self.resnet(&h, &format!("decoder.up.{lvl}.block.{b}"))?; }
            if lvl != 0 {
                let up = Im { t: ops::upsample2x(&h.t, h.h, h.w)?, h: h.h * 2, w: h.w * 2 };
                h = self.conv(&up, &format!("decoder.up.{lvl}.upsample.conv"), 1)?;
            }
        }
        let h = Im { t: self.gn_silu(&h.t, "decoder.norm_out", true)?, h: h.h, w: h.w };
        let o = self.conv(&h, "decoder.conv_out", 1)?;
        Ok(Act { c: o.t.rows, h: o.h, w: o.w, d: o.t.into_host()? })
    }
}

/// Tile origins along one latent axis of length n: stride tile - overlap, the last tile flush with the end.
pub fn tile_starts(n: usize, tile: usize, overlap: usize) -> Vec<usize> {
    if n <= tile { return vec![0]; }
    let mut v: Vec<usize> = (0..).map(|i| i * (tile - overlap)).take_while(|&s| s + tile < n).collect();
    v.push(n - tile);
    v
}

impl VaeDecoder<'_> {
    /// Decode in latent tiles of `tile` (with `overlap`) so memory stays at one tile's size; tiles are blended with
    /// linear ramps over the overlap (weight 1 at image borders), then normalized. Small latents decode in one piece.
    pub fn decode_tiled(&self, latent: &Act, tile: usize, overlap: usize) -> Result<Act, String> {
        let (c, h, w) = (latent.c, latent.h, latent.w);
        if h <= tile && w <= tile {
            progress("decode", 0, 1);
            let r = self.decode(latent);
            progress("decode", 1, 1);
            return r;
        }
        let (ys, xs) = (tile_starts(h, tile, overlap), tile_starts(w, tile, overlap));
        let (hh, ww, ramp) = (h * 8, w * 8, (overlap * 8) as f32);
        let mut acc = vec![0f32; 3 * hh * ww];
        let mut wsum = vec![0f32; hh * ww];
        // weight of pixel i in a tile of n pixels: ramps in from edges that have a neighbour
        let wt = |i: usize, n: usize, lo: bool, hi: bool| -> f32 {
            let mut v = 1f32;
            if lo { v = v.min((i as f32 + 0.5) / ramp); }
            if hi { v = v.min((n as f32 - i as f32 - 0.5) / ramp); }
            v
        };
        let (n_tiles, mut done_tiles) = (ys.len() * xs.len(), 0usize);
        progress("decode", 0, n_tiles);
        for &y0 in &ys { for &x0 in &xs {
            check_cancel()?;
            let (th, tw) = (tile.min(h), tile.min(w));
            let mut d = Vec::with_capacity(c * th * tw);
            for ch in 0..c { for y in y0..y0 + th { d.extend_from_slice(&latent.d[(ch * h + y) * w + x0..(ch * h + y) * w + x0 + tw]); } }
            let img = self.decode(&Act { c, h: th, w: tw, d })?;
            let (ph, pw) = (img.h, img.w);
            done_tiles += 1;
            progress("decode", done_tiles, n_tiles);
            for y in 0..ph {
                let wy = wt(y, ph, y0 > 0, y0 + th < h);
                for x in 0..pw {
                    let wgt = wy * wt(x, pw, x0 > 0, x0 + tw < w);
                    let o = (y0 * 8 + y) * ww + x0 * 8 + x;
                    wsum[o] += wgt;
                    for ch in 0..3 { acc[ch * hh * ww + o] += wgt * img.d[(ch * ph + y) * pw + x]; }
                }
            }
        } }
        for ch in 0..3 { for (a, s) in acc[ch * hh * ww..(ch + 1) * hh * ww].iter_mut().zip(&wsum) { *a /= s; } }
        Ok(Act { c: 3, h: hh, w: ww, d: acc })
    }
}

/// Default VAE tiling: 64-latent (512 px) tiles, 16-latent (128 px) overlap.
pub const VAE_TILE: (usize, usize) = (64, 16);
/// Overlap of two neighbouring VAE tiles for a tile of `tile` latent pixels: a quarter (64 -> 16, the measured default).
pub fn vae_overlap(tile: usize) -> usize { (tile / 4).max(4) }

/// Euler sampling from `latent` (already scaled by the initial sigma) for the given timesteps.
/// `uncond = Some((empty-prompt context, g))`: classifier-free guidance, eps = eps_u + g (eps_c - eps_u), two UNet calls per step.
/// What the user changed while a job runs (P5.1.7), asked for at a step boundary: `cond` = a new text conditioning
/// (SD / SDXL: the context vector; FLUX: (text, pooled)), `cfg` = a new guidance scale.
#[derive(Default)]
pub struct Steer<T> { pub cond: Option<T>, pub cfg: Option<f32> }

/// `steer`: asked at the start of every step; may block (pause) and returns the changes to apply from this step on.
/// `on_x0`: called after every step but the last with the step index and the denoised estimate x0 = x - sigma * eps
/// (what the image would be if sampling stopped here): the live preview. Costs nothing when None.
pub fn euler_sample(unet: &UNet, latent: Act, ctx: &[f32], uncond: Option<(&[f32], f32)>, steps: &[usize],
                    mut on_x0: Option<&mut dyn FnMut(usize, &Act) -> Result<(), String>>,
                    mut steer: Option<&mut dyn FnMut(usize) -> Result<Steer<Vec<f32>>, String>>) -> Result<Act, String> {
    let mut x = latent;
    let mut ctx_cur: std::borrow::Cow<[f32]> = std::borrow::Cow::Borrowed(ctx);
    let mut guide = uncond;
    for (i, &t) in steps.iter().enumerate() {
        check_cancel()?;
        if let Some(st) = steer.as_mut() {
            let s = st(i)?;
            if let Some(c) = s.cond { ctx_cur = std::borrow::Cow::Owned(c); }
            if let (Some(g), Some((uc, _))) = (s.cfg, guide) { guide = Some((uc, g)); }       // only a model that uses guidance
        }
        progress("denoise", i, steps.len());
        let sigma = sigma_at(t);
        let scale = 1.0 / (sigma * sigma + 1.0).sqrt();
        let inp = Act { d: x.d.iter().map(|v| v * scale).collect(), ..x.clone() };
        let mut eps = unet.forward(&inp, t as f32, &ctx_cur)?;
        if let Some((uc, g)) = guide {
            let eu = unet.forward(&inp, t as f32, uc)?;     // ponytail: two batch-1 calls; one batch-2 pass would read the weights once
            for (e, u) in eps.d.iter_mut().zip(&eu.d) { *e = u + g * (*e - u); }
        }
        if let (Some(cb), true) = (on_x0.as_mut(), i + 1 < steps.len()) {
            let x0 = Act { d: x.d.iter().zip(&eps.d).map(|(xv, e)| xv - sigma * e).collect(), ..x.clone() };
            cb(i, &x0)?;
        }
        let sigma_next = steps.get(i + 1).map_or(0.0, |&tn| sigma_at(tn));
        for (xv, e) in x.d.iter_mut().zip(&eps.d) { *xv += e * (sigma_next - sigma); }   // x + d*dt, d = eps
    }
    progress("denoise", steps.len(), steps.len());
    Ok(x)
}

#[cfg(test)]
mod tests {
    #[test]
    fn tile_starts_cover_the_axis() {
        assert_eq!(super::tile_starts(64, 64, 16), vec![0]);
        assert_eq!(super::tile_starts(128, 64, 16), vec![0, 48, 64]);
        assert_eq!(super::tile_starts(96, 64, 16), vec![0, 32]);
        for n in [65usize, 100, 128, 160, 200] {
            let s = super::tile_starts(n, 64, 16);
            assert_eq!(*s.last().unwrap() + 64, n);                          // last tile flush with the end
            assert!(s.windows(2).all(|w| w[1] > w[0] && w[1] - w[0] <= 48)); // overlap >= 16 everywhere
        }
    }
}
