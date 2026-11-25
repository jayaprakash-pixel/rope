//! Model profile: what a model file costs, per block — weight bytes and stored types (from the file) and FLOPs
//! per call plus the activation peak (from one dry counting run of rope's own model code, see `ffi::Counter`).
//! The same code profiles every model rope runs: blocks come from tensor names, FLOPs from the kernels, so a new
//! architecture needs no profile code of its own. The planner places blocks on devices from this profile.
#![forbid(unsafe_code)]

use crate::weights::Weights;
use std::collections::BTreeMap;

/// Component a tensor belongs to: "text" (runs once), "denoiser" (runs every step), "vae" (runs once).
pub fn component(name: &str) -> &'static str {
    const DENOISER: [&str; 8] = ["model.diffusion_model.", "double_blocks.", "single_blocks.", "img_in.", "txt_in.", "time_in.", "vector_in.", "final_layer."];
    const TEXT: [&str; 6] = ["cond_stage_model.", "conditioner.", "text_model.", "enc.", "token_embd.", "text_projection"];
    const VAE: [&str; 5] = ["first_stage_model.", "decoder.", "encoder.", "post_quant_conv.", "quant_conv."];
    if DENOISER.iter().any(|p| name.starts_with(p)) || name.starts_with("guidance_in.") { "denoiser" }
    else if TEXT.iter().any(|p| name.starts_with(p)) { "text" }
    else if VAE.iter().any(|p| name.starts_with(p)) { "vae" }
    else { "other" }
}

/// Block of a tensor: the name up to its first numbered segment ("model.diffusion_model.input_blocks.3",
/// "double_blocks.7", "enc.blk.4", "first_stage_model.decoder.up.2"); without a number, its first two segments.
pub fn block_of(name: &str) -> String {
    let segs: Vec<&str> = name.split('.').collect();
    // the checkpoint wrapper prefixes are not blocks
    let skip = match segs.as_slice() {
        ["model", "diffusion_model", ..] | ["cond_stage_model", "transformer", ..] => 2,
        ["first_stage_model", ..] | ["cond_stage_model", ..] => 1,
        ["conditioner", "embedders", _, "model" | "transformer", ..] => 4,
        _ => 0,
    };
    let body = &segs[skip..segs.len().saturating_sub(1).max(skip)];   // drop ".weight"/".bias"
    let cut = match body.iter().position(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())) {
        Some(i) => i + 1,
        None => body.len().min(2).max(1),
    };
    segs[..skip + cut.min(body.len()).max(1)].join(".")
}

#[derive(Default, Debug, Clone)]
pub struct Block { pub component: &'static str, pub bytes: u64, pub types: BTreeMap<String, u64>, pub flops: f64 }

#[derive(Default, Debug)]
pub struct Profile {
    pub blocks: BTreeMap<String, Block>,
    pub peak_private: BTreeMap<&'static str, u64>,     // per component, bytes above the start of its run
    pub file_bytes: u64,                                // sum of the files' tensor bytes (check)
    pub timed: BTreeMap<&'static str, (f64, BTreeMap<&'static str, (f64, f64)>)>,  // component -> (wall s, kind -> (s, FLOPs))
    pub dry: bool,
    pub shapes: std::collections::HashMap<(usize, usize, usize, u32), (u32, f64)>,
}

impl Profile {
    /// Blocks, bytes and types of every tensor in the given files (no FLOPs yet).
    pub fn from_files(files: &[&Weights]) -> Profile {
        let mut p = Profile { dry: true, ..Default::default() };
        for w in files {
            for t in &w.mf.tensors {
                let b = p.blocks.entry(block_of(&t.name)).or_insert_with(|| Block { component: component(&t.name), ..Default::default() });
                b.bytes += t.nbytes;
                *b.types.entry(t.dtype.clone()).or_default() += t.nbytes;
                p.file_bytes += t.nbytes;
            }
        }
        p
    }

    /// Weight address ranges -> block, for the counter.
    pub fn ranges(files: &[&Weights]) -> Vec<(usize, usize, String)> {
        files.iter().flat_map(|w| w.mf.tensors.iter().filter_map(move |t| {
            w.raw(&t.name).ok().map(|(b, _)| (b.as_ptr() as usize, b.len(), block_of(&t.name)))
        })).collect()
    }

    /// Run `f` (one call of one component) as a dry counting run and add its FLOPs and activation peak.
    pub fn count(&mut self, files: &[&Weights], comp: &'static str, f: impl FnOnce() -> Result<(), String>) -> Result<(), String> {
        let base = crate::ffi::private_bytes();
        crate::ffi::profile_begin(self.dry, Self::ranges(files));
        let t0 = std::time::Instant::now();
        let r = f();
        let wall = t0.elapsed().as_secs_f64();
        let c = crate::ffi::profile_end().ok_or("profiler was not running")?;
        r?;
        for (blk, fl) in c.flops {
            let comp_of = self.blocks.get(&blk).map(|b| b.component).unwrap_or(comp);
            self.blocks.entry(blk).or_insert_with(|| Block { component: comp_of, ..Default::default() }).flops += fl;
        }
        if !self.dry {
            let e = self.timed.entry(comp).or_default();
            e.0 += wall;
            for (k, (s, fl)) in &c.kernels { let x = e.1.entry(k).or_default(); x.0 += s; x.1 += fl; }
            for (k, (n, s)) in &c.shapes { let x = self.shapes.entry(*k).or_default(); x.0 += n; x.1 += s; }
        }
        let e = self.peak_private.entry(comp).or_default();
        *e = (*e).max(c.peak_private.saturating_sub(base));
        Ok(())
    }

    /// (blocks, bytes, GFLOP per call) of one component.
    pub fn totals(&self, comp: &str) -> (usize, u64, f64) {
        self.blocks.values().filter(|b| b.component == comp)
            .fold((0, 0, 0.0), |(n, by, fl), b| (n + 1, by + b.bytes, fl + b.flops / 1e9))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn blocks_and_components() {
        let cases = [
            ("model.diffusion_model.input_blocks.3.1.transformer_blocks.0.attn1.to_q.weight", "model.diffusion_model.input_blocks.3", "denoiser"),
            ("model.diffusion_model.time_embed.0.weight", "model.diffusion_model.time_embed.0", "denoiser"),
            ("model.diffusion_model.out.2.bias", "model.diffusion_model.out.2", "denoiser"),
            ("double_blocks.7.img_attn.qkv.weight", "double_blocks.7", "denoiser"),
            ("img_in.weight", "img_in", "denoiser"),
            ("final_layer.adaLN_modulation.1.bias", "final_layer.adaLN_modulation.1", "denoiser"),
            ("enc.blk.4.attn_q.weight", "enc.blk.4", "text"),
            ("token_embd.weight", "token_embd", "text"),
            ("cond_stage_model.transformer.text_model.encoder.layers.5.mlp.fc1.weight", "cond_stage_model.transformer.text_model.encoder.layers.5", "text"),
            ("conditioner.embedders.0.model.transformer.resblocks.9.attn.in_proj_weight", "conditioner.embedders.0.model.transformer.resblocks.9", "text"),
            ("first_stage_model.decoder.up.2.block.0.conv1.weight", "first_stage_model.decoder.up.2", "vae"),
            ("first_stage_model.decoder.mid.attn_1.q.weight", "first_stage_model.decoder.mid", "vae"),
            ("decoder.conv_in.weight", "decoder.conv_in", "vae"),
        ];
        for (n, b, c) in cases { assert_eq!((block_of(n).as_str(), component(n)), (b, c), "{n}"); }
    }
}
