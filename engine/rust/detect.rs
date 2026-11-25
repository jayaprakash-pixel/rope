//! Architecture detection: what is inside a model file, and can rope run it?
//! Ordered rules (most specific first) over tensor names, a few shapes, and the GGUF general.architecture.
//! Works on the parsed header only (no tensor data), so it is fast and testable without real files.
#![forbid(unsafe_code)]

use crate::modelfile::{ModelFile, Tensor};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role { Diffusion, Video, Vae, TextEncoder, Llm, Adapter, Upscaler, Control }

impl Role {
    pub fn name(self) -> &'static str {
        match self { Role::Diffusion => "image model", Role::Video => "video model", Role::Vae => "VAE / decoder",
                     Role::TextEncoder => "text encoder", Role::Llm => "text LLM", Role::Adapter => "LoRA add-on",
                     Role::Upscaler => "upscaler add-on", Role::Control => "ControlNet add-on" }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    pub family: String,           // stable id, e.g. "sd2", "sd1", "sdxl", "flux", "llm:llama"
    pub title: String,            // human name, may include size
    pub role: Role,
    pub evidence: String,         // the tensor / metadata that decided it
}

/// Families rope can run today, and the phase that adds the others.
pub fn support(family: &str) -> Result<(), &'static str> {
    match family {
        "sd1" | "sd2" | "sdxl" | "sd-vae" | "taesd" | "flux" => Ok(()),
        "flux-vae" | "t5" | "clip-l" => Err("this is a FLUX companion file; run the FLUX model file and rope fetches its companions"),
        "clip-g" => Err("CLIP-G belongs to SDXL / SD3, not supported yet"),
        "lora" => Err("this is a LoRA add-on, not a model: give it with --lora FILE next to an SD / SDXL model"),
        "esrgan" => Err("this is an upscaler add-on, not a model: use it with --upscale FILE"),
        "controlnet" => Err("ControlNet add-ons are not supported yet"),
        "sd3" => Err("SD3 is not planned yet"),
        "wan" | "wan-vae" | "taehv" | "umt5" => Err("video (Wan) is Phase 3"),
        _ if family.starts_with("llm:") => Err("text LLMs are Phase 4"),
        _ => Err("unknown architecture"),
    }
}

/// Find a tensor by exact name, optionally under one of the usual checkpoint prefixes.
fn find<'a>(mf: &'a ModelFile, name: &str) -> Option<&'a Tensor> {
    const PREFIXES: [&str; 3] = ["", "model.diffusion_model.", "first_stage_model."];
    PREFIXES.iter().find_map(|p| mf.tensors.iter().find(|t| t.name == format!("{p}{name}")))
}
fn has(mf: &ModelFile, name: &str) -> bool { find(mf, name).is_some() }
fn has_prefix(mf: &ModelFile, prefix: &str) -> bool { mf.tensors.iter().any(|t| t.name.starts_with(prefix)) }

/// safetensors shapes are outermost-first; GGUF shapes are innermost-first. Return outermost-first.
fn dims(t: &Tensor) -> Vec<u64> { if t.ggml_type.is_some() { t.shape.iter().rev().copied().collect() } else { t.shape.clone() } }

pub fn detect(mf: &ModelFile) -> Option<Detection> {
    let d = |family: &str, title: String, role, evidence: String| Some(Detection { family: family.to_string(), title, role, evidence });

    // ---- LLM GGUF files carry their architecture in metadata
    const ENCODER_ARCHS: [&str; 2] = ["t5encoder", "t5"];
    if !mf.arch.is_empty() && !ENCODER_ARCHS.contains(&mf.arch.as_str()) && has_prefix(mf, "blk.") {
        return d(&format!("llm:{}", mf.arch), format!("{} language model", mf.arch), Role::Llm, format!("general.architecture = {}", mf.arch));
    }
    // ---- T5 / umT5 encoders (GGUF "enc.blk.*"): umT5 has a ~256k vocabulary, T5 v1.1 has 32k
    if has_prefix(mf, "enc.blk.") {
        let vocab = mf.tensors.iter().find(|t| t.name == "token_embd.weight").map(|t| dims(t)[0]).unwrap_or(0);
        return if vocab > 100_000 { d("umt5", format!("umT5-XXL text encoder (vocab {vocab})"), Role::TextEncoder, "enc.blk.* + token_embd vocab > 100k".into()) }
               else { d("t5", format!("T5-XXL text encoder (vocab {vocab})"), Role::TextEncoder, "enc.blk.* + token_embd vocab".into()) };
    }
    // ---- add-on files (not models): LoRA adapters, ControlNet, upscalers. Before the UNet rule: an original-format ControlNet also has input_blocks.0.0
    if let Some(t) = mf.tensors.iter().find(|t| t.name.contains(".lora_down.weight") || t.name.contains(".lora_A.weight")) {
        let n = mf.tensors.iter().filter(|t| t.name.ends_with(".lora_down.weight") || t.name.ends_with(".lora_A.weight")).count();
        let sdxl = mf.tensors.iter().any(|t| t.name.contains("transformer_blocks_1_") || t.name.contains("transformer_blocks.1."));   // SD 1.x / 2.x blocks have depth 1
        let for_ = if t.name.contains("lora_unet_") { if sdxl { "SDXL UNet" } else { "SD 1.x / 2.x UNet" } } else { "a model (diffusers naming, not applied yet)" };
        return d("lora", format!("LoRA adapter, {n} layers, for {for_}"), Role::Adapter, t.name.clone());
    }
    if has(mf, "controlnet_cond_embedding.conv_in.weight") || has(mf, "input_hint_block.0.weight") {
        return d("controlnet", "ControlNet".into(), Role::Control, "controlnet_cond_embedding / input_hint_block".into());
    }
    if has(mf, "conv_first.weight") && has(mf, "body.0.rdb1.conv1.weight") && has(mf, "conv_up2.weight") {
        let ch = find(mf, "conv_first.weight").map(|t| dims(t)[1]).unwrap_or(0);
        return d("esrgan", format!("Real-ESRGAN x4 upscaler (RRDBNet, {} input channels)", ch), Role::Upscaler, "conv_first + body.N.rdbK + conv_up2".into());
    }
    // ---- diffusion transformers
    if has(mf, "double_blocks.0.img_attn.qkv.weight") {
        return d("flux", "FLUX.1 diffusion transformer".into(), Role::Diffusion, "double_blocks.0.img_attn.qkv.weight".into());
    }
    if has(mf, "joint_blocks.0.x_block.attn.qkv.weight") {
        return d("sd3", "Stable Diffusion 3 MMDiT".into(), Role::Diffusion, "joint_blocks.0.x_block.attn.qkv.weight".into());
    }
    if has(mf, "patch_embedding.weight") && has(mf, "blocks.0.cross_attn.k.weight") {
        let dim = find(mf, "blocks.0.cross_attn.k.weight").map(|t| dims(t)[0]).unwrap_or(0);
        let size = match dim { 1536 => " 1.3B", 3072 => " 5B", 5120 => " 14B", _ => "" };
        return d("wan", format!("Wan 2.x video transformer{size}"), Role::Video, "patch_embedding + blocks.0.cross_attn".into());
    }
    // ---- UNet checkpoints (SD 1.x / 2.x / XL): cross-attention width tells them apart
    if has(mf, "input_blocks.0.0.weight") {
        if has(mf, "label_emb.0.0.weight") || has_prefix(mf, "conditioner.embedders.1.") {
            return d("sdxl", "Stable Diffusion XL".into(), Role::Diffusion, "label_emb / conditioner.embedders.1".into());
        }
        let ctx = find(mf, "input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight").map(|t| *dims(t).last().unwrap_or(&0)).unwrap_or(0);
        return match ctx {
            1024 => d("sd2", "Stable Diffusion 2.x (OpenCLIP-H, 1024-wide cross-attention)".into(), Role::Diffusion, "attn2.to_k in 1024".into()),
            768 => d("sd1", "Stable Diffusion 1.x (CLIP-L, 768-wide cross-attention)".into(), Role::Diffusion, "attn2.to_k in 768".into()),
            _ => d("sd-unknown", format!("SD-style UNet with {ctx}-wide cross-attention"), Role::Diffusion, "input_blocks.0.0 + attn2.to_k".into()),
        };
    }
    // ---- decoders / VAEs
    if has(mf, "decoder.layers.0.weight") {
        return d("taesd", "TAESD tiny image decoder".into(), Role::Vae, "decoder.layers.0.weight".into());
    }
    if let Some(t) = find(mf, "decoder.conv_in.weight") {
        let zc = dims(t).get(1).copied().unwrap_or(0);
        return if zc == 16 { d("flux-vae", "FLUX / SD3 16-channel VAE".into(), Role::Vae, "decoder.conv_in 16 input channels".into()) }
               else { d("sd-vae", "SD 4-channel VAE".into(), Role::Vae, format!("decoder.conv_in {zc} input channels")) };
    }
    if has_prefix(mf, "decoder.upsamples.") || (has(mf, "conv1.weight") && has_prefix(mf, "decoder.middle.")) {
        return d("wan-vae", "Wan video VAE".into(), Role::Vae, "decoder.upsamples / decoder.middle".into());
    }
    if has(mf, "decoder.1.weight") && find(mf, "decoder.1.weight").map(|t| dims(t).get(1) == Some(&16)).unwrap_or(false) {
        return d("taehv", "TAEHV tiny video decoder".into(), Role::Vae, "decoder.1.weight with 16 latent channels".into());
    }
    // ---- CLIP text encoders
    if let Some(t) = find(mf, "text_model.embeddings.token_embedding.weight") {
        let w = *dims(t).last().unwrap_or(&0);
        return if w == 1280 { d("clip-g", "CLIP ViT-bigG text encoder".into(), Role::TextEncoder, "text_model.* width 1280".into()) }
               else { d("clip-l", format!("CLIP ViT-L text encoder (width {w})"), Role::TextEncoder, "text_model.* embeddings".into()) };
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modelfile::Format;

    fn mf(names: &[(&str, &[u64])], arch: &str, gguf: bool) -> ModelFile {
        ModelFile { format: if gguf { Format::Gguf } else { Format::Safetensors }, version: 3, n_kv: 0, alignment: 32, data_offset: 0,
                    file_size: 0, arch: arch.into(),
                    tensors: names.iter().map(|(n, s)| Tensor { name: n.to_string(), dtype: "F16".into(), ggml_type: gguf.then_some(1),
                                                               shape: s.to_vec(), offset: 0, nbytes: 0 }).collect() }
    }
    fn fam(m: &ModelFile) -> String { detect(m).map(|d| d.family).unwrap_or_else(|| "unknown".into()) }

    #[test]
    fn families() {
        let unet = |ctx: u64| vec![("model.diffusion_model.input_blocks.0.0.weight", &[320u64, 4, 3, 3][..]),
                                   ("model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight", if ctx == 768 { &[320u64, 768][..] } else { &[320u64, 1024][..] })];
        assert_eq!(fam(&mf(&unet(1024), "", false)), "sd2");
        assert_eq!(fam(&mf(&unet(768), "", false)), "sd1");
        let mut xl = unet(1024); xl.push(("model.diffusion_model.label_emb.0.0.weight", &[1280, 2816]));
        assert_eq!(fam(&mf(&xl, "", false)), "sdxl");
        assert_eq!(fam(&mf(&[("double_blocks.0.img_attn.qkv.weight", &[3072, 9216])], "", true)), "flux");
        assert_eq!(fam(&mf(&[("model.diffusion_model.joint_blocks.0.x_block.attn.qkv.weight", &[4608, 1536])], "", false)), "sd3");
        assert_eq!(fam(&mf(&[("blk.0.attn_q.weight", &[4096, 4096])], "llama", true)), "llm:llama");
        assert_eq!(fam(&mf(&[("enc.blk.0.attn_q.weight", &[4096, 4096]), ("token_embd.weight", &[4096, 256384])], "t5encoder", true)), "umt5");
        assert_eq!(fam(&mf(&[("enc.blk.0.attn_q.weight", &[4096, 4096]), ("token_embd.weight", &[4096, 32128])], "t5encoder", true)), "t5");
        assert_eq!(fam(&mf(&[("some.random.tensor", &[10, 10])], "", false)), "unknown");
        // add-ons
        assert_eq!(fam(&mf(&[("lora_unet_input_blocks_4_1_proj_in.lora_down.weight", &[32, 640]), ("lora_unet_input_blocks_4_1_proj_in.lora_up.weight", &[640, 32])], "", false)), "lora");
        let d = detect(&mf(&[("lora_unet_input_blocks_4_1_transformer_blocks_1_ff_net_2.lora_down.weight", &[32, 640])], "", false)).unwrap();
        assert!(d.title.contains("SDXL") && d.role == Role::Adapter);
        assert_eq!(fam(&mf(&[("unet.up_blocks.0.attn.to_q.lora_A.weight", &[8, 640])], "", false)), "lora");
        assert_eq!(fam(&mf(&[("input_hint_block.0.weight", &[16, 3, 3, 3]), ("input_blocks.0.0.weight", &[320, 4, 3, 3])], "", false)), "controlnet");
        assert_eq!(fam(&mf(&[("conv_first.weight", &[64, 3, 3, 3]), ("body.0.rdb1.conv1.weight", &[32, 64, 3, 3]), ("conv_up2.weight", &[64, 64, 3, 3])], "", false)), "esrgan");
        assert!(support("lora").is_err() && support("esrgan").is_err());
        assert!(support("sd2").is_ok() && support("sd1").is_ok() && support("flux").is_ok() && support("t5").is_err() && support("llm:qwen3").is_err());
    }
}
