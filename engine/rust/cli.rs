//! The `rope` command line (the bin target calls `run`). Run big image, video and text AI models on low-end laptops.
//! Platform in safe Rust; math kernels in C/AVX2 and CUDA (engine/src), called through ffi.rs.


use std::path::Path;
use std::process::{Command, ExitCode};
use std::time::Duration;

use crate::VERSION;
use crate::{alloc, detect, ffi, flux, gguf_write, json, lora, mmap, modelfile, models, placement, planner, png, profile, scan, sd, selftest, serve, t5, tokenizer, tokenizer_t5, upscale, weights};

/// Canonical listing, identical to bench/tools/ref_modelfile.py, so the two can be diffed byte for byte.
fn inspect(path: &str) -> Result<(), String> {
    let m = modelfile::open(Path::new(path))?;
    let mut out = String::with_capacity(m.tensors.len() * 80);
    let fmt = if m.format == modelfile::Format::Gguf { "gguf" } else { "safetensors" };
    out += &format!("format={fmt} n_tensors={} data_offset={}\n", m.tensors.len(), m.data_offset);
    for t in &m.tensors {
        let shape: Vec<String> = t.shape.iter().map(|d| d.to_string()).collect();
        out += &format!("{}\t{}\t{}\t{}\t{}\n", t.name, t.dtype, shape.join(","), t.offset, t.nbytes);
    }
    use std::io::Write;
    std::io::stdout().write_all(out.as_bytes()).map_err(|e| e.to_string())
}

/// First `count` dequantized values of a tensor (for the gguf-py cross-check).
fn dump(path: &str, name: &str, count: usize) -> Result<(), String> {
    let m = modelfile::open(Path::new(path))?;
    let t = m.tensors.iter().find(|t| t.name == name).ok_or("tensor not found")?;
    let ty = t.ggml_type.filter(|&ty| ffi::wtype_supported(ty)).ok_or_else(|| format!("type {} not supported", t.dtype))?;
    let map = mmap::Map::open(Path::new(path)).map_err(|e| e.to_string())?;
    let data = &map.bytes()[t.offset as usize..(t.offset + t.nbytes) as usize];   // bounds checked by the reader
    let row = t.shape[0] as usize;
    let rb = ffi::row_bytes(ty, row as u64);
    let total: u64 = t.shape.iter().product();
    let count = count.min(total as usize);
    let mut out = String::new();
    for r in 0..count.div_ceil(row) {
        let vals = ffi::dequant_row(ty, &data[r * rb..(r + 1) * rb], row);
        for v in vals.iter().take(count - r * row) { out += &format!("{v:.9e}\n"); }
    }
    print!("{out}");
    Ok(())
}

/// Text encoder output [77, width] as raw little-endian f32 (for the diffusers cross-check).
/// SDXL (`tokdir2` = CLIP-G tokenizer): the 2048-wide context followed by the 2816-wide conditioning vector, 512 size.
fn encode_text(model: &str, tokdir: &str, prompt: &str, out: &str, tokdir2: Option<&str>) -> Result<(), String> {
    let t0 = std::time::Instant::now();
    let w = weights::Weights::open(Path::new(model))?;
    let tok = tokenizer::ClipTokenizer::load(Path::new(tokdir))?;
    let ids = tok.encode_77(prompt);
    let t1 = std::time::Instant::now();
    let h = match tokdir2 {
        Some(d2) => {
            let tg = tokenizer::ClipTokenizer::load(Path::new(d2))?;
            sd::sdxl_condition(&w, Some((&ids, &tg.encode_77(prompt), tg.eos)), 512)?
        }
        None => sd::TextEncoder::new(&w)?.encode(&ids)?,
    };
    let enc = t1.elapsed().as_secs_f64();
    let bytes: Vec<u8> = h.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(out, bytes).map_err(|e| e.to_string())?;
    let pooled = sd::TextEncoder::pooled(&h, &ids, tok.eos);           // also written: <out>.pooled
    std::fs::write(format!("{out}.pooled"), pooled.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).map_err(|e| e.to_string())?;
    eprintln!("encode-text: open {:.2} s, encode {:.2} s", (t1 - t0).as_secs_f64(), enc);
    Ok(())
}

/// rope encode-t5 <t5.gguf> <spiece-dir> <prompt> <out.f32>: T5 hidden states [256, 4096] (FLUX conditioning).
fn encode_t5(model: &str, tokdir: &str, prompt: &str, out: &str) -> Result<(), String> {
    let t0 = std::time::Instant::now();
    let w = weights::Weights::open(Path::new(model))?;
    let ids = tokenizer_t5::T5Tokenizer::load(Path::new(tokdir))?.encode_padded(prompt, 256);
    let t1 = std::time::Instant::now();
    let h = t5::T5Encoder::new(&w)?.encode(&ids)?;
    std::fs::write(out, h.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).map_err(|e| e.to_string())?;
    eprintln!("encode-t5: open {:.2} s, encode {:.2} s", (t1 - t0).as_secs_f64(), t1.elapsed().as_secs_f64());
    Ok(())
}

/// rope flux <model.gguf> <latent.f32 [16,h,w]> <t 0..1> <t5.f32 [n,4096]> <pooled.f32 [768]> <out.f32>: one transformer call.
fn flux_once(a: &[String]) -> Result<(), String> {
    let w = weights::Weights::open(Path::new(&a[0]))?;
    if std::env::var("ROPE_GPU").as_deref() == Ok("all") { ffi::gpu_set_ranges(w.ranges("")); }
    let m = flux::Flux::new(&w)?;
    let lat = read_f32(&a[1])?;
    let side = ((lat.len() / 16) as f64).sqrt() as usize;
    let t: f32 = a[2].parse().map_err(|_| "bad t")?;
    let (txt, pooled) = (read_f32(&a[3])?, read_f32(&a[4])?);
    let t0 = std::time::Instant::now();
    let v = m.forward(&lat, side, side, t, &txt, &pooled)?;
    eprintln!("flux: {:.2} s for a {side}x{side} latent, {} text tokens", t0.elapsed().as_secs_f64(), txt.len() / 4096);
    #[cfg(feature = "cuda")]
    if let Some((b, peak, hits, st, res)) = ffi::gpu_stats() {
        eprintln!("vram: budget {:.2} GiB, peak used {:.2} GiB, resident {:.2} GiB, {hits} resident hits, {:.2} GB uploaded",
                  b as f64 / 1073741824.0, peak as f64 / 1073741824.0, res as f64 / 1073741824.0, st as f64 / 1e9);
    }
    std::fs::write(&a[5], v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).map_err(|e| e.to_string())
}

fn read_f32(path: &str) -> Result<Vec<f32>, String> {
    let b = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

/// One UNet call on files: latent [4,s,s] f32, timestep, context [77,1024] f32 -> noise prediction f32.
fn unet_once(model: &str, latent: &str, t: f32, ctx: &str, out: &str) -> Result<(), String> {
    let w = weights::Weights::open(Path::new(model))?;
    if std::env::var("ROPE_GPU").as_deref() == Ok("all") { ffi::gpu_set_ranges(w.ranges("model.diffusion_model.")); }
    let u = sd::UNet::new(&w)?;
    let l = read_f32(latent)?;
    let side = ((l.len() / 4) as f64).sqrt() as usize;
    let x = sd::Act { c: 4, h: side, w: side, d: l };
    let c = read_f32(ctx)?;
    let t0 = std::time::Instant::now();
    let y = u.forward(&x, t, &c)?;
    eprintln!("unet: {:.2} s for a {side}x{side} latent", t0.elapsed().as_secs_f64());
    std::fs::write(out, y.d.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).map_err(|e| e.to_string())
}

/// TAESD decode on files: latent [4,s,s] f32 -> image [3,8s,8s] f32 in [-1,1].
fn taesd_once(model: &str, latent: &str, out: &str) -> Result<(), String> {
    let w = weights::Weights::open(Path::new(model))?;
    let l = read_f32(latent)?;
    let side = ((l.len() / 4) as f64).sqrt() as usize;
    let t0 = std::time::Instant::now();
    let y = sd::Taesd::new(&w)?.decode(&sd::Act { c: 4, h: side, w: side, d: l })?;
    eprintln!("taesd: {:.2} s -> {}x{}", t0.elapsed().as_secs_f64(), y.w, y.h);
    std::fs::write(out, y.d.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).map_err(|e| e.to_string())
}

/// rope image --prompt P [--quality fast|balanced|best] [--models-dir D] [--seed S] [--out x.png] [device overrides]
/// Automatic mode: the planner chooses model, steps, size and decoder for this device; files are verified first.
fn image_auto(args: &[String]) -> Result<(), String> {
    if args.iter().any(|a| a == "--no-download") { models::set_no_download(true); }    // never use the network (selftest)
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let models_dir = get("--models-dir").unwrap_or_else(|| "models".into());
    let cat_path = get("--catalog").unwrap_or_else(|| "catalog/models.json".into());
    let cat = json::parse(&std::fs::read_to_string(&cat_path).map_err(|e| format!("{cat_path}: {e}"))?)?;
    let (models, cal) = load_catalog(&cat_path)?;
    let q = planner::Quality::parse(&get("--quality").unwrap_or_else(|| "balanced".into())).ok_or("quality must be fast|balanced|best")?;
    let dev = device_from_args(args, &models_dir)?;

    // 1. Which model: the file the user brings (main path), or the catalog's suggestion.
    let (spec, model_path) = match get("--model") {
        Some(mp) => {
            let w = weights::Weights::open(Path::new(&mp))?;
            let (mut spec, d, adm, unet) = spec_for_file(&cat, &models, &models_dir, &mp, &w, args)?;
            // full-precision file on a small laptop: stream it, or quantize once (it was just scanned) and cache the copy
            let st = source_for(&spec, &w, &dev, cal, q, args, unet)?;
            let mp = match st {
                planner::Source::AsIs => mp,
                planner::Source::Stream(why) => { eprintln!("  source: {why}"); mp }
                planner::Source::Quantize { ty, out_gib, why } => {
                    eprintln!("  source: {why}");
                    let dir = Path::new(&models_dir).join(".rope").join("quantized");
                    let stem = Path::new(&mp).file_stem().unwrap().to_string_lossy().into_owned();
                    let out = dir.join(format!("{stem}.{}.{}.gguf", &adm.sha256[..16], if ty == ffi::RT_Q8_0 { "q8_0" } else { "q4_k" }));
                    if out.exists() { eprintln!("  using the cached copy {}", out.display()); } else {
                        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
                        eprintln!("  quantizing once ({out_gib:.1} GiB) -> {}", out.display());
                        let t0 = std::time::Instant::now();
                        let (nq, nf, by) = gguf_write::quantize_file(Path::new(&mp), &out, ty, &d.family)?;
                        eprintln!("  quantized {nq} tensors ({nf} kept as stored), {:.2} GB in {:.0} s", by as f64 / 1e9, t0.elapsed().as_secs_f64());
                    }
                    let qw = weights::Weights::open(&out)?;
                    spec.unet_bytes_set(qw.mf.tensors.iter().filter(|t| profile::component(&t.name) == "denoiser").map(|t| t.nbytes as f64).sum());
                    out.to_string_lossy().into_owned()
                }
            };
            (spec, Some(mp))
        }
        None => (models.iter().find(|m| m.kind == "image" && m.supported).ok_or("no image model runs in rope yet")?.clone(), None),
    };

    // 2. Plan this model for this device, then make the companion files (and catalog model file) present and verified.
    let dir = models_dir.clone();
    let p = planner::plan_model(&spec, &dev, cal, q, &overrides_from_args(args), &move |n: &str| Path::new(&dir).join(n).exists())?;
    let cache = planner::buffer_cache_gib(&dev, p.commit_gib);
    alloc::set_cap((cache * 1073741824.0) as usize);
    eprintln!("plan: {} {}x{}, {} step(s){}, decoder {} (tile {}), predicted peak {:.2} GiB, buffer cache {:.2} GiB, ~{:.0} s estimated", p.model, p.size, p.size, p.steps,
              if p.guidance > 1.0 { format!(" with guidance {}", p.guidance) } else { String::new() }, p.decoder.name(), p.vae_tile, p.peak_gib, cache, p.estimate.total);
    for n in &p.notes { eprintln!("  note: {n}"); }
    let mut log = Vec::new();
    for f in &p.files {
        let r = models::ensure(Path::new(&models_dir), f, &mut log);
        for l in log.drain(..) { eprintln!("  {l}"); }
        r?;
    }
    let path_of = |role: &str| p.files.iter().find(|f| f.role == role).map(|f| Path::new(&models_dir).join(&f.name).to_string_lossy().into_owned());
    let model_path = model_path.or_else(|| path_of("model")).ok_or("plan has no model file")?;
    if path_of("tokenizer").is_none() { return Err("plan has no tokenizer".into()); }

    // 2b. Placement: on a GPU laptop (CUDA build), the rules decide which blocks run on the GPU and which stay
    //     resident; the executor follows (weight files register their GPU ranges when they open).
    let mut estimate = p.estimate.total;
    let mut decode_est = p.estimate.decode;                       // the decode's share, for the app's ETA
    let vram = get("--vram-gib").and_then(|v| v.parse().ok()).or(dev.gpu.as_ref().map(|g| g.vram_free_gib)).unwrap_or(0.0);
    if let Some(g) = gpu_estimate(&models_dir, &model_path, &spec, &p, &dev, vram)? {
        (estimate, decode_est) = (g.total + p.estimate.upscale, g.decode);
        eprintln!("  placement: {} blocks on the GPU ({} resident, {} streamed), VRAM {:.2} of {:.2} GiB; ~{:.0} s estimated with the GPU",
                  g.gpu.len(), g.resident.len(), g.gpu.len() - g.resident.len(), g.pl.vram_used_gib, g.pl.vram_budget_gib, estimate);
        ffi::gpu_plan(g.gpu, g.resident);
    }

    if std::env::var("ROPE_PROGRESS").as_deref() == Ok("1") {
        eprintln!("progress: estimate {estimate:.1} {decode_est:.1} {} {:.1}", p.steps, p.estimate.upscale);
    }

    // 3. Run. Every companion role becomes a flag of the low-level path: --tokenizer DIR, --taesd F (SD);
    //    --t5 F, --t5-tokenizer DIR, --clip F, --tokenizer DIR, --vae F (FLUX). Tokenizer roles pass their folder.
    let mut a: Vec<String> = vec!["--model".into(), model_path,
                                  "--prompt".into(), get("--prompt").ok_or("missing --prompt")?, "--steps".into(), p.steps.to_string(),
                                  "--size".into(), p.size.to_string(), "--cfg".into(), p.guidance.to_string()];
    let mut roles: Vec<&str> = p.files.iter().map(|f| f.role.as_str()).filter(|r| *r != "model").collect();
    roles.sort_unstable();
    roles.dedup();
    for role in roles {
        let v = path_of(role).unwrap();
        let v = if role.ends_with("tokenizer") { Path::new(&v).parent().map(|d| d.to_string_lossy().into_owned()).unwrap_or(v) } else { v };
        a.extend([format!("--{role}"), v]);
    }
    for k in ["--seed", "--out", "--lora", "--lora-scale", "--upscale"] { if let Some(v) = get(k) { a.extend([k.to_string(), v]); } }
    if args.iter().any(|x| x == "--control") { a.push("--control".into()); }        // commands on stdin (rope serve)
    a.extend(["--vae-tile".to_string(), p.vae_tile.to_string()]);                   // the tile the plan chose for the memory budget
    if let Some(dir) = get("--preview-dir") {
        a.extend(["--preview-budget-s".to_string(), get("--preview-budget-s").unwrap_or_else(|| format!("{:.2}", 0.03 * estimate))]);
        // previews need the family's tiny decoder (taesd for SD 1.x / 2.x, taesdxl for SDXL, taef1 for FLUX: about 10 MB each); a family without one runs without previews
        if let Some(f) = spec.files.iter().find(|f| f.role == "taesd" || f.role == "preview-taesd") {
            let mut log = Vec::new();
            let r = models::ensure(Path::new(&models_dir), f, &mut log);
            for l in log.drain(..) { eprintln!("  {l}"); }
            if r.is_ok() {
                a.extend(["--preview-dir".to_string(), dir]);
                if !a.iter().any(|x| x == "--taesd" || x == "--preview-taesd") { a.extend(["--preview-taesd".to_string(), Path::new(&models_dir).join(&f.name).to_string_lossy().into_owned()]); }
            }
        }
    }
    let t0 = std::time::Instant::now();
    image(&a)?;
    let took = t0.elapsed().as_secs_f64();
    eprintln!("done in {took:.1} s (planner estimate {estimate:.1} s, {:+.0}%)", (estimate - took) / took * 100.0);
    Ok(())
}

/// Open an upscaler add-on: same outlier scan as for models, and it must be a Real-ESRGAN x4 file.
fn open_upscaler(path: &str) -> Result<weights::Weights, String> {
    let w = weights::Weights::open(Path::new(path))?;
    if w.tensor("conv_first.weight").is_err() || w.tensor("conv_up2.weight").is_err() { return Err(format!("{path}: not a Real-ESRGAN x4 upscaler file")); }
    if let Some(bad) = scan::scan(&w)?.iter().find(|r| r.stats.flagged()) { return Err(format!("{path}: SUSPICIOUS add-on file, refused ({} has an outlier injection pattern)", bad.name)); }
    Ok(w)
}

/// One upscale on files (tests): raw f32 image [3,h,w] in [-1,1] -> [3,4h,4w]. One network pass over the whole image when `tile` is 0.
fn upscale_once(model: &str, input: &str, h: usize, wd: usize, out: &str) -> Result<(), String> {
    let w = open_upscaler(model)?;
    let d = read_f32(input)?;
    if d.len() != 3 * h * wd { return Err("input size does not match H W".into()); }
    let img = sd::Act { c: 3, h, w: wd, d };
    let y = if std::env::var("ROPE_UPSCALE_TILE").as_deref() == Ok("0") {
        let u = sd::Act { c: 3, h, w: wd, d: img.d.iter().map(|v| (v + 1.0) * 0.5).collect() };       // one tile: the reference comparison
        let mut r = upscale::tile_forward(&w, &u)?;
        for v in &mut r.d { *v = *v * 2.0 - 1.0; }
        r
    } else { upscale::run(&w, &img)? };
    std::fs::write(out, y.d.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).map_err(|e| e.to_string())
}

/// Full VAE decode on files: latent [z,s,s] f32 -> image [3,8s,8s] f32 in [-1,1]. A 16-channel VAE file is FLUX's.
/// `tile`: latent tile size (0 = decode in one piece).
fn vae_once(model: &str, latent: &str, out: &str, tile: usize) -> Result<(), String> {
    let w = weights::Weights::open(Path::new(model))?;
    let l = read_f32(latent)?;
    if std::env::var("ROPE_GPU").as_deref() == Ok("all") { ffi::gpu_set_ranges(w.ranges("")); }   // test hook
    let z = sd::VaeDecoder::new(&w, sd::SD_VAE)?.z;
    let dec = sd::VaeDecoder::new(&w, if z == 16 { sd::FLUX_VAE } else { sd::SD_VAE })?;
    let side = ((l.len() / z) as f64).sqrt() as usize;
    let t0 = std::time::Instant::now();
    let lat = sd::Act { c: z, h: side, w: side, d: l };
    let y = if tile == 0 { dec.decode(&lat)? } else { dec.decode_tiled(&lat, tile, sd::VAE_TILE.1)? };
    eprintln!("vae: {:.2} s -> {}x{}", t0.elapsed().as_secs_f64(), y.w, y.h);
    std::fs::write(out, y.d.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).map_err(|e| e.to_string())
}

/// rope image --model M [--taesd T]  --tokenizer DIR --prompt P [--seed S] [--steps N] [--size 512] [--out x.png] [--latent-in f] [--latent-out f]
pub fn image(args: &[String]) -> Result<(), String> {
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    if get("--tokenizer").is_none() { return image_auto(args); }   // automatic: catalog choice or the user's own --model
    let need = |k: &str| get(k).ok_or_else(|| format!("missing {k}"));
    let model = need("--model")?;
    let fam = detect::detect(&modelfile::open(Path::new(&model))?).map(|d| d.family);
    if fam.as_deref() == Some("flux") {
        if get("--lora").is_some() { return Err("LoRA is supported for SD 1.x / 2.x / SDXL models, not FLUX yet".into()); }
        return image_flux(args);
    }
    let (tokdir, prompt) = (need("--tokenizer")?, need("--prompt")?);
    let taesd = get("--taesd");   // tiny decoder if given, else the full VAE from the model file
    let seed: u64 = get("--seed").and_then(|s| s.parse().ok()).unwrap_or(42);
    let steps: usize = get("--steps").and_then(|s| s.parse().ok()).unwrap_or(1);
    let size: usize = get("--size").and_then(|s| s.parse().ok()).unwrap_or(512);
    let out = get("--out").unwrap_or_else(|| "rope.png".into());
    let vae_tile: usize = get("--vae-tile").and_then(|s| s.parse().ok()).unwrap_or(sd::VAE_TILE.0);   // latent pixels per VAE tile
    if size % 64 != 0 || !(1..=50).contains(&steps) { return Err("size must be a multiple of 64, steps 1..50".into()); }
    let side = size / 8;
    let tick = std::time::Instant::now();
    let w = weights::Weights::open(Path::new(&model))?;
    if get("--lora").is_none() { lora::clear(); }                     // a library user runs many jobs: no adapter left from the last one
    if let Some(l) = get("--lora") {                                  // add-on: LoRA file (--lora-scale S, default 1.0)
        let s: f32 = get("--lora-scale").and_then(|s| s.parse().ok()).unwrap_or(1.0);
        let i = lora::load(Path::new(&l), s)?;
        lora::verify(&w)?;
        eprintln!("lora: {} layers, rank {}..{}, strength {s}{}", i.modules, i.rank_min, i.rank_max, if i.skipped > 0 { format!(", {} entries not applied (text encoder / convolution)", i.skipped) } else { String::new() });
    }
    if std::env::var("ROPE_GPU").as_deref() == Ok("all") { ffi::gpu_set_ranges(w.ranges("model.diffusion_model.")); }   // test hook
    let tw = match &taesd { Some(t) => Some(weights::Weights::open(Path::new(t))?), None => None };
    let tok = tokenizer::ClipTokenizer::load(Path::new(&tokdir))?;
    let t_open = tick.elapsed().as_secs_f64();
    let t0 = std::time::Instant::now();
    sd::progress("text", 0, 1);
    let cfg: f32 = get("--cfg").and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let sdxl = fam.as_deref() == Some("sdxl");
    // SDXL: CLIP-L + CLIP-G; the empty-prompt condition is all zeros (SDXL's zero-for-empty rule)
    let tg = if sdxl { Some(tokenizer::ClipTokenizer::load(Path::new(&need("--clip-g-tokenizer")?))?) } else { None };
    let encode = |p: &str| -> Result<Vec<f32>, String> {
        match &tg {
            Some(tg) => sd::sdxl_condition(&w, Some((&tok.encode_77(p), &tg.encode_77(p), tg.eos)), size),
            None => sd::TextEncoder::new(&w)?.encode(&tok.encode_77(p)),
        }
    };
    let ctx = encode(&prompt)?;
    let uctx = if cfg > 1.0 { Some(if sdxl { sd::sdxl_condition(&w, None, size)? } else { encode("")? }) } else { None };
    let t_text = t0.elapsed().as_secs_f64();
    sd::progress("text", 1, 1);
    let ts = sd::trailing_timesteps(steps);
    let s0 = sd::sigma_at(ts[0]);
    let noise = match get("--latent-in") { Some(f) => read_f32(&f)?, None => sd::gaussian_noise(4 * side * side, seed) };
    if noise.len() != 4 * side * side { return Err("latent-in has the wrong size".into()); }
    let latent = sd::Act { c: 4, h: side, w: side, d: noise.iter().map(|v| v * s0).collect() };
    // live previews: --preview-dir DIR plus the family's tiny decoder file (--preview-taesd, or --taesd): TAESD for SD 1.x / 2.x, TAESDXL for SDXL
    let prev_dir = get("--preview-dir");
    let prev_file = get("--preview-taesd").or_else(|| taesd.clone());
    let ptw = match (&prev_dir, &prev_file) { (Some(_), Some(f)) => Some(weights::Weights::open(Path::new(f))?), _ => None };
    let pdec = match &ptw { Some(t) => Some(sd::Taesd::new(t)?), None => None };
    // time budget (--preview-budget-s, set by auto mode to 3% of the plan's estimate): each preview is measured and one is
    // skipped when it would exceed the budget (a lower-quality fallback would be worse than the previous preview). So short
    // jobs get few or no previews and long jobs get one after every step
    let budget: f64 = get("--preview-budget-s").and_then(|s| s.parse().ok()).unwrap_or(f64::INFINITY);
    let (mut n_prev, mut t_prev) = (0usize, 0f64);
    let mut cost_half = 0.35 * (size as f64 / 512.0).powi(2);          // first guess for a half-size preview (CPU), then measured
    let mut preview_cb = |i: usize, x0: &sd::Act| -> Result<(), String> {
        let (Some(d), Some(dir)) = (&pdec, &prev_dir) else { return Ok(()) };
        if t_prev + cost_half > budget { return Ok(()); }
        let t = std::time::Instant::now();
        let r = write_preview(d, x0, dir, i, ts.len(), 2);
        cost_half = t.elapsed().as_secs_f64();
        n_prev += 1; t_prev += cost_half;
        r
    };
    let t0 = std::time::Instant::now();
    let on_x0: Option<&mut dyn FnMut(usize, &sd::Act) -> Result<(), String>> = if pdec.is_some() { Some(&mut preview_cb) } else { None };
    let rx = control_channel(args);
    let mut steer_cb = |i: usize| -> Result<sd::Steer<Vec<f32>>, String> {
        let Some(rx) = &rx else { return Ok(Default::default()) };
        let (p, cfg) = poll_control(rx, i, ts.len());
        let cond = match p { Some(p) => { let c = encode(&p)?; sd::progress("prompt", i, ts.len()); Some(c) } None => None };
        Ok(sd::Steer { cond, cfg })
    };
    let steer: Option<&mut dyn FnMut(usize) -> Result<sd::Steer<Vec<f32>>, String>> = if rx.is_some() { Some(&mut steer_cb) } else { None };
    let x0 = sd::euler_sample(&sd::UNet::new(&w)?, latent, &ctx, uctx.as_deref().map(|u| (u, cfg)), &ts, on_x0, steer)?;
    let t_unet = t0.elapsed().as_secs_f64();
    if pdec.is_some() { eprintln!("previews: {n_prev} in {t_prev:.2} s"); }
    if let Some(f) = get("--latent-out") { std::fs::write(f, x0.d.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).map_err(|e| e.to_string())?; }
    let t0 = std::time::Instant::now();
    let img = match &tw { Some(t) => sd::Taesd::new(t)?.decode(&x0)?, None => sd::VaeDecoder::new(&w, if sdxl { sd::SDXL_VAE } else { sd::SD_VAE })?.decode_tiled(&x0, vae_tile, sd::vae_overlap(vae_tile))? };
    let t_vae = t0.elapsed().as_secs_f64();
    let img = match get("--upscale") {                                 // add-on: Real-ESRGAN x4 on tiles (output 4x larger)
        Some(f) => {
            let t = std::time::Instant::now();
            let uw = open_upscaler(&f)?;
            alloc::trim();                                             // the stage before gives its cached buffers back
            let u = upscale::run(&uw, &img)?;
            eprintln!("upscale: {}x{} -> {}x{} in {:.1} s", img.w, img.h, u.w, u.h, t.elapsed().as_secs_f64());
            u
        }
        None => img,
    };
    save_png(&img, &out)?;
    #[cfg(feature = "cuda")]
    if let Some((b, peak, hits, st, res)) = ffi::gpu_stats() {
        eprintln!("vram: budget {:.2} GiB, peak used {:.2} GiB, resident {:.2} GiB, {hits} resident hits, {:.2} GB uploaded in {:.1} s, {:.2} GB staged ahead",
                  b as f64 / 1073741824.0, peak as f64 / 1073741824.0, res as f64 / 1073741824.0, st as f64 / 1e9, ffi::gpu_upload_seconds(),
                  ffi::gpu_staged_bytes() as f64 / 1e9);
    }
    eprintln!("image: open {t_open:.2} s, text {t_text:.2} s, unet {t_unet:.2} s ({steps} step), decode {t_vae:.2} s, total {:.2} s -> {out}", tick.elapsed().as_secs_f64());
    Ok(())
}

/// With `--control`: commands from `rope serve` on stdin, one per line (`pause`, `resume`, `prompt <text>`, `cfg <x>`),
/// read on a thread so a step is never interrupted, only the boundary between two steps (P5.1.7).
fn control_channel(args: &[String]) -> Option<std::sync::mpsc::Receiver<String>> {
    if !args.iter().any(|a| a == "--control") { return None; }
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        use std::io::BufRead;
        for l in std::io::stdin().lock().lines().map_while(Result::ok) { if tx.send(l).is_err() { break; } }
    });
    Some(rx)
}

/// At a step boundary: take what the app asked for. `pause` blocks here until `resume` (or until the pipe closes).
/// Returns the new prompt text and the new guidance scale, if any (the last of each wins).
fn poll_control(rx: &std::sync::mpsc::Receiver<String>, i: usize, n: usize) -> (Option<String>, Option<f32>) {
    let (mut prompt, mut cfg, mut paused) = (None, None, false);
    loop {
        let line = if paused { match rx.recv() { Ok(l) => l, Err(_) => break } } else { match rx.try_recv() { Ok(l) => l, Err(_) => break } };
        let (cmd, arg) = line.split_once(' ').unwrap_or((line.as_str(), ""));
        match cmd {
            "pause" if !paused => { paused = true; sd::progress("paused", i, n); }
            "resume" if paused => { paused = false; sd::progress("resumed", i, n); }
            "prompt" if !arg.trim().is_empty() => prompt = Some(arg.trim().to_string()),
            "cfg" => cfg = arg.trim().parse::<f32>().ok().filter(|g| (0.0..=30.0).contains(g)),
            _ => {}
        }
    }
    (prompt, cfg)
}

/// Live preview after step `i` (not the last): the denoised estimate with its latent averaged `pool` x `pool` (2: half-size
/// PNG; 4: quarter size, 4x cheaper again), written to <dir>/preview.png (atomically: temp file, then rename), then a
/// `progress: preview k/n` line for rope serve.
fn write_preview(dec: &sd::Taesd, x0: &sd::Act, dir: &str, i: usize, n: usize, pool: usize) -> Result<(), String> {
    let (h2, w2) = (x0.h / pool, x0.w / pool);
    let mut d = vec![0f32; x0.c * h2 * w2];
    let inv = 1.0 / (pool * pool) as f32;
    for c in 0..x0.c { for y in 0..h2 { for x in 0..w2 {
        let mut acc = 0f32;
        for dy in 0..pool { for dx in 0..pool { acc += x0.d[(c * x0.h + pool * y + dy) * x0.w + pool * x + dx]; } }
        d[(c * h2 + y) * w2 + x] = acc * inv;
    } } }
    let img = dec.decode(&sd::Act { c: x0.c, h: h2, w: w2, d })?;
    std::fs::create_dir_all(dir).map_err(|e| format!("{dir}: {e}"))?;
    let (tmp, dst) = (format!("{dir}/preview.tmp"), format!("{dir}/preview.png"));
    save_png(&img, &tmp)?;
    if std::fs::rename(&tmp, &dst).is_ok() { sd::progress("preview", i + 1, n); }       // a reader holding the file: skip this one
    Ok(())
}

/// Image [3, h, w] in [-1, 1] -> PNG file.
fn save_png(img: &sd::Act, out: &str) -> Result<(), String> {
    let plane = img.w * img.h;
    let mut rgb = vec![0u8; plane * 3];
    for i in 0..plane { for c in 0..3 { rgb[i * 3 + c] = ((img.d[c * plane + i] / 2.0 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8; } }
    std::fs::write(out, png::encode_rgb(img.w, img.h, &rgb)).map_err(|e| e.to_string())
}

/// Low-level FLUX image: rope image --model flux.gguf --t5 t5.gguf --t5-tokenizer DIR --clip clip_l.safetensors
///   --tokenizer DIR --vae ae.safetensors --prompt P [--steps 4] [--size 512] [--seed S] [--latent-in f] [--out x.png]
fn image_flux(args: &[String]) -> Result<(), String> {
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let need = |k: &str| get(k).ok_or_else(|| format!("FLUX needs {k}"));
    let prompt = need("--prompt")?;
    let steps: usize = get("--steps").and_then(|s| s.parse().ok()).unwrap_or(4);
    let size: usize = get("--size").and_then(|s| s.parse().ok()).unwrap_or(512);
    let seed: u64 = get("--seed").and_then(|s| s.parse().ok()).unwrap_or(42);
    let out = get("--out").unwrap_or_else(|| "rope.png".into());
    let vae_tile: usize = get("--vae-tile").and_then(|s| s.parse().ok()).unwrap_or(sd::VAE_TILE.0);
    if size % 64 != 0 || !(1..=50).contains(&steps) { return Err("size must be a multiple of 64, steps 1..50".into()); }
    let side = size / 8;
    let tick = std::time::Instant::now();
    // text: CLIP-L pooled + T5 hidden states; each encoder file is unmapped right after use
    sd::progress("text", 0, 1);
    let (clip_f, tok_dir, t5_f, t5_dir) = (need("--clip")?, need("--tokenizer")?, need("--t5")?, need("--t5-tokenizer")?);
    // text for a prompt: used for the first prompt and again when the user changes it mid-run (the files open for the call only)
    let encode = |p: &str| -> Result<(Vec<f32>, Vec<f32>), String> {
        let cw = weights::Weights::open(Path::new(&clip_f))?;
        let tok = tokenizer::ClipTokenizer::load(Path::new(&tok_dir))?;
        let ids = tok.encode_77(p);
        let pooled = sd::TextEncoder::pooled(&sd::TextEncoder::new(&cw)?.encode(&ids)?, &ids, tok.eos);
        let tw = weights::Weights::open(Path::new(&t5_f))?;
        let ids = tokenizer_t5::T5Tokenizer::load(Path::new(&t5_dir))?.encode_padded(p, 256);
        Ok((pooled, t5::T5Encoder::new(&tw)?.encode(&ids)?))
    };
    let (pooled, txt) = encode(&prompt)?;
    sd::progress("text", 1, 1);
    let t_text = tick.elapsed().as_secs_f64();
    let noise = match get("--latent-in") { Some(f) => read_f32(&f)?, None => sd::gaussian_noise(16 * side * side, seed) };
    if noise.len() != 16 * side * side { return Err("latent-in has the wrong size".into()); }
    // live previews through TAEF1 (the FLUX tiny decoder, 16 latent channels): same rules as the SD path
    let prev_dir = get("--preview-dir");
    let ptw = match (&prev_dir, get("--preview-taesd")) { (Some(_), Some(f)) => Some(weights::Weights::open(Path::new(&f))?), _ => None };
    let pdec = match &ptw { Some(t) => Some(sd::Taesd::new(t)?), None => None };
    let budget: f64 = get("--preview-budget-s").and_then(|s| s.parse().ok()).unwrap_or(f64::INFINITY);
    let (mut n_prev, mut t_prev) = (0usize, 0f64);
    let mut cost_half = 0.35 * (size as f64 / 512.0).powi(2);
    let mut preview_cb = |i: usize, x0: &sd::Act| -> Result<(), String> {
        let (Some(d), Some(dir)) = (&pdec, &prev_dir) else { return Ok(()) };
        if t_prev + cost_half > budget { return Ok(()); }
        let t = std::time::Instant::now();
        let r = write_preview(d, x0, dir, i, steps, 2);
        cost_half = t.elapsed().as_secs_f64();
        n_prev += 1; t_prev += cost_half;
        r
    };
    let t0 = std::time::Instant::now();
    let x0 = {
        let w = weights::Weights::open(Path::new(&need("--model")?))?;
        if std::env::var("ROPE_GPU").as_deref() == Ok("all") { ffi::gpu_set_ranges(w.ranges("")); }   // test hook until P2.3.3
        let rx = control_channel(args);
        let mut steer_cb = |i: usize| -> Result<sd::Steer<(Vec<f32>, Vec<f32>)>, String> {
            let Some(rx) = &rx else { return Ok(Default::default()) };
            let (p, _) = poll_control(rx, i, steps);                 // FLUX is guidance-distilled: no guidance scale to change
            let cond = match p { Some(p) => { let (pl, t) = encode(&p)?; sd::progress("prompt", i, steps); Some((t, pl)) } None => None };
            Ok(sd::Steer { cond, cfg: None })
        };
        let steer: Option<&mut dyn FnMut(usize) -> Result<sd::Steer<(Vec<f32>, Vec<f32>)>, String>> = if rx.is_some() { Some(&mut steer_cb) } else { None };
        let on_x0: Option<&mut dyn FnMut(usize, &sd::Act) -> Result<(), String>> = if pdec.is_some() { Some(&mut preview_cb) } else { None };
        let x = flux::sample(&flux::Flux::new(&w)?, noise, side, side, &txt, &pooled, steps, on_x0, steer)?;
        #[cfg(feature = "cuda")]
        if let Some((b, peak, hits, st, res)) = ffi::gpu_stats() {
            eprintln!("vram: budget {:.2} GiB, peak used {:.2} GiB, resident {:.2} GiB, {hits} resident hits, {:.2} GB uploaded",
                      b as f64 / 1073741824.0, peak as f64 / 1073741824.0, res as f64 / 1073741824.0, st as f64 / 1e9);
        }
        x
    };
    let t_dit = t0.elapsed().as_secs_f64();
    if pdec.is_some() { eprintln!("previews: {n_prev} in {t_prev:.2} s"); }
    if let Some(f) = get("--latent-out") { std::fs::write(f, x0.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).map_err(|e| e.to_string())?; }
    let t0 = std::time::Instant::now();
    let vw = weights::Weights::open(Path::new(&need("--vae")?))?;
    let img = sd::VaeDecoder::new(&vw, sd::FLUX_VAE)?.decode_tiled(&sd::Act { c: 16, h: side, w: side, d: x0 }, vae_tile, sd::vae_overlap(vae_tile))?;
    let t_vae = t0.elapsed().as_secs_f64();
    let img = match get("--upscale") {                                 // add-on: Real-ESRGAN x4 on tiles (output 4x larger)
        Some(f) => {
            let t = std::time::Instant::now();
            let uw = open_upscaler(&f)?;
            alloc::trim();                                             // the stage before gives its cached buffers back
            let u = upscale::run(&uw, &img)?;
            eprintln!("upscale: {}x{} -> {}x{} in {:.1} s", img.w, img.h, u.w, u.h, t.elapsed().as_secs_f64());
            u
        }
        None => img,
    };
    save_png(&img, &out)?;
    eprintln!("image: text {t_text:.2} s, unet {t_dit:.2} s ({steps} step), decode {t_vae:.2} s, total {:.2} s -> {out}", tick.elapsed().as_secs_f64());
    Ok(())
}

/// User overrides of the plan: --size N, --steps N, --decoder vae|taesd.
fn overrides_from_args(args: &[String]) -> planner::Overrides {
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    planner::Overrides {
        size: get("--size").and_then(|s| s.parse().ok()),
        steps: get("--steps").and_then(|s| s.parse().ok()),
        decoder: get("--decoder").map(|d| if d == "vae" { planner::Decoder::Vae } else { planner::Decoder::Taesd }),
        vae_tile: get("--vae-tile").and_then(|s| s.parse().ok()),
        upscale: args.iter().any(|a| a == "--upscale"),        // `rope image --upscale FILE` and `rope plan --upscale`: add the upscaler stage
    }
}

/// rope profile <model> [--size N] [--t5 F] [--clip F] [--vae F] [--blocks]: weight bytes, FLOPs per call and
/// activation peak of every block, from one dry counting run per component (no arithmetic is done).
fn profile_cmd(args: &[String]) -> Result<(), String> {
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let size: usize = get("--size").and_then(|s| s.parse().ok()).unwrap_or(512);
    let open = |p: &str| weights::Weights::open(Path::new(p));
    let main_w = open(&args[0])?;
    let comp: Vec<weights::Weights> = ["--t5", "--clip", "--vae"].iter().filter_map(|k| get(k)).map(|p| open(&p)).collect::<Result<_, _>>()?;
    if std::env::var("ROPE_GPU").as_deref() == Ok("all") { ffi::gpu_set_ranges(main_w.ranges("")); }   // test hook until P2.3.3
    let t0 = std::time::Instant::now();
    let (fam, p) = profile_model(&main_w, &comp, size, !args.iter().any(|a| a == "--timed"))?;
    print_profile(args, &fam, size, &p, t0.elapsed().as_secs_f64());
    Ok(())
}

/// Dry counting run of one model (+ companion files) at `size`: (family, profile).
/// `dry`: count only (seconds); else really compute and time every kernel kind (`--timed`).
fn profile_model(main_w: &weights::Weights, comp: &[weights::Weights], size: usize, dry: bool) -> Result<(String, profile::Profile), String> {
    let side = size / 8;
    let files: Vec<&weights::Weights> = std::iter::once(main_w).chain(comp.iter()).collect();
    let fam = detect::detect(&main_w.mf).map(|d| d.family).ok_or("unknown architecture")?;
    let mut p = profile::Profile::from_files(&files);
    p.dry = dry;
    let by = |pre: &str| -> Option<&weights::Weights> { comp.iter().find(|w| w.mf.tensors.iter().any(|t| t.name.starts_with(pre))) };
    match fam.as_str() {
        "sd1" | "sd2" | "sdxl" => {
            let sdxl = fam == "sdxl";
            let mut ctx = Vec::new();
            p.count(&files, "text", || {
                ctx = if sdxl { sd::sdxl_condition(main_w, Some((&[0u32; 77], &[0u32; 77], 49407)), size)? } else { sd::TextEncoder::new(main_w)?.encode(&[0u32; 77])? };
                Ok(())
            })?;
            let lat = sd::Act { c: 4, h: side, w: side, d: vec![0.0; 4 * side * side] };
            p.count(&files, "denoiser", || sd::UNet::new(main_w)?.forward(&lat, 999.0, &ctx).map(|_| ()))?;
            p.count(&files, "vae", || sd::VaeDecoder::new(main_w, if sdxl { sd::SDXL_VAE } else { sd::SD_VAE })?.decode_tiled(&lat, sd::VAE_TILE.0, sd::VAE_TILE.1).map(|_| ()))?;
        }
        "flux" => {
            let lat = vec![0f32; 16 * side * side];
            p.count(&files, "denoiser", || flux::Flux::new(main_w)?.forward(&lat, side, side, 1.0, &vec![0.0; 256 * 4096], &[0.0; 768]).map(|_| ()))?;
            if let Some(w) = by("enc.blk.") { p.count(&files, "text", || t5::T5Encoder::new(w)?.encode(&[0u32; 256]).map(|_| ()))?; }
            if let Some(w) = by("text_model.") { p.count(&files, "text", || sd::TextEncoder::new(w)?.encode(&[0u32; 77]).map(|_| ()))?; }
            if let Some(w) = by("decoder.conv_in") {
                let a = sd::Act { c: 16, h: side, w: side, d: lat.clone() };
                p.count(&files, "vae", || sd::VaeDecoder::new(w, sd::FLUX_VAE)?.decode_tiled(&a, sd::VAE_TILE.0, sd::VAE_TILE.1).map(|_| ()))?;
            }
        }
        f => return Err(format!("profiling {f} models is not supported yet")),
    }
    Ok((fam, p))
}

/// `rope profile --json` document (also the profile cache format).
fn profile_json(fam: &str, size: usize, p: &profile::Profile) -> String {
    let blocks: Vec<String> = p.blocks.iter().map(|(n, b)| format!("{{\"name\": \"{n}\", \"component\": \"{}\", \"bytes\": {}, \"flops\": {:.0}}}", b.component, b.bytes, b.flops)).collect();
    let peaks: Vec<String> = p.peak_private.iter().map(|(c, v)| format!("\"{c}\": {v}")).collect();
    format!("{{\"family\": \"{fam}\", \"size\": {size}, \"peak_private\": {{{}}}, \"blocks\": [\n  {}\n]}}", peaks.join(", "), blocks.join(",\n  "))
}

/// Profile of a model (+ companions) at `size`: cached in <models>/.rope/profiles/ (key: file name, size, file length,
/// mtime, companion count), else one dry counting run (seconds) that fills the cache.
fn cached_profile(models_dir: &str, model_path: &str, main_w: &weights::Weights, comp: &[weights::Weights], size: usize)
    -> Result<(Vec<placement::BlockCost>, [f64; 3]), String> {
    let md = std::fs::metadata(model_path).map_err(|e| e.to_string())?;
    let mt = md.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_secs());
    let name = Path::new(model_path).file_name().unwrap().to_string_lossy().into_owned();
    let f = Path::new(models_dir).join(".rope").join("profiles").join(format!("{name}_{size}_{}_{mt}_{}.json", md.len(), comp.len()));
    if let Ok(t) = std::fs::read_to_string(&f) { if let Ok(v) = json::parse(&t) { return placement::from_profile_json(&v); } }
    let (fam, p) = profile_model(main_w, comp, size, true)?;
    let txt = profile_json(&fam, size, &p);
    let _ = std::fs::create_dir_all(f.parent().unwrap()).and_then(|_| std::fs::write(&f, &txt));
    placement::from_profile_json(&json::parse(&txt)?)
}

/// Where every block of this plan runs on this device (the same answer for `rope plan --explain` and `rope image`).
/// Blocks are GPU-capable only if their component is in the family's `gpu_components` (code on the op interface).
#[allow(clippy::too_many_arguments)]
/// A plan's GPU placement on this device (CUDA build, a GPU, a family with GPU components), shared by `rope image`
/// and `rope suggest`: the placement, the blocks it puts on the GPU / keeps resident, and the estimate with the GPU
/// (total, decode share). None when nothing runs on the GPU.
struct GpuEstimate { pl: placement::Placement, gpu: std::collections::HashSet<String>, resident: std::collections::HashSet<String>, total: f64, decode: f64 }

fn gpu_estimate(models_dir: &str, model_path: &str, spec: &planner::ModelSpec, p: &planner::Plan, dev: &planner::Device, vram: f64)
                -> Result<Option<GpuEstimate>, String> {
    if !cfg!(feature = "cuda") || vram <= 0.0 || spec.gpu_components.is_empty() { return Ok(None); }
    let path_of = |role: &str| p.files.iter().find(|f| f.role == role).map(|f| Path::new(models_dir).join(&f.name).to_string_lossy().into_owned());
    let (pl, _) = {
        let w = weights::Weights::open(Path::new(model_path))?;
        let comp: Vec<weights::Weights> = ["t5", "clip", "vae"].iter().filter_map(|r| path_of(r)).filter_map(|f| weights::Weights::open(Path::new(&f)).ok()).collect();
        placement_for(models_dir, model_path, &w, &comp, p, &spec.gpu_components, &spec.gpu_eff, dev, vram)?
    };
    let pick = |f: fn(placement::Where) -> bool| pl.blocks.iter().filter(|b| f(b.at)).map(|b| b.name.clone()).collect::<std::collections::HashSet<_>>();
    let (gpu, resident) = (pick(|w| matches!(w, placement::Where::GpuResident | placement::Where::GpuStreamed)), pick(|w| w == placement::Where::GpuResident));
    if gpu.is_empty() { return Ok(None); }
    let taesd = p.decoder == planner::Decoder::Taesd;
    let total = p.estimate.open + pl.seconds + pl.upload_once_s + if taesd { p.estimate.decode } else { 0.0 };
    // the VAE's blocks as placed (GPU or CPU); TAESD runs on the CPU at the planner's estimate
    let decode = if taesd { p.estimate.decode } else { pl.blocks.iter().filter(|b| b.component == "vae").map(|b| b.seconds).sum() };
    Ok(Some(GpuEstimate { pl, gpu, resident, total, decode }))
}

fn placement_for(models_dir: &str, model_path: &str, main_w: &weights::Weights, comp: &[weights::Weights], p: &planner::Plan,
                 gpu_components: &[String], gpu_eff: &[(String, f64)], dev: &planner::Device, vram: f64) -> Result<(placement::Placement, [f64; 3]), String> {
    let (mut blocks, peaks) = cached_profile(models_dir, model_path, main_w, comp, p.size as usize)?;
    // effective CPU speed per component on this device = profiled FLOPs / the planner's calibrated stage time
    let e = &p.estimate;
    let cfg = if p.guidance > 1.0 { 2.0 } else { 1.0 };
    let fl = |c: &str| blocks.iter().filter(|b| b.component == c).map(|b| b.flops / 1e9).sum::<f64>();
    let rate = |gf: f64, s: f64| if s > 0.0 && gf > 0.0 { gf / s } else { 100.0 };
    let cpu = [rate(fl("text"), e.text / cfg), rate(fl("denoiser"), e.unet_step / cfg), rate(fl("vae"), e.decode)];
    // the plan's decoder: with TAESD (a tiny separate file) the full VAE's blocks are not used at all
    let taesd = p.decoder == planner::Decoder::Taesd;
    blocks.retain(|b| !(taesd && b.component == "vae"));
    for b in &mut blocks {
        b.gpu_ok = gpu_components.iter().any(|c| *c == b.component);
        b.gpu_eff = gpu_eff.iter().find(|(c, _)| *c == b.component).map_or(1.0, |e| e.1);
    }
    let g = dev.gpu.as_ref();
    let pd = placement::Dev { ram_gib: dev.ram_usable_gib, vram_gib: vram, cpu_gflops: cpu,
                              gpu_gflops: g.map_or(0.0, |g| g.gflops), h2d_gbs: g.map_or(0.0, |g| g.h2d_gbs), disk_gbs: dev.disk_read_gbs };
    let pk = [peaks[0], peaks[1], if taesd { 0.0 } else { peaks[2] }];
    let pl = placement::place(&blocks, &pd, &placement::Use { denoiser_calls: p.steps as f64 * cfg, peak_private: pk })?;
    Ok((pl, cpu))
}

fn print_profile(args: &[String], fam: &str, size: usize, p: &profile::Profile, secs: f64) {
    let gib = |b: u64| b as f64 / 1073741824.0;
    if args.iter().any(|a| a == "--json") { println!("{}", profile_json(fam, size, p)); return; }
    println!("profile of {} [{fam}] at {size}x{size} (counting run {secs:.1} s)", args[0]);
    println!("  {:<9} {:>6} {:>10} {:>14} {:>13}  runs", "component", "blocks", "weights", "GFLOP/call", "activations");
    for (c, runs) in [("text", "once"), ("denoiser", "every step (x2 with CFG)"), ("vae", "once"), ("other", "-")] {
        let (n, b, fl) = p.totals(c);
        if n == 0 { continue; }
        let act = p.peak_private.get(c).map_or("-".to_string(), |v| format!("{:.2} GiB", gib(*v)));
        println!("  {c:<9} {n:>6} {:>8.2} GiB {fl:>14.1} {act:>13}  {runs}", gib(b));
    }
    for (c, (wall, kinds)) in &p.timed {
        let ks: f64 = kinds.values().map(|v| v.0).sum();
        let parts: Vec<String> = kinds.iter().map(|(k, (s, fl))| format!("{k} {s:.2} s ({:.0} GFLOPS)", fl / s.max(1e-9) / 1e9)).collect();
        println!("  timed {c:<9} {wall:>7.2} s = kernels {ks:.2} s [{}] + other (Rust) {:.2} s ({:.0}%)", parts.join(", "), wall - ks, (wall - ks) / wall * 100.0);
    }
    if !p.shapes.is_empty() {
        let mut v: Vec<_> = p.shapes.iter().collect();
        v.sort_by(|a, b| b.1 .1.total_cmp(&a.1 .1));
        println!("  slowest matmul shapes (m x k x n, type): calls, seconds, GFLOPS");
        for ((m, k, n, t), (c, s)) in v.into_iter().take(10) {
            println!("    {m:>5} x {k:>5} x {n:>5} t{t:<2} {c:>4} calls {s:>7.2} s {:>6.0} GFLOPS", 2.0 * (*m * *k * *n) as f64 * *c as f64 / s / 1e9);
        }
    }
    let sum: u64 = p.blocks.values().map(|b| b.bytes).sum();
    println!("  check: blocks hold {} of {} tensor bytes in the file(s)", sum, p.file_bytes);
    if args.iter().any(|a| a == "--blocks") {
        for (name, b) in &p.blocks {
            let types: Vec<String> = b.types.keys().cloned().collect();
            println!("    {:<9} {name:<70} {:>9.1} MiB {:>9.2} GFLOP  {}", b.component, b.bytes as f64 / 1048576.0, b.flops / 1e9, types.join("+"));
        }
    }
}

/// Load the catalog: (model specs, calibration GFLOPS).
/// How to read a user's big file on this device (the source rule): as it is, streamed from disk, or a quantized copy made once.
fn source_for(spec: &planner::ModelSpec, w: &weights::Weights, dev: &planner::Device, cal: planner::Cal, q: planner::Quality, args: &[String], unet: f64)
              -> Result<planner::Source, String> {
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let bits = stored_bits(w);
    let force = match get("--quantize").as_deref() { Some("q8_0") => Some(ffi::RT_Q8_0), Some("q4_k") => Some(ffi::RT_Q4_K), Some(o) => return Err(format!("--quantize {o}: use q8_0 or q4_k")), None => None };
    let per_call = if spec.guidance > 1.0 { 2.0 } else { 1.0 };
    let size = overrides_from_args(args).size.unwrap_or(spec.native);
    let steps = overrides_from_args(args).steps.unwrap_or(spec.steps_for(q));
    let e = planner::estimate(spec, dev, cal, size, steps, planner::Decoder::Vae, 64);
    let gib = 1073741824.0;
    let m = planner::SourceIn { denoiser_gib: unet / gib, rest_gib: (w.mf.file_size as f64 - unet).max(0.0) / gib, stored_bits: bits,
                                commit_gib: spec.commit_for_vae(), calls: steps as f64 * per_call, compute_s: e.unet_step / per_call,
                                image_s: e.open + e.text + steps as f64 * e.unet_step + e.decode };
    Ok(planner::source_strategy(m, dev, args.iter().any(|a| a == "--keep-precision"), spec.can_quantize(), force))
}

/// Bits per weight of the file's full-precision denoiser tensors (0 = already quantized).
fn stored_bits(w: &weights::Weights) -> f64 {
    w.mf.tensors.iter().filter(|t| profile::component(&t.name) == "denoiser").find(|t| t.ggml_type.is_none())
        .map_or(0.0, |t| match t.dtype.as_str() { "F32" => 32.0, "F16" | "BF16" => 16.0, _ => 0.0 })
}

/// Why rope cannot run this whole-model file beyond its architecture: FLUX-dev, or an SD checkpoint without text encoder or VAE.
fn file_gap(mf: &modelfile::ModelFile, family: &str) -> Option<String> {
    let any = |p: &str| mf.tensors.iter().any(|t| t.name.starts_with(p));
    if family == "flux" && any("guidance_in.") {
        return Some("this is FLUX.1-dev (guidance-distilled, 20+ steps): rope runs FLUX.1-schnell only for now".into());
    }
    // the same name prefixes the loaders accept (sd.rs): original (cond_stage_model.) and SGM (conditioner.) layouts; VAE with or without first_stage_model.
    let vae = any("first_stage_model.") || any("decoder.conv_in.");
    let (text, vae) = match family {
        "sd1" | "sd2" | "sdxl" => (any("cond_stage_model.") || any("conditioner."), vae),
        _ => (true, true),
    };
    match (text, vae) {
        (true, true) => None,
        (false, _) => Some("the file has no text encoder inside (a UNet-only file): rope needs a single-file checkpoint with text encoder and VAE".into()),
        (_, false) => Some("the file has no VAE inside: rope needs a single-file checkpoint with text encoder and VAE".into()),
    }
}

/// rope check-model <file> [--id X] [--quality q] [--models-dir D] [--catalog C] [device overrides] [--json]:
/// can rope run a model file the user brings, on this laptop, and how? Reads the header only (no hash, no scan:
/// those run at the first picture). If the default size does not fit, it uses the largest size that does.
fn check_model_cmd(args: &[String]) -> Result<(), String> {
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let path = args.first().filter(|a| !a.starts_with("--")).ok_or("usage: rope check-model <file>")?.clone();
    let models_dir = get("--models-dir").unwrap_or_else(|| "models".into());
    let cat_path = get("--catalog").unwrap_or_else(|| "catalog/models.json".into());
    let name = Path::new(&path).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    let id = get("--id").unwrap_or_else(|| name.clone());
    let as_json = args.iter().any(|a| a == "--json");
    let head = |title: &str, family: &str| format!("\"id\": {}, \"kind\": \"image\", \"mine\": true, \"title\": {}, \"file\": {}, \"path\": {}, \"bytes\": {bytes}, \"family\": {}, \"on_disk\": true, \"tested\": false",
                                                   json::quote(&id), json::quote(title), json::quote(&name), json::quote(&path), json::quote(family));
    let no = |title: &str, family: &str, why: &str| -> Result<(), String> {
        if as_json { println!("{{{}, \"runs\": false, \"why\": {}}}", head(title, family), json::quote(why)); } else { println!("{name}: cannot run here: {why}"); }
        Ok(())
    };
    let ext = Path::new(&path).extension().map(|e| e.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
    if !matches!(ext.as_str(), "safetensors" | "gguf") {
        return no(&name, "", if matches!(ext.as_str(), "ckpt" | "pt" | "pth" | "bin") {
            "a .ckpt / .pt / .bin file is a Python pickle, and opening one can run code: rope opens .safetensors and .gguf only. Convert the file to .safetensors first"
        } else { "not a model file rope reads (.safetensors or .gguf)" });
    }
    let w = match weights::Weights::open(Path::new(&path)) { Ok(w) => w, Err(e) => return no(&name, "", &format!("cannot read the file: {e}")) };
    let Some(d) = detect::detect(&w.mf) else { return no(&name, "", "unknown architecture: rope runs SD 1.x, SD 2.x, SDXL and FLUX.1-schnell models") };
    let title = format!("{} ({name})", d.title);
    match d.role {
        detect::Role::Diffusion => {}
        detect::Role::Adapter if d.family == "lora" => return no(&title, &d.family, "this is a LoRA add-on, not a whole model: put it in the add-ons folder (models/addons) and choose it under Add-ons"),
        detect::Role::Upscaler => return no(&title, &d.family, "this is an upscaler add-on, not a whole model: put it in the add-ons folder (models/addons)"),
        _ => return no(&title, &d.family, &format!("this is a part of a model ({}), not a whole image model", d.role.name())),
    }
    if let Err(why) = detect::support(&d.family) { return no(&title, &d.family, &format!("architecture {} is not supported yet: {why}", d.title)); }
    if let Some(why) = file_gap(&w.mf, &d.family) { return no(&title, &d.family, &why); }
    let cat = json::parse(&std::fs::read_to_string(&cat_path).map_err(|e| format!("{cat_path}: {e}"))?)?;
    let (models, cal) = load_catalog(&cat_path)?;
    let q = planner::Quality::parse(&get("--quality").unwrap_or_else(|| "balanced".into())).ok_or("quality must be fast|balanced|best")?;
    let dev = device_from_args(args, &models_dir)?;
    let unet: f64 = w.mf.tensors.iter().filter(|t| profile::component(&t.name) == "denoiser").map(|t| t.nbytes as f64).sum();
    // a catalog file (same name and size) keeps its tuned settings; the hash check comes at the first picture
    let known = models.iter().find(|m| m.files.iter().any(|f| f.role == "model" && f.name == name && f.bytes == bytes));
    let mut spec = match known { Some(k) => k.clone(), None => planner::family_spec(&cat, &d.family, &name, unet)? };
    spec.files.retain(|f| f.role != "model");
    let mut notes = Vec::new();
    let source = match source_for(&spec, &w, &dev, cal, q, args, unet)? {
        planner::Source::AsIs => "as-is",
        planner::Source::Stream(why) => { notes.push(format!("big file: {why}")); "stream" }
        planner::Source::Quantize { ty, why, .. } => {
            notes.push(format!("big file: {why}. Rope makes this copy at the first picture and keeps it"));
            spec.unet_bytes_set(unet * if ty == ffi::RT_Q8_0 { 8.5 } else { 4.5 } / stored_bits(&w).max(1.0));
            "quantize"
        }
    };
    let dir = models_dir.clone();
    let have = move |n: &str| Path::new(&dir).join(n).exists();
    let ov = overrides_from_args(args);
    let p = match planner::plan_model(&spec, &dev, cal, q, &ov, &have) {
        Ok(p) => p,
        Err(why) => match planner::max_plan(&spec, &dev, cal, q, &planner::Overrides { size: None, ..ov.clone() }, &have) {
            Some(x) => { notes.push(format!("the default size does not fit here ({why}); using {0}x{0}, the largest size that fits", x.size)); x }
            None => return no(&title, &d.family, &why),
        },
    };
    notes.extend(p.notes.iter().cloned());
    if as_json {
        println!("{{{}, \"runs\": true, \"license\": {}, \"source\": \"{source}\", \"size\": {}, \"steps\": {}, \"guidance\": {}, \"decoder\": {}, \"commit_gib\": {:.2}, \"peak_gib\": {:.2}, \"download_gib\": {:.2}, \"estimate_s\": {:.1}, \"notes\": [{}]}}",
                 head(&title, &d.family), json::quote(known.map_or("", |k| k.license.as_str())), p.size, p.steps, p.guidance, json::quote(p.decoder.name()), p.commit_gib, p.peak_gib,
                 p.download_bytes as f64 / 1073741824.0, p.estimate.total, notes.iter().map(|n| json::quote(n)).collect::<Vec<_>>().join(", "));
    } else {
        println!("{name}: {} - runs here: {}x{}, {} step(s), ~{:.0} s; file used {source}; helper files to download {:.2} GiB", d.title, p.size, p.size, p.steps, p.estimate.total, p.download_bytes as f64 / 1073741824.0);
        for n in &notes { println!("  note: {n}"); }
    }
    Ok(())
}

/// The spec rope runs a user's file with, shared by `rope image` and `rope plan --explain`: detect the architecture,
/// admit the file (sha256 + safety scan, cached), then the catalog entry whose sha256 matches (tuned settings) or the
/// family defaults. Returns (spec without the model file to download, detection, admission, denoiser bytes).
fn spec_for_file(cat: &json::Value, models: &[planner::ModelSpec], models_dir: &str, mp: &str, w: &weights::Weights, args: &[String])
                 -> Result<(planner::ModelSpec, detect::Detection, models::Admission, f64), String> {
    let d = detect::detect(&w.mf).ok_or("UNSUPPORTED: unknown architecture (see rope detect)")?;
    eprintln!("detected: {} [{}]", d.title, d.family);
    detect::support(&d.family).map_err(|why| format!("UNSUPPORTED: architecture {} ({}) is not supported yet: {why}", d.family, d.title))?;
    if let Some(why) = file_gap(&w.mf, &d.family) { return Err(format!("UNSUPPORTED: {why}")); }
    let t0 = std::time::Instant::now();
    let adm = models::admit(Path::new(models_dir), Path::new(mp), w, args.iter().any(|a| a == "--allow-suspicious"))?;
    if adm.first_use { eprintln!("first use: sha256 + safety scan in {:.1} s (scan: {})", t0.elapsed().as_secs_f64(), adm.scan); }
    let unet: f64 = w.mf.tensors.iter().filter(|t| profile::component(&t.name) == "denoiser").map(|t| t.nbytes as f64).sum();
    let mut spec = match models.iter().find(|m| m.files.iter().any(|f| f.role == "model" && f.sha256 == adm.sha256)) {
        Some(known) => { eprintln!("recognized: {} (catalog settings)", known.id); known.clone() }
        None => planner::family_spec(cat, &d.family, &Path::new(mp).file_name().unwrap().to_string_lossy(), unet)?,
    };
    spec.files.retain(|f| f.role != "model");                    // the user's file is used in place, never downloaded
    Ok((spec, d, adm, unet))
}

fn load_catalog(path: &str) -> Result<(Vec<planner::ModelSpec>, planner::Cal), String> {
    let j = json::parse(&std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?)?;
    let c = j.get("calibration");
    let num = |k: &str| c.and_then(|c| c.get(k)).and_then(json::Value::as_f64).unwrap_or(0.0);
    let cal = planner::Cal { gflops: num("gemm_gflops"), gflops_1t: num("gemm_gflops_1t") };
    if cal.gflops <= 0.0 || cal.gflops_1t <= 0.0 { return Err("catalog calibration needs gemm_gflops and gemm_gflops_1t".into()); }
    let models = planner::catalog_specs(&j)?;
    Ok((models, cal))
}

/// The device for planning: live probe + GEMM speed, with optional overrides that simulate another laptop:
/// --ram-gib X, --ram-free-gib X, --gflops G, --gflops-1t G, --no-gpu, --disk-gbs X.
fn device_from_args(args: &[String], models_dir: &str) -> Result<planner::Device, String> {
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).and_then(|s| s.parse::<f64>().ok());
    let probe = json::parse(&ffi::probe_json(models_dir, get("--disk-mb").map(|v| v as i32).unwrap_or(256))?)?;
    let g = get("--gflops").unwrap_or_else(ffi::gemm_gflops);
    let g1 = get("--gflops-1t").unwrap_or_else(|| ffi::gemm_gflops_threads(1));
    let mut dev = planner::Device::from_probe(&probe, g, g1)?;
    if let Some(r) = get("--ram-gib") { dev.ram_usable_gib = r; dev.ram_free_gib = dev.ram_free_gib.min(r); }
    if let Some(r) = get("--ram-free-gib") { dev.ram_free_gib = r; }
    if let Some(d) = get("--disk-gbs") { dev.disk_read_gbs = d; }
    if args.iter().any(|a| a == "--no-gpu") { dev.gpu = None; }
    Ok(dev)
}

/// rope detect <file>: which architecture is this, and can rope run it? Exit 0 recognized, 3 unknown.
fn detect_cmd(path: &str) -> i32 {
    let mf = match modelfile::open(Path::new(path)) {
        Ok(m) => m,
        Err(e) => { println!("unknown: not a readable GGUF/safetensors model file ({e})"); return 3; }
    };
    match detect::detect(&mf) {
        Some(d) => {
            let addon = matches!(d.role, detect::Role::Adapter | detect::Role::Upscaler | detect::Role::Control);
            let sup = match detect::support(&d.family) { Ok(()) => "supported by rope".to_string(), Err(why) if addon && d.family != "controlnet" => format!("add-on: {why}"), Err(why) => format!("not supported yet: {why}") };
            println!("{}
  family   {}
  role     {}
  status   {}
  evidence {}", d.title, d.family, d.role.name(), sup, d.evidence);
            0
        }
        None => { println!("unknown architecture ({} tensors; first: {})", mf.tensors.len(), mf.tensors.first().map_or("-", |t| t.name.as_str())); 3 }
    }
}

/// rope scan <model>: outlier-injection scan of full-precision weights. Exit 1 if any tensor is flagged.
fn scan_cmd(path: &str) -> Result<usize, String> {
    let w = weights::Weights::open(Path::new(path))?;
    let t0 = std::time::Instant::now();
    let reps = scan::scan(&w)?;
    let flagged: Vec<_> = reps.iter().filter(|r| r.stats.flagged()).collect();
    let groups: u64 = reps.iter().map(|r| r.stats.groups).sum();
    let worst = reps.iter().map(|r| r.stats.max_ratio).fold(0f32, f32::max);
    println!("scanned {} float tensors, {} groups of {} in {:.1} s; largest group ratio {:.1} (limit {})",
             reps.len(), groups, scan::GROUP, t0.elapsed().as_secs_f64(), worst, scan::RATIO_LIMIT);
    if let Some(top) = reps.iter().max_by(|a, b| a.stats.spiky.cmp(&b.stats.spiky).then(a.stats.max_ratio.total_cmp(&b.stats.max_ratio))) {
        if top.stats.spiky > 0 { println!("  most spiky tensor: {} ({} of {} groups above {}, max ratio {:.0}){}", top.name, top.stats.spiky, top.stats.groups,
                                          scan::RATIO_LIMIT, top.stats.max_ratio, if top.stats.flagged() { "" } else { " - below the flag rule" }); }
    }
    for r in &flagged {
        println!("  FLAGGED {}: {} of {} groups above {} (max ratio {:.0})", r.name, r.stats.spiky, r.stats.groups, scan::RATIO_LIMIT, r.stats.max_ratio);
    }
    println!("{}", if flagged.is_empty() { "CLEAN: no outlier injection pattern" } else { "SUSPICIOUS: do not quantize this model" });
    Ok(flagged.len())
}

/// rope verify [--models-dir D] [--catalog C]: re-hash every catalog file present in the folder (ignores markers).
fn verify_cmd(args: &[String]) -> Result<(), String> {
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let dir = get("--models-dir").unwrap_or_else(|| "models".into());
    let (models, _) = load_catalog(&get("--catalog").unwrap_or_else(|| "catalog/models.json".into()))?;
    let mut bad = 0;
    for f in models.iter().flat_map(|m| m.files.iter()) {
        let p = Path::new(&dir).join(&f.name);
        if !p.exists() { continue; }
        match models::verify(Path::new(&dir), f, &p) {
            Ok(()) => println!("  OK        {}", f.name),
            Err(e) => { bad += 1; println!("  REJECTED  {e}"); }
        }
    }
    if bad > 0 { Err(format!("{bad} file(s) failed verification")) } else { Ok(()) }
}

/// rope get [--quality q] [--models-dir D] [--catalog C] [device overrides]: fetch + verify exactly the files the plan needs.
fn get_cmd(args: &[String]) -> Result<(), String> {
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let models_dir = get("--models-dir").unwrap_or_else(|| "models".into());
    let (models, cal) = load_catalog(&get("--catalog").unwrap_or_else(|| "catalog/models.json".into()))?;
    let q = planner::Quality::parse(&get("--quality").unwrap_or_else(|| "balanced".into())).ok_or("quality must be fast|balanced|best")?;
    let dev = device_from_args(args, &models_dir)?;
    let dir = models_dir.clone();
    let p = planner::plan(&models, &dev, cal, &get("--kind").unwrap_or_else(|| "image".into()), q, &planner::Overrides::default(),
                          &move |n: &str| Path::new(&dir).join(n).exists())?;
    println!("plan: {} {}x{} {} step(s), decoder {} -> needs {} file(s)", p.model, p.size, p.size, p.steps, p.decoder.name(), p.files.len());
    let mut log = Vec::new();
    for f in &p.files {
        let r = models::ensure(Path::new(&models_dir), f, &mut log);
        for l in log.drain(..) { println!("  {l}"); }
        match r {
            Ok(o) => println!("  {:<40} {:?}", f.name, o),
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// rope plan [--kind image] [--quality fast|balanced|best] [--size N] [--steps N] [--decoder taesd|vae]
///           [--models-dir models] [--catalog catalog/models.json] [--gflops G] [--disk-mb N] [--json]
fn plan_cmd(args: &[String]) -> Result<(), String> {
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let models_dir = get("--models-dir").unwrap_or_else(|| "models".into());
    let (models, cal) = load_catalog(&get("--catalog").unwrap_or_else(|| "catalog/models.json".into()))?;
    let q = planner::Quality::parse(&get("--quality").unwrap_or_else(|| "balanced".into())).ok_or("quality must be fast|balanced|best")?;
    let dev = device_from_args(args, &models_dir)?;
    let ov = overrides_from_args(args);
    if args.iter().any(|a| a == "--explain") { return explain_cmd(args, &models_dir, cal, q, &dev, &ov); }
    if args.iter().any(|a| a == "--max") { return plan_max_cmd(args, &models, cal, q, &dev, &ov); }
    let dir = models_dir.clone();
    let have = move |n: &str| Path::new(&dir).join(n).exists();
    let p = match get("--model-id") {                           // a given catalog model, else the planner's first choice
        Some(id) => planner::plan_model(models.iter().find(|m| m.id == id && m.supported).ok_or_else(|| format!("no supported catalog model {id}"))?, &dev, cal, q, &ov, &have)?,
        None => planner::plan(&models, &dev, cal, &get("--kind").unwrap_or_else(|| "image".into()), q, &ov, &have)?,
    };
    let e = &p.estimate;
    if args.iter().any(|a| a == "--json") {
        println!("{{\"model\": \"{}\", \"size\": {}, \"steps\": {}, \"decoder\": \"{}\", \"placement\": \"{}\", \"threads\": {}, \"commit_gib\": {:.2}, \"peak_gib\": {:.2}, \"vae_tile\": {}, \"download_gib\": {:.2}, \
                  \"device_gflops\": {:.1}, \"device_gflops_1t\": {:.1}, \"estimate_s\": {{\"open\": {:.2}, \"text\": {:.2}, \"unet_step\": {:.2}, \"paging_step\": {:.2}, \"decode\": {:.2}, \"total\": {:.2}}}}}",
                 p.model, p.size, p.steps, p.decoder.name(), p.placement, p.threads, p.commit_gib, p.peak_gib, p.vae_tile, p.download_bytes as f64 / 1073741824.0,
                 dev.gemm_gflops, dev.gemm_gflops_1t, e.open, e.text, e.unet_step, e.paging_step, e.decode, e.total);
    } else {
        println!("plan ({} quality) on this device ({:.0} GFLOPS, {:.1} GiB RAM, disk {:.1} GB/s):", planner::quality_key(q), dev.gemm_gflops, dev.ram_usable_gib, dev.disk_read_gbs);
        println!("  model    {} — {}", p.model, p.title);
        println!("  image    {}x{}, {} step(s), decoder {}, on {} with {} threads", p.size, p.size, p.steps, p.decoder.name(), p.placement, p.threads);
        println!("  memory   predicted peak {:.2} GiB (decoder tile {}x{} px); downloads needed {:.2} GiB", p.peak_gib, p.vae_tile * 8, p.vae_tile * 8, p.download_bytes as f64 / 1073741824.0);
        println!("  time     ~{:.0} s  (text {:.1} + {} x {:.1} s UNet{} + decode {:.1})", e.total, e.text, e.steps, e.unet_step,
                 if e.paging_step > 0.0 { format!(" + {:.1} s disk", e.paging_step) } else { String::new() }, e.decode);
        for n in &p.notes { println!("  note     {n}"); }
    }
    Ok(())
}

/// rope plan --max [--quality q] [--model-id ID] [--json] [device overrides]: for every supported catalog model (or one), the
/// largest image size this laptop can run, what it costs (time, predicted peak memory, VAE tile), and why the next size up is refused.
fn plan_max_cmd(args: &[String], models: &[planner::ModelSpec], cal: planner::Cal, q: planner::Quality, dev: &planner::Device, ov: &planner::Overrides) -> Result<(), String> {
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let only = get("--model-id");
    let have = |_: &str| true;
    let ov0 = planner::Overrides { size: None, ..ov.clone() };
    let rows: Vec<(&planner::ModelSpec, Option<planner::Plan>, Option<String>)> = models.iter().filter(|m| m.supported && m.kind == "image" && only.as_deref().map_or(true, |o| m.id == o))
        .map(|m| {
            let mp = planner::max_plan(m, dev, cal, q, &ov0, &have);
            let why = match &mp {
                Some(x) if x.size + 64 <= m.max => planner::plan_model(m, dev, cal, q, &planner::Overrides { size: Some(x.size + 64), ..ov0.clone() }, &have).err(),
                Some(_) => None,
                None => planner::plan_model(m, dev, cal, q, &planner::Overrides { size: Some(64), ..ov0.clone() }, &have).err(),
            };
            (m, mp, why)
        }).collect();
    if args.iter().any(|a| a == "--json") {
        let items: Vec<String> = rows.iter().map(|(m, mp, why)| format!("{{\"id\": {}, \"max_size\": {}, \"model_max\": {}, \"estimate_s\": {:.1}, \"peak_gib\": {:.2}, \"vae_tile\": {}, \"limit\": {}}}",
            json::quote(&m.id), mp.as_ref().map_or(0, |p| p.size), m.max, mp.as_ref().map_or(0.0, |p| p.estimate.total), mp.as_ref().map_or(0.0, |p| p.peak_gib),
            mp.as_ref().map_or(0, |p| p.vae_tile), why.as_ref().map_or("null".to_string(), |w| json::quote(w)))).collect();
        println!("{{\"quality\": {}, \"ram_gib\": {:.2}, \"models\": [{}]}}", json::quote(planner::quality_key(q)), dev.ram_usable_gib, items.join(", "));
        return Ok(());
    }
    println!("largest image per model on this device ({:.1} GiB RAM usable, {} quality):", dev.ram_usable_gib, planner::quality_key(q));
    for (m, mp, why) in &rows {
        match mp {
            Some(p) => {
                println!("  {:<18} {}x{}  ~{:.0} s  peak {:.2} GiB, decoder tile {}x{} px", m.id, p.size, p.size, p.estimate.total, p.peak_gib, p.vae_tile * 8, p.vae_tile * 8);
                match why { Some(w) => println!("  {:<18} next size up: {w}", ""), None => println!("  {:<18} (the model's own maximum, {})", "", m.max) }
            }
            None => println!("  {:<18} does not fit: {}", m.id, why.as_deref().unwrap_or("no reason")),
        }
    }
    Ok(())
}

/// rope suggest [--kind image|video|text] [--quality q] [device overrides]: which catalog models suit this laptop.
fn suggest_cmd(args: &[String]) -> Result<(), String> { suggest_impl(args, None) }

/// The `suggest --json` answer as a string (for the library): every catalog model, whether it runs here, its plan.
pub fn suggest_json(args: &[String]) -> Result<String, String> {
    let mut a = args.to_vec();
    if !a.iter().any(|x| x == "--json") { a.push("--json".into()); }
    let mut out = String::new();
    suggest_impl(&a, Some(&mut out))?;
    Ok(out)
}

fn suggest_impl(args: &[String], json_out: Option<&mut String>) -> Result<(), String> {
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let models_dir = get("--models-dir").unwrap_or_else(|| "models".into());
    let (models, cal) = load_catalog(&get("--catalog").unwrap_or_else(|| "catalog/models.json".into()))?;
    let q = planner::Quality::parse(&get("--quality").unwrap_or_else(|| "balanced".into())).ok_or("quality must be fast|balanced|best")?;
    let dev = device_from_args(args, &models_dir)?;
    let dir = models_dir.clone();
    let ov = overrides_from_args(args);
    let only = get("--only");                                     // one catalog model (the app refreshes the chosen one at another size)
    let have = move |n: &str| Path::new(&dir).join(n).exists();
    let list: Vec<_> = planner::suggest(&models, &dev, cal, get("--kind").as_deref(), q, &ov, &have).into_iter()
        .filter(|s| only.as_deref().map_or(true, |o| s.model.id == o)).collect();
    if args.iter().any(|a| a == "--json") {
        // for the app (rope serve): every catalog model, whether it runs here, its plan, and whether its file is on disk
        let items: Vec<String> = list.iter().map(|s| {
            let m = s.model;
            let file = m.files.iter().find(|f| f.role == "model").map(|f| f.name.clone()).unwrap_or_default();
            let on_disk = !file.is_empty() && Path::new(&models_dir).join(&file).exists();
            let head = format!("\"id\": {}, \"kind\": {}, \"title\": {}, \"file\": {}, \"on_disk\": {on_disk}, \"tested\": {}, \"license\": {}",
                               json::quote(&m.id), json::quote(&m.kind), json::quote(&m.title), json::quote(&file), m.tested, json::quote(&m.license));
            // with a GPU (CUDA build): the placement's estimate, as `rope image` will use it (profiles are cached)
            let vram = dev.gpu.as_ref().map_or(0.0, |g| g.vram_free_gib);
            let gpu = match &s.plan { Ok(p) if on_disk => gpu_estimate(&models_dir, &Path::new(&models_dir).join(&file).to_string_lossy(), m, p, &dev, vram).ok().flatten(), _ => None };
            match &s.plan {
                Ok(p) => {
                    // the ceiling on this device and the sizes that fit (CPU planner estimates; the chosen size is refreshed with --size)
                    let ov0 = planner::Overrides { size: None, ..ov.clone() };
                    let mp = planner::max_plan(m, &dev, cal, q, &ov0, &have);
                    // multiples of 128 up to the model's maximum that fit, plus the ceiling itself when it is not one of them
                    let mut grid: Vec<u32> = (2..=m.max / 128).map(|k| k * 128).collect();
                    if let Some(x) = &mp { if !grid.contains(&x.size) && x.size >= 256 { grid.push(x.size); grid.sort_unstable(); } }
                    let sizes: Vec<String> = grid.into_iter().filter_map(|sz| planner::plan_model(m, &dev, cal, q, &planner::Overrides { size: Some(sz), ..ov0.clone() }, &have).ok())
                        .map(|sp| format!("{{\"size\": {}, \"estimate_s\": {:.1}, \"peak_gib\": {:.2}, \"vae_tile\": {}}}", sp.size, sp.estimate.total, sp.peak_gib, sp.vae_tile)).collect();
                    let next = mp.as_ref().map(|x| x.size + 64).filter(|s| *s <= m.max)
                        .and_then(|s| planner::plan_model(m, &dev, cal, q, &planner::Overrides { size: Some(s), ..ov0.clone() }, &have).err());
                    format!("{{{head}, \"runs\": true, \"on_gpu\": {}, \"cpu_estimate_s\": {:.1},", gpu.is_some(), p.estimate.total) + &format!(" \"size\": {}, \"steps\": {}, \"guidance\": {}, \"decoder\": {}, \"commit_gib\": {:.2}, \"peak_gib\": {:.2}, \"vae_tile\": {}, \"download_gib\": {:.2}, \"estimate_s\": {:.1}, \"upscale_s\": {:.1}, \"max_size\": {}, \"max_estimate_s\": {:.1}, \"max_native\": {}, \"sizes\": [{}], \"limit\": {}, \"notes\": [{}]}}",
                                 p.size, p.steps, p.guidance, json::quote(p.decoder.name()), p.commit_gib, p.peak_gib, p.vae_tile, p.download_bytes as f64 / 1073741824.0,
                                 gpu.as_ref().map_or(p.estimate.total, |g| g.total + p.estimate.upscale), p.estimate.upscale, mp.as_ref().map_or(0, |x| x.size), mp.as_ref().map_or(0.0, |x| x.estimate.total), m.max,
                                 sizes.join(", "), next.map_or("null".to_string(), |w| json::quote(&w)), p.notes.iter().map(|n| json::quote(n)).collect::<Vec<_>>().join(", "))
                }
                Err(why) => format!("{{{head}, \"runs\": false, \"supported\": {}, \"why\": {}}}", m.supported, json::quote(why)),
            }
        }).collect();
        let text = format!("{{\"quality\": {}, \"device\": {{\"gflops\": {:.1}, \"ram_gib\": {:.2}, \"gpu\": {}}}, \"models\": [{}]}}",
                 json::quote(planner::quality_key(q)), dev.gemm_gflops, dev.ram_usable_gib,
                 dev.gpu.as_ref().map_or("null".to_string(), |g| json::quote(&g.name)), items.join(", "));
        match json_out { Some(o) => *o = text, None => println!("{text}") }
        return Ok(());
    }
    println!("models for this device ({:.0} GFLOPS, {:.1} GiB RAM, {}; {} quality):", dev.gemm_gflops, dev.ram_usable_gib,
             dev.gpu.as_ref().map_or("no GPU".to_string(), |g| g.name.clone()), planner::quality_key(q));
    for (i, s) in list.iter().enumerate() {
        let m = s.model;
        match &s.plan {
            Ok(p) => println!("  {}. {:<22} {:<6} ~{:>4.0} s  {}x{}, {} step(s){}, {}  memory {:.2} GiB  download {:.2} GiB{}",
                              i + 1, m.id, m.kind, p.estimate.total, p.size, p.size, p.steps,
                              if p.guidance > 1.0 { " +CFG" } else { "" }, p.decoder.name(), p.commit_gib, p.download_bytes as f64 / 1073741824.0,
                              if m.tested { "" } else { "  (not tested on rope yet)" }),
            Err(why) if m.supported => println!("  -  {:<18} {:<6} does not fit here: {why}", m.id, m.kind),
            Err(why) => println!("  -  {:<18} {:<6} not runnable in rope yet: {why}", m.id, m.kind),
        }
    }
    println!("any other model file: rope detect <file>, then rope image --model <file>");
    Ok(())
}

/// rope plan --model <file> --explain [--quality q] [--size N] [device overrides incl. --vram-gib X]:
/// profile the file at the planned size, then show where every block runs on this device and why.
fn explain_cmd(args: &[String], models_dir: &str, cal: planner::Cal, q: planner::Quality, dev: &planner::Device, ov: &planner::Overrides) -> Result<(), String> {
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let mp = get("--model").ok_or("--explain needs --model <file>")?;
    let cat_path = get("--catalog").unwrap_or_else(|| "catalog/models.json".into());
    let cat = json::parse(&std::fs::read_to_string(&cat_path).map_err(|e| e.to_string())?)?;
    let (models, _) = load_catalog(&cat_path)?;
    let w = weights::Weights::open(Path::new(&mp))?;
    let (spec, d, _, _) = spec_for_file(&cat, &models, models_dir, &mp, &w, args)?;
    let p = planner::plan_model(&spec, dev, cal, q, ov, &|_: &str| true)?;
    // companions the family needs for a full profile (FLUX: T5, CLIP-L, VAE), when they are on disk
    let comp: Vec<weights::Weights> = p.files.iter().filter(|f| matches!(f.role.as_str(), "t5" | "clip" | "vae"))
        .filter_map(|f| weights::Weights::open(&Path::new(models_dir).join(&f.name)).ok()).collect();
    let g = dev.gpu.as_ref();
    let vram = get("--vram-gib").and_then(|v| v.parse().ok()).or(g.map(|g| g.vram_free_gib)).unwrap_or(0.0);
    let (pl, cpu) = placement_for(models_dir, &mp, &w, &comp, &p, &spec.gpu_components, &spec.gpu_eff, dev, vram)?;
    let e = &p.estimate;
    let taesd = p.decoder == planner::Decoder::Taesd;
    let pd_gflops = g.map_or(0.0, |g| g.gflops);
    let pd_h2d = g.map_or(0.0, |g| g.h2d_gbs);
    println!("{} [{}] {}x{}, {} step(s){} on {:.1} GiB RAM, {} (VRAM {:.1} GiB, GPU ~{:.0} GFLOPS, RAM->VRAM {:.1} GB/s), disk {:.1} GB/s",
             mp, d.family, p.size, p.size, p.steps, if p.guidance > 1.0 { " with CFG" } else { "" }, dev.ram_usable_gib,
             g.map_or("no GPU".to_string(), |g| g.name.clone()), vram, pd_gflops, pd_h2d, dev.disk_read_gbs);
    println!("  CPU speed per component (from the calibrated plan): text {:.0}, denoiser {:.0}, VAE {:.0} GFLOPS", cpu[0], cpu[1], cpu[2]);
    for comp in ["text", "denoiser", "vae"] {
        let mine: Vec<&placement::Placed> = pl.blocks.iter().filter(|b| b.component == comp).collect();
        if mine.is_empty() { continue; }
        let secs: f64 = mine.iter().map(|b| b.seconds).sum();
        println!("  {comp} ({} blocks, {secs:.1} s per image):", mine.len());
        let mut by: std::collections::BTreeMap<&str, (usize, f64, String)> = Default::default();
        for b in &mine { let e = by.entry(b.at.name()).or_insert((0, 0.0, format!("{}: {}", b.name, b.why))); e.0 += 1; e.1 += b.seconds; }
        for (wh, (n, s, why)) in by { println!("    {wh:<14} {n:>3} blocks {s:>8.1} s   e.g. {why}"); }
        if args.iter().any(|a| a == "--blocks") {
            for b in &mine { println!("      {:<55} {:<14} {:>7.2} s  {}", b.name, b.at.name(), b.seconds, b.why); }
        }
    }
    println!("  total ~{:.0} s per image (+{:.1} s one-time upload to VRAM); VRAM used {:.2} of {:.2} GiB; RAM cache {:.2} of {:.2} GiB",
             pl.seconds + if taesd { e.decode } else { 0.0 } + e.upscale, pl.upload_once_s.max(0.0), pl.vram_used_gib, pl.vram_budget_gib, pl.ram_cache_gib, pl.ram_cache_budget_gib);
    if taesd { println!("  decoder: TAESD (tiny, {:.1} s on the CPU, not placed); the full VAE in the file is unused", e.decode); }
    // the stage timeline: one stage at a time is in memory; the run's peak is the largest stage
    println!("  stages (one at a time; predicted peak {:.2} GiB = the largest stage, decoder tile {} px):", p.peak_gib, p.vae_tile * 8);
    for st in &p.stages { println!("    {:<8} {:>7.1} s   {:.2} GiB while it runs   then frees: {}", st.name, st.seconds, st.peak_gib, st.frees); }
    Ok(())
}

/// rope app [--port 4003] [--home DIR] [--no-browser]: the installed app. Starts `rope serve` with the folders of the
/// installation (program folder: app/, catalog/; data folder: models/, gallery/) and opens it in the browser.
/// If Rope already runs on one of the ports, only the browser opens. Data folder: --home, else ROPE_DATA_DIR,
/// else %LOCALAPPDATA%\Rope (Linux: ~/.local/share/rope).
fn app_cmd(args: &[String]) -> Result<(), String> {
    use std::io::{Read, Write};
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let prog = exe.parent().ok_or("no program folder")?.to_path_buf();
    let home = get("--home").map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("ROPE_DATA_DIR").map(Into::into))
        .or_else(|| std::env::var_os("LOCALAPPDATA").map(|d| Path::new(&d).join("Rope")))
        .or_else(|| std::env::var_os("HOME").map(|d| Path::new(&d).join(".local/share/rope")))
        .ok_or("no data folder: pass --home DIR")?;
    let (models, gallery) = (home.join("models"), home.join("gallery"));
    std::fs::create_dir_all(&models).map_err(|e| format!("{}: {e}", models.display()))?;
    // Is Rope (API 1) already answering on this port?
    let is_rope = |port: u16| -> bool {
        let Ok(mut c) = std::net::TcpStream::connect_timeout(&([127, 0, 0, 1], port).into(), Duration::from_millis(300)) else { return false };
        let _ = c.set_read_timeout(Some(Duration::from_secs(2)));
        let _ = write!(c, "GET /api/health HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n");
        let mut s = String::new();
        let _ = c.take(4096).read_to_string(&mut s);
        s.contains("\"api\"")
    };
    let open = |port: u16| {
        let url = format!("http://127.0.0.1:{port}/");
        let r = if cfg!(windows) { Command::new("cmd").args(["/c", "start", "", &url]).status() } else { Command::new("xdg-open").arg(&url).status() };
        if r.is_err() { eprintln!("open {url} in your browser"); }
    };
    let first: u16 = get("--port").and_then(|p| p.parse().ok()).unwrap_or(4003);
    let browser = !args.iter().any(|a| a == "--no-browser");
    let mut port = None;
    for p in first..first.saturating_add(20) {
        if is_rope(p) { println!("Rope already runs at http://127.0.0.1:{p}/"); if browser { open(p); } return Ok(()); }
        if std::net::TcpListener::bind(("127.0.0.1", p)).is_ok() { port = Some(p); break; }
    }
    let port = port.ok_or(format!("no free port between {first} and {}", first.saturating_add(19)))?;
    println!("Rope is running at http://127.0.0.1:{port}/");
    println!("Keep this window open while you use Rope. Close it to stop Rope.");
    println!("Pictures: {}   Models: {}", gallery.display(), models.display());
    if browser {
        std::thread::spawn(move || {
            for _ in 0..100 { if is_rope(port) { open(port); return; } std::thread::sleep(Duration::from_millis(100)); }
        });
    }
    let s = |p: &Path| p.to_string_lossy().into_owned();
    let sa: Vec<String> = vec!["--port".into(), port.to_string(), "--static".into(), s(&prog.join("app")), "--gallery".into(), s(&gallery),
                               "--models-dir".into(), s(&models), "--catalog".into(), s(&prog.join("catalog").join("models.json"))];
    serve::serve(&sa)
}

fn usage() -> ExitCode {
    eprintln!("usage: rope version | test [filter] | inspect <model> | dump <model> <tensor> <n> | probe [--dir D] [--disk-mb N] | tokenize <tokenizer-dir> <text>
\n                           | encode-text <model> <tokenizer-dir> <prompt> <out.f32>\n\
               \x20           | bench <M> <K> <N> <type> | bench-cuda <M> <K> <N> <type> <mode>");
    ExitCode::from(2)
}

pub fn run() -> ExitCode {
    let a: Vec<String> = std::env::args().collect();
    if let Ok(n) = std::env::var("ROPE_THREADS") { ffi::set_threads(n.parse().unwrap_or(0)); }
    let num = |i: usize| -> i64 { a.get(i).and_then(|s| s.parse().ok()).unwrap_or(-1) };
    let code = match a.get(1).map(String::as_str) {
        Some("version") => { println!("rope {VERSION} ({})", if ffi::has_cuda() { "cpu+cuda" } else { "cpu" }); 0 }
        Some("test") => ffi::run_c_tests(a.get(2).map(String::as_str)),
        Some("inspect") if a.len() > 2 => match inspect(&a[2]) { Ok(()) => 0, Err(e) => { eprintln!("rope inspect: {e}"); 1 } },
        Some("dump") if a.len() > 4 => match dump(&a[2], &a[3], num(4).max(0) as usize) { Ok(()) => 0, Err(e) => { eprintln!("rope dump: {e}"); 1 } },
        Some("probe") => {
            let (mut dir, mut mb) = (".".to_string(), 1024);
            let mut i = 2;
            while i + 1 < a.len() {
                match a[i].as_str() { "--dir" => dir = a[i + 1].clone(), "--disk-mb" => mb = a[i + 1].parse().unwrap_or(1024), _ => {} }
                i += 2;
            }
            ffi::probe(&dir, mb)
        }
        Some("bench") if a.len() > 5 => ffi::bench(num(2), num(3), num(4), num(5) as i32),
        #[cfg(feature = "cuda")]
        Some("bench-cuda") if a.len() > 6 => ffi::bench_cuda(num(2), num(3), num(4), num(5) as i32, num(6) as i32),
        Some("flux") if a.len() > 7 => match flux_once(&a[2..8]) { Ok(()) => 0, Err(e) => { eprintln!("rope flux: {e}"); 1 } },
        Some("encode-t5") if a.len() > 5 => match encode_t5(&a[2], &a[3], &a[4], &a[5]) {
            Ok(()) => 0, Err(e) => { eprintln!("rope encode-t5: {e}"); 1 }
        },
        Some("tokenize-t5") if a.len() > 3 => match tokenizer_t5::T5Tokenizer::load(Path::new(&a[2])) {
            Ok(t) => { println!("{}", t.encode_padded(&a[3], 256).iter().map(u32::to_string).collect::<Vec<_>>().join(" ")); 0 }
            Err(e) => { eprintln!("rope tokenize-t5: {e}"); 1 }
        },
        Some("tokenize") if a.len() > 3 => match tokenizer::ClipTokenizer::load(Path::new(&a[2])) {
            Ok(t) => { let ids: Vec<String> = t.encode_77(&a[3]).iter().map(|i| i.to_string()).collect(); println!("{}", ids.join(" ")); 0 }
            Err(e) => { eprintln!("rope tokenize: {e}"); 1 }
        },
        Some("encode-text") if a.len() > 5 => match encode_text(&a[2], &a[3], &a[4], &a[5], a.get(6).map(String::as_str)) {
            Ok(()) => 0, Err(e) => { eprintln!("rope encode-text: {e}"); 1 }
        },
        Some("unet") if a.len() > 6 => match unet_once(&a[2], &a[3], a[4].parse().unwrap_or(999.0), &a[5], &a[6]) {
            Ok(()) => 0, Err(e) => { eprintln!("rope unet: {e}"); 1 }
        },
        Some("taesd") if a.len() > 4 => match taesd_once(&a[2], &a[3], &a[4]) {
            Ok(()) => 0, Err(e) => { eprintln!("rope taesd: {e}"); 1 }
        },
        Some("vae") if a.len() > 4 => match vae_once(&a[2], &a[3], &a[4], a.get(5).and_then(|t| t.parse().ok()).unwrap_or(sd::VAE_TILE.0)) {
            Ok(()) => 0, Err(e) => { eprintln!("rope vae: {e}"); 1 }
        },
        // rope upscale MODEL IN.f32 H W OUT.f32 [TILE]: files in the style of `vae`; image [3,H,W] in [-1,1] -> [3,4H,4W]; TILE 0 = one piece
        Some("upscale") if a.len() > 6 => match upscale_once(&a[2], &a[3], a[4].parse().unwrap_or(0), a[5].parse().unwrap_or(0), &a[6])
            .inspect(|_| { let (c, w) = ffi::peak_memory(); eprintln!("memory: peak commit {:.2} GiB, peak working set {:.2} GiB", c as f64 / 1073741824.0, w as f64 / 1073741824.0); }) {
            Ok(()) => 0, Err(e) => { eprintln!("rope upscale: {e}"); 1 }
        },
        Some("selftest") => match selftest::selftest(&a[2..]) { Ok(c) => c, Err(e) => { eprintln!("rope selftest: {e}"); 2 } },
        Some("image") => match image(&a[2..]).inspect(|_| { let (c, w) = ffi::peak_memory(); eprintln!("memory: peak commit {:.2} GiB, peak working set {:.2} GiB", c as f64 / 1073741824.0, w as f64 / 1073741824.0); }) {
            Ok(()) => 0,
            Err(e) => { eprintln!("rope image: {e}"); if e.starts_with("UNSUPPORTED") { 3 } else if e.starts_with("SUSPICIOUS") { 4 } else { 1 } }
        },
        Some("gflops") => { println!("{:.1} {:.1}", ffi::gemm_gflops(), ffi::gemm_gflops_threads(1)); 0 }
        Some("detect") if a.len() > 2 => detect_cmd(&a[2]),
        Some("scan") if a.len() > 2 => match scan_cmd(&a[2]) { Ok(n) => (n > 0) as i32, Err(e) => { eprintln!("rope scan: {e}"); 2 } },
        Some("verify") => match verify_cmd(&a[2..]) { Ok(()) => 0, Err(e) => { eprintln!("rope verify: {e}"); 1 } },
        Some("get") => match get_cmd(&a[2..]) { Ok(()) => 0, Err(e) => { eprintln!("rope get: {e}"); 1 } },
        Some("quantize") if a.len() > 3 => {
            let ty = match a.iter().position(|x| x == "--type").and_then(|i| a.get(i + 1)).map(String::as_str).unwrap_or("q4_k") {
                "q8_0" => ffi::RT_Q8_0, _ => ffi::RT_Q4_K };
            let arch = weights::Weights::open(Path::new(&a[2])).ok().and_then(|w| detect::detect(&w.mf)).map_or("unknown".into(), |d| d.family);
            match gguf_write::quantize_file(Path::new(&a[2]), Path::new(&a[3]), ty, &arch) {
                Ok((q, f, b)) => { eprintln!("rope quantize: {q} tensors quantized, {f} kept as stored, {:.2} GB -> {}", b as f64 / 1e9, a[3]); 0 }
                Err(e) => { eprintln!("rope quantize: {e}"); 1 }
            }
        }
        // test hook: quantize a raw f32 matrix, write the bytes and rope's own dequantization (<out>.deq)
        Some("quant-test") if a.len() > 6 => {
            let ty: u32 = a[2].parse().unwrap_or(0);
            let (rows, k): (usize, usize) = (a[4].parse().unwrap_or(0), a[5].parse().unwrap_or(0));
            let x = read_f32(&a[3]).unwrap_or_default();
            match ffi::quantize_rows(ty, &x, rows, k) {
                Ok(q) => {
                    let deq: Vec<u8> = q.chunks_exact(ffi::row_bytes(ty, k as u64)).flat_map(|r| ffi::dequant_row(ty, r, k)).flat_map(|v| v.to_le_bytes()).collect();
                    let ok = std::fs::write(&a[6], &q).is_ok() && std::fs::write(format!("{}.deq", a[6]), deq).is_ok();
                    if ok { 0 } else { 1 }
                }
                Err(e) => { eprintln!("rope quant-test: {e}"); 1 }
            }
        }
        Some("profile") if a.len() > 2 => match profile_cmd(&a[2..]) { Ok(()) => 0, Err(e) => { eprintln!("rope profile: {e}"); 1 } },
        Some("plan") => match plan_cmd(&a[2..]) { Ok(()) => 0, Err(e) => { eprintln!("rope plan: {e}"); 1 } },
        Some("suggest") => match suggest_cmd(&a[2..]) { Ok(()) => 0, Err(e) => { eprintln!("rope suggest: {e}"); 1 } },
        Some("serve") => match serve::serve(&a[2..]) { Ok(()) => 0, Err(e) => { eprintln!("rope serve: {e}"); 1 } },
        Some("check-model") => match check_model_cmd(&a[2..]) { Ok(()) => 0, Err(e) => { eprintln!("rope check-model: {e}"); 1 } },
        Some("app") => match app_cmd(&a[2..]) { Ok(()) => 0, Err(e) => { eprintln!("rope app: {e}"); 1 } },
        Some("help") => { usage(); 0 }
        _ => return usage(),
    };
    ExitCode::from(code.clamp(0, 255) as u8)
}
