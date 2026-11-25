//! Planner: given the device (probe) and the model catalog, decide what runs and how, and estimate the time.
//! Time model = per stage: parallel part x (cal GEMM GFLOPS / device GEMM GFLOPS)
//!                      + serial part x (cal 1-thread GFLOPS / device 1-thread GFLOPS)   (Amdahl), x resolution scaling
//!              + paging: UNet weights that do not fit in the RAM left for the page cache are re-read from disk each step.
#![forbid(unsafe_code)]

use crate::json::Value;
use crate::models::FileSpec;

const OS_RESERVE_GIB: f64 = 1.5;   // RAM Windows and other apps keep for themselves on a small laptop (file cache budget)
const PRIVATE_RESERVE_GIB: f64 = 1.0;
/// The memory model was fitted to measurements (errors up to about 7% too low, 12% too high): a size or tile is accepted only
/// when its predicted peak plus this margin fits the budget.
const MEM_MARGIN: f64 = 1.10;   // the part of RAM Windows needs for itself: an image's private memory must fit beside it
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

#[derive(Debug, Clone)]
#[allow(dead_code)] // vram/tensor cores drive GPU placement in P2
pub struct Gpu { pub name: String, pub vram_free_gib: f64, pub tensor_cores: bool, pub gflops: f64, pub h2d_gbs: f64 }

/// GPU speed rule, calibrated on the GTX 1650 (rope FP16 GEMM 1700 GFLOPS = 31% of 14 SM x 256 x 1.515 GHz):
/// FP16-capable SMs do 256 FLOP/clock; Pascal consumer cards (sm_61) and older have slow FP16, so their FP32 rate
/// (128 FLOP/clock) is used. The planner's GPU estimates are only as good as this rule until P2.3.3 measures them.
pub fn gpu_gflops(sm: f64, clock_mhz: f64, _cc: &str) -> f64 {
    // rope's GPU path runs FP32 math (FP16 failed the image check: 29.9 dB). Calibrated on the whole FLUX step on the
    // GTX 1650 (GEMM + attention + light ops, activations on the device): 17,686 GFLOP in 25.7 s = 688 GFLOPS
    // = 25% of 14 SM x 128 FLOP/clock x 1.515 GHz.
    sm * 128.0 * clock_mhz * 1e6 * 0.25 / 1e9
}

#[derive(Debug, Clone)]
pub struct Device {
    pub ram_usable_gib: f64,
    /// RAM free right now (Windows: available, Linux: MemAvailable). Under LOW_FREE_GIB the run is slower: see `pressure`.
    pub ram_free_gib: f64,
    /// Running on battery (the probe's power.on_ac is false). Only a note: the measured effect goes both ways.
    pub on_battery: bool,
    pub cores: u32,
    pub avx2: bool,
    pub gpu: Option<Gpu>,
    pub disk_read_gbs: f64,
    pub disk_free_gib: f64,
    pub gemm_gflops: f64,
    pub gemm_gflops_1t: f64,
}

impl Device {
    pub fn from_probe(j: &Value, gemm_gflops: f64, gemm_gflops_1t: f64) -> Result<Device, String> {
        let num = |v: Option<&Value>, what: &str| v.and_then(Value::as_f64).ok_or(format!("probe: missing {what}"));
        let gpu = match j.get("gpu") {
            Some(g @ Value::Obj(_)) => Some(Gpu {
                name: g.get("name").and_then(Value::as_str).unwrap_or("?").to_string(),
                vram_free_gib: num(g.get("vram_free_gib"), "vram_free_gib")?,
                tensor_cores: g.get("tensor_cores").and_then(Value::as_bool).unwrap_or(false),
                gflops: gpu_gflops(g.get("multiprocessors").and_then(Value::as_f64).unwrap_or(0.0), g.get("clock_mhz").and_then(Value::as_f64).unwrap_or(0.0),
                                   g.get("compute_capability").and_then(Value::as_str).unwrap_or("")),
                h2d_gbs: g.get("h2d_gbs").and_then(Value::as_f64).unwrap_or(0.0),
            }),
            _ => None,
        };
        let disk = j.get("disk").ok_or("probe: missing disk")?;
        let read = num(disk.get("seq_read_gbs"), "seq_read_gbs")?;
        Ok(Device {
            ram_usable_gib: num(j.get("ram_usable_gib"), "ram_usable_gib")?,
            ram_free_gib: j.get("ram_free_gib").and_then(Value::as_f64).unwrap_or(f64::INFINITY),
            on_battery: j.get("power").and_then(|p| p.get("on_ac")).and_then(Value::as_bool) == Some(false),
            cores: num(j.get("cores"), "cores")? as u32,
            avx2: j.get("cpu_flags").and_then(|f| f.get("avx2")).and_then(Value::as_bool).unwrap_or(false),
            gpu,
            disk_read_gbs: if read > 0.0 { read } else { 0.5 },   // not measured -> assume a SATA SSD
            disk_free_gib: num(disk.get("free_gib"), "free_gib")?,
            gemm_gflops,
            gemm_gflops_1t,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quality { Fast, Balanced, Best }

impl Quality {
    pub fn parse(s: &str) -> Option<Quality> { match s { "fast" => Some(Quality::Fast), "balanced" => Some(Quality::Balanced), "best" => Some(Quality::Best), _ => None } }
    fn key(self) -> &'static str { match self { Quality::Fast => "fast", Quality::Balanced => "balanced", Quality::Best => "best" } }
}

/// Peak private (committed) memory of one image, in GiB, from measurements of the family (catalog `memory_gib`):
///   peak = base + the largest stage: text, denoiser at this size, decoder at this size and VAE tile.
/// The stages run one after the other and free their buffers, so the largest one decides. Measured on the 4 GB profile
/// (CPU only): SDXL-Turbo 1.43 / 1.46 / 1.47 GiB at 512 / 768 / 1024, FLUX 1.37 GiB at 512, SD-Turbo 0.69 GiB (TAESD).
#[derive(Debug, Clone, Copy, Default)]
pub struct MemModel { pub base: f64, pub text: f64, pub den_512: f64, pub den_exp: f64, pub vae_tile64: f64, pub taesd_512: f64 }

/// VAE tiles Rope can choose (latent pixels per side), largest first; a smaller tile needs less memory and more time.
/// Activations of one upscaler tile (128 px + 16 px of context on each side), GiB.
pub const UPSCALE_TILE_GIB: f64 = 0.0;   // measured: with the stage's 256 MB cache cap the tile buffers fit inside the base (0.30 GiB at 256 px)
/// Upscaler time per input pixel on the calibration device (measured: 512x512 -> 2048x2048 in 128 s, CPU, all threads).
const UPSCALE_S_PER_PX: f64 = 128.1 / (512.0 * 512.0);

pub const VAE_TILES: [u32; 5] = [64, 48, 32, 24, 16];

impl MemModel {
    fn from_family(fam: &Value) -> MemModel {
        let g = |k: &str| fam.get("memory_gib").and_then(|m| m.get(k)).and_then(Value::as_f64).unwrap_or(0.0);
        MemModel { base: g("base"), text: g("text"), den_512: g("denoiser_at_512"), den_exp: g("denoiser_exp"), vae_tile64: g("vae_tile64"), taesd_512: g("taesd_at_512") }
    }
    /// Denoiser activations at `size` (a power law through the profiled sizes).
    pub fn denoiser(&self, size: u32) -> f64 { self.den_512 * (size as f64 / 512.0).powf(self.den_exp) }
    /// Decoder: the full VAE holds one tile (a 64-latent tile = 512x512 px is `vae_tile64`; memory grows with the tile's area),
    /// an image smaller than the tile is one smaller tile; TAESD holds the whole image.
    pub fn decoder(&self, size: u32, dec: Decoder, tile: u32) -> f64 {
        match dec {
            Decoder::Taesd => self.taesd_512 * (size as f64 / 512.0).powi(2),
            Decoder::Vae => self.vae_tile64 * ((size / 8).min(tile) as f64 / 64.0).powi(2),
        }
    }
    /// Upscaler stage (Real-ESRGAN x4 on tiles): one tile's activations plus the 4x output image and its PNG bytes.
    /// Measured on the stage alone (base included): 0.30 GiB for 256 -> 1024, 0.55 GiB for 512 -> 2048; the growth is the output image, its copies and the PNG (about 5 copies of the f32 output).
    pub fn upscaler(&self, size: u32) -> f64 { UPSCALE_TILE_GIB + 5.0 * 3.0 * 4.0 * (4.0 * size as f64).powi(2) / GIB }
    /// Predicted peak private memory of a whole image.
    pub fn peak(&self, size: u32, dec: Decoder, tile: u32) -> f64 {
        self.base + self.text.max(self.denoiser(size)).max(self.decoder(size, dec, tile))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decoder { Taesd, Vae }

impl Decoder { pub fn name(self) -> &'static str { match self { Decoder::Taesd => "taesd", Decoder::Vae => "vae" } } }

#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub id: String, pub kind: String, pub title: String, pub supported: bool, pub why_not: String,
    pub native: u32, pub max: u32, steps: [u32; 3],
    calib: u32,              // size the family's stage timings were measured at (the family's native size)
    par: [f64; 5],           // open, text, unet_step, taesd, vae: parallel seconds at native size (calibration device)
    ser: [f64; 5],           // same order: serial seconds
    att_share: f64,
    commit: [f64; 2],        // taesd, vae (GiB)
    pub mem: MemModel,       // private memory as a function of image size and VAE tile (P2.3.4)
    unet_bytes: f64,
    pub guidance: f64,       // classifier-free guidance scale; > 1 means two UNet calls per step
    pub gpu_components: Vec<String>,   // components whose code runs on the op interface (GPU-capable)
    pub local_quantize: bool,          // the family's code can run a locally quantized GGUF copy
    pub gpu_eff: Vec<(String, f64)>,   // measured GPU efficiency per component (relative to the device GPU rule)
    pub files: Vec<FileSpec>,
    pub tested: bool,        // false: added from its header and metadata only (same architecture as a tested model), never run on rope
    pub license: String,     // the model's licence as its publisher states it ("" = not given in the catalog)
}

fn file_specs(v: Option<&Value>) -> Vec<FileSpec> {
    v.and_then(Value::as_arr).unwrap_or(&[]).iter().map(|x| {
        let t = |k: &str| x.get(k).and_then(Value::as_str).unwrap_or("").to_string();
        FileSpec { role: t("role"), name: t("name"), repo: t("repo"), path: t("path"), sha256: t("sha256"),
                   bytes: x.get("bytes").and_then(Value::as_f64).unwrap_or(0.0) as u64 }
    }).collect()
}

impl ModelSpec {
    /// A runnable spec = family profile (how to run the architecture) + identity (id, title, tuned steps, files).
    /// `unet_bytes`: measured from the actual file when known (FP16 and FP32 files differ), else the family reference.
    pub fn from_family(fam_id: &str, fam: &Value, id: &str, title: &str, steps: Option<&Value>, mut files: Vec<FileSpec>,
                       unet_bytes: Option<f64>) -> ModelSpec {
        let f = |o: Option<&Value>, k: &str| o.and_then(|v| v.get(k)).and_then(Value::as_f64).unwrap_or(0.0);
        let five = |o: Option<&Value>| ["open", "text", "unet_step", "taesd", "vae"].map(|k| f(o, k));
        let st = steps.or(fam.get("steps"));
        files.extend(file_specs(fam.get("companions")));
        let sup = crate::detect::support(fam_id);
        ModelSpec {
            id: id.into(), title: title.into(), kind: fam.get("kind").and_then(Value::as_str).unwrap_or("image").into(),
            supported: sup.is_ok(), why_not: sup.err().unwrap_or("").into(),
            native: f(Some(fam), "native_size") as u32, calib: f(Some(fam), "native_size") as u32, max: f(Some(fam), "max_size").max(f(Some(fam), "native_size")) as u32,
            steps: [f(st, "fast") as u32, f(st, "balanced") as u32, f(st, "best") as u32],
            par: five(fam.get("stage_parallel_s")), ser: five(fam.get("stage_serial_s")),
            att_share: f(Some(fam), "unet_attention_share"),
            commit: [f(fam.get("commit_gib"), "taesd"), f(fam.get("commit_gib"), "vae")],
            mem: MemModel::from_family(fam),
            unet_bytes: unet_bytes.unwrap_or_else(|| f(Some(fam), "unet_bytes_reference")),
            guidance: f(Some(fam), "guidance").max(1.0),
            gpu_components: fam.get("gpu_components").and_then(Value::as_arr).unwrap_or(&[]).iter()
                .filter_map(|v| v.as_str().map(str::to_string)).collect(),
            local_quantize: fam.get("local_quantize").and_then(Value::as_bool).unwrap_or(false),
            gpu_eff: fam.get("gpu_efficiency").and_then(Value::as_obj).map(|o| o.iter().filter_map(|(k, v)| v.as_f64().map(|x| (k.clone(), x))).collect()).unwrap_or_default(),
            files, tested: true, license: String::new(),
        }
    }

    /// Steps this spec plans for a quality.
    pub fn steps_for(&self, q: Quality) -> u32 { self.steps[q as usize].max(1) }
    /// Private memory of the full-VAE plan (the larger of the two decoders).
    pub fn commit_for_vae(&self) -> f64 { self.commit[1] }
    /// The denoiser's size changed (the file was replaced by a local quantized copy).
    pub fn unet_bytes_set(&mut self, b: f64) { self.unet_bytes = b; }
    /// The family's model code can run a locally quantized (GGUF) copy.
    pub fn can_quantize(&self) -> bool { self.local_quantize }

    /// A catalog model whose family has no run profile yet: listed, never planned.
    fn unsupported(id: &str, title: &str, fam_id: &str, kind: &str) -> ModelSpec {
        ModelSpec { id: id.into(), title: title.into(), kind: kind.into(), supported: false,
                    why_not: crate::detect::support(fam_id).err().unwrap_or("no run profile").into(),
                    native: 0, calib: 0, max: 0, steps: [1; 3], par: [0.0; 5], ser: [0.0; 5], att_share: 0.0, commit: [0.0; 2], mem: MemModel::default(), unet_bytes: 0.0, guidance: 1.0, gpu_components: vec![], local_quantize: false, gpu_eff: vec![], files: vec![], tested: true, license: String::new() }
    }
}

/// All catalog models as specs (catalog v2: `families` + `models` pointing to a family).
pub fn catalog_specs(cat: &Value) -> Result<Vec<ModelSpec>, String> {
    let fams = cat.get("families");
    cat.get("models").and_then(Value::as_arr).ok_or("catalog has no models")?.iter().map(|m| {
        let s = |k: &str| m.get(k).and_then(Value::as_str).unwrap_or("").to_string();
        let fam_id = s("family");
        Ok(match fams.and_then(|f| f.get(&fam_id)) {
            Some(fam) => {
                let mut spec = ModelSpec::from_family(&fam_id, fam, &s("id"), &s("title"), m.get("steps"), file_specs(m.get("files")), None);
                // a distilled model of the family can run at another size or without guidance (SDXL-Turbo: 512, no CFG)
                if let Some(n) = m.get("native_size").and_then(Value::as_f64) { spec.native = n as u32; spec.max = spec.max.max(spec.native); }
                if let Some(g) = m.get("guidance").and_then(Value::as_f64) { spec.guidance = g.max(1.0); }
                // the exact denoiser size of this file (read from its header): sizes the paging estimate
                if let Some(b) = m.get("unet_bytes").and_then(Value::as_f64) { spec.unet_bytes = b; }
                spec.tested = m.get("tested").and_then(Value::as_bool).unwrap_or(true);
                spec.license = s("license");
                spec
            }
            None => ModelSpec::unsupported(&s("id"), &s("title"), &fam_id, &s("kind")),
        })
    }).collect()
}

/// Spec for a file the user brings: family profile + the file's own UNet size. Err if the family cannot run yet.
pub fn family_spec(cat: &Value, fam_id: &str, name: &str, unet_bytes: f64) -> Result<ModelSpec, String> {
    crate::detect::support(fam_id).map_err(|why| format!("UNSUPPORTED: architecture {fam_id} is not supported yet ({why})"))?;
    let fam = cat.get("families").and_then(|f| f.get(fam_id)).ok_or_else(|| format!("UNSUPPORTED: no run profile for {fam_id}"))?;
    let title = fam.get("title").and_then(Value::as_str).unwrap_or(fam_id);
    Ok(ModelSpec::from_family(fam_id, fam, name, &format!("{title} (your file)"), None, vec![], Some(unet_bytes)))
}

/// Calibration device speeds (all threads, one thread).
#[derive(Debug, Clone, Copy)]
pub struct Cal { pub gflops: f64, pub gflops_1t: f64 }

#[derive(Debug, Clone)]
pub struct Estimate { pub open: f64, pub text: f64, pub unet_step: f64, pub paging_step: f64, pub steps: u32, pub decode: f64, pub upscale: f64, pub total: f64 }

/// Free RAM below which a run is slower than its plan, and by how much (P2.5.1, real laptops, SD-Turbo 512 on the CPU).
/// Laptops with under 2 GiB free when the run started took longer than planned on every run: 1.14x, 1.15x, 1.18x, 1.68x
/// (median 1.17x; the OS frees memory, trims and compresses pages while the image runs). Laptops with 2.5 GiB or more
/// free took 0.82x to 0.96x of the plan. The factor applies to the CPU work only.
pub const LOW_FREE_GIB: f64 = 2.0;
pub const LOW_FREE_FACTOR: f64 = 1.2;
pub fn pressure(dev: &Device) -> f64 { if dev.ram_free_gib < LOW_FREE_GIB { LOW_FREE_FACTOR } else { 1.0 } }

/// Time estimate for one image with this model on this device.
pub fn estimate(m: &ModelSpec, dev: &Device, cal: Cal, size: u32, steps: u32, dec: Decoder, tile: u32) -> Estimate {
    let ratio = |c: f64, d: f64| if d > 0.0 { c / d } else { 1.0 };
    let (sp, ss) = (ratio(cal.gflops, dev.gemm_gflops), ratio(cal.gflops_1t, dev.gemm_gflops_1t));
    let f = pressure(dev);
    let t = |i: usize| (m.par[i] * sp + m.ser[i] * ss) * f;
    let r = (size as f64 / m.calib as f64).powi(2);
    let unet_r = (1.0 - m.att_share) * r + m.att_share * r * r;
    let commit = m.commit[(dec == Decoder::Vae) as usize];
    let cache_bytes = ((dev.ram_usable_gib - commit - OS_RESERVE_GIB).max(0.0)) * GIB;
    let miss = (m.unet_bytes - cache_bytes).max(0.0);
    let calls = if m.guidance > 1.0 { 2.0 } else { 1.0 };     // CFG: conditional + unconditional UNet call
    let e = Estimate {
        open: t(0),
        text: t(1) * calls,
        unet_step: t(2) * unet_r * calls,
        paging_step: calls * miss / (dev.disk_read_gbs * 1e9),
        steps,
        decode: if dec == Decoder::Vae && size as usize / 8 > tile as usize {
            // tiled VAE: every tile costs like a 512x512 decode (a 64-latent tile) scaled by the tile's area
            let n = crate::sd::tile_starts(size as usize / 8, tile as usize, crate::sd::vae_overlap(tile as usize)).len() as f64;
            n * n * t(4) * (512.0 / m.calib as f64).powi(2) * (tile as f64 / 64.0).powi(2)
        } else { t(if dec == Decoder::Vae { 4 } else { 3 }) * r },
        upscale: 0.0,
        total: 0.0,
    };
    // the next block's weights are read on a background thread while this block computes: reads and compute overlap
    // (measured, FLUX 512 on the 4 GB profile: BF16 streamed 315 s, Q4_K 305 s; the sum model said 468 / 360 s)
    Estimate { total: e.open + e.text + steps as f64 * e.unet_step.max(e.paging_step) + e.decode, ..e }
}

/// One stage of the run: what it costs while it runs and what it gives back when it ends (P2.3.5).
#[derive(Debug, Clone)]
pub struct Stage { pub name: &'static str, pub seconds: f64, pub peak_gib: f64, pub frees: &'static str }

#[derive(Debug, Clone)]
pub struct Plan {
    pub model: String, pub title: String, pub size: u32, pub steps: u32, pub decoder: Decoder, pub guidance: f64,
    pub placement: &'static str, pub threads: u32, pub commit_gib: f64, pub download_bytes: u64, pub files: Vec<FileSpec>,
    pub estimate: Estimate, pub notes: Vec<String>,
    pub vae_tile: u32,       // latent pixels per side of one VAE tile (64 unless the memory budget asks for less)
    pub peak_gib: f64,       // predicted peak private memory of the whole image (MemModel): the largest stage + base
    pub stages: Vec<Stage>,  // the timeline: text, denoise, decode, optional upscale; one stage at a time is in memory
}

#[derive(Debug, Clone, Default)]
pub struct Overrides { pub size: Option<u32>, pub steps: Option<u32>, pub decoder: Option<Decoder>, pub vae_tile: Option<u32>, pub upscale: bool }

const BALANCED_BUDGET_S: f64 = 60.0;

/// Choose model, steps, size, decoder and placement for a request. `have(name)` says if a file is already local.
pub fn plan(models: &[ModelSpec], dev: &Device, cal: Cal, kind: &str, q: Quality, ov: &Overrides,
            have: &dyn Fn(&str) -> bool) -> Result<Plan, String> {
    let m = models.iter().find(|m| m.kind == kind && m.supported).ok_or_else(|| {
        let why: Vec<String> = models.iter().filter(|m| m.kind == kind).map(|m| format!("{}: {}", m.id, m.why_not)).collect();
        format!("no {kind} model runs in rope yet ({})", why.join("; "))
    })?;
    plan_model(m, dev, cal, q, ov, have)
}

/// Plan one given model (catalog choice or a file the user brought) for this device.
pub fn plan_model(m: &ModelSpec, dev: &Device, cal: Cal, q: Quality, ov: &Overrides,
                  have: &dyn Fn(&str) -> bool) -> Result<Plan, String> {
    let mut notes = Vec::new();
    let steps = ov.steps.unwrap_or(m.steps[q as usize]).max(1);
    let size = ov.size.unwrap_or(m.native);
    if size % 64 != 0 || size > m.max { return Err(format!("size {size} not supported by {} (multiple of 64, max {})", m.id, m.max)); }
    let budget = dev.ram_usable_gib - PRIVATE_RESERVE_GIB;            // private memory this image may use
    // the largest VAE tile whose predicted peak fits the budget (or the one the user forced); None = nothing fits
    let tile_for = |d: Decoder| -> Option<u32> {
        if d == Decoder::Taesd { return (m.mem.peak(size, d, 64) * MEM_MARGIN <= budget).then_some(64); }
        match ov.vae_tile { Some(t) => (m.mem.peak(size, d, t) * MEM_MARGIN <= budget).then_some(t), None => VAE_TILES.iter().copied().find(|&t| m.mem.peak(size, d, t) * MEM_MARGIN <= budget) }
    };
    let has_taesd = m.files.iter().any(|f| f.role == "taesd");
    let mut dec = ov.decoder.unwrap_or(if !has_taesd { Decoder::Vae } else { match q {
        Quality::Fast => Decoder::Taesd,
        Quality::Best => Decoder::Vae,
        Quality::Balanced => {
            let t = estimate(m, dev, cal, size, steps, Decoder::Vae, tile_for(Decoder::Vae).unwrap_or(64)).total;
            if t <= BALANCED_BUDGET_S { Decoder::Vae } else {
                notes.push(format!("full VAE would take ~{t:.0} s here (balanced budget {BALANCED_BUDGET_S:.0} s); using the tiny decoder (TAESD)"));
                Decoder::Taesd
            }
        }
    } });
    if tile_for(dec).is_none() && dec == Decoder::Vae && ov.decoder.is_none() && has_taesd {
        notes.push("not enough free RAM for the full VAE, even in the smallest tiles; using TAESD".into());
        dec = Decoder::Taesd;
    }
    let Some(tile) = tile_for(dec) else {
        // say which stage does not fit, and by how much
        let (what, need) = if m.mem.denoiser(size) >= m.mem.decoder(size, dec, *VAE_TILES.last().unwrap()) { ("the denoiser's activations", m.mem.denoiser(size)) }
                           else { ("the image decoder", m.mem.decoder(size, dec, *VAE_TILES.last().unwrap())) };
        return Err(format!("{size}x{size} needs about {:.2} GiB of private memory ({what} {need:.2} GiB + {:.2} GiB base, plus a {:.0}% safety margin) and 1 GiB is kept for Windows; this device has {:.2} GiB usable",
                           (m.mem.base + need) * MEM_MARGIN, m.mem.base, (MEM_MARGIN - 1.0) * 100.0, dev.ram_usable_gib));
    };
    if dec == Decoder::Vae && tile < 64 && (size / 8) > tile {
        // image difference to a full-tile decode, measured (SDXL, same latent): 48 -> 42.5 dB, 32 -> 37.5, 24 -> 34.9, 16 -> 31.1
        let psnr = match tile { 48.. => 42, 32.. => 37, 24.. => 35, _ => 31 };
        notes.push(format!("memory is tight: the image decoder works in {}x{} px tiles (less memory, more time; the picture differs slightly from a full-tile decode, PSNR about {psnr} dB)", tile * 8, tile * 8));
    }
    let files: Vec<FileSpec> = m.files.iter().filter(|f| f.role != "taesd" || dec == Decoder::Taesd).cloned().collect();
    let download: u64 = files.iter().filter(|f| !have(&f.name)).map(|f| f.bytes).sum();
    if download as f64 / GIB + 1.0 > dev.disk_free_gib { return Err(format!("needs {:.1} GiB of downloads, only {:.1} GiB free", download as f64 / GIB, dev.disk_free_gib)); }
    if let (Some(g), false) = (&dev.gpu, crate::ffi::has_cuda()) { notes.push(format!("GPU {} detected, but this rope build has no GPU support: the plan runs on the CPU", g.name)); }
    if !dev.avx2 { notes.push("CPU has no AVX2: the kernels run on the SSE4.1 path (~3x slower than AVX2 on the same CPU; the measured speed above already is this path's)".into()); }
    let mut est = estimate(m, dev, cal, size, steps, dec, tile);
    if ov.upscale {
        if (m.mem.base + m.mem.upscaler(size)) * MEM_MARGIN > budget {
            return Err(format!("the 4x upscaler needs about {:.2} GiB of private memory after a {size}x{size} image; this device has {:.2} GiB usable (1 GiB is kept for Windows)", (m.mem.base + m.mem.upscaler(size)) * MEM_MARGIN, dev.ram_usable_gib));
        }
        est.upscale = UPSCALE_S_PER_PX * (size as f64).powi(2) * cal.gflops / dev.gemm_gflops.max(1e-9) * pressure(dev);
        est.total += est.upscale;
        notes.push(format!("4x upscale after the decode: {size}x{size} -> {}x{} (+{:.0} s)", size * 4, size * 4, est.upscale));
    }
    let mut stages = vec![
        Stage { name: "text", seconds: est.open + est.text, peak_gib: m.mem.base + m.mem.text, frees: "the text encoder (its weights are file pages: they leave memory under pressure)" },
        Stage { name: "denoise", seconds: steps as f64 * est.unet_step.max(est.paging_step), peak_gib: m.mem.base + m.mem.denoiser(size), frees: "the denoiser's activations; its weights stay file pages" },
        Stage { name: "decode", seconds: est.decode, peak_gib: m.mem.base + m.mem.decoder(size, dec, tile), frees: "the decoder and the latent" },
    ];
    if ov.upscale { stages.push(Stage { name: "upscale", seconds: est.upscale, peak_gib: m.mem.base + m.mem.upscaler(size), frees: "the upscaler tile buffers" }); }
    if pressure(dev) > 1.0 {
        notes.push(format!("only {:.1} GiB RAM free now: the system frees memory while the image runs, so the estimate is {:.0}% longer (measured 1.1x to 1.7x on laptops with under {LOW_FREE_GIB:.0} GiB free); close other programs for the normal speed",
                           dev.ram_free_gib, (LOW_FREE_FACTOR - 1.0) * 100.0));
    }
    if dev.on_battery { notes.push("on battery: the speed can differ a lot from this estimate (measured 0.7x to 1.7x on battery); connect the charger for a dependable time".into()); }
    if est.paging_step > 0.5 {
        let hidden = if est.paging_step <= est.unet_step { "hidden behind the compute" } else { "not fully hidden by the compute" };
        notes.push(format!("weights do not fit in RAM: ~{:.1} s per step re-reading them from disk ({hidden})", est.paging_step));
    }
    Ok(Plan { model: m.id.clone(), title: m.title.clone(), size, steps, decoder: dec, guidance: m.guidance, placement: "cpu", threads: dev.cores,
              commit_gib: m.commit[(dec == Decoder::Vae) as usize], download_bytes: download, files, estimate: est, notes,
              vae_tile: tile, peak_gib: stages.iter().map(|s| s.peak_gib).fold(0.0, f64::max).max(m.mem.peak(size, dec, tile)), stages })
}

/// The largest size (a multiple of 64, up to the model's own maximum) this device can run, with its plan; None when not even
/// the smallest size fits. `ov.size` is ignored. This is the "ceiling" the app and `rope plan --max` report.
pub fn max_plan(m: &ModelSpec, dev: &Device, cal: Cal, q: Quality, ov: &Overrides, have: &dyn Fn(&str) -> bool) -> Option<Plan> {
    (1..=m.max / 64).rev().find_map(|k| plan_model(m, dev, cal, q, &Overrides { size: Some(k * 64), ..ov.clone() }, have).ok())
}

pub fn quality_key(q: Quality) -> &'static str { q.key() }

/// What to do with a model file before running it.
#[derive(Debug, Clone, PartialEq)]
pub enum Source { AsIs, Stream(String), Quantize { ty: u32, out_gib: f64, why: String } }

/// Weights converted per second by `rope quantize` (read + quantize + write), measured: FLUX BF16 11.9 G weights in 150 s.
pub const QUANT_WPS: f64 = 8.0e7;

/// What the source rule needs to know about one model file on one device.
#[derive(Debug, Clone, Copy)]
pub struct SourceIn {
    pub denoiser_gib: f64,   // denoiser as stored
    pub rest_gib: f64,       // the rest of the file (text encoders, VAE): a copy keeps it as stored
    pub stored_bits: f64,    // bits per denoiser weight as stored (16 BF16/F16, 32 F32; 0 = already quantized)
    pub commit_gib: f64,     // the plan's private memory
    pub calls: f64,          // denoiser calls per image
    pub compute_s: f64,      // denoiser compute per call on this device (estimate)
    pub image_s: f64,        // whole image without disk reads (estimate)
}

/// One rule for any full-precision model whose denoiser cannot stay in RAM (every call re-reads it from disk; the
/// reads overlap with the compute): use the most precise source whose unhidden reads cost at most 10% of the image
/// time — the file itself streamed at full precision, then a Q8_0 copy, then a Q4_K copy (made once after the outlier
/// scan, cached). If none does, the fastest one. A copy needs disk room for all of it (the rest of the file is kept as
/// stored); `force`: the copy type the user asked for. Measured (FLUX 512, 4 GB profile): BF16 streamed 315 s, Q8_0
/// 329 s (SSIM 0.93 vs BF16), Q4_K 305 s (SSIM 0.62; the published Q4_K GGUF 0.54).
pub fn source_strategy(m: SourceIn, dev: &Device, keep_precision: bool, can_quantize: bool, force: Option<u32>) -> Source {
    let SourceIn { denoiser_gib, rest_gib, stored_bits, commit_gib, calls, compute_s, image_s } = m;
    if stored_bits == 0.0 { return Source::AsIs; }
    let cache = (dev.ram_usable_gib - OS_RESERVE_GIB - commit_gib).max(0.0);
    if denoiser_gib <= cache { return Source::AsIs; }
    let read = |gib: f64| (gib - cache).max(0.0) * GIB / (dev.disk_read_gbs * 1e9);       // per call
    let extra = |gib: f64| (read(gib) - compute_s).max(0.0) * calls;                       // per image, not hidden
    let stream = |why: String| Source::Stream(format!("{why}: streaming the {denoiser_gib:.1} GiB file at full precision (~{:.0} s of reads per call, ~{:.0} s per image not hidden by the compute)", read(denoiser_gib), extra(denoiser_gib)));
    if !can_quantize { return stream("this family cannot run a locally quantized copy yet".into()); }
    if keep_precision { return stream("--keep-precision".into()); }
    // candidates, most precise first: (copy type, denoiser GiB); None = the file itself
    let mut cands: Vec<(Option<u32>, f64)> = vec![];
    if force.is_none() { cands.push((None, denoiser_gib)); }
    for (ty, bits) in [(crate::ffi::RT_Q8_0, 8.5), (crate::ffi::RT_Q4_K, 4.5)] {
        let g = denoiser_gib * bits / stored_bits;
        if force.map_or(true, |f| f == ty) && rest_gib + g + 1.0 <= dev.disk_free_gib { cands.push((Some(ty), g)); }
    }
    if cands.is_empty() { return stream("not enough disk for a quantized copy".into()); }
    let budget = 0.1 * image_s;
    let pick = cands.iter().find(|c| extra(c.1) <= budget)
        .or_else(|| cands.iter().min_by(|a, b| extra(a.1).total_cmp(&extra(b.1)))).copied().unwrap();
    match pick {
        (None, g) if extra(g) <= budget => stream(format!("the denoiser does not fit in RAM ({cache:.1} GiB cache), but its unhidden reads (~{:.0} s) are within 10% of the ~{image_s:.0} s image", extra(g))),
        (None, _) => stream("not enough disk for a quantized copy".into()),
        (Some(ty), g) => {
            // ponytail: one-time cost = weights / QUANT_WPS, measured on the dev laptop; it does not scale with the CPU
            // yet — add a per-core rate when real laptops (P2.5.1) differ.
            let once = denoiser_gib * GIB * 8.0 / stored_bits / QUANT_WPS;
            let out_gib = rest_gib + g;
            Source::Quantize { ty, out_gib, why: format!(
                "the {denoiser_gib:.1} GiB full-precision denoiser does not fit in RAM ({cache:.1} GiB cache): streaming it adds ~{:.0} s to a ~{image_s:.0} s image; a {} copy ({out_gib:.1} GiB on disk, made once in ~{once:.0} s) adds ~{:.0} s",
                extra(denoiser_gib), if ty == crate::ffi::RT_Q8_0 { "Q8_0" } else { "Q4_K" }, extra(g)) }
        }
    }
}

/// Buffer-reuse cache for this device (GiB): half of the RAM left after Windows and the model's own private memory,
/// between 0.125 and 1. Measured on the 4 GB profile: FLUX 274 s with a 768 MB cache vs 338 s with 256 MB.
pub fn buffer_cache_gib(dev: &Device, commit_gib: f64) -> f64 { ((dev.ram_usable_gib - OS_RESERVE_GIB - commit_gib) * 0.5).clamp(0.125, 1.0) }

/// One catalog model judged for this device: a plan if it runs here, else the reason it does not.
pub struct Suggestion<'m> { pub model: &'m ModelSpec, pub plan: Result<Plan, String> }

/// Recommend catalog models for this device. Order: models that fit (fastest first), then supported models that
/// do not fit this device, then models rope cannot run yet. `kind` filters (image / video / text); None = all.
pub fn suggest<'m>(models: &'m [ModelSpec], dev: &Device, cal: Cal, kind: Option<&str>, q: Quality, ov: &Overrides,
                   have: &dyn Fn(&str) -> bool) -> Vec<Suggestion<'m>> {
    let mut out: Vec<Suggestion> = models.iter().filter(|m| kind.map_or(true, |k| m.kind == k)).map(|m| Suggestion {
        model: m,
        plan: if m.supported { plan_model(m, dev, cal, q, ov, have) } else { Err(m.why_not.clone()) },
    }).collect();
    let rank = |s: &Suggestion| match &s.plan { Ok(p) => (0, p.estimate.total), Err(_) => (if s.model.supported { 1 } else { 2 }, 0.0) };
    out.sort_by(|a, b| { let (x, y) = (rank(a), rank(b)); x.0.cmp(&y.0).then(x.1.total_cmp(&y.1)) });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json;

    #[test]
    fn low_free_ram_makes_the_cpu_work_slower() {
        let (models, cal) = catalog();
        let m = models.iter().find(|m| m.id == "sd-turbo").unwrap();
        let mut d = dev(7.7, 6, 1.0, 2.0, None, cal);
        d.ram_free_gib = 3.0;
        let a = estimate(m, &d, cal, 512, 1, Decoder::Taesd, 64);
        d.ram_free_gib = 1.2;
        let b = estimate(m, &d, cal, 512, 1, Decoder::Taesd, 64);
        assert!((b.total / a.total - LOW_FREE_FACTOR).abs() < 1e-9, "{} vs {}", a.total, b.total);
        let p = plan_model(m, &d, cal, Quality::Fast, &Overrides::default(), &|_| true).unwrap();
        assert!(p.notes.iter().any(|n| n.contains("RAM free now")));
    }

    fn catalog() -> (Vec<ModelSpec>, Cal) {
        let txt = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../catalog/models.json")).unwrap();
        let j = json::parse(&txt).unwrap();
        let c = j.get("calibration").unwrap();
        let cal = Cal { gflops: c.get("gemm_gflops").unwrap().as_f64().unwrap(), gflops_1t: c.get("gemm_gflops_1t").unwrap().as_f64().unwrap() };
        (catalog_specs(&j).unwrap(), cal)
    }
    /// speed: device speed relative to the calibration laptop (all-thread and 1-thread scaled together here)
    fn dev(ram: f64, cores: u32, speed: f64, disk: f64, gpu: Option<(&str, f64)>, cal: Cal) -> Device {
        Device { ram_usable_gib: ram, ram_free_gib: ram, on_battery: false, cores, avx2: true, disk_read_gbs: disk, disk_free_gib: 50.0,
                 gemm_gflops: cal.gflops * speed, gemm_gflops_1t: cal.gflops_1t * speed,
                 gpu: gpu.map(|(n, v)| Gpu { name: n.into(), vram_free_gib: v, tensor_cores: false, gflops: 1700.0, h2d_gbs: 6.67 }) }
    }

    #[test]
    fn four_profiles() {
        let (m, cal) = catalog();
        let all = |_: &str| true;
        let none = Overrides::default();
        // 4 GB laptop, 2 old cores, SATA SSD, no GPU: full VAE too slow -> TAESD; paging cost is predicted
        let p = plan(&m, &dev(3.4, 2, 1.0 / 6.0, 0.45, None, cal), cal, "image", Quality::Balanced, &none, &all).unwrap();
        assert_eq!((p.model.as_str(), p.steps, p.decoder), ("sd-turbo", 1, Decoder::Taesd));
        assert!(p.estimate.paging_step > 0.0 && p.commit_gib + 1.0 <= 3.4);
        // 8 GB laptop, 6 cores like the dev machine, no GPU: full VAE fits the 60 s budget
        let p = plan(&m, &dev(7.35, 6, 1.0, 3.5, None, cal), cal, "image", Quality::Balanced, &none, &all).unwrap();
        assert_eq!((p.steps, p.decoder, p.placement), (1, Decoder::Vae, "cpu"));
        assert!(p.estimate.total < 60.0);
        // 8 GB + GTX 1650: same plan on CPU, with a note about the GPU
        let p = plan(&m, &dev(7.35, 6, 1.0, 3.5, Some(("GTX 1650", 3.2)), cal), cal, "image", Quality::Balanced, &none, &all).unwrap();
        assert_eq!(p.decoder, Decoder::Vae);
        assert!(p.notes.iter().any(|n| n.contains("GTX 1650")));
        // 16 GB + 8 GB GPU, faster CPU: best = 4 steps + VAE, fast = 1 step + TAESD
        let d = dev(15.7, 8, 1.6, 5.0, Some(("RTX 3060", 7.5)), cal);
        let p = plan(&m, &d, cal, "image", Quality::Best, &none, &all).unwrap();
        assert_eq!((p.steps, p.decoder), (4, Decoder::Vae));
        let p = plan(&m, &d, cal, "image", Quality::Fast, &none, &all).unwrap();
        assert_eq!((p.steps, p.decoder), (1, Decoder::Taesd));
        // no plan exceeds the RAM budget, and impossible requests are refused with a reason
        assert!(plan(&m, &dev(1.2, 2, 1.0 / 6.0, 0.45, None, cal), cal, "image", Quality::Balanced, &none, &all).is_err());
        assert!(plan(&m, &dev(7.35, 6, 1.0, 3.5, None, cal), cal, "video", Quality::Balanced, &none, &all).unwrap_err().contains("Phase 3"));
        assert!(plan(&m, &dev(7.35, 6, 1.0, 3.5, None, cal), cal, "image", Quality::Balanced, &Overrides { size: Some(1024), ..none.clone() }, &all).is_err());
    }

    /// Peak private memory against what was measured on the 4 GB profile (CPU only; data in bench/rope/memory_4gb_cpu_2026-11-19.jsonl):
    /// SDXL-Turbo 1.43 / 1.46 / 1.47 GiB at 512 / 768 / 1024, FLUX 1.37 / 1.38 / 1.39, SD-Turbo with TAESD 0.69; SDXL at 512 with
    /// VAE tiles 48 / 32 / 24 / 16: 1.01 / 0.60 / 0.43 / 0.34; denoiser-bound (tile 16) SDXL 960 0.65, FLUX 576 0.69. Every one within 15%.
    #[test]
    fn memory_model_matches_measurements() {
        let (m, cal) = catalog();
        let d = dev(3.4, 6, 1.0, 3.5, None, cal);
        let spec = |id: &str| m.iter().find(|x| x.id == id).unwrap().clone();
        let peak = |id: &str, size: u32, tile: u32, q: Quality| {
            plan_model(&spec(id), &d, cal, q, &Overrides { size: Some(size), vae_tile: Some(tile), ..Overrides::default() }, &|_| true).unwrap().peak_gib
        };
        for (id, size, tile, measured) in [
            ("sdxl-turbo", 512, 64, 1.43), ("sdxl-turbo", 768, 64, 1.46), ("sdxl-turbo", 1024, 64, 1.47),
            ("flux-schnell-q4k", 512, 64, 1.37), ("flux-schnell-q4k", 768, 64, 1.38), ("flux-schnell-q4k", 1024, 64, 1.39),
            ("sdxl-turbo", 512, 48, 1.01), ("sdxl-turbo", 512, 32, 0.60), ("sdxl-turbo", 512, 24, 0.43), ("sdxl-turbo", 512, 16, 0.34),
            ("sdxl-turbo", 960, 16, 0.65), ("flux-schnell-q4k", 576, 16, 0.69),
        ] {
            let got = peak(id, size, tile, Quality::Best);
            assert!((got - measured).abs() / measured <= 0.15, "{id} {size} tile {tile}: predicted {got:.2}, measured {measured:.2}");
        }
        // SD-Turbo with the tiny decoder (quality fast): measured 0.69
        let got = peak("sd-turbo", 512, 64, Quality::Fast);
        assert!((got - 0.69).abs() / 0.69 <= 0.15, "sd-turbo taesd: {got:.2}");
    }

    /// A smaller laptop gets a smaller VAE tile (more time, less memory) and a ceiling below the model's maximum;
    /// the next size up is refused with the reason; a laptop with enough RAM is unchanged.
    #[test]
    fn tight_memory_shrinks_the_tile_then_the_size() {
        let (m, cal) = catalog();
        let spec = m.iter().find(|x| x.id == "sdxl-turbo").unwrap();
        let q = Quality::Best;
        let ov = Overrides::default();
        let big = max_plan(spec, &dev(3.4, 6, 1.0, 3.5, None, cal), cal, q, &ov, &|_| true).unwrap();
        assert_eq!((big.size, big.vae_tile), (1024, 64));                       // 4 GB laptop: the model's maximum, full tile
        let mid = max_plan(spec, &dev(2.4, 6, 1.0, 3.5, None, cal), cal, q, &ov, &|_| true).unwrap();
        assert_eq!(mid.size, 1024);
        assert!(mid.vae_tile < 64 && mid.peak_gib * MEM_MARGIN <= 1.4 + 1e-9);   // smaller tile, still the full size
        // decode time by tile at 512 (one 64-latent tile = a single decode): measured 11.2 s (64), 22.7 (48), 22.6 (32), 23.0 (24), 17.6 (16)
        let dec = |t: u32| plan_model(spec, &dev(3.4, 6, 1.0, 3.5, None, cal), cal, q, &Overrides { size: Some(512), vae_tile: Some(t), ..ov.clone() }, &|_| true).unwrap().estimate.decode;
        for (t, measured_ratio) in [(48u32, 2.03), (32, 2.02), (24, 2.05), (16, 1.57)] {
            let ratio = dec(t) / dec(64);
            assert!((ratio - measured_ratio).abs() / measured_ratio <= 0.15, "tile {t}: decode time x{ratio:.2}, measured x{measured_ratio:.2}");
        }
        let small = max_plan(spec, &dev(1.6, 6, 1.0, 3.5, None, cal), cal, q, &ov, &|_| true).unwrap();
        assert!(small.size < 1024 && small.size >= 704, "ceiling {}", small.size);
        let why = plan_model(spec, &dev(1.6, 6, 1.0, 3.5, None, cal), cal, q, &Overrides { size: Some(small.size + 64), ..ov.clone() }, &|_| true).unwrap_err();
        assert!(why.contains("needs about") && why.contains("1 GiB is kept for Windows"), "{why}");
        // a laptop too small for even the smallest image: refused with the reason, no ceiling
        assert!(max_plan(spec, &dev(1.2, 6, 1.0, 3.5, None, cal), cal, q, &ov, &|_| true).is_none());
    }

    #[test]
    fn suggest_ranks_fitting_models_first() {
        let (m, cal) = catalog();
        // 4 GB laptop, no GPU: every supported model fits, fastest first (SD-Turbo, 1 step); Wan last, with its phase
        let s = suggest(&m, &dev(3.4, 6, 1.0, 3.5, None, cal), cal, None, Quality::Balanced, &Overrides::default(), &|_| true);
        assert_eq!(s[0].model.id, "sd-turbo");
        let fit: Vec<&Suggestion> = s.iter().take_while(|x| x.plan.is_ok()).collect();
        assert_eq!(fit.len(), m.iter().filter(|x| x.supported).count());
        assert!(fit.windows(2).all(|w| w[0].plan.as_ref().unwrap().estimate.total <= w[1].plan.as_ref().unwrap().estimate.total));
        let last = s.last().unwrap();
        assert!(last.model.id.starts_with("wan") && last.plan.as_ref().unwrap_err().contains("Phase"));
        // a device too small for any plan: supported models come back as "does not fit" with the reason, before unsupported ones
        let s = suggest(&m, &dev(1.2, 2, 1.0 / 6.0, 0.45, None, cal), cal, Some("image"), Quality::Balanced, &Overrides::default(), &|_| true);
        assert!(s[0].model.supported && s[0].plan.as_ref().unwrap_err().contains("GiB"));
        assert!(s.iter().all(|x| x.model.kind == "image"));
    }

    #[test]
    fn full_precision_strategy() {
        let (_, cal) = catalog();
        let d4 = dev(3.4, 6, 1.0, 1.0, None, cal);       // 1 GB/s disk: 22.1 GiB BF16 = ~22 s of reads per call
        let q = |s: &Source| if let Source::Quantize { ty, .. } = s { *ty } else { 0 };
        // FLUX-like file: 22.1 GiB denoiser, 4 calls; image time = compute x calls + 20 s of other stages
        let f = |compute: f64| SourceIn { denoiser_gib: 22.1, rest_gib: 0.0, stored_bits: 16.0, commit_gib: 1.2, calls: 4.0,
                                          compute_s: compute, image_s: compute * 4.0 + 20.0 };
        let run = |m: SourceIn, d: &Device| source_strategy(m, d, false, true, None);
        // fast compute (5 s per call): no source hides its reads -> the fastest, Q4_K
        assert_eq!(q(&run(f(5.0), &d4)), crate::ffi::RT_Q4_K);
        // ~13 s per call: Q8_0's reads (~11 s) hide, BF16's (~22 s) do not -> Q8_0
        assert_eq!(q(&run(f(13.0), &d4)), crate::ffi::RT_Q8_0);
        // slow compute (FLUX on this laptop, ~70 s per call): even BF16's reads hide -> stream at full precision
        assert!(matches!(run(f(70.0), &d4), Source::Stream(_)));
        // SDXL-Turbo 512 on this laptop: 4 GiB of reads (~8.6 s at 0.5 GB/s) vs 7.4 s compute, 20 s image:
        // streaming adds ~1 s (5%) -> stream, no 4.8 GB copy for a 1 s gain
        let mut slow = dev(3.4, 6, 1.0, 0.5, None, cal);
        let xl = SourceIn { denoiser_gib: 4.8, rest_gib: 1.7, stored_bits: 16.0, commit_gib: 1.1, calls: 1.0, compute_s: 7.4, image_s: 20.0 };
        assert!(matches!(run(xl, &slow), Source::Stream(_)));
        // --quantize q8_0 forces the type
        assert_eq!(q(&source_strategy(f(70.0), &d4, false, true, Some(crate::ffi::RT_Q8_0))), crate::ffi::RT_Q8_0);
        // --keep-precision: stream
        assert!(matches!(source_strategy(f(5.0), &d4, true, true, None), Source::Stream(_)));
        // already quantized: as is
        assert_eq!(run(SourceIn { stored_bits: 0.0, ..f(5.0) }, &d4), Source::AsIs);
        // fits in RAM (big laptop): as is, quantizing would only cost quality
        assert_eq!(run(SourceIn { denoiser_gib: 3.2, stored_bits: 32.0, ..f(5.0) }, &dev(31.0, 8, 1.0, 3.4, None, cal)), Source::AsIs);
        // the family cannot run a quantized copy: stream
        assert!(matches!(source_strategy(f(5.0), &d4, false, false, None), Source::Stream(_)));
        // disk room counts the whole copy: 8 GiB free fits a Q4_K denoiser (6.2) but not with a 2 GiB rest -> stream
        slow.disk_free_gib = 8.0;
        let mut tight = dev(3.4, 6, 1.0, 1.0, None, cal); tight.disk_free_gib = 8.0;
        assert_eq!(q(&run(f(13.0), &tight)), crate::ffi::RT_Q4_K);
        assert!(matches!(run(SourceIn { rest_gib: 2.0, ..f(13.0) }, &tight), Source::Stream(_)));
    }

    #[test]
    fn missing_files_need_disk() {
        let (m, cal) = catalog();
        let mut d = dev(7.35, 6, 1.0, 3.5, None, cal);
        d.disk_free_gib = 3.0;
        let e = plan(&m, &d, cal, "image", Quality::Balanced, &Overrides::default(), &|_| false).unwrap_err();
        assert!(e.contains("downloads"), "{e}");
    }
}
