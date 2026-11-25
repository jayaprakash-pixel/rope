//! Placement rules (P2.3.2): one set of rules decides, for ANY device and ANY profiled model, where every block
//! of weights lives and runs. Nothing here knows a model family or a laptop model; inputs are numbers only:
//!   device:  RAM budget, VRAM budget, CPU speed per component, GPU speed, RAM->VRAM copy speed, disk speed
//!   model:   per block — component, stored bytes, FLOPs per call (from `rope profile`), calls per image
//! Places:   GpuResident  stored bytes kept in VRAM (uploaded once), computed on the GPU
//!           GpuStreamed  copied RAM->VRAM through a two-slot ring each call, computed on the GPU
//!           CpuRam       computed on the CPU, weights stay in the OS file cache
//!           CpuDisk      computed on the CPU, weights re-read from disk each call (RAM cache full)
//! VRAM holds weights in their stored (quantized) form; FP16 exists only as a per-layer scratch (GPU_SCRATCH).
//! Resident weights are uploaded once per session (reported as `upload_once_s`), not once per image.
#![forbid(unsafe_code)]

/// VRAM kept free for the CUDA context, kernels' workspace and the FP16 per-layer scratch.
pub const GPU_RESERVE_GIB: f64 = 0.4;
pub const GPU_SCRATCH_GIB: f64 = 0.25;
/// RAM kept free for Windows and other programs (same rule as the planner).
pub const OS_RESERVE_GIB: f64 = 1.0;
const GIB: f64 = 1073741824.0;
/// Mapped-file -> pinned host copy speed (dev laptop: 16 GB uploaded in 6.8 s end to end at 6.67 GB/s DMA => ~3.6 GB/s).
pub const HOST_COPY_GBS: f64 = 3.6;
/// VRAM kept for streaming, in multiples of the largest streamed block. The executor streams through one raw-weight
/// scratch (uploads on the compute stream, the CPU copying ahead); block-ahead staging (two slots, ROPE_STAGE=1) gave
/// no gain on the dev laptop (SD 1.5: 72.2 s staged vs 68.6 s), so it is off and one slot is reserved.
pub const RING_SLOTS: f64 = 1.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Where { GpuResident, GpuStreamed, CpuRam, CpuDisk }

impl Where {
    pub fn name(self) -> &'static str {
        match self { Where::GpuResident => "GPU, resident", Where::GpuStreamed => "GPU, streamed", Where::CpuRam => "CPU, RAM", Where::CpuDisk => "CPU, disk" }
    }
    #[cfg(test)]
    fn gpu(self) -> bool { matches!(self, Where::GpuResident | Where::GpuStreamed) }
}

#[derive(Debug, Clone)]
/// `gpu_ok`: the executor can run this block well on the GPU (its model code is on the op interface).
/// `gpu_eff`: measured GPU efficiency of this kind of block relative to the device's GPU rate (calibrated per family).
pub struct BlockCost { pub name: String, pub component: String, pub bytes: f64, pub flops: f64, pub gpu_ok: bool, pub gpu_eff: f64 }

/// Device numbers the rules need. `cpu_gflops`: effective CPU speed per component (text, denoiser, vae) on this
/// device — rope's measured model speed, not the kernel peak. No GPU: `vram_gib` 0.
#[derive(Debug, Clone)]
pub struct Dev { pub ram_gib: f64, pub vram_gib: f64, pub cpu_gflops: [f64; 3], pub gpu_gflops: f64, pub h2d_gbs: f64, pub disk_gbs: f64 }

/// How often each component runs per image: text once, denoiser steps x (2 with CFG), VAE once (tiles are inside
/// its profile). Activation peaks (private bytes) per component.
#[derive(Debug, Clone)]
pub struct Use { pub denoiser_calls: f64, pub peak_private: [f64; 3] }

#[derive(Debug, Clone)]
pub struct Placed { pub name: String, pub component: String, pub at: Where, pub seconds: f64, pub why: String }

#[derive(Debug, Clone)]
pub struct Placement {
    pub blocks: Vec<Placed>,
    pub seconds: f64,            // per image
    pub upload_once_s: f64,      // first image only: resident weights RAM -> VRAM
    pub vram_used_gib: f64,      // resident weights + ring + reserve + scratch
    pub vram_budget_gib: f64,
    pub ram_cache_gib: f64,      // weights kept in the OS file cache
    pub ram_cache_budget_gib: f64,
}

fn comp_idx(c: &str) -> usize { match c { "text" => 0, "denoiser" => 1, _ => 2 } }

/// Place every block. Err with a reason when the device cannot hold even the activations.
pub fn place(blocks: &[BlockCost], dev: &Dev, u: &Use) -> Result<Placement, String> {
    let act = u.peak_private.iter().cloned().fold(0.0, f64::max) / GIB;
    let cache_budget = dev.ram_gib - OS_RESERVE_GIB - act;
    if cache_budget < 0.0 {
        return Err(format!("needs {:.2} GiB for activations + {OS_RESERVE_GIB:.0} GiB for Windows; this device has {:.2} GiB usable", act, dev.ram_gib));
    }
    let gpu_act = u.peak_private[1] / GIB;                     // the denoiser's activations live on the GPU when it runs there
    let vram_budget = (dev.vram_gib - GPU_RESERVE_GIB - GPU_SCRATCH_GIB - gpu_act).max(0.0);
    let has_gpu = dev.vram_gib > 0.0 && dev.gpu_gflops > 0.0 && dev.h2d_gbs > 0.0 && vram_budget > 0.0;
    let calls = |b: &BlockCost| if b.component == "denoiser" { u.denoiser_calls } else { 1.0 };
    let t_cpu = |b: &BlockCost| b.flops / (dev.cpu_gflops[comp_idx(&b.component)] * 1e9);
    let t_gpu = |b: &BlockCost| if has_gpu && b.gpu_ok { b.flops / (dev.gpu_gflops * b.gpu_eff.max(0.05) * 1e9) } else { f64::INFINITY };
    // an upload = mapped file -> pinned copy on the CPU, then DMA: the two stages in series
    let up_gbs = 1.0 / (1.0 / dev.h2d_gbs + 1.0 / HOST_COPY_GBS);
    let up = |b: &BlockCost| b.bytes / (up_gbs * 1e9);
    let disk = |b: &BlockCost| b.bytes / (dev.disk_gbs * 1e9);

    // 1. VRAM: resident where it saves the most time per byte versus the best other choice (CPU or streamed).
    //    Ties (streaming would hide its upload anyway) go to the blocks read most per image: resident weights need
    //    no RAM cache, so idle VRAM relieves RAM and disk.
    let mut at: Vec<Where> = vec![Where::CpuRam; blocks.len()];
    let mut vram = 0.0;
    if has_gpu {
        let mut order: Vec<(f64, f64, usize)> = blocks.iter().enumerate().map(|(i, b)| {
            let other = (calls(b) * t_cpu(b)).min(calls(b) * t_gpu(b).max(up(b)));
            ((other - calls(b) * t_gpu(b)) / b.bytes.max(1.0), calls(b), i)
        }).collect();
        order.sort_by(|a, b| b.0.total_cmp(&a.0).then(b.1.total_cmp(&a.1)));
        for (gain, _, i) in order {
            let g = blocks[i].bytes / GIB;
            if gain >= 0.0 && vram + g <= vram_budget { at[i] = Where::GpuResident; vram += g; }
        }
        // 2. the rest: streamed when faster than the CPU, keeping room for a two-slot ring of the largest streamed block
        loop {
            for (i, b) in blocks.iter().enumerate() {
                if at[i] == Where::GpuResident { continue; }
                at[i] = if calls(b) * t_gpu(b).max(up(b)) < calls(b) * t_cpu(b) { Where::GpuStreamed } else { Where::CpuRam };
            }
            let ring = RING_SLOTS * blocks.iter().zip(&at).filter(|(_, w)| **w == Where::GpuStreamed).map(|(b, _)| b.bytes).fold(0.0, f64::max) / GIB;
            if vram + ring <= vram_budget { break; }
            // no room for the ring: give back the resident block that saves the least per byte, then re-decide
            let worst = blocks.iter().enumerate().filter(|(i, _)| at[*i] == Where::GpuResident)
                .min_by(|(_, a), (_, b)| ((calls(a) * t_cpu(a) - calls(a) * t_gpu(a)) / a.bytes).total_cmp(&((calls(b) * t_cpu(b) - calls(b) * t_gpu(b)) / b.bytes)))
                .map(|(i, _)| i);
            match worst {
                Some(i) => { at[i] = Where::CpuRam; vram -= blocks[i].bytes / GIB; }
                None => { for w in at.iter_mut() { if *w == Where::GpuStreamed { *w = Where::CpuRam; } } break; }   // ring alone does not fit
            }
        }
    }
    let ring = RING_SLOTS * blocks.iter().zip(&at).filter(|(_, w)| **w == Where::GpuStreamed).map(|(b, _)| b.bytes).fold(0.0, f64::max) / GIB;

    // 3. RAM file cache (page granular, so a block can be partly cached) for the weights read from RAM, most reads
    //    per image first: a fractional fill, optimal for "disk seconds saved", and monotone in RAM and VRAM.
    let mut order: Vec<usize> = (0..blocks.len()).filter(|&i| at[i] != Where::GpuResident).collect();
    order.sort_by(|&a, &b| calls(&blocks[b]).total_cmp(&calls(&blocks[a])));
    let mut cache = 0.0;
    let mut frac = vec![0f64; blocks.len()];                 // cached fraction of each block
    for i in order {
        let g = blocks[i].bytes / GIB;
        let take = (cache_budget - cache).clamp(0.0, g);
        frac[i] = if g > 0.0 { take / g } else { 1.0 };
        cache += take;
    }

    // 4. times and reasons
    let mut out = Vec::with_capacity(blocks.len());
    let mut total = 0.0;
    for (i, b) in blocks.iter().enumerate() {
        let n = calls(b);
        let (w, s, why) = match at[i] {
            Where::GpuResident => (Where::GpuResident, n * t_gpu(b),
                format!("{:.0} MiB kept in VRAM, read {n:.0}x per image; saves {:.2} s/image vs CPU", b.bytes / 1048576.0, n * (t_cpu(b) - t_gpu(b)))),
            Where::GpuStreamed => {
                let src = (1.0 - frac[i]) * disk(b);
                (Where::GpuStreamed, n * (t_gpu(b).max(up(b) + src)), format!("upload {:.3} s vs GPU {:.3} s per call; VRAM full", up(b) + src, t_gpu(b)))
            }
            _ if frac[i] >= 0.999 => (Where::CpuRam, n * t_cpu(b),
                if !has_gpu { "no usable GPU".into() } else if !b.gpu_ok { "no GPU code for this component yet".into() } else { format!("CPU {:.3} s <= GPU path per call", t_cpu(b)) }),
            _ => (Where::CpuDisk, n * (t_cpu(b) + (1.0 - frac[i]) * disk(b)),
                  format!("RAM cache full ({:.0}% cached): +{:.2} s disk per call", frac[i] * 100.0, (1.0 - frac[i]) * disk(b))),
        };
        total += s;
        out.push(Placed { name: b.name.clone(), component: b.component.clone(), at: w, seconds: s, why });
    }
    let used = if vram + ring > 0.0 { vram + ring + GPU_RESERVE_GIB + GPU_SCRATCH_GIB + gpu_act } else { 0.0 };
    let upload_once_s = blocks.iter().zip(&at).filter(|(_, w)| **w == Where::GpuResident).map(|(b, _)| up(b)).sum();
    Ok(Placement { blocks: out, seconds: total, upload_once_s, vram_used_gib: used, vram_budget_gib: dev.vram_gib, ram_cache_gib: cache, ram_cache_budget_gib: cache_budget })
}

/// Read a `rope profile --json` document into block costs + activation peaks (every block GPU-capable;
/// the caller narrows `gpu_ok`). Used for the profile cache and the test fixtures.
pub fn from_profile_json(v: &crate::json::Value) -> Result<(Vec<BlockCost>, [f64; 3]), String> {
    let num = |o: &crate::json::Value, k: &str| o.get(k).and_then(crate::json::Value::as_f64).unwrap_or(0.0);
    let blocks = v.get("blocks").and_then(crate::json::Value::as_arr).ok_or("profile has no blocks")?.iter().map(|b| BlockCost {
        name: b.get("name").and_then(crate::json::Value::as_str).unwrap_or("").into(),
        component: b.get("component").and_then(crate::json::Value::as_str).unwrap_or("other").into(),
        bytes: num(b, "bytes"), flops: num(b, "flops"), gpu_ok: true, gpu_eff: 1.0,
    }).collect();
    let pk = v.get("peak_private").ok_or("profile has no peak_private")?;
    Ok((blocks, [num(pk, "text"), num(pk, "denoiser"), num(pk, "vae")]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> (Vec<BlockCost>, [f64; 3]) {
        let txt = match name {
            "sd15" => include_str!("fixtures/profile_sd15_512.json"),
            "sdturbo" => include_str!("fixtures/profile_sdturbo_512.json"),
            _ => include_str!("fixtures/profile_flux_512.json"),
        };
        from_profile_json(&crate::json::parse(txt).unwrap()).unwrap()
    }
    /// The dev laptop's measured speeds (CPU per component from rope's own runs; GPU FP16 path; PCIe from the probe).
    fn dev(ram: f64, vram: f64) -> Dev {
        Dev { ram_gib: ram, vram_gib: vram, cpu_gflops: [280.0, 150.0, 114.0], gpu_gflops: 1700.0, h2d_gbs: 6.67, disk_gbs: 3.4 }
    }

    #[test]
    fn matrix_respects_budgets_and_never_gets_slower() {
        for (model, calls) in [("sd15", 40.0), ("sdturbo", 1.0), ("flux", 4.0)] {
            let (blocks, peak) = fixture(model);
            let u = Use { denoiser_calls: calls, peak_private: peak };
            let mut by_ram: Vec<Option<f64>> = Vec::new();
            for ram in [2.0, 4.0, 8.0, 16.0] {
                let mut prev = f64::INFINITY;
                let mut cpu_only = None;
                for vram in [0.0, 2.0, 4.0, 6.0, 8.0] {
                    let tag = format!("{model} ram {ram} vram {vram}");
                    let p = match place(&blocks, &dev(ram, vram), &u) {
                        Ok(p) => p,
                        Err(e) => { assert!(e.contains("GiB"), "{tag}: refusal without a reason: {e}"); continue; }
                    };
                    assert_eq!(p.blocks.len(), blocks.len(), "{tag}: every block placed once");
                    assert!(p.vram_used_gib <= vram + 1e-9, "{tag}: VRAM {:.2} > {vram}", p.vram_used_gib);
                    assert!(p.ram_cache_gib <= p.ram_cache_budget_gib + 1e-9, "{tag}: RAM cache over budget");
                    if vram == 0.0 { assert!(p.blocks.iter().all(|b| !b.at.gpu()), "{tag}: GPU used without VRAM"); cpu_only = Some(p.seconds); }
                    if let Some(c) = cpu_only { assert!(p.seconds <= c * 1.0001, "{tag}: GPU plan slower than CPU-only ({:.1} > {c:.1})", p.seconds); }
                    assert!(p.seconds <= prev * 1.0001, "{tag}: more VRAM made it slower ({:.1} > {prev:.1})", p.seconds);
                    assert!(p.blocks.iter().all(|b| !b.why.is_empty()), "{tag}: every placement explains itself");
                    prev = p.seconds;
                    if vram == 0.0 { by_ram.push(Some(p.seconds)); }
                }
            }
            // more RAM never makes the CPU-only plan slower
            let t: Vec<f64> = by_ram.into_iter().flatten().collect();
            assert!(t.windows(2).all(|w| w[1] <= w[0] * 1.0001), "{model}: more RAM made it slower: {t:?}");
        }
    }

    #[test]
    fn flux_on_a_small_gpu_streams_and_a_big_gpu_holds_it() {
        let (blocks, peak) = fixture("flux");
        let u = Use { denoiser_calls: 4.0, peak_private: peak };
        let small = place(&blocks, &dev(8.0, 4.0), &u).unwrap();
        assert!(small.blocks.iter().any(|b| b.at == Where::GpuResident) && small.blocks.iter().any(|b| b.at == Where::GpuStreamed),
                "4 GB VRAM: part resident, part streamed");
        let big = place(&blocks, &dev(16.0, 16.0), &u).unwrap();
        assert!(big.blocks.iter().filter(|b| b.component == "denoiser").all(|b| b.at == Where::GpuResident), "16 GB VRAM holds the whole DiT");
        assert!(big.seconds < small.seconds);
        // 2 GB laptop: refused with the reason
        assert!(place(&blocks, &dev(2.0, 0.0), &u).unwrap_err().contains("GiB"));
    }
}
