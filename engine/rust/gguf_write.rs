//! GGUF v3 writer and `rope quantize`: one-time local quantization of a full-precision model file. The repeated
//! blocks' 2-D (and conv) weights — almost all of the bytes — become Q4_K or Q8_0; stems, heads, norms and biases stay
//! F32 (small and more sensitive). Streams tensor by tensor, so a 24 GB input needs one tensor in RAM at a time.
#![forbid(unsafe_code)]

use crate::ffi;
use crate::weights::Weights;
use std::io::Write;
use std::path::Path;

const ALIGN: u64 = 32;

fn put_str(o: &mut Vec<u8>, s: &str) { o.extend((s.len() as u64).to_le_bytes()); o.extend(s.as_bytes()); }

/// Should this tensor be quantized? A denoiser weight (the copy exists for the denoiser; text encoders and VAE stay as
/// stored), 2+ dims, inside a numbered block, and its innermost dimension a multiple of the format's block (GGUF blocks
/// run along the innermost dimension: a 3x3 conv kernel, innermost 3, cannot be quantized).
pub fn quantizable(name: &str, shape_outer_first: &[u64], block: u64) -> bool {
    if shape_outer_first.len() < 2 || crate::profile::component(name) != "denoiser" { return false; }
    let n: u64 = shape_outer_first.iter().product();
    let in_block = name.split('.').any(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()));
    in_block && shape_outer_first.last().unwrap() % block == 0 && n >= 4096
}

/// Quantize `input` into a GGUF file at `out` (type: ffi::RT_Q4_K or RT_Q8_0; Q4_K falls back to Q8_0 per tensor
/// when the rows do not split into 256). Returns (quantized, kept as stored, bytes).
pub fn quantize_file(input: &Path, out: &Path, wtype: u32, arch: &str) -> Result<(usize, usize, u64), String> {
    let w = Weights::open(input)?;
    let block = ffi::wtype_block(wtype);
    // plan every tensor: name, shape (outermost first), output type, output bytes
    struct T { name: String, shape: Vec<u64>, ty: u32, bytes: u64, q: bool }
    let mut ts = Vec::new();
    for t in &w.mf.tensors {
        let mut shape = t.dims();
        if shape.is_empty() { shape.push(1); }                             // a scalar (logit_scale): GGUF needs 1+ dims
        // Q4_K needs rows of 256: other block weights (SDXL's 640-wide rows, 3x3 convs of 2880) fall back to Q8_0
        // (rows of 32) instead of F32, which would be larger than a 16-bit source
        // not quantized: the source bytes as they are (F32 / F16 / BF16), lossless and no larger than the source.
        // ponytail: an F32 source keeps its 3x3 convs in F32 (SD 2.x copy 2.98 GB); write them as F16 if that matters.
        let (ty, q) = if quantizable(&t.name, &shape, block) { (wtype, true) }
                      else if wtype != ffi::RT_Q8_0 && quantizable(&t.name, &shape, ffi::wtype_block(ffi::RT_Q8_0)) { (ffi::RT_Q8_0, true) }
                      else { (crate::weights::type_id(t).ok_or_else(|| format!("{}: unsupported dtype {}", t.name, t.dtype))?, false) };
        let bytes = if q { shape[0] * ffi::row_bytes(ty, shape[1..].iter().product()) as u64 } else { t.nbytes };
        ts.push(T { name: t.name.clone(), shape, ty, bytes, q });
    }
    // header + kv + tensor infos
    let mut h = Vec::new();
    h.extend(b"GGUF");
    h.extend(3u32.to_le_bytes());
    h.extend((ts.len() as u64).to_le_bytes());
    h.extend(3u64.to_le_bytes());
    put_str(&mut h, "general.architecture"); h.extend(8u32.to_le_bytes()); put_str(&mut h, arch);
    put_str(&mut h, "general.alignment"); h.extend(4u32.to_le_bytes()); h.extend((ALIGN as u32).to_le_bytes());
    put_str(&mut h, "general.quantized_by"); h.extend(8u32.to_le_bytes()); put_str(&mut h, "rope quantize");
    let mut off = 0u64;
    for t in &ts {
        put_str(&mut h, &t.name);
        h.extend((t.shape.len() as u32).to_le_bytes());
        for d in t.shape.iter().rev() { h.extend(d.to_le_bytes()); }        // GGUF: innermost dimension first
        h.extend(t.ty.to_le_bytes());
        h.extend(off.to_le_bytes());
        off += t.bytes.div_ceil(ALIGN) * ALIGN;
    }
    while h.len() as u64 % ALIGN != 0 { h.push(0); }
    let tmp = out.with_extension("gguf.part");
    let f = std::fs::File::create(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
    let mut o = std::io::BufWriter::with_capacity(8 << 20, f);
    o.write_all(&h).map_err(|e| e.to_string())?;
    let (mut nq, mut nf) = (0, 0);
    let t0 = std::time::Instant::now();
    let (mut t_read, mut t_quant, mut t_write) = (0f64, 0f64, 0f64);
    for (i, t) in ts.iter().enumerate() {
        let ta = std::time::Instant::now();
        let x = if t.q { w.f32(&t.name)? } else { Vec::new() };            // any stored type -> f32 (reads the file)
        let tb = std::time::Instant::now();
        let data: Vec<u8> = if !t.q { nf += 1; w.raw(&t.name)?.0.to_vec() } else {
            nq += 1;
            let k: u64 = t.shape[1..].iter().product();
            ffi::quantize_rows(t.ty, &x, t.shape[0] as usize, k as usize)?
        };
        let tc = std::time::Instant::now();
        o.write_all(&data).map_err(|e| e.to_string())?;
        let pad = (ALIGN - data.len() as u64 % ALIGN) % ALIGN;
        o.write_all(&vec![0u8; pad as usize]).map_err(|e| e.to_string())?;
        t_read += (tb - ta).as_secs_f64(); t_quant += (tc - tb).as_secs_f64(); t_write += tc.elapsed().as_secs_f64();
        if (i + 1) % 50 == 0 || i + 1 == ts.len() {
            eprintln!("  quantize: {}/{} tensors, {:.0} s", i + 1, ts.len(), t0.elapsed().as_secs_f64());
        }
    }
    o.flush().map_err(|e| e.to_string())?;
    drop(o);
    eprintln!("  quantize time: read + convert {t_read:.0} s, quantize {t_quant:.0} s, write {t_write:.0} s");
    std::fs::rename(&tmp, out).map_err(|e| e.to_string())?;
    let bytes = std::fs::metadata(out).map_err(|e| e.to_string())?.len();
    Ok((nq, nf, bytes))
}

#[cfg(test)]
mod tests {
    #[test]
    fn which_tensors_are_quantized() {
        assert!(super::quantizable("double_blocks.3.img_attn.qkv.weight", &[9216, 3072], 256));
        assert!(!super::quantizable("double_blocks.3.img_attn.qkv.bias", &[9216], 256));            // 1-D
        assert!(!super::quantizable("img_in.weight", &[3072, 64], 32));                           // stem: no block number
        assert!(!super::quantizable("final_layer.linear.weight", &[64, 3072], 256));             // head
        assert!(!super::quantizable("conditioner.embedders.1.model.transformer.resblocks.3.attn.in_proj_weight", &[3840, 1280], 256)); // text encoder
        assert!(!super::quantizable("model.diffusion_model.input_blocks.1.0.in_layers.2.weight", &[320, 320, 3, 3], 32)); // innermost 3
        assert!(super::quantizable("model.diffusion_model.input_blocks.4.1.proj_in.weight", &[640, 640], 32));      // SDXL 640: Q8_0
        assert!(!super::quantizable("model.diffusion_model.input_blocks.4.1.proj_in.weight", &[640, 640], 256));    // not Q4_K
    }
}
