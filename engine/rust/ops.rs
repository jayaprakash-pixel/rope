//! Op interface (G3 / P2.3.3): model code is written once against these ops; each op runs where its data lives
//! (weights decide for `lin`: a GPU-placed weight pulls its input to the GPU). Activations stay on the device across
//! consecutive GPU ops; transfers happen only at CPU/GPU boundaries. The CPU side of every op is exactly the code
//! the models used before (bit-identical results); any device failure falls back to the CPU.
#![forbid(unsafe_code)]

use crate::ffi;
use crate::weights::Weights;

pub enum Data {
    H(Vec<f32>),
    #[cfg(feature = "cuda")]
    D(ffi::DevBuf),
}

/// A row-major activation matrix [rows, cols], on the host or on the GPU.
pub struct T { pub rows: usize, pub cols: usize, pub data: Data }

impl T {
    pub fn host(v: Vec<f32>, rows: usize, cols: usize) -> T { debug_assert_eq!(v.len(), rows * cols); T { rows, cols, data: Data::H(v) } }
    pub fn is_dev(&self) -> bool { !matches!(self.data, Data::H(_)) }

    /// Host copy (downloads a device tensor).
    pub fn into_host(self) -> Result<Vec<f32>, String> {
        match self.data {
            Data::H(v) => Ok(v),
            #[cfg(feature = "cuda")]
            Data::D(d) => { let mut v = vec![0f32; d.len]; if ffi::dev_download(&d, &mut v) { Ok(v) } else { Err("GPU download failed".into()) } }
        }
    }
    fn host_ref(&self) -> Result<std::borrow::Cow<'_, [f32]>, String> {
        match &self.data {
            Data::H(v) => Ok(std::borrow::Cow::Borrowed(v)),
            #[cfg(feature = "cuda")]
            Data::D(d) => { let mut v = vec![0f32; d.len]; if ffi::dev_download(d, &mut v) { Ok(std::borrow::Cow::Owned(v)) } else { Err("GPU download failed".into()) } }
        }
    }
    #[cfg(feature = "cuda")]
    fn dev(d: ffi::DevBuf, rows: usize, cols: usize) -> T { T { rows, cols, data: Data::D(d) } }
    /// The host vector of a tensor the caller has just made host-resident.
    fn hv(&mut self) -> &mut Vec<f32> {
        match &mut self.data { Data::H(v) => v, #[cfg(feature = "cuda")] Data::D(_) => unreachable!("host data expected") }
    }
}

/// Device view of a tensor: borrowed if it lives there, else a temporary upload.
#[cfg(feature = "cuda")]
enum DRef<'a> { B(&'a ffi::DevBuf), O(ffi::DevBuf) }
#[cfg(feature = "cuda")]
impl DRef<'_> { fn get(&self) -> &ffi::DevBuf { match self { DRef::B(b) => b, DRef::O(o) => o } } }
#[cfg(feature = "cuda")]
fn on_dev(x: &T) -> Option<DRef<'_>> {
    match &x.data { Data::D(d) => Some(DRef::B(d)), Data::H(v) => ffi::dev_upload(v).map(DRef::O) }
}
#[cfg(feature = "cuda")]
fn up(v: &[f32]) -> Option<ffi::DevBuf> { ffi::dev_upload(v) }

/// y[m, n] = x[m, k] W^T (+ bias): GGUF weights have ne = [k, n]; safetensors weights are [n, k] (or [n, c, kh, kw]).
pub fn lin(w: &Weights, x: &T, name: &str, bias: bool) -> Result<T, String> {
    let t = w.tensor(&format!("{name}.weight"))?;
    let (k, n) = if t.ggml_type.is_some() { (t.shape[0] as usize, t.shape[1] as usize) } else { (t.shape[1..].iter().product::<u64>() as usize, t.shape[0] as usize) };
    let (wb, ty) = w.raw(&format!("{name}.weight"))?;
    let b = if bias { Some(w.f32(&format!("{name}.bias"))?) } else { None };
    let m = x.rows;
    let lora = crate::lora::module(name);                       // adapter layer: low-rank add on the host (x comes down from the GPU)
    #[cfg(feature = "cuda")]
    if lora.is_none() && m >= 4 && ffi::weight_on_gpu(wb) {
        let r = (|| {
            let xd = on_dev(x)?;
            let bd = match &b { Some(b) => Some(up(&b[..n])?), None => None };
            ffi::dev_matmul(xd.get(), m, k, wb, ty, n, bd.as_ref())
        })();
        if let Some(y) = r { return Ok(T::dev(y, m, n)); }
    }
    let xh = x.host_ref()?;
    let mut y = vec![0f32; m * n];
    ffi::matmul(&xh, m, k, wb, ty, n, &mut y);
    if let Some(b) = b { ffi::add_rows(&mut y, &b[..n]); }
    if let Some(l) = lora { if l.k == k && l.n == n { l.apply(&xh, m, &mut y); } }
    Ok(T::host(y, m, n))
}

/// (1 + scale) * LN(x) + shift, LN without affine, eps 1e-6.
pub fn modulate(x: &T, shift: &[f32], scale: &[f32]) -> Result<T, String> {
    let (m, n) = (x.rows, x.cols);
    #[cfg(feature = "cuda")]
    if let Data::D(d) = &x.data {
        if let Some(y) = (|| ffi::dev_modulate(d, m, n, &up(shift)?, &up(scale)?))() { return Ok(T::dev(y, m, n)); }
    }
    let mut y = x.host_ref()?.into_owned();
    ffi::layernorm(&mut y, m, n, None, None, 1e-6);
    for row in y.chunks_exact_mut(n) { for i in 0..n { row[i] = row[i] * (1.0 + scale[i]) + shift[i]; } }
    Ok(T::host(y, m, n))
}

/// x += gate * y (gate per column).
pub fn add_gated(x: &mut T, gate: &[f32], y: &T) -> Result<(), String> {
    let n = x.cols;
    #[cfg(feature = "cuda")]
    if let Data::D(d) = &x.data {
        let ok = (|| { let yd = on_dev(y)?; let g = up(gate)?; ffi::dev_add_gated(d, &g, yd.get(), x.rows, n).then_some(()) })();
        if ok.is_some() { return Ok(()); }
        let v = std::mem::replace(x, T::host(vec![], 0, 0)).into_host()?;     // device failed: continue on the host
        *x = T::host(v, y.rows, n);
    }
    let yh = y.host_ref()?;
    let xv = x.hv();
    for (xr, yr) in xv.chunks_exact_mut(n).zip(yh.chunks_exact(n)) { for i in 0..n { xr[i] += gate[i] * yr[i]; } }
    Ok(())
}

pub fn gelu_tanh(x: &mut T) -> Result<(), String> {
    #[cfg(feature = "cuda")]
    if let Data::D(d) = &x.data { if ffi::dev_gelu_tanh(d) { return Ok(()); } }
    if x.is_dev() { let (r, c) = (x.rows, x.cols); let v = std::mem::replace(x, T::host(vec![], 0, 0)).into_host()?; *x = T::host(v, r, c); }
    let v = x.hv();
    ffi::gelu_tanh(v);
    Ok(())
}

/// Per-head RMSNorm (learned scales) then interleaved-pair RoPE on q (cols [0, d)) and k (cols [d, 2d)) of a fused
/// buffer; rows use RoPE table rows row0.. (cos/sin: [*, hd/2]).
#[allow(clippy::too_many_arguments)]
pub fn qk_norm_rope(buf: &mut T, d: usize, heads: usize, hd: usize, qs: &[f32], ks: &[f32], tab: &Rope, row0: usize) -> Result<(), String> {
    let (rows, rs) = (buf.rows, buf.cols);
    #[cfg(feature = "cuda")]
    if let Data::D(b) = &buf.data {
        let ok = (|| { let (c, s) = tab.dev()?; ffi::dev_qk_norm_rope(b, rows, rs, d, heads, hd, &up(qs)?, &up(ks)?, c, s, row0).then_some(()) })();
        if ok.is_some() { return Ok(()); }
        let v = std::mem::replace(buf, T::host(vec![], 0, 0)).into_host()?;
        *buf = T::host(v, rows, rs);
    }
    let b = buf.hv();
    for r in 0..rows {
        let (c, s) = (&tab.cos[(row0 + r) * hd / 2..], &tab.sin[(row0 + r) * hd / 2..]);
        for (off, sc) in [(0, qs), (d, ks)] {
            for h in 0..heads {
                let v = &mut b[r * rs + off + h * hd..r * rs + off + (h + 1) * hd];
                let ms = v.iter().map(|x| x * x).sum::<f32>() / hd as f32;
                let inv = 1.0 / (ms + 1e-6).sqrt();
                for (x, g) in v.iter_mut().zip(sc.iter()) { *x *= inv * g; }
                for p in 0..hd / 2 {
                    let (a, bb) = (v[2 * p], v[2 * p + 1]);
                    v[2 * p] = c[p] * a - s[p] * bb;
                    v[2 * p + 1] = s[p] * a + c[p] * bb;
                }
            }
        }
    }
    Ok(())
}

/// RoPE cos/sin tables, uploaded to the GPU once when a GPU op first needs them.
pub struct Rope {
    pub cos: Vec<f32>, pub sin: Vec<f32>,
    #[cfg(feature = "cuda")]
    dev: std::cell::OnceCell<Option<(ffi::DevBuf, ffi::DevBuf)>>,
}
impl Rope {
    pub fn new(cos: Vec<f32>, sin: Vec<f32>) -> Rope {
        Rope { cos, sin, #[cfg(feature = "cuda")] dev: std::cell::OnceCell::new() }
    }
    #[cfg(feature = "cuda")]
    fn dev(&self) -> Option<(&ffi::DevBuf, &ffi::DevBuf)> {
        self.dev.get_or_init(|| Some((up(&self.cos)?, up(&self.sin)?))).as_ref().map(|(c, s)| (c, s))
    }
}

/// Multi-head self-attention over a fused buffer: q/k/v at columns qo/ko/vo -> [rows, heads*hd].
pub fn attention(buf: &T, qo: usize, ko: usize, vo: usize, heads: usize, hd: usize, scale: f32) -> Result<T, String> {
    let (n, rs, d) = (buf.rows, buf.cols, heads * hd);
    #[cfg(feature = "cuda")]
    if let Data::D(b) = &buf.data {
        if let Some(o) = ffi::dev_attention(b, n, rs, qo, ko, vo, heads, hd, scale) { return Ok(T::dev(o, n, d)); }
    }
    let b = buf.host_ref()?;
    let mut o = vec![0f32; n * d];
    ffi::attention_mh(&b[qo..], &b[ko..], &b[vo..], &mut o, n, n, heads, hd, rs, rs, rs, d, scale);
    Ok(T::host(o, n, d))
}

/// Rows [r0, r1) as a new tensor.
pub fn rows(x: &T, r0: usize, r1: usize) -> Result<T, String> {
    let c = x.cols;
    #[cfg(feature = "cuda")]
    if let Data::D(d) = &x.data {
        if let Some(y) = (|| { let y = ffi::dev_alloc((r1 - r0) * c)?; ffi::dev_copy2d(&y, c, 0, d, c, r0 * c, c, r1 - r0).then_some(y) })() {
            return Ok(T::dev(y, r1 - r0, c));
        }
    }
    Ok(T::host(x.host_ref()?[r0 * c..r1 * c].to_vec(), r1 - r0, c))
}

/// Columns [c0, c1) as a new tensor.
pub fn cols(x: &T, c0: usize, c1: usize) -> Result<T, String> {
    let (r, c) = (x.rows, x.cols);
    #[cfg(feature = "cuda")]
    if let Data::D(d) = &x.data {
        if let Some(y) = (|| { let y = ffi::dev_alloc(r * (c1 - c0))?; ffi::dev_copy2d(&y, c1 - c0, 0, d, c, c0, c1 - c0, r).then_some(y) })() {
            return Ok(T::dev(y, r, c1 - c0));
        }
    }
    Ok(T::host(x.host_ref()?.chunks_exact(c).flat_map(|row| row[c0..c1].iter().copied()).collect(), r, c1 - c0))
}

/// [a; b]: rows of a, then rows of b (same width).
pub fn cat_rows(a: T, b: &T) -> Result<T, String> {
    let c = a.cols;
    #[cfg(feature = "cuda")]
    if a.is_dev() || b.is_dev() {
        let r = (|| {
            let (ad, bd) = (on_dev(&a)?, on_dev(b)?);
            let y = ffi::dev_alloc((a.rows + b.rows) * c)?;
            (ffi::dev_copy2d(&y, c, 0, ad.get(), c, 0, c, a.rows) && ffi::dev_copy2d(&y, c, a.rows * c, bd.get(), c, 0, c, b.rows)).then_some(y)
        })();
        if let Some(y) = r { return Ok(T::dev(y, a.rows + b.rows, c)); }
    }
    let rows = a.rows + b.rows;
    let mut v = a.into_host()?;
    v.extend_from_slice(&b.host_ref()?);
    Ok(T::host(v, rows, c))
}

/// [a | b]: columns of a, then columns of b (same rows).
pub fn cat_cols(a: &T, b: &T) -> Result<T, String> {
    let (r, ca, cb) = (a.rows, a.cols, b.cols);
    #[cfg(feature = "cuda")]
    if a.is_dev() || b.is_dev() {
        let y = (|| {
            let (ad, bd) = (on_dev(a)?, on_dev(b)?);
            let y = ffi::dev_alloc(r * (ca + cb))?;
            (ffi::dev_copy2d(&y, ca + cb, 0, ad.get(), ca, 0, ca, r) && ffi::dev_copy2d(&y, ca + cb, ca, bd.get(), cb, 0, cb, r)).then_some(y)
        })();
        if let Some(y) = y { return Ok(T::dev(y, r, ca + cb)); }
    }
    let (ah, bh) = (a.host_ref()?, b.host_ref()?);
    let mut v = vec![0f32; r * (ca + cb)];
    for i in 0..r {
        v[i * (ca + cb)..i * (ca + cb) + ca].copy_from_slice(&ah[i * ca..(i + 1) * ca]);
        v[i * (ca + cb) + ca..(i + 1) * (ca + cb)].copy_from_slice(&bh[i * cb..(i + 1) * cb]);
    }
    Ok(T::host(v, r, ca + cb))
}

/// Make a tensor host-resident in place (after a device op failed, or for a CPU-only op).
fn make_host(x: &mut T) -> Result<(), String> {
    if x.is_dev() { let (r, c) = (x.rows, x.cols); let v = std::mem::replace(x, T::host(vec![], 0, 0)).into_host()?; *x = T::host(v, r, c); }
    Ok(())
}

/// A copy of a tensor (same device).
pub fn clone(x: &T) -> Result<T, String> { rows(x, 0, x.rows) }

/// `lin` with the bias added only if the file has one.
pub fn lin_opt(w: &Weights, x: &T, name: &str) -> Result<T, String> { lin(w, x, name, w.tensor(&format!("{name}.bias")).is_ok()) }

/// conv2d of x [c, h*w] (rows = channels) -> ([o, ho*wo], ho, wo); bias added if the file has one.
pub fn conv(w: &Weights, x: &T, h: usize, wd: usize, name: &str, stride: usize, pad: usize) -> Result<(T, usize, usize), String> {
    let t = w.tensor(&format!("{name}.weight"))?;
    let s = t.dims();
    let (o, c, kh, kw) = (s[0] as usize, s[1] as usize, s[2] as usize, s[3] as usize);
    if c != x.rows { return Err(format!("{name}: expects {c} channels, got {}", x.rows)); }
    let (wb, ty) = w.raw(&format!("{name}.weight"))?;
    let b = w.f32(&format!("{name}.bias")).ok();
    #[cfg(feature = "cuda")]
    if ffi::weight_on_gpu(wb) {
        let r = (|| {
            let xd = on_dev(x)?;
            let bd = match &b { Some(b) => Some(up(b)?), None => None };
            ffi::dev_conv2d(xd.get(), c, h, wd, wb, ty, bd.as_ref(), o, kh, kw, stride, pad)
        })();
        if let Some((y, ho, wo)) = r { return Ok((T::dev(y, o, ho * wo), ho, wo)); }
    }
    let xh = x.host_ref()?;
    let mut out = Vec::new();
    let (ho, wo) = ffi::conv2d(&xh, c, h, wd, wb, ty, b.as_deref(), o, kh, kw, stride, pad, &mut out);
    Ok((T::host(out, o, ho * wo), ho, wo))
}

/// GroupNorm over x [c, hw] with per-channel affine.
pub fn groupnorm(x: &mut T, groups: usize, g: &[f32], b: &[f32], eps: f32) -> Result<(), String> {
    let (c, hw) = (x.rows, x.cols);
    #[cfg(feature = "cuda")]
    if let Data::D(d) = &x.data { if (|| ffi::dev_groupnorm(d, c, hw, groups, &up(g)?, &up(b)?, eps).then_some(()))().is_some() { return Ok(()); } }
    make_host(x)?;
    ffi::groupnorm(x.hv(), c, hw, groups, g, b, eps);
    Ok(())
}

/// LayerNorm over the rows of x with optional affine.
pub fn layernorm(x: &mut T, g: Option<&[f32]>, b: Option<&[f32]>, eps: f32) -> Result<(), String> {
    let (m, n) = (x.rows, x.cols);
    #[cfg(feature = "cuda")]
    if let Data::D(d) = &x.data {
        let r = (|| {
            let gd = match g { Some(g) => Some(up(g)?), None => None };
            let bd = match b { Some(b) => Some(up(b)?), None => None };
            ffi::dev_layernorm(d, m, n, gd.as_ref(), bd.as_ref(), eps).then_some(())
        })();
        if r.is_some() { return Ok(()); }
    }
    make_host(x)?;
    ffi::layernorm(x.hv(), m, n, g, b, eps);
    Ok(())
}

pub fn silu(x: &mut T) -> Result<(), String> {
    #[cfg(feature = "cuda")]
    if let Data::D(d) = &x.data { if ffi::dev_silu(d) { return Ok(()); } }
    make_host(x)?;
    ffi::silu(x.hv());
    Ok(())
}

pub fn gelu_erf(x: &mut T) -> Result<(), String> {
    #[cfg(feature = "cuda")]
    if let Data::D(d) = &x.data { if ffi::dev_gelu_erf(d) { return Ok(()); } }
    make_host(x)?;
    ffi::gelu_erf(x.hv());
    Ok(())
}

/// x += y (same shape).
pub fn add(x: &mut T, y: &T) -> Result<(), String> {
    #[cfg(feature = "cuda")]
    if let Data::D(d) = &x.data { if (|| { let yd = on_dev(y)?; ffi::dev_add(d, yd.get()).then_some(()) })().is_some() { return Ok(()); } }
    make_host(x)?;
    let yh = y.host_ref()?;
    for (a, b) in x.hv().iter_mut().zip(yh.iter()) { *a += b; }
    Ok(())
}

/// x *= y (same shape).
pub fn mul(x: &mut T, y: &T) -> Result<(), String> {
    #[cfg(feature = "cuda")]
    if let Data::D(d) = &x.data { if (|| { let yd = on_dev(y)?; ffi::dev_mul(d, yd.get()).then_some(()) })().is_some() { return Ok(()); } }
    make_host(x)?;
    let yh = y.host_ref()?;
    for (a, b) in x.hv().iter_mut().zip(yh.iter()) { *a *= b; }
    Ok(())
}

/// x[i, :] += s[i] (a per-channel value on [c, hw]).
pub fn add_row_scalars(x: &mut T, s: &[f32]) -> Result<(), String> {
    let (r, c) = (x.rows, x.cols);
    #[cfg(feature = "cuda")]
    if let Data::D(d) = &x.data { if (|| ffi::dev_add_row_scalars(d, &up(s)?, r, c).then_some(()))().is_some() { return Ok(()); } }
    make_host(x)?;
    let v = x.hv();
    for i in 0..r { for e in &mut v[i * c..(i + 1) * c] { *e += s[i]; } }
    Ok(())
}

/// [r, c] -> [c, r].
pub fn transpose(x: &T) -> Result<T, String> {
    let (r, c) = (x.rows, x.cols);
    #[cfg(feature = "cuda")]
    if let Data::D(d) = &x.data { if let Some(y) = ffi::dev_transpose(d, r, c) { return Ok(T::dev(y, c, r)); } }
    let xh = x.host_ref()?;
    let mut t = vec![0f32; r * c];
    for i in 0..r { for j in 0..c { t[j * r + i] = xh[i * c + j]; } }
    Ok(T::host(t, c, r))
}

/// Nearest 2x upsample of x [ch, h*w] -> [ch, 2h*2w].
pub fn upsample2x(x: &T, h: usize, w: usize) -> Result<T, String> {
    let ch = x.rows;
    #[cfg(feature = "cuda")]
    if let Data::D(d) = &x.data { if let Some(y) = ffi::dev_upsample2x(d, ch, h, w) { return Ok(T::dev(y, ch, 4 * h * w)); } }
    let xh = x.host_ref()?;
    let (h2, w2) = (h * 2, w * 2);
    let mut d = vec![0f32; ch * h2 * w2];
    for c in 0..ch { for y in 0..h2 { for xx in 0..w2 { d[(c * h2 + y) * w2 + xx] = xh[(c * h + y / 2) * w + xx / 2]; } } }
    Ok(T::host(d, ch, h2 * w2))
}

/// Multi-head attention with separate q [nq, d], k [nk, d], v [nk, d] -> [nq, d].
pub fn attention3(q: &T, k: &T, v: &T, heads: usize, hd: usize, scale: f32) -> Result<T, String> {
    let (nq, nk, d) = (q.rows, k.rows, heads * hd);
    #[cfg(feature = "cuda")]
    if q.is_dev() || k.is_dev() {
        let r = (|| { let (qd, kd, vd) = (on_dev(q)?, on_dev(k)?, on_dev(v)?); ffi::dev_attention3(qd.get(), kd.get(), vd.get(), nq, nk, heads, hd, scale) })();
        if let Some(o) = r { return Ok(T::dev(o, nq, d)); }
    }
    let (qh, kh, vh) = (q.host_ref()?, k.host_ref()?, v.host_ref()?);
    let mut o = vec![0f32; nq * d];
    ffi::attention_mh(&qh, &kh, &vh, &mut o, nq, nk, heads, hd, d, d, d, d, scale);
    Ok(T::host(o, nq, d))
}

#[cfg(all(test, feature = "cuda"))]
mod gpu_vs_cpu {
    //! Every device op against its CPU twin on random data (run: cargo test --features cuda -- --test-threads=1).
    use super::*;
    fn rnd(n: usize, seed: u64) -> Vec<f32> { crate::sd::gaussian_noise(n, seed) }
    fn dev(v: &[f32], r: usize, c: usize) -> T { T::dev(ffi::dev_upload(v).unwrap(), r, c) }
    fn close(name: &str, gpu: T, cpu: T) {
        assert_eq!((gpu.rows, gpu.cols), (cpu.rows, cpu.cols), "{name}: shape");
        assert!(gpu.is_dev(), "{name}: GPU op fell back to the host");
        let (g, c) = (gpu.into_host().unwrap(), cpu.into_host().unwrap());
        let err = g.iter().zip(&c).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max) / c.iter().map(|v| v.abs()).fold(1e-9f32, f32::max);
        println!("{name}: max rel err {err:.2e}");
        assert!(err < 1e-4, "{name}: GPU differs from CPU, max rel err {err:.2e}");
    }
    #[test]
    fn each_op_matches_the_cpu() {
        let (r, c) = (37, 3 * 256);
        let x = rnd(r * c, 1);
        let (sh, sc, gate) = (rnd(c, 2), rnd(c, 3), rnd(c, 4));
        close("modulate", modulate(&dev(&x, r, c), &sh, &sc).unwrap(), modulate(&T::host(x.clone(), r, c), &sh, &sc).unwrap());
        let y = rnd(r * c, 5);
        let (mut a, mut b) = (dev(&x, r, c), T::host(x.clone(), r, c));
        add_gated(&mut a, &gate, &dev(&y, r, c)).unwrap(); add_gated(&mut b, &gate, &T::host(y.clone(), r, c)).unwrap();
        close("add_gated", a, b);
        let (mut a, mut b) = (dev(&x, r, c), T::host(x.clone(), r, c));
        gelu_tanh(&mut a).unwrap(); gelu_tanh(&mut b).unwrap();
        close("gelu_tanh", a, b);
        close("rows", rows(&dev(&x, r, c), 5, 20).unwrap(), rows(&T::host(x.clone(), r, c), 5, 20).unwrap());
        close("cols", cols(&dev(&x, r, c), 100, 600).unwrap(), cols(&T::host(x.clone(), r, c), 100, 600).unwrap());
        close("cat_rows", cat_rows(dev(&x, r, c), &dev(&y, r, c)).unwrap(), cat_rows(T::host(x.clone(), r, c), &T::host(y.clone(), r, c)).unwrap());
        let z = rnd(r * 40, 6);
        close("cat_cols", cat_cols(&dev(&x, r, c), &dev(&z, r, 40)).unwrap(), cat_cols(&T::host(x.clone(), r, c), &T::host(z.clone(), r, 40)).unwrap());
        // fused q|k|v with d = 2 heads x 128
        let (heads, hd) = (2, 128);
        let d = heads * hd;
        let (qs, ks) = (rnd(hd, 7), rnd(hd, 8));
        let tab = Rope::new(rnd((r + 3) * hd / 2, 9), rnd((r + 3) * hd / 2, 10));
        let (mut a, mut b) = (dev(&x, r, c), T::host(x.clone(), r, c));
        qk_norm_rope(&mut a, d, heads, hd, &qs, &ks, &tab, 3).unwrap(); qk_norm_rope(&mut b, d, heads, hd, &qs, &ks, &tab, 3).unwrap();
        close("qk_norm_rope", a, b);
        close("attention", attention(&dev(&x, r, c), 0, d, 2 * d, heads, hd, 0.088).unwrap(),
              attention(&T::host(x.clone(), r, c), 0, d, 2 * d, heads, hd, 0.088).unwrap());
        // SD UNet / VAE ops: images as [channels = 64, pixels = 12 x 12]
        let (ch, hh, ww) = (64, 12, 12);
        let im = rnd(ch * hh * ww, 11);
        let (g, b, sc) = (rnd(ch, 12), rnd(ch, 13), rnd(ch, 14));
        let (mut a, mut bb) = (dev(&im, ch, hh * ww), T::host(im.clone(), ch, hh * ww));
        groupnorm(&mut a, 32, &g, &b, 1e-5).unwrap(); groupnorm(&mut bb, 32, &g, &b, 1e-5).unwrap(); close("groupnorm", a, bb);
        let (lg, lb) = (rnd(c, 15), rnd(c, 16));
        let (mut a, mut bb) = (dev(&x, r, c), T::host(x.clone(), r, c));
        layernorm(&mut a, Some(&lg), Some(&lb), 1e-5).unwrap(); layernorm(&mut bb, Some(&lg), Some(&lb), 1e-5).unwrap(); close("layernorm", a, bb);
        let (mut a, mut bb) = (dev(&x, r, c), T::host(x.clone(), r, c)); silu(&mut a).unwrap(); silu(&mut bb).unwrap(); close("silu", a, bb);
        let (mut a, mut bb) = (dev(&x, r, c), T::host(x.clone(), r, c)); gelu_erf(&mut a).unwrap(); gelu_erf(&mut bb).unwrap(); close("gelu_erf", a, bb);
        let (mut a, mut bb) = (dev(&x, r, c), T::host(x.clone(), r, c)); add(&mut a, &dev(&y, r, c)).unwrap(); add(&mut bb, &T::host(y.clone(), r, c)).unwrap(); close("add", a, bb);
        let (mut a, mut bb) = (dev(&x, r, c), T::host(x.clone(), r, c)); mul(&mut a, &dev(&y, r, c)).unwrap(); mul(&mut bb, &T::host(y.clone(), r, c)).unwrap(); close("mul", a, bb);
        let (mut a, mut bb) = (dev(&im, ch, hh * ww), T::host(im.clone(), ch, hh * ww));
        add_row_scalars(&mut a, &sc).unwrap(); add_row_scalars(&mut bb, &sc).unwrap(); close("add_row_scalars", a, bb);
        close("transpose", transpose(&dev(&x, r, c)).unwrap(), transpose(&T::host(x.clone(), r, c)).unwrap());
        close("upsample2x", upsample2x(&dev(&im, ch, hh * ww), hh, ww).unwrap(), upsample2x(&T::host(im.clone(), ch, hh * ww), hh, ww).unwrap());
        let (kq, kk, kv) = (rnd(50 * d, 17), rnd(9 * d, 18), rnd(9 * d, 19));
        close("attention3", attention3(&dev(&kq, 50, d), &dev(&kk, 9, d), &dev(&kv, 9, d), heads, hd, 0.1).unwrap(),
              attention3(&T::host(kq.clone(), 50, d), &T::host(kk.clone(), 9, d), &T::host(kv.clone(), 9, d), heads, hd, 0.1).unwrap());
    }
}
