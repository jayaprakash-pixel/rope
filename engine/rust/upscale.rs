//! Upscaler add-on (P2.4.5): Real-ESRGAN x4 (RRDBNet) on tiles.
//!
//! The net: conv_first -> 23 RRDB (3 residual dense blocks, each 5 convs with dense channel concatenation, residual x0.2)
//! -> conv_body, skip add -> nearest x2, conv_up1, leaky ReLU 0.2 -> nearest x2, conv_up2, leaky ReLU -> conv_hr, leaky ReLU -> conv_last.
//! Input and output in [0, 1]. The image is cut into tiles (TILE px, PAD px of context on each side that are cut off again), so
//! the memory is one tile's activations, not the image's; the image border uses the net's own zero padding, like a whole-image run.
//! Runs on the host with the conv kernel of the platform (the add-on file is not placed on the GPU).
#![forbid(unsafe_code)]

use crate::ffi;
use crate::sd::{self, Act};
use crate::weights::Weights;

pub const SCALE: usize = 4;
/// Input tile (core) and context pad in input pixels. A tile of 128+2*16 needs about 0.35 GiB of activations.
pub const TILE: usize = 128;
pub const PAD: usize = 16;
/// Allocator cache while this stage runs: enough to reuse the big conv buffers between tiles.
const CACHE_CAP: usize = 256 << 20;

fn conv(w: &Weights, x: &Act, name: &str) -> Result<Act, String> {
    let t = w.tensor(&format!("{name}.weight"))?;
    let s = t.dims();
    let (o, c, kh, kw) = (s[0] as usize, s[1] as usize, s[2] as usize, s[3] as usize);
    if c != x.c { return Err(format!("{name}: expects {c} channels, got {}", x.c)); }
    let (wb, ty) = w.raw(&format!("{name}.weight"))?;
    let b = w.f32(&format!("{name}.bias")).ok();
    let mut out = Vec::new();
    let (h, ww) = ffi::conv2d(&x.d, c, x.h, x.w, wb, ty, b.as_deref(), o, kh, kw, 1, kh / 2, &mut out);
    Ok(Act { c: o, h, w: ww, d: out })
}

fn lrelu(x: &mut Act) { for v in &mut x.d { if *v < 0.0 { *v *= 0.2; } } }

/// Channel concatenation (channel-major layout: append).
fn cat(parts: &[&Act]) -> Act {
    let mut d = Vec::with_capacity(parts.iter().map(|p| p.d.len()).sum());
    for p in parts { d.extend_from_slice(&p.d); }
    Act { c: parts.iter().map(|p| p.c).sum(), h: parts[0].h, w: parts[0].w, d }
}

fn up2(x: &Act) -> Act {
    let (h2, w2) = (x.h * 2, x.w * 2);
    let mut d = vec![0f32; x.c * h2 * w2];
    for c in 0..x.c { for y in 0..h2 { for xx in 0..w2 { d[(c * h2 + y) * w2 + xx] = x.d[(c * x.h + y / 2) * x.w + xx / 2]; } } }
    Act { c: x.c, h: h2, w: w2, d }
}

fn rdb(w: &Weights, x: &Act, name: &str) -> Result<Act, String> {
    let mut x1 = conv(w, x, &format!("{name}.conv1"))?; lrelu(&mut x1);
    let mut x2 = conv(w, &cat(&[x, &x1]), &format!("{name}.conv2"))?; lrelu(&mut x2);
    let mut x3 = conv(w, &cat(&[x, &x1, &x2]), &format!("{name}.conv3"))?; lrelu(&mut x3);
    let mut x4 = conv(w, &cat(&[x, &x1, &x2, &x3]), &format!("{name}.conv4"))?; lrelu(&mut x4);
    let mut x5 = conv(w, &cat(&[x, &x1, &x2, &x3, &x4]), &format!("{name}.conv5"))?;
    for (a, b) in x5.d.iter_mut().zip(&x.d) { *a = *a * 0.2 + b; }
    Ok(x5)
}

/// One tile through the net: x [3, h, w] in [0, 1] -> [3, 4h, 4w].
pub fn tile_forward(w: &Weights, x: &Act) -> Result<Act, String> {
    let fea = conv(w, x, "conv_first")?;
    let n = (0..).take_while(|i| w.tensor(&format!("body.{i}.rdb1.conv1.weight")).is_ok()).count();
    let mut t = Act { c: fea.c, h: fea.h, w: fea.w, d: fea.d.clone() };
    for i in 0..n {
        let r = rdb(w, &rdb(w, &rdb(w, &t, &format!("body.{i}.rdb1"))?, &format!("body.{i}.rdb2"))?, &format!("body.{i}.rdb3"))?;
        for (a, b) in t.d.iter_mut().zip(&r.d) { *a = *b * 0.2 + *a; }
    }
    let mut f = conv(w, &t, "conv_body")?;
    for (a, b) in f.d.iter_mut().zip(&fea.d) { *a += b; }
    let mut f = conv(w, &up2(&f), "conv_up1")?; lrelu(&mut f);
    let mut f = conv(w, &up2(&f), "conv_up2")?; lrelu(&mut f);
    let mut f = conv(w, &f, "conv_hr")?; lrelu(&mut f);
    conv(w, &f, "conv_last")
}

/// Upscale an image [3, h, w] in [-1, 1] (the pipeline's convention) 4x, tile by tile.
pub fn run(w: &Weights, img: &Act) -> Result<Act, String> {
    if img.c != 3 { return Err("upscaler expects a 3-channel image".into()); }
    let (h, wd) = (img.h, img.w);
    let (oh, ow) = (h * SCALE, wd * SCALE);
    // This stage owns the memory now: the last stage's cached buffers go back, and the cache is small (tiles at the image edge have
    // other sizes than the inner ones, and a big cache would fill up with blocks of every size)
    crate::alloc::trim();
    let old_cap = crate::alloc::cap();
    crate::alloc::set_cap(CACHE_CAP);
    let mut out = vec![0f32; 3 * oh * ow];
    let (ty, tx) = (h.div_ceil(TILE), wd.div_ceil(TILE));
    for j in 0..ty { for i in 0..tx {
        sd::check_cancel()?;
        sd::progress("upscale", j * tx + i, ty * tx);
        let (y0, x0) = (j * TILE, i * TILE);
        let (y1, x1) = ((y0 + TILE).min(h), (x0 + TILE).min(wd));
        let (cy0, cx0) = (y0.saturating_sub(PAD), x0.saturating_sub(PAD));
        let (cy1, cx1) = ((y1 + PAD).min(h), (x1 + PAD).min(wd));
        let (th, tw) = (cy1 - cy0, cx1 - cx0);
        let mut d = vec![0f32; 3 * th * tw];
        for c in 0..3 { for y in 0..th { for x in 0..tw { d[(c * th + y) * tw + x] = (img.d[(c * h + cy0 + y) * wd + cx0 + x] + 1.0) * 0.5; } } }
        let r = tile_forward(w, &Act { c: 3, h: th, w: tw, d })?;
        let (rw, rh) = (tw * SCALE, th * SCALE);
        for c in 0..3 { for y in y0..y1 { for x in x0..x1 {
            for dy in 0..SCALE { for dx in 0..SCALE {
                let (sy, sx) = ((y - cy0) * SCALE + dy, (x - cx0) * SCALE + dx);
                let v = r.d[(c * rh + sy) * rw + sx];
                out[(c * oh + y * SCALE + dy) * ow + x * SCALE + dx] = v * 2.0 - 1.0;
            } }
        } } }
    } }
    sd::progress("upscale", ty * tx, ty * tx);
    crate::alloc::trim();
    crate::alloc::set_cap(old_cap);
    Ok(Act { c: 3, h: oh, w: ow, d: out })
}
