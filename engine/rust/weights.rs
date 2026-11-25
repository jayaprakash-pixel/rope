//! Named tensors of one model file, read in place from a memory map (no copy of large matrices).
#![forbid(unsafe_code)]

use crate::mmap::Map;
use crate::modelfile::{self, ModelFile, Tensor};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;
use std::thread::JoinHandle;

/// `prefetchers`: background threads reading this file's pages; joined before the map is dropped, so no prefetch
/// can ever touch an unmapped page (a thread outliving the map crashed the process at exit).
pub struct Weights { map: Map, pub mf: ModelFile, index: HashMap<String, usize>, prefetchers: Mutex<Vec<JoinHandle<()>>> }

impl Drop for Weights {
    fn drop(&mut self) {
        // runs before the fields (and so the map) are dropped
        for h in self.prefetchers.get_mut().map(std::mem::take).unwrap_or_default() { let _ = h.join(); }
        // a later file may be mapped at these addresses: they must not stay in the GPU dispatch tables
        let b = self.map.bytes();
        crate::ffi::gpu_forget(b.as_ptr() as usize, b.as_ptr() as usize + b.len());
    }
}

/// GGUF type id for a tensor (safetensors dtypes mapped to the same ids).
pub fn type_id(t: &Tensor) -> Option<u32> {
    t.ggml_type.or(match t.dtype.as_str() { "F32" => Some(0), "F16" => Some(1), "BF16" => Some(30), _ => None })
}

impl Weights {
    pub fn open(path: &Path) -> Result<Self, String> {
        let mf = modelfile::open(path)?;
        let map = Map::open(path).map_err(|e| format!("map {}: {e}", path.display()))?;
        if map.bytes().len() as u64 != mf.file_size { return Err("file changed while opening".into()); }
        let index = mf.tensors.iter().enumerate().map(|(i, t)| (t.name.clone(), i)).collect();
        let w = Weights { map, mf, index, prefetchers: Mutex::new(Vec::new()) };
        let base = w.map.bytes().as_ptr() as usize;
        crate::ffi::gpu_register(&w.mf.tensors.iter().map(|t| (t.name.clone(), base + t.offset as usize, t.nbytes as usize)).collect::<Vec<_>>());
        Ok(w)
    }

    pub fn tensor(&self, name: &str) -> Result<&Tensor, String> {
        self.index.get(name).map(|&i| &self.mf.tensors[i]).ok_or_else(|| format!("tensor {name} not found"))
    }

    /// Raw bytes and GGUF type id of a tensor, straight from the map (bounds were checked by the reader).
    pub fn raw(&self, name: &str) -> Result<(&[u8], u32), String> {
        let t = self.tensor(name)?;
        let ty = type_id(t).ok_or_else(|| format!("{name}: unsupported dtype {}", t.dtype))?;
        Ok((&self.map.bytes()[t.offset as usize..(t.offset + t.nbytes) as usize], ty))
    }

    /// Start reading every tensor whose name starts with `prefix` into RAM, on a background thread, so the
    /// block runs from memory instead of page-faulting from disk (the next block is prefetched while this one runs).
    pub fn prefetch_async(&self, prefix: &str) {
        let base = self.map.bytes().as_ptr() as usize;
        let ranges: Vec<(usize, usize)> = self.mf.tensors.iter().filter(|t| t.name.starts_with(prefix))
            .map(|t| (base + t.offset as usize, t.nbytes as usize)).collect();
        if ranges.is_empty() { return; }
        crate::ffi::gpu_stage(ranges.clone());                         // GPU-streamed weights of the block: block-ahead ring
        let h = std::thread::spawn(move || { crate::ffi::prefetch(&ranges); crate::ffi::touch_pages(&ranges); });
        if let Ok(mut v) = self.prefetchers.lock() { v.retain(|h| !h.is_finished()); v.push(h); }
    }

    /// Address ranges of every tensor whose name starts with `prefix` ("" = all), e.g. for the GPU dispatch table.
    pub fn ranges(&self, prefix: &str) -> Vec<(usize, usize)> {
        let base = self.map.bytes().as_ptr() as usize;
        self.mf.tensors.iter().filter(|t| t.name.starts_with(prefix)).map(|t| (base + t.offset as usize, t.nbytes as usize)).collect()
    }

    /// A small tensor as f32 values (biases, norms, embeddings rows): copied and converted.
    pub fn f32(&self, name: &str) -> Result<Vec<f32>, String> {
        let (b, ty) = self.raw(name)?;
        let n: u64 = self.tensor(name)?.shape.iter().product::<u64>().max(1);
        match ty {
            0 => Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()),
            _ => Ok(crate::ffi::dequant_row(ty, b, n as usize)),
        }
    }

    /// Rows [r0, r0+count) of a 2-D f32 tensor (embedding lookup without converting the whole table).
    pub fn f32_rows(&self, name: &str, rows: &[u32], width: usize) -> Result<Vec<f32>, String> {
        // Any stored type (F32/F16/BF16/quantized): row size in bytes = tensor bytes / number of rows.
        let (b, ty) = self.raw(name)?;
        let elems: u64 = self.tensor(name)?.shape.iter().product();
        let row_bytes = b.len() / (elems as usize / width);
        let mut out = Vec::with_capacity(rows.len() * width);
        for &r in rows {
            let s = r as usize * row_bytes;
            let row = b.get(s..s + row_bytes).ok_or_else(|| format!("{name}: row {r} out of range"))?;
            if ty == 0 { out.extend(row.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))); }
            else { out.extend(crate::ffi::dequant_row(ty, row, width)); }
        }
        Ok(out)
    }
}
