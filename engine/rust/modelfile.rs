//! Model-file reader: GGUF (v2/v3) and safetensors. Safe Rust only: model files come from the internet
//! and are untrusted input. Every length, count and offset is validated against the file size.
#![forbid(unsafe_code)]

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

pub const MAX_DIMS: usize = 8;
const MAX_NAME: u64 = 4096;
const MAX_STR: u64 = 16 << 20;
const MAX_TENSORS: u64 = 1_000_000;
const MAX_KV: u64 = 1_000_000;
const MAX_JSON: u64 = 256 << 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format { Gguf, Safetensors }

#[derive(Debug, Clone)]
pub struct Tensor {
    pub name: String,
    pub dtype: String,           // "Q4_K", "F16", "BF16", ... as named by the file format
    pub ggml_type: Option<u32>,  // GGUF only
    pub shape: Vec<u64>,         // as stored: GGUF ne[] (innermost first); safetensors outermost first
    pub offset: u64,             // absolute byte offset in the file
    pub nbytes: u64,
}

impl Tensor {
    /// Shape with the outermost dimension first, for both formats (GGUF stores the innermost first).
    pub fn dims(&self) -> Vec<u64> { if self.ggml_type.is_some() { self.shape.iter().rev().copied().collect() } else { self.shape.clone() } }
}

#[derive(Debug)]
#[allow(dead_code)] // version, n_kv, alignment, file_size, arch: read by the planner (P1.2) and covered by tests
pub struct ModelFile {
    pub format: Format,
    pub version: u32,
    pub n_kv: u64,
    pub alignment: u64,
    pub data_offset: u64,
    pub file_size: u64,
    pub arch: String,
    pub tensors: Vec<Tensor>,
}

/// ggml type id -> (name, elements per block, bytes per block). From the GGUF specification.
pub fn ggml_type_info(t: u32) -> Option<(&'static str, u64, u64)> {
    Some(match t {
        0 => ("F32", 1, 4), 1 => ("F16", 1, 2), 2 => ("Q4_0", 32, 18), 3 => ("Q4_1", 32, 20),
        6 => ("Q5_0", 32, 22), 7 => ("Q5_1", 32, 24), 8 => ("Q8_0", 32, 34), 9 => ("Q8_1", 32, 36),
        10 => ("Q2_K", 256, 84), 11 => ("Q3_K", 256, 110), 12 => ("Q4_K", 256, 144), 13 => ("Q5_K", 256, 176),
        14 => ("Q6_K", 256, 210), 15 => ("Q8_K", 256, 292), 16 => ("IQ2_XXS", 256, 66), 17 => ("IQ2_XS", 256, 74),
        18 => ("IQ3_XXS", 256, 98), 19 => ("IQ1_S", 256, 50), 20 => ("IQ4_NL", 32, 18), 21 => ("IQ3_S", 256, 110),
        22 => ("IQ2_S", 256, 82), 23 => ("IQ4_XS", 256, 136), 24 => ("I8", 1, 1), 25 => ("I16", 1, 2),
        26 => ("I32", 1, 4), 27 => ("I64", 1, 8), 28 => ("F64", 1, 8), 29 => ("IQ1_M", 256, 56),
        30 => ("BF16", 1, 2), 34 => ("TQ1_0", 256, 54), 35 => ("TQ2_0", 256, 66), 39 => ("MXFP4", 32, 17),
        40 => ("NVFP4", 64, 36), 41 => ("Q1_0", 128, 18), 42 => ("Q2_0", 64, 18), 43 => ("F8_E4M3", 1, 1),
        _ => return None,
    })
}

fn st_elem_size(dt: &str) -> Option<u64> {
    Some(match dt {
        "F64" | "I64" | "U64" => 8, "F32" | "I32" | "U32" => 4, "F16" | "BF16" | "I16" | "U16" => 2,
        "I8" | "U8" | "BOOL" | "F8_E4M3" | "F8_E5M2" => 1,
        _ => return None,
    })
}

type Res<T> = Result<T, String>;

/// Bounds-checked reader: never reads or seeks past the declared file size.
struct Rd<R> { r: R, size: u64, pos: u64 }

impl<R: Read + Seek> Rd<R> {
    fn bytes(&mut self, n: u64) -> Res<Vec<u8>> {
        if n > self.size - self.pos { return Err("truncated file".into()); }
        let mut v = vec![0u8; n as usize];
        self.r.read_exact(&mut v).map_err(|e| e.to_string())?;
        self.pos += n;
        Ok(v)
    }
    fn u32(&mut self) -> Res<u32> { let b = self.bytes(4)?; Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]])) }
    fn u64(&mut self) -> Res<u64> { let b = self.bytes(8)?; Ok(u64::from_le_bytes(b.try_into().unwrap())) }
    fn skip(&mut self, n: u64) -> Res<()> {
        if n > self.size - self.pos { return Err("truncated file".into()); }
        self.r.seek(SeekFrom::Current(n as i64)).map_err(|e| e.to_string())?;
        self.pos += n;
        Ok(())
    }
    /// GGUF string: u64 length + bytes.
    fn string(&mut self, max: u64) -> Res<String> {
        let n = self.u64()?;
        if n > max { return Err(format!("string length {n} exceeds limit")); }
        Ok(String::from_utf8_lossy(&self.bytes(n)?).into_owned())
    }
    fn skip_string(&mut self, max: u64) -> Res<()> {
        let n = self.u64()?;
        if n > max { return Err(format!("string length {n} exceeds limit")); }
        self.skip(n)
    }
}

const GGUF_SCALAR: [u64; 13] = [1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8]; // by value type 0..12 (8 string, 9 array)

fn gguf_skip_value<R: Read + Seek>(r: &mut Rd<R>, t: u32, depth: u32) -> Res<()> {
    if t > 12 || depth > 2 { return Err(format!("bad GGUF value type {t}")); }
    match t {
        8 => r.skip_string(MAX_STR),
        9 => {
            let et = r.u32()?;
            let n = r.u64()?;
            if et > 12 { return Err("bad GGUF array type".into()); }
            if et != 8 && et != 9 {
                let b = n.checked_mul(GGUF_SCALAR[et as usize]).ok_or("array size overflow")?;
                return r.skip(b);
            }
            for _ in 0..n { gguf_skip_value(r, et, depth + 1)?; }
            Ok(())
        }
        _ => r.skip(GGUF_SCALAR[t as usize]),
    }
}

fn parse_gguf<R: Read + Seek>(r: &mut Rd<R>) -> Res<ModelFile> {
    let version = r.u32()?;
    if !(2..=3).contains(&version) { return Err(format!("unsupported GGUF version {version}")); }
    let nt = r.u64()?;
    let n_kv = r.u64()?;
    if nt > MAX_TENSORS || n_kv > MAX_KV { return Err("bad GGUF counts".into()); }
    let (mut alignment, mut arch) = (32u64, String::new());
    for _ in 0..n_kv {
        let key = r.string(MAX_NAME)?;
        let t = r.u32()?;
        if key == "general.alignment" && t == 4 {
            alignment = r.u32()? as u64;
            if alignment == 0 || !alignment.is_power_of_two() { return Err("bad alignment".into()); }
        } else if key == "general.architecture" && t == 8 {
            arch = r.string(MAX_NAME)?;
        } else {
            gguf_skip_value(r, t, 0)?;
        }
    }
    let mut tensors = Vec::with_capacity(nt.min(65536) as usize);
    for i in 0..nt {
        let name = r.string(MAX_NAME)?;
        let nd = r.u32()? as usize;
        if !(1..=MAX_DIMS).contains(&nd) { return Err(format!("bad tensor info #{i}")); }
        let mut shape = Vec::with_capacity(nd);
        let mut ne: u64 = 1;
        for _ in 0..nd {
            let v = r.u64()?;
            if v > i64::MAX as u64 { return Err(format!("bad shape in {name}")); }
            ne = ne.checked_mul(v).ok_or_else(|| format!("bad shape in {name}"))?;
            shape.push(v);
        }
        let t = r.u32()?;
        let offset = r.u64()?;
        let (tn, blk, bytes) = ggml_type_info(t).ok_or_else(|| format!("unknown ggml type {t} in {name}"))?;
        if shape[0] % blk != 0 { return Err(format!("{name}: row {} not a multiple of block {blk}", shape[0])); }
        let nbytes = (ne / blk).checked_mul(bytes).ok_or_else(|| format!("size overflow in {name}"))?;
        tensors.push(Tensor { name, dtype: tn.into(), ggml_type: Some(t), shape, offset, nbytes });
    }
    let data_offset = r.pos.div_ceil(alignment) * alignment;
    for t in &mut tensors {
        if t.offset % alignment != 0 { return Err(format!("{}: unaligned offset", t.name)); }
        if t.offset > r.size.saturating_sub(data_offset) { return Err(format!("{}: offset past end of file", t.name)); }
        t.offset += data_offset;
        if t.nbytes > r.size - t.offset { return Err(format!("{}: data past end of file", t.name)); }
    }
    Ok(ModelFile { format: Format::Gguf, version, n_kv, alignment, data_offset, file_size: r.size, arch, tensors })
}

// ---------------- minimal JSON (what a safetensors header needs), safe and bounded ----------------
struct Js<'a> { s: &'a [u8], p: usize }

impl<'a> Js<'a> {
    fn ws(&mut self) { while self.p < self.s.len() && matches!(self.s[self.p], b' ' | b'\n' | b'\r' | b'\t') { self.p += 1; } }
    fn eat(&mut self, c: u8) -> bool { self.ws(); if self.s.get(self.p) == Some(&c) { self.p += 1; true } else { false } }
    fn next(&mut self) -> Res<u8> { let c = *self.s.get(self.p).ok_or("unexpected end of JSON")?; self.p += 1; Ok(c) }
    fn hex4(&mut self) -> Res<u32> {
        let mut v = 0u32;
        for _ in 0..4 { v = v * 16 + (self.next()? as char).to_digit(16).ok_or("bad \\u escape")?; }
        Ok(v)
    }
    fn string(&mut self) -> Res<String> {
        if !self.eat(b'"') { return Err("expected string".into()); }
        let mut out: Vec<u8> = Vec::new();
        loop {
            let c = self.next()?;
            match c {
                b'"' => break,
                b'\\' => match self.next()? {
                    b'n' => out.push(b'\n'), b't' => out.push(b'\t'), b'r' => out.push(b'\r'),
                    b'b' => out.push(8), b'f' => out.push(12),
                    b'u' => {
                        let mut cp = self.hex4()?;
                        if (0xD800..0xDC00).contains(&cp) && self.s[self.p..].starts_with(b"\\u") {
                            let save = self.p;
                            self.p += 2;
                            match self.hex4() {
                                Ok(lo) if (0xDC00..0xE000).contains(&lo) => cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00),
                                _ => self.p = save,
                            }
                        }
                        let ch = char::from_u32(cp).unwrap_or('\u{FFFD}');
                        let mut b = [0u8; 4];
                        out.extend_from_slice(ch.encode_utf8(&mut b).as_bytes());
                    }
                    e => out.push(e),
                },
                _ => out.push(c),
            }
            if out.len() as u64 > MAX_STR { return Err("JSON string too long".into()); }
        }
        Ok(String::from_utf8_lossy(&out).into_owned())
    }
    fn u64(&mut self) -> Res<u64> {
        self.ws();
        let start = self.p;
        let mut v: u64 = 0;
        while let Some(d) = self.s.get(self.p).filter(|c| c.is_ascii_digit()) {
            v = v.checked_mul(10).and_then(|v| v.checked_add((d - b'0') as u64)).ok_or("number overflow")?;
            self.p += 1;
        }
        if self.p == start { return Err("expected number".into()); }
        Ok(v)
    }
    fn skip(&mut self, depth: u32) -> Res<()> {
        self.ws();
        if depth > 64 { return Err("JSON nested too deeply".into()); }
        match *self.s.get(self.p).ok_or("unexpected end of JSON")? {
            b'"' => { self.string()?; }
            c @ (b'{' | b'[') => {
                let close = if c == b'{' { b'}' } else { b']' };
                self.p += 1;
                if self.eat(close) { return Ok(()); }
                loop {
                    if c == b'{' { self.string()?; if !self.eat(b':') { return Err("expected ':'".into()); } }
                    self.skip(depth + 1)?;
                    if !self.eat(b',') { break; }
                }
                if !self.eat(close) { return Err("unterminated JSON container".into()); }
            }
            _ => while self.p < self.s.len() && !matches!(self.s[self.p], b',' | b'}' | b']') { self.p += 1; },
        }
        Ok(())
    }
}

fn parse_safetensors<R: Read + Seek>(r: &mut Rd<R>, hlen: u64) -> Res<ModelFile> {
    if hlen > MAX_JSON || hlen > r.size - 8 { return Err("bad safetensors header length".into()); }
    let buf = r.bytes(hlen)?;
    let data_offset = 8 + hlen;
    let mut j = Js { s: &buf, p: 0 };
    let (mut tensors, mut n_kv) = (Vec::new(), 0u64);
    if !j.eat(b'{') { return Err("header is not a JSON object".into()); }
    if !j.eat(b'}') {
        loop {
            let name = j.string()?;
            if !j.eat(b':') { return Err("bad header key".into()); }
            if name == "__metadata__" {
                if !j.eat(b'{') { return Err("bad __metadata__".into()); }
                if !j.eat(b'}') {
                    loop {
                        j.skip(1)?;
                        if !j.eat(b':') { return Err("bad __metadata__".into()); }
                        j.skip(1)?;
                        n_kv += 1;
                        if !j.eat(b',') { break; }
                    }
                    if !j.eat(b'}') { return Err("bad __metadata__".into()); }
                }
            } else {
                if tensors.len() as u64 >= MAX_TENSORS { return Err("too many tensors".into()); }
                let (mut dtype, mut shape, mut offs) = (None, None, None);
                if !j.eat(b'{') { return Err(format!("bad entry for {name}")); }
                loop {
                    let k = j.string()?;
                    if !j.eat(b':') { return Err(format!("bad field in {name}")); }
                    match k.as_str() {
                        "dtype" => dtype = Some(j.string()?),
                        "shape" => {
                            if !j.eat(b'[') { return Err(format!("malformed entry for {name}")); }
                            let mut v = Vec::new();
                            if !j.eat(b']') {
                                loop {
                                    let d = j.u64()?;
                                    if v.len() >= MAX_DIMS || d > i64::MAX as u64 { return Err(format!("malformed entry for {name}")); }
                                    v.push(d);
                                    if !j.eat(b',') { break; }
                                }
                                if !j.eat(b']') { return Err(format!("malformed entry for {name}")); }
                            }
                            shape = Some(v);
                        }
                        "data_offsets" => {
                            if !j.eat(b'[') { return Err(format!("malformed entry for {name}")); }
                            let b = j.u64()?;
                            if !j.eat(b',') { return Err(format!("malformed entry for {name}")); }
                            let e = j.u64()?;
                            if !j.eat(b']') { return Err(format!("malformed entry for {name}")); }
                            offs = Some((b, e));
                        }
                        _ => j.skip(1)?,
                    }
                    if !j.eat(b',') { break; }
                }
                if !j.eat(b'}') { return Err(format!("malformed entry for {name}")); }
                let (dtype, shape, (b, e)) = match (dtype, shape, offs) {
                    (Some(d), Some(s), Some(o)) => (d, s, o),
                    _ => return Err(format!("malformed entry for {name}")),
                };
                if e < b || e > r.size - data_offset { return Err(format!("{name}: data past end of file")); }
                let mut n: u64 = 1;
                for d in &shape { n = n.checked_mul(*d).ok_or_else(|| format!("malformed entry for {name}"))?; }
                if let Some(es) = st_elem_size(&dtype) {
                    if n.checked_mul(es) != Some(e - b) { return Err(format!("{name}: size does not match dtype x shape")); }
                }
                tensors.push(Tensor { name, dtype, ggml_type: None, shape, offset: data_offset + b, nbytes: e - b });
            }
            if !j.eat(b',') { break; }
        }
        if !j.eat(b'}') { return Err("malformed safetensors header".into()); }
    }
    Ok(ModelFile { format: Format::Safetensors, version: 0, n_kv, alignment: 1, data_offset, file_size: r.size, arch: String::new(), tensors })
}

/// Parse a model file from any seekable reader of known size.
pub fn parse<R: Read + Seek>(reader: R, size: u64) -> Res<ModelFile> {
    let mut r = Rd { r: reader, size, pos: 0 };
    let magic = r.bytes(8).map_err(|_| "file too short".to_string())?;
    if &magic[..4] == b"GGUF" {
        r.r.seek(SeekFrom::Start(4)).map_err(|e| e.to_string())?;
        r.pos = 4;
        parse_gguf(&mut r)
    } else {
        parse_safetensors(&mut r, u64::from_le_bytes(magic.try_into().unwrap()))
    }
}

pub fn open(path: &Path) -> Res<ModelFile> {
    let f = File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let size = f.metadata().map_err(|e| e.to_string())?.len();
    parse(BufReader::with_capacity(1 << 20, f), size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn gstr(v: &mut Vec<u8>, s: &str) { v.extend((s.len() as u64).to_le_bytes()); v.extend(s.as_bytes()); }
    fn u32(v: &mut Vec<u8>, x: u32) { v.extend(x.to_le_bytes()); }
    fn u64(v: &mut Vec<u8>, x: u64) { v.extend(x.to_le_bytes()); }
    fn pf(b: &[u8]) -> Res<ModelFile> { parse(Cursor::new(b.to_vec()), b.len() as u64) }

    /// GGUF v3: 2 KV (arch string + u32 array), 2 tensors (F32 [4,2], Q8_0 [32,1]).
    fn make_gguf(q8_off: u64) -> Vec<u8> {
        let mut v = b"GGUF".to_vec();
        u32(&mut v, 3); u64(&mut v, 2); u64(&mut v, 2);
        gstr(&mut v, "general.architecture"); u32(&mut v, 8); gstr(&mut v, "testarch");
        gstr(&mut v, "some.array"); u32(&mut v, 9); u32(&mut v, 4); u64(&mut v, 3); for i in 1..=3 { u32(&mut v, i); }
        gstr(&mut v, "w.f32"); u32(&mut v, 2); u64(&mut v, 4); u64(&mut v, 2); u32(&mut v, 0); u64(&mut v, 0);
        gstr(&mut v, "w.q8"); u32(&mut v, 1); u64(&mut v, 32); u32(&mut v, 8); u64(&mut v, q8_off);
        while v.len() % 32 != 0 { v.push(0); }
        for i in 0..8 { v.extend((i as f32).to_le_bytes()); }
        for i in 0..34 { v.push(100 + i as u8); }
        v
    }

    #[test]
    fn gguf_valid() {
        let b = make_gguf(32);
        let m = pf(&b).unwrap();
        assert_eq!((m.tensors.len(), m.arch.as_str(), m.n_kv), (2, "testarch", 2));
        assert_eq!((m.tensors[0].dtype.as_str(), m.tensors[0].nbytes, m.tensors[0].offset), ("F32", 32, m.data_offset));
        assert_eq!(m.tensors[0].shape, vec![4, 2]);
        assert_eq!((m.tensors[1].dtype.as_str(), m.tensors[1].nbytes, m.tensors[1].offset), ("Q8_0", 34, m.data_offset + 32));
        assert_eq!(b[m.tensors[1].offset as usize], 100);
    }

    #[test]
    fn gguf_hostile() {
        assert!(pf(&make_gguf(4096)).is_err(), "offset past end");
        let mut b = make_gguf(32); b.truncate(b.len() - 10); assert!(pf(&b).is_err(), "truncated data");
        let mut b = make_gguf(32); b[8..16].copy_from_slice(&u64::MAX.to_le_bytes()); assert!(pf(&b).is_err(), "huge tensor count");
        let mut b = make_gguf(32); b.truncate(40); assert!(pf(&b).is_err(), "truncated metadata");
        let mut b = b"GGUF".to_vec(); u32(&mut b, 3); u64(&mut b, 0); u64(&mut b, 1); u64(&mut b, 1 << 40);
        assert!(pf(&b).is_err(), "giant key length");
    }

    fn make_st(json: &str, data: usize) -> Vec<u8> {
        let mut v = (json.len() as u64).to_le_bytes().to_vec();
        v.extend(json.as_bytes());
        v.extend((0..data).map(|i| i as u8));
        v
    }

    #[test]
    fn safetensors_valid() {
        let js = r#"{"__metadata__":{"format":"pt","url":"http://a:b"},"aé":{"dtype":"F16","shape":[2,3],"data_offsets":[0,12]},"b":{"shape":[],"dtype":"F32","data_offsets":[12,16]}}"#;
        let m = pf(&make_st(js, 16)).unwrap();
        assert_eq!((m.tensors.len(), m.n_kv, m.data_offset), (2, 2, 8 + js.len() as u64));
        assert_eq!((m.tensors[0].name.as_str(), m.tensors[0].dtype.as_str(), m.tensors[0].nbytes), ("aé", "F16", 12));
        assert_eq!(m.tensors[0].shape, vec![2, 3]);
        assert_eq!((m.tensors[1].shape.len(), m.tensors[1].offset, m.tensors[1].nbytes), (0, m.data_offset + 12, 4));
    }

    #[test]
    fn safetensors_hostile() {
        assert!(pf(&make_st(r#"{"a":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}"#, 8)).is_err(), "data past end");
        assert!(pf(&make_st(r#"{"a":{"dtype":"F32","shape":[5],"data_offsets":[0,16]}}"#, 16)).is_err(), "size mismatch");
        assert!(pf(&make_st(r#"{"a":{"dtype":"F32","shape":[4],"data_offsets":[0,16]"#, 16)).is_err(), "unterminated");
        assert!(pf(&make_st(r#"{"a":{"dtype":"F32","shape":[99999999999999999999],"data_offsets":[0,16]}}"#, 16)).is_err(), "overflow");
        assert!(pf(&(1u64 << 50).to_le_bytes()).is_err(), "giant header length");
    }
}
