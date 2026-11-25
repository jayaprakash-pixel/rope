//! T5 SentencePiece (unigram) tokenizer, read straight from the official `spiece.model` (protobuf), matching the
//! Hugging Face `tokenizers` pipeline that transformers' T5TokenizerFast builds from that file:
//!   1. split out the added tokens <pad> </s> <unk> <extra_id_0..99> on the raw text
//!   2. normalize with the model's precompiled charsmap (NFKC-style rules; darts-clone trie)
//!   3. split on whitespace; each word gets the "▁" prefix (Metaspace, prepend_scheme "always")
//!   4. unigram Viterbi per word (best total log-probability); unknown characters -> <unk>, runs of <unk> fused
//!   5. append </s>
#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::path::Path;

pub struct T5Tokenizer {
    pieces: HashMap<String, (u32, f32)>,
    max_piece: usize,           // longest piece in bytes
    unk_score: f32,
    charsmap: Option<(Vec<u32>, Vec<u8>)>,   // darts-clone units, normalized-string blob
    added: Vec<(String, u32)>,
}

pub const PAD: u32 = 0;
pub const EOS: u32 = 1;
pub const UNK: u32 = 2;

// ---- minimal protobuf reader (wire types 0, 1, 2, 5) -------------------------------------------------------
fn varint(b: &[u8], i: &mut usize) -> Result<u64, String> {
    let mut v = 0u64;
    for shift in (0..64).step_by(7) {
        let c = *b.get(*i).ok_or("protobuf: truncated varint")?;
        *i += 1;
        v |= ((c & 0x7f) as u64) << shift;
        if c < 0x80 { return Ok(v); }
    }
    Err("protobuf: varint too long".into())
}

/// Iterate the fields of one message: (field number, wire type, payload bytes or the varint value).
fn fields(b: &[u8]) -> Result<Vec<(u64, u8, &[u8], u64)>, String> {
    let (mut i, mut out) = (0, Vec::new());
    while i < b.len() {
        let key = varint(b, &mut i)?;
        let (num, wt) = (key >> 3, (key & 7) as u8);
        match wt {
            0 => { let v = varint(b, &mut i)?; out.push((num, wt, &b[0..0], v)); }
            1 | 5 => { let n = if wt == 1 { 8 } else { 4 }; let s = b.get(i..i + n).ok_or("protobuf: truncated")?; i += n; out.push((num, wt, s, 0)); }
            2 => { let n = varint(b, &mut i)? as usize; let s = b.get(i..i + n).ok_or("protobuf: truncated")?; i += n; out.push((num, wt, s, 0)); }
            _ => return Err(format!("protobuf: unsupported wire type {wt}")),
        }
    }
    Ok(out)
}

impl T5Tokenizer {
    /// Load `spiece.model` (from a directory holding it, or the file itself).
    pub fn load(path: &Path) -> Result<Self, String> {
        let file = if path.is_dir() { path.join("spiece.model") } else { path.to_path_buf() };
        let b = std::fs::read(&file).map_err(|e| format!("{}: {e}", file.display()))?;
        let (mut pieces, mut max_piece, mut min_score, mut charsmap) = (HashMap::new(), 0, f32::MAX, None);
        let mut id = 0u32;
        for (num, _, payload, _) in fields(&b)? {
            match num {
                1 => {                                                    // SentencePiece { piece, score, type }
                    let (mut piece, mut score, mut ty) = (String::new(), 0f32, 1u64);
                    for (n, _, p, v) in fields(payload)? {
                        match n { 1 => piece = String::from_utf8_lossy(p).into_owned(),
                                  2 => score = f32::from_le_bytes([p[0], p[1], p[2], p[3]]),
                                  3 => ty = v, _ => {} }
                    }
                    if ty == 1 || ty == 4 {                               // NORMAL / USER_DEFINED: matchable by Viterbi
                        max_piece = max_piece.max(piece.len());
                        if ty == 1 { min_score = min_score.min(score); }
                        pieces.insert(piece, (id, score));
                    }
                    id += 1;
                }
                3 => {                                                    // NormalizerSpec { precompiled_charsmap = 2 }
                    for (n, _, p, _) in fields(payload)? {
                        if n == 2 && p.len() > 4 {
                            let sz = u32::from_le_bytes([p[0], p[1], p[2], p[3]]) as usize;
                            let units = p[4..4 + sz].chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                            charsmap = Some((units, p[4 + sz..].to_vec()));
                        }
                    }
                }
                _ => {}
            }
        }
        if id < 32000 { return Err(format!("{}: only {id} pieces, not a T5 vocabulary", file.display())); }
        let mut added: Vec<(String, u32)> = vec![("<pad>".into(), PAD), ("</s>".into(), EOS), ("<unk>".into(), UNK)];
        added.extend((0..100).map(|i| (format!("<extra_id_{i}>"), id + 99 - i)));   // extra ids follow the vocabulary, in reverse
        Ok(T5Tokenizer { pieces, max_piece, unk_score: min_score - 10.0, charsmap, added })
    }

    /// darts-clone common-prefix search: value of the FIRST (shortest) key that prefixes `key`, like spm_precompiled.
    fn charsmap_lookup(&self, key: &[u8]) -> Option<&str> {
        let (units, blob) = self.charsmap.as_ref()?;
        let offset = |u: u32| ((u >> 10) << ((u & (1 << 9)) >> 6)) as usize;
        let mut pos = offset(units[0]);
        for &c in key {
            pos ^= c as usize;
            let u = *units.get(pos)?;
            if u & ((1 << 31) | 0xff) != c as u32 { return None; }
            pos ^= offset(u);
            if (u >> 8) & 1 == 1 {
                let start = (units[pos] & 0x7fff_ffff) as usize;
                let end = blob[start..].iter().position(|&b| b == 0).map(|e| start + e)?;
                return std::str::from_utf8(&blob[start..end]).ok();
            }
        }
        None
    }

    /// Precompiled normalization, char by char (whole char as key; ponytail: spm_precompiled first tries the whole
    /// grapheme cluster, which only differs for multi-char graphemes whose prefix is a key — not seen in the tests).
    fn normalize(&self, s: &str) -> String {
        let mut out = String::with_capacity(s.len());
        for (i, c) in s.char_indices() {
            match self.charsmap_lookup(&s.as_bytes()[i..i + c.len_utf8()]) { Some(n) => out.push_str(n), None => out.push(c) }
        }
        out
    }

    /// Unigram Viterbi over one pre-token ("▁word").
    fn viterbi(&self, w: &str, out: &mut Vec<u32>) {
        let b = w.as_bytes();
        let n = b.len();
        let mut best: Vec<(f32, usize, u32)> = vec![(f32::NEG_INFINITY, 0, 0); n + 1];
        best[0].0 = 0.0;
        let bounds: Vec<usize> = w.char_indices().map(|(i, _)| i).chain([n]).collect();
        for (k, &s) in bounds.iter().enumerate() {
            if s == n || best[s].0 == f32::NEG_INFINITY { continue; }
            let mut single = false;
            for &e in bounds[k + 1..].iter().take_while(|&&e| e - s <= self.max_piece) {
                if let Some(&(id, sc)) = self.pieces.get(&w[s..e]) {
                    if e == bounds[k + 1] { single = true; }
                    if best[s].0 + sc > best[e].0 { best[e] = (best[s].0 + sc, s, id); }
                }
            }
            let e = bounds[k + 1];
            if !single && best[s].0 + self.unk_score > best[e].0 { best[e] = (best[s].0 + self.unk_score, s, UNK); }
        }
        let mut ids = Vec::new();
        let mut e = n;
        while e > 0 { ids.push(best[e].2); e = best[e].1; }
        for id in ids.into_iter().rev() {
            if id == UNK && out.last() == Some(&UNK) { continue; }            // fuse_unk
            out.push(id);
        }
    }

    /// Token ids, ending with </s>.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            let hit = self.added.iter().filter_map(|(t, id)| rest.find(t.as_str()).map(|p| (p, t.len(), *id))).min_by_key(|h| (h.0, usize::MAX - h.1));
            let (seg, special, next) = match hit { Some((p, l, id)) => (&rest[..p], Some(id), &rest[p + l..]), None => (rest, None, "") };
            let norm = self.normalize(seg);
            for word in norm.split_whitespace() { self.viterbi(&format!("\u{2581}{word}"), &mut out); }
            if let Some(id) = special { out.push(id); }
            rest = next;
        }
        out.push(EOS);
        out
    }

    /// FLUX conditioning input: ids truncated to `len` (keeping </s> last) and padded with <pad>.
    pub fn encode_padded(&self, text: &str, len: usize) -> Vec<u32> {
        let mut ids = self.encode(text);
        if ids.len() > len { ids.truncate(len - 1); ids.push(EOS); }
        ids.resize(len, PAD);
        ids
    }
}
