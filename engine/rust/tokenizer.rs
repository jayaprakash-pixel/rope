//! CLIP byte-level BPE tokenizer (SD 1.x/2.x, OpenCLIP), matching the Hugging Face `tokenizers` pipeline that
//! transformers uses for CLIPTokenizer:
//!   1. if the pad token is an added token (SD 2.x: "!", id 0), split it out of the raw text first (SD 1.x pads with <|endoftext|>)
//!   2. normalize each remaining segment: NFC (not applied, see note), collapse whitespace runs to " ", lowercase
//!   3. split out the normalized special tokens <|startoftext|> / <|endoftext|>
//!   4. CLIP split: 's|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+ (whitespace removed)
//!   5. byte-level BPE per piece, "</w>" on the last symbol.
//! Note: NFC normalization is not applied (Rust std has no NFC tables); prompts typed on keyboards are NFC already.
#![forbid(unsafe_code)]

use crate::json;
use crate::unicode_tables::{is_letter, is_number};
use std::collections::HashMap;
use std::path::Path;

pub struct ClipTokenizer {
    vocab: HashMap<String, u32>,
    ranks: HashMap<(String, String), u32>,
    byte_char: [char; 256],
    pub bos: u32,
    pub eos: u32,
    pub pad: u32,
    /// Pad token that is an added token outside the BPE vocabulary rules (SD 2.x: "!"): split on it first.
    added_pad: Option<String>,
}

/// GPT-2/CLIP byte -> printable unicode char table.
fn bytes_to_unicode() -> [char; 256] {
    let mut table = ['\0'; 256];
    let printable = |b: u32| (0x21..=0x7E).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b);
    let mut n = 0;
    for b in 0..256u32 {
        table[b as usize] = if printable(b) { char::from_u32(b).unwrap() } else { n += 1; char::from_u32(255 + n).unwrap() };
    }
    table
}

/// Normalizer: collapse every whitespace run to one space, then lowercase.
fn normalize(seg: &str) -> String {
    let mut s = String::with_capacity(seg.len());
    let mut in_ws = false;
    for c in seg.chars() {
        if c.is_whitespace() { if !in_ws { s.push(' '); } in_ws = true; } else { s.push(c); in_ws = false; }
    }
    s.to_lowercase()
}

/// CLIP split rule; whitespace between matches is removed.
fn clip_split(s: &str) -> Vec<&str> {
    const CONTRACTIONS: [&str; 7] = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"];
    let other = |c: char| !c.is_whitespace() && !is_letter(c) && !is_number(c);
    let mut out = Vec::new();
    let mut i = 0;
    while i < s.len() {
        let rest = &s[i..];
        let c = rest.chars().next().unwrap();
        if let Some(k) = CONTRACTIONS.iter().find(|k| rest.starts_with(**k)) { out.push(&rest[..k.len()]); i += k.len(); continue; }
        let run = |pred: &dyn Fn(char) -> bool| rest.char_indices().find(|&(_, ch)| !pred(ch)).map_or(rest.len(), |(j, _)| j);
        let n = if is_letter(c) { run(&is_letter) }
                else if is_number(c) { c.len_utf8() }
                else if c.is_whitespace() { i += c.len_utf8(); continue; }
                else { run(&other) };
        out.push(&rest[..n]);
        i += n;
    }
    out
}

impl ClipTokenizer {
    /// dir holds vocab.json and merges.txt (Hugging Face CLIP tokenizer layout).
    pub fn load(dir: &Path) -> Result<Self, String> {
        let vtxt = std::fs::read_to_string(dir.join("vocab.json")).map_err(|e| format!("vocab.json: {e}"))?;
        let vjson = json::parse(&vtxt)?;
        let mut vocab = HashMap::new();
        for (k, v) in vjson.as_obj().ok_or("vocab.json is not an object")? {
            vocab.insert(k.clone(), v.as_f64().ok_or("vocab id is not a number")? as u32);
        }
        let mtxt = std::fs::read_to_string(dir.join("merges.txt")).map_err(|e| format!("merges.txt: {e}"))?;
        let mut ranks = HashMap::new();
        for (r, line) in mtxt.trim().lines().skip(1).take(49152 - 256 - 2).enumerate() {
            let (a, b) = line.split_once(' ').ok_or("bad merges line")?;
            ranks.insert((a.to_string(), b.to_string()), r as u32);
        }
        let id = |t: &str| vocab.get(t).copied().ok_or_else(|| format!("token {t} missing from vocab"));
        // Pad token from special_tokens_map.json (string or {"content": ..}); CLIP's default is <|endoftext|> (SD 1.x).
        let pad_text = std::fs::read_to_string(dir.join("special_tokens_map.json")).ok()
            .and_then(|t| json::parse(&t).ok())
            .and_then(|j| j.get("pad_token").and_then(|p| p.as_str().or_else(|| p.get("content").and_then(json::Value::as_str))).map(str::to_string))
            .unwrap_or_else(|| "<|endoftext|>".into());
        let added_pad = (!["<|startoftext|>", "<|endoftext|>"].contains(&pad_text.as_str())).then(|| pad_text.clone());
        Ok(ClipTokenizer { bos: id("<|startoftext|>")?, eos: id("<|endoftext|>")?, pad: id(&pad_text)?, added_pad, vocab, ranks, byte_char: bytes_to_unicode() })
    }

    fn bpe(&self, word: &str, out: &mut Vec<u32>) {
        let mut parts: Vec<String> = word.bytes().map(|b| self.byte_char[b as usize].to_string()).collect();
        if let Some(last) = parts.last_mut() { last.push_str("</w>"); }
        while parts.len() > 1 {
            let best = (0..parts.len() - 1)
                .filter_map(|i| self.ranks.get(&(parts[i].clone(), parts[i + 1].clone())).map(|&r| (r, i)))
                .min();
            let Some((_, bi)) = best else { break };
            let (a, b) = (parts[bi].clone(), parts[bi + 1].clone());
            let mut merged = Vec::with_capacity(parts.len());
            let mut i = 0;
            while i < parts.len() {
                if i + 1 < parts.len() && parts[i] == a && parts[i + 1] == b { merged.push(format!("{a}{b}")); i += 2; }
                else { merged.push(parts[i].clone()); i += 1; }
            }
            parts = merged;
        }
        out.extend(parts.iter().map(|p| self.vocab.get(p).copied().unwrap_or(self.eos)));
    }

    /// Token ids without BOS/EOS framing.
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let parts: Vec<&str> = match &self.added_pad { Some(t) => text.split(t.as_str()).collect(), None => vec![text] };
        for (k, raw) in parts.into_iter().enumerate() {
            if k > 0 { out.push(self.pad); }                     // each added pad token (SD 2.x "!") stands alone
            let norm = normalize(raw);
            let mut rest = norm.as_str();
            while !rest.is_empty() {                             // normalized special tokens
                let hit = ["<|startoftext|>", "<|endoftext|>"].iter().filter_map(|t| rest.find(t).map(|p| (p, *t))).min();
                let (seg, special, next) = match hit {
                    Some((p, t)) => (&rest[..p], Some(t), &rest[p + t.len()..]),
                    None => (rest, None, ""),
                };
                for piece in clip_split(seg) { self.bpe(piece, &mut out); }
                if let Some(t) = special { out.push(if t == "<|startoftext|>" { self.bos } else { self.eos }); }
                rest = next;
            }
        }
        out
    }

    /// SD conditioning input: [BOS] + up to 75 tokens + [EOS], padded with `pad` to 77.
    pub fn encode_77(&self, text: &str) -> [u32; 77] {
        let mut ids = [self.pad; 77];
        ids[0] = self.bos;
        let toks = self.tokenize(text);
        let n = toks.len().min(75);
        ids[1..1 + n].copy_from_slice(&toks[..n]);
        ids[1 + n] = self.eos;
        ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn split_rule() {
        assert_eq!(clip_split("it's 2024, ok"), vec!["it", "'s", "2", "0", "2", "4", ",", "ok"]);
        assert_eq!(normalize("  A\tB\n\nC "), " a b c ");
        assert!(!is_letter('\u{093F}') && is_letter('\u{0915}'), "Devanagari vowel sign is a mark, KA is a letter");
    }
    #[test]
    fn byte_table_is_bijective() {
        let t = bytes_to_unicode();
        let mut seen = std::collections::HashSet::new();
        assert!(t.iter().all(|c| seen.insert(*c)));
        assert_eq!(t[b'a' as usize], 'a');
        assert_eq!(t[b' ' as usize] as u32, 0x120);
    }
}
