//! Small JSON parser (safe Rust, bounded depth) for config files: tokenizer vocab, model catalog, probe output.
#![forbid(unsafe_code)]

#[derive(Debug, Clone, PartialEq)]
pub enum Value { Null, Bool(bool), Num(f64), Str(String), Arr(Vec<Value>), Obj(Vec<(String, Value)>) }

#[allow(dead_code)] // accessors used by the planner and model catalog (P1.2, P1.3)
impl Value {
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self { Value::Obj(v) => v.iter().find(|(k, _)| k == key).map(|(_, v)| v), _ => None }
    }
    pub fn as_str(&self) -> Option<&str> { if let Value::Str(s) = self { Some(s) } else { None } }
    pub fn as_f64(&self) -> Option<f64> { if let Value::Num(n) = self { Some(*n) } else { None } }
    pub fn as_bool(&self) -> Option<bool> { if let Value::Bool(b) = self { Some(*b) } else { None } }
    pub fn as_arr(&self) -> Option<&[Value]> { if let Value::Arr(a) = self { Some(a) } else { None } }
    pub fn as_obj(&self) -> Option<&[(String, Value)]> { if let Value::Obj(o) = self { Some(o) } else { None } }
}

/// `s` as a JSON string literal (quotes included), for building JSON output by hand.
pub fn quote(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""), '\\' => o.push_str("\\\\"), '\n' => o.push_str("\\n"), '\r' => o.push_str("\\r"), '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

pub fn parse(s: &str) -> Result<Value, String> {
    let mut p = P { s: s.as_bytes(), i: 0 };
    let v = p.value(0)?;
    p.ws();
    if p.i != p.s.len() { return Err(format!("trailing data at byte {}", p.i)); }
    Ok(v)
}

struct P<'a> { s: &'a [u8], i: usize }

impl P<'_> {
    fn ws(&mut self) { while self.i < self.s.len() && matches!(self.s[self.i], b' ' | b'\n' | b'\r' | b'\t') { self.i += 1; } }
    fn err<T>(&self, m: &str) -> Result<T, String> { Err(format!("{m} at byte {}", self.i)) }
    fn lit(&mut self, w: &str, v: Value) -> Result<Value, String> {
        if self.s[self.i..].starts_with(w.as_bytes()) { self.i += w.len(); Ok(v) } else { self.err("bad literal") }
    }
    fn value(&mut self, depth: u32) -> Result<Value, String> {
        if depth > 128 { return self.err("nested too deeply"); }
        self.ws();
        match self.s.get(self.i).copied() {
            Some(b'{') => {
                self.i += 1;
                let mut o = Vec::new();
                self.ws();
                if self.s.get(self.i) == Some(&b'}') { self.i += 1; return Ok(Value::Obj(o)); }
                loop {
                    self.ws();
                    let k = self.string()?;
                    self.ws();
                    if self.s.get(self.i) != Some(&b':') { return self.err("expected ':'"); }
                    self.i += 1;
                    o.push((k, self.value(depth + 1)?));
                    self.ws();
                    match self.s.get(self.i) { Some(b',') => self.i += 1, Some(b'}') => { self.i += 1; return Ok(Value::Obj(o)); } _ => return self.err("expected ',' or '}'") }
                }
            }
            Some(b'[') => {
                self.i += 1;
                let mut a = Vec::new();
                self.ws();
                if self.s.get(self.i) == Some(&b']') { self.i += 1; return Ok(Value::Arr(a)); }
                loop {
                    a.push(self.value(depth + 1)?);
                    self.ws();
                    match self.s.get(self.i) { Some(b',') => self.i += 1, Some(b']') => { self.i += 1; return Ok(Value::Arr(a)); } _ => return self.err("expected ',' or ']'") }
                }
            }
            Some(b'"') => Ok(Value::Str(self.string()?)),
            Some(b't') => self.lit("true", Value::Bool(true)),
            Some(b'f') => self.lit("false", Value::Bool(false)),
            Some(b'n') => self.lit("null", Value::Null),
            Some(c) if c == b'-' || c.is_ascii_digit() => {
                let st = self.i;
                while self.i < self.s.len() && matches!(self.s[self.i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') { self.i += 1; }
                std::str::from_utf8(&self.s[st..self.i]).ok().and_then(|t| t.parse().ok()).map(Value::Num).ok_or_else(|| format!("bad number at byte {st}"))
            }
            _ => self.err("unexpected character"),
        }
    }
    fn hex4(&mut self) -> Result<u32, String> {
        let h = self.s.get(self.i..self.i + 4).ok_or("truncated \\u escape")?;
        self.i += 4;
        u32::from_str_radix(std::str::from_utf8(h).map_err(|_| "bad \\u escape")?, 16).map_err(|_| "bad \\u escape".into())
    }
    fn string(&mut self) -> Result<String, String> {
        if self.s.get(self.i) != Some(&b'"') { return self.err("expected string"); }
        self.i += 1;
        let mut out = Vec::new();
        loop {
            let c = *self.s.get(self.i).ok_or("unterminated string")?;
            self.i += 1;
            match c {
                b'"' => return Ok(String::from_utf8_lossy(&out).into_owned()),
                b'\\' => {
                    let e = *self.s.get(self.i).ok_or("unterminated escape")?;
                    self.i += 1;
                    match e {
                        b'n' => out.push(b'\n'), b't' => out.push(b'\t'), b'r' => out.push(b'\r'), b'b' => out.push(8), b'f' => out.push(12),
                        b'u' => {
                            let mut cp = self.hex4()?;
                            if (0xD800..0xDC00).contains(&cp) && self.s[self.i..].starts_with(b"\\u") {
                                let save = self.i; self.i += 2;
                                match self.hex4() { Ok(lo) if (0xDC00..0xE000).contains(&lo) => cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00), _ => self.i = save }
                            }
                            let mut b = [0u8; 4];
                            out.extend_from_slice(char::from_u32(cp).unwrap_or('\u{FFFD}').encode_utf8(&mut b).as_bytes());
                        }
                        o => out.push(o),
                    }
                }
                _ => out.push(c),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_nested() {
        let v = parse(r#"{"a":[1,2.5,-3e2,true,null],"b":{"c":"xé😀\n"}}"#).unwrap();
        assert_eq!(v.get("a").unwrap().as_arr().unwrap()[2].as_f64(), Some(-300.0));
        assert_eq!(v.get("b").unwrap().get("c").unwrap().as_str(), Some("xé😀\n"));
    }
    #[test]
    fn rejects_bad() {
        for bad in [r#"{"a":1"#, r#"[1,]"#, r#"{"a" 1}"#, "tru", r#""\u12""#, "[1] x"] { assert!(parse(bad).is_err(), "{bad}"); }
        let deep = "[".repeat(200) + &"]".repeat(200);
        assert!(parse(&deep).is_err(), "depth limit");
    }
}
