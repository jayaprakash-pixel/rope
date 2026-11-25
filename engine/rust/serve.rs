//! `rope serve`: the local API the app talks to (P5.1.3). Standard library only, like the rest of the crate.
//!
//! - Binds 127.0.0.1 only. Requests whose Host is not this machine's loopback address are refused (DNS rebinding),
//!   and a POST from another web origin is refused (a web page cannot start jobs on the user's laptop).
//! - Each image runs in a child process (`rope image`, the same executable, ROPE_PROGRESS=1): on a 4 GB laptop all
//!   model memory returns to the OS after every image, and a failing model cannot take the server down.
//! - One job at a time. Progress comes from the child's `progress:` lines and reaches the app as server-sent events,
//!   with an ETA from the measured step time and the plan's decode estimate. Finished images go to the gallery folder
//!   (PNG + a JSON sidecar with prompt, model, seed and times).
#![forbid(unsafe_code)]

use crate::json;
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const MAX_HEAD: usize = 16 << 10;
const MAX_BODY: usize = 64 << 10;
const MAX_PROMPT: usize = 2000;
/// Version of the HTTP API (docs/HTTP_API.md). A route, field or meaning is never removed or changed inside one version; new fields may be added.
const API_VERSION: u32 = 1;

struct Job {
    id: String,
    events: Vec<String>,        // SSE "data:" payloads, replayable from any index
    done: bool,
    child: Option<Child>,
    stdin: Option<std::process::ChildStdin>,   // control channel into the job (pause, resume, prompt, cfg)
    png: PathBuf,
    preview_dir: PathBuf,       // live previews of this job (removed when it ends)
    meta: String,               // gallery sidecar JSON fields (without braces)
}

struct Server {
    port: u16,
    exe: PathBuf,
    models_dir: String,
    catalog: String,
    gallery: PathBuf,
    static_dir: Option<PathBuf>,
    dev_args: Vec<String>,      // device overrides for the planner and the jobs (--ram-gib X, --no-gpu): to try a smaller laptop
    origins: Vec<String>,       // web origins allowed to POST (the app itself, plus --dev-origin)
    job: Mutex<Option<Arc<Mutex<Job>>>>,
    cache: Mutex<HashMap<String, String>>,
}

/// rope serve [--port 4003] [--models-dir models] [--catalog catalog/models.json] [--gallery gallery] [--ram-gib X] [--no-gpu]
///            [--static app/dist] [--dev-origin http://127.0.0.1:3005]
pub fn serve(args: &[String]) -> Result<(), String> {
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let port: u16 = get("--port").and_then(|p| p.parse().ok()).unwrap_or(4003);
    let mut origins = vec![format!("http://127.0.0.1:{port}"), format!("http://localhost:{port}")];
    if let Some(o) = get("--dev-origin") { origins.push(o); }
    let s = Arc::new(Server {
        port,
        exe: std::env::current_exe().map_err(|e| e.to_string())?,
        models_dir: get("--models-dir").unwrap_or_else(|| "models".into()),
        catalog: get("--catalog").unwrap_or_else(|| "catalog/models.json".into()),
        gallery: PathBuf::from(get("--gallery").unwrap_or_else(|| "gallery".into())),
        static_dir: get("--static").map(PathBuf::from),
        dev_args: { let mut v = Vec::new(); if let Some(r) = get("--ram-gib") { v.extend(["--ram-gib".to_string(), r]); } if args.iter().any(|a| a == "--no-gpu") { v.push("--no-gpu".into()); } v },
        origins,
        job: Mutex::new(None),
        cache: Mutex::new(HashMap::new()),
    });
    std::fs::create_dir_all(&s.gallery).map_err(|e| format!("{}: {e}", s.gallery.display()))?;
    let l = TcpListener::bind(("127.0.0.1", port)).map_err(|e| format!("bind 127.0.0.1:{port}: {e}"))?;
    eprintln!("rope serve: http://127.0.0.1:{port}  (models {}, gallery {})", s.models_dir, s.gallery.display());
    for c in l.incoming() {
        let Ok(c) = c else { continue };
        let s = s.clone();
        std::thread::spawn(move || { let _ = handle(&s, c); });
    }
    Ok(())
}

struct Req { method: String, path: String, query: HashMap<String, String>, headers: HashMap<String, String>, body: Vec<u8> }

fn read_req(c: &mut TcpStream) -> Result<Req, String> {
    c.set_read_timeout(Some(Duration::from_secs(10))).ok();
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    let head_end = loop {
        if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") { break i; }
        if buf.len() > MAX_HEAD { return Err("header too large".into()); }
        let n = c.read(&mut tmp).map_err(|e| e.to_string())?;
        if n == 0 { return Err("closed".into()); }
        buf.extend_from_slice(&tmp[..n]);
    };
    let head = std::str::from_utf8(&buf[..head_end]).map_err(|_| "bad header")?;
    let mut lines = head.split("\r\n");
    let mut first = lines.next().unwrap_or("").split(' ');
    let (method, target) = (first.next().unwrap_or("").to_string(), first.next().unwrap_or("/").to_string());
    let headers: HashMap<String, String> = lines.filter_map(|l| l.split_once(':')).map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_string())).collect();
    let len: usize = headers.get("content-length").and_then(|v| v.parse().ok()).unwrap_or(0);
    if len > MAX_BODY { return Err("body too large".into()); }
    let mut body = buf[head_end + 4..].to_vec();
    while body.len() < len {
        let n = c.read(&mut tmp).map_err(|e| e.to_string())?;
        if n == 0 { break; }
        body.extend_from_slice(&tmp[..n]);
    }
    body.truncate(len);
    let (path, q) = target.split_once('?').unwrap_or((&target, ""));
    let query = q.split('&').filter_map(|kv| kv.split_once('=')).map(|(k, v)| (k.to_string(), unescape(v))).collect();
    Ok(Req { method, path: path.to_string(), query, headers, body })
}

fn unescape(s: &str) -> String {
    let b = s.as_bytes();
    let mut o = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'+' => o.push(b' '),
            b'%' if i + 2 < b.len() => {
                match u8::from_str_radix(std::str::from_utf8(&b[i + 1..i + 3]).unwrap_or("zz"), 16) { Ok(v) => { o.push(v); i += 2; } Err(_) => o.push(b'%') }
            }
            c => o.push(c),
        }
        i += 1;
    }
    String::from_utf8_lossy(&o).into_owned()
}

/// Content-Security-Policy of the app: everything from this server only (no CDN; no request can leave for a third party).
const CSP: &str = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; font-src 'self'; \
                   connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'";

fn send(c: &mut TcpStream, status: &str, ctype: &str, body: &[u8]) -> std::io::Result<()> {
    write!(c, "HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n\
               Content-Security-Policy: {CSP}\r\nReferrer-Policy: no-referrer\r\nConnection: close\r\n\r\n", body.len())?;
    c.write_all(body)
}
fn json_err(c: &mut TcpStream, status: &str, msg: &str) -> std::io::Result<()> {
    send(c, status, "application/json", format!("{{\"error\": {}}}", json::quote(msg)).as_bytes())
}

fn handle(s: &Server, mut c: TcpStream) -> std::io::Result<()> {
    let r = match read_req(&mut c) { Ok(r) => r, Err(e) => return json_err(&mut c, "400 Bad Request", &e) };
    // loopback Host only (a DNS name pointing at 127.0.0.1 must not reach this API)
    let host = r.headers.get("host").map(String::as_str).unwrap_or("");
    if host != format!("127.0.0.1:{}", s.port) && host != format!("localhost:{}", s.port) { return json_err(&mut c, "403 Forbidden", "bad host"); }
    if r.method == "POST" {
        if let Some(o) = r.headers.get("origin") { if !s.origins.iter().any(|a| a == o) { return json_err(&mut c, "403 Forbidden", "bad origin"); } }
    }
    let parts: Vec<&str> = r.path.trim_start_matches('/').split('/').collect();
    match (r.method.as_str(), parts.as_slice()) {
        ("GET", ["api", "health"]) => send(&mut c, "200 OK", "application/json", format!("{{\"ok\": true, \"api\": {API_VERSION}, \"version\": {}}}", json::quote(crate::VERSION)).as_bytes()),
        ("GET", ["api", "device"]) => cached(s, &mut c, "device", &["probe", "--disk-mb", "256"]),
        ("GET", ["api", "models"]) => {
            let q = r.query.get("quality").map(String::as_str).unwrap_or("balanced");
            if !["fast", "balanced", "best"].contains(&q) { return json_err(&mut c, "400 Bad Request", "quality must be fast|balanced|best"); }
            // optional: one size (a multiple of 64, 64..2048) and one catalog model, to refresh the chosen model's plan at that size
            let size = match r.query.get("size") { None => None, Some(v) => match v.parse::<u32>() { Ok(n) if n % 64 == 0 && (64..=2048).contains(&n) => Some(n), _ => return json_err(&mut c, "400 Bad Request", "size must be a multiple of 64 from 64 to 2048") } };
            let only = r.query.get("only").filter(|o| safe_name(o));
            let up = r.query.get("upscale").map(String::as_str) == Some("1");     // plan with the 4x upscaler stage (time, memory, ceiling)
            let key = format!("models-{q}-{}-{}-{up}", size.unwrap_or(0), only.map_or("", |o| o.as_str()));
            let mut args: Vec<&str> = vec!["suggest", "--json", "--quality", q, "--disk-mb", "0", "--models-dir", &s.models_dir, "--catalog", &s.catalog];
            let size_s = size.map(|n| n.to_string());
            if let Some(n) = &size_s { args.extend(["--size", n]); }
            if let Some(o) = only { args.extend(["--only", o]); }
            if up { args.push("--upscale"); }
            args.extend(s.dev_args.iter().map(String::as_str));
            cached(s, &mut c, &key, &args)
        }
        ("GET", ["api", "addons"]) => addons_list(s, &mut c),
        ("GET", ["api", "mine"]) => {
            let q = r.query.get("quality").map(String::as_str).filter(|q| ["fast", "balanced", "best"].contains(q)).unwrap_or("balanced");
            let items: Vec<String> = mine_list(s).iter().filter_map(|(id, path)| mine_check(s, id, path, q)).collect();
            send(&mut c, "200 OK", "application/json", format!("{{\"models\": [{}]}}", items.join(", ")).as_bytes())
        }
        ("POST", ["api", "mine"]) => mine_add(s, &mut c, &r.body),
        ("POST", ["api", "mine", "remove"]) => {
            let id = std::str::from_utf8(&r.body).ok().and_then(|b| json::parse(b).ok()).and_then(|v| v.get("id").and_then(json::Value::as_str).map(str::to_string)).unwrap_or_default();
            let mut l = mine_list(s);
            let n = l.len();
            l.retain(|(i, _)| *i != id);
            if l.len() == n { return json_err(&mut c, "404 Not Found", "no such model in the list"); }
            match mine_save(s, &l) { Ok(()) => send(&mut c, "200 OK", "application/json", b"{\"ok\": true}"), Err(e) => json_err(&mut c, "500 Internal Server Error", &e) }
        }
        ("POST", ["api", "jobs"]) => start_job(s, &mut c, &r.body),
        ("GET", ["api", "jobs", id, "events"]) => events(s, &mut c, id),
        ("GET", ["api", "jobs", id, "image"]) => {
            let j = s.job.lock().unwrap().clone();
            match j { Some(j) if j.lock().unwrap().id == *id => {
                let p = j.lock().unwrap().png.clone();
                match std::fs::read(&p) { Ok(b) => send(&mut c, "200 OK", "image/png", &b), Err(_) => json_err(&mut c, "404 Not Found", "no image yet") }
            } _ => json_err(&mut c, "404 Not Found", "no such job") }
        }
        ("GET", ["api", "jobs", id, "preview"]) => {
            let j = s.job.lock().unwrap().clone();
            let dir = j.and_then(|j| { let j = j.lock().unwrap(); (j.id == *id).then(|| j.preview_dir.clone()) });
            match dir.and_then(|d| std::fs::read(d.join("preview.png")).ok()) { Some(b) => send(&mut c, "200 OK", "image/png", &b), None => json_err(&mut c, "404 Not Found", "no preview yet") }
        }
        ("POST", ["api", "jobs", id, "control"]) => control(s, &mut c, id, &r.body),
        ("POST", ["api", "jobs", id, "cancel"]) => {
            let j = s.job.lock().unwrap().clone();
            if let Some(j) = j { let mut j = j.lock().unwrap(); if j.id == *id { if let Some(ch) = j.child.as_mut() { let _ = ch.kill(); } } }
            send(&mut c, "200 OK", "application/json", b"{\"ok\": true}")
        }
        ("GET", ["api", "gallery"]) => gallery_list(s, &mut c),
        ("GET", ["api", "gallery", name]) => {
            if !safe_name(name) || !name.ends_with(".png") { return json_err(&mut c, "400 Bad Request", "bad name"); }
            match std::fs::read(s.gallery.join(name)) { Ok(b) => send(&mut c, "200 OK", "image/png", &b), Err(_) => json_err(&mut c, "404 Not Found", "no such image") }
        }
        ("GET", _) if !r.path.starts_with("/api/") => static_file(s, &mut c, &r.path),
        _ => json_err(&mut c, "404 Not Found", "no such route"),
    }
}

/// Output of a rope sub-command (JSON on stdout), computed once per server run.
fn cached(s: &Server, c: &mut TcpStream, key: &str, args: &[&str]) -> std::io::Result<()> {
    if let Some(v) = s.cache.lock().unwrap().get(key).cloned() { return send(c, "200 OK", "application/json", v.as_bytes()); }
    match Command::new(&s.exe).args(args).stdin(Stdio::null()).stderr(Stdio::null()).output() {
        Ok(o) if o.status.success() => {
            let v = String::from_utf8_lossy(&o.stdout).trim().to_string();
            if json::parse(&v).is_err() { return json_err(c, "500 Internal Server Error", "sub-command gave no JSON"); }
            s.cache.lock().unwrap().insert(key.to_string(), v.clone());
            send(c, "200 OK", "application/json", v.as_bytes())
        }
        _ => json_err(c, "500 Internal Server Error", "sub-command failed"),
    }
}

fn safe_name(n: &str) -> bool { !n.is_empty() && n.len() < 128 && n.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b'.') && !n.contains("..") }

fn now_ms() -> u128 { SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0) }

/// POST /api/jobs {"model": catalog id, "prompt": "...", "quality": "fast|balanced|best", "seed": n, "size": n?, "steps": n?}
fn start_job(s: &Server, c: &mut TcpStream, body: &[u8]) -> std::io::Result<()> {
    let v = match std::str::from_utf8(body).ok().and_then(|b| json::parse(b).ok()) { Some(v) => v, None => return json_err(c, "400 Bad Request", "body must be JSON") };
    let st = |k: &str| v.get(k).and_then(json::Value::as_str).map(str::to_string);
    let num = |k: &str| v.get(k).and_then(json::Value::as_f64);
    let prompt = st("prompt").unwrap_or_default();
    if prompt.trim().is_empty() || prompt.chars().count() > MAX_PROMPT { return json_err(c, "400 Bad Request", "prompt must be 1..2000 characters"); }
    let quality = st("quality").unwrap_or_else(|| "balanced".into());
    if !["fast", "balanced", "best"].contains(&quality.as_str()) { return json_err(c, "400 Bad Request", "quality must be fast|balanced|best"); }
    let model = st("model").unwrap_or_default();
    // the model must be a catalog model whose file is on disk (the list the app shows): no arbitrary paths from the web
    let file = {
        let cat = std::fs::read_to_string(&s.catalog).ok().and_then(|t| json::parse(&t).ok());
        cat.as_ref().and_then(|c| c.get("models")).and_then(json::Value::as_arr).and_then(|ms| ms.iter().find(|m| m.get("id").and_then(json::Value::as_str) == Some(&model)))
            .and_then(|m| m.get("files")).and_then(json::Value::as_arr)
            .and_then(|fs| fs.iter().find(|f| f.get("role").and_then(json::Value::as_str) == Some("model")))
            .and_then(|f| f.get("name")).and_then(json::Value::as_str).map(str::to_string)
    };
    // or a model file the user added to "your own models" (its path is in models/mine.json, put there from this app)
    let mine = mine_list(s).into_iter().find(|(id, _)| *id == model).map(|(_, p)| std::path::PathBuf::from(p));
    let path = match (mine, file.filter(|f| safe_name(f))) {
        (Some(p), _) => p,
        (None, Some(f)) => Path::new(&s.models_dir).join(f),
        (None, None) => return json_err(c, "400 Bad Request", "unknown model"),
    };
    if !path.exists() { return json_err(c, "409 Conflict", "model file not downloaded"); }
    // add-ons: a file name inside models/addons, and its detected kind must match (no arbitrary paths from the web)
    let (mut lora, mut upscale) = (None, None);
    for (key, want, out) in [("lora", "lora", &mut lora), ("upscale", "esrgan", &mut upscale)] {
        let Some(name) = st(key).filter(|n| !n.is_empty()) else { continue };
        match addon_path(s, &name, want) { Ok(p) => *out = Some((name, p)), Err(e) => return json_err(c, "400 Bad Request", &e) }
    }
    let lora_scale = num("lora_scale").unwrap_or(1.0);
    if !(0.0..=2.0).contains(&lora_scale) { return json_err(c, "400 Bad Request", "lora_scale must be from 0 to 2"); }
    let seed = num("seed").map(|x| x.max(0.0) as u64).unwrap_or_else(|| (now_ms() % 1_000_000_007) as u64);
    let mut slot = s.job.lock().unwrap();
    if let Some(j) = slot.as_ref() { if !j.lock().unwrap().done { return json_err(c, "409 Conflict", "a job is already running"); } }
    let id = format!("{}", now_ms());
    let png = s.gallery.join(format!("{id}-{model}.png"));
    let preview_dir = s.gallery.join(format!(".preview-{id}"));
    let mut cmd = Command::new(&s.exe);
    cmd.args(["image", "--model"]).arg(&path).args(["--prompt", &prompt, "--quality", &quality, "--seed", &seed.to_string(),
              "--models-dir", &s.models_dir, "--catalog", &s.catalog, "--disk-mb", "0", "--preview-dir"]).arg(&preview_dir).arg("--out").arg(&png);
    for k in ["size", "steps"] { if let Some(x) = num(k) { cmd.args([format!("--{k}"), (x as u64).to_string()]); } }
    if let Some((_, p)) = &lora { cmd.arg("--lora").arg(p).args(["--lora-scale", &lora_scale.to_string()]); }
    if let Some((_, p)) = &upscale { cmd.arg("--upscale").arg(p); }
    cmd.args(&s.dev_args).arg("--control").env("ROPE_PROGRESS", "1").stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::piped());
    let mut child = match cmd.spawn() { Ok(ch) => ch, Err(e) => return json_err(c, "500 Internal Server Error", &e.to_string()) };
    let err = child.stderr.take().unwrap();
    let stdin = child.stdin.take();
    let addons = format!("{}{}", lora.as_ref().map_or(String::new(), |(n, _)| format!(", \"lora\": {}, \"lora_scale\": {lora_scale}", json::quote(n))),
                         upscale.as_ref().map_or(String::new(), |(n, _)| format!(", \"upscale\": {}", json::quote(n))));
    let meta = format!("\"id\": {}, \"model\": {}, \"prompt\": {}, \"quality\": {}, \"seed\": {seed}{addons}, \"image\": {}",
                       json::quote(&id), json::quote(&model), json::quote(&prompt), json::quote(&quality),
                       json::quote(&format!("/api/gallery/{}", png.file_name().unwrap().to_string_lossy())));
    let job = Arc::new(Mutex::new(Job { id: id.clone(), events: vec![format!("{{\"type\": \"start\", {meta}}}")], done: false, child: Some(child), stdin, png, preview_dir, meta }));
    *slot = Some(job.clone());
    drop(slot);
    std::thread::spawn(move || watch(job, err));
    send(c, "202 Accepted", "application/json", format!("{{\"id\": {}, \"seed\": {seed}}}", json::quote(&id)).as_bytes())
}

/// "Your own models": model files anywhere on this laptop that the user added by path (models/mine.json). The file
/// stays where it is: rope reads it in place and never deletes it. (id, absolute path); the id is a hash of the path.
fn mine_list(s: &Server) -> Vec<(String, String)> {
    let t = std::fs::read_to_string(Path::new(&s.models_dir).join("mine.json")).unwrap_or_default();
    json::parse(&t).ok().and_then(|v| v.get("models").and_then(json::Value::as_arr).map(|a| a.iter().filter_map(|m| {
        Some((m.get("id")?.as_str()?.to_string(), m.get("path")?.as_str()?.to_string()))
    }).collect())).unwrap_or_default()
}

fn mine_save(s: &Server, l: &[(String, String)]) -> Result<(), String> {
    let items: Vec<String> = l.iter().map(|(i, p)| format!("{{\"id\": {}, \"path\": {}}}", json::quote(i), json::quote(p))).collect();
    std::fs::create_dir_all(&s.models_dir).map_err(|e| e.to_string())?;
    std::fs::write(Path::new(&s.models_dir).join("mine.json"), format!("{{\"models\": [{}]}}\n", items.join(", "))).map_err(|e| e.to_string())
}

fn mine_id(path: &str) -> String {
    let mut h: u32 = 0x811c9dc5;                                    // FNV-1a of the lower-case path
    for b in path.to_lowercase().bytes() { h = (h ^ b as u32).wrapping_mul(0x01000193); }
    format!("mine-{h:08x}")
}

/// `rope check-model` for one file, cached per path, size and quality (a changed file gets a fresh check).
fn mine_check(s: &Server, id: &str, path: &str, q: &str) -> Option<String> {
    let len = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let key = format!("mine-{id}-{len}-{q}");
    if let Some(v) = s.cache.lock().unwrap().get(&key).cloned() { return Some(v); }
    let mut args: Vec<&str> = vec!["check-model", path, "--json", "--id", id, "--quality", q, "--disk-mb", "0", "--models-dir", &s.models_dir, "--catalog", &s.catalog];
    args.extend(s.dev_args.iter().map(String::as_str));
    let o = Command::new(&s.exe).args(&args).stdin(Stdio::null()).stderr(Stdio::null()).output().ok()?;
    let v = String::from_utf8_lossy(&o.stdout).trim().to_string();
    if !o.status.success() || json::parse(&v).is_err() { return None; }
    if len > 0 { s.cache.lock().unwrap().insert(key, v.clone()); }
    Some(v)
}

/// POST /api/mine {"path": "D:\\models\\x.safetensors"}: add a model file by its full path on this laptop.
/// Only requests from the app itself reach this (the loopback Host and Origin rules); the file is read, never changed.
fn mine_add(s: &Server, c: &mut TcpStream, body: &[u8]) -> std::io::Result<()> {
    let path = std::str::from_utf8(body).ok().and_then(|b| json::parse(b).ok()).and_then(|v| v.get("path").and_then(json::Value::as_str).map(|p| p.trim().trim_matches('"').to_string())).unwrap_or_default();
    if path.is_empty() || path.len() > 1024 || path.chars().any(char::is_control) { return json_err(c, "400 Bad Request", "path must be the full path of a model file"); }
    let p = Path::new(&path);
    if !p.is_absolute() { return json_err(c, "400 Bad Request", "use the full path, for example D:\\models\\my-model.safetensors"); }
    if !p.is_file() { return json_err(c, "404 Not Found", "no file at this path"); }
    let mut l = mine_list(s);
    let id = mine_id(&path);
    if !l.iter().any(|(i, _)| *i == id) {
        if l.len() >= 50 { return json_err(c, "409 Conflict", "the list holds 50 files: remove one first"); }
        l.push((id.clone(), path.clone()));
        if let Err(e) = mine_save(s, &l) { return json_err(c, "500 Internal Server Error", &e); }
    }
    match mine_check(s, &id, &path, "balanced") {
        Some(v) => send(c, "200 OK", "application/json", v.as_bytes()),
        None => json_err(c, "500 Internal Server Error", "the file check failed"),
    }
}

/// The add-on files in models/addons that rope can use: (file name, kind "lora" | "esrgan", title, bytes).
fn addon_files(s: &Server) -> Vec<(String, String, String, u64)> {
    let dir = Path::new(&s.models_dir).join("addons");
    let mut v: Vec<_> = std::fs::read_dir(&dir).into_iter().flatten().flatten().filter_map(|e| {
        let name = e.file_name().to_string_lossy().into_owned();
        if !safe_name(&name) || !name.ends_with(".safetensors") { return None; }
        let mf = crate::modelfile::open(&e.path()).ok()?;
        let d = crate::detect::detect(&mf)?;
        matches!(d.family.as_str(), "lora" | "esrgan").then(|| (name, d.family, d.title, e.metadata().map_or(0, |m| m.len())))
    }).collect();
    v.sort();
    v
}

/// Path of an add-on file by name, only if it is in the list above with the wanted kind.
fn addon_path(s: &Server, name: &str, want: &str) -> Result<std::path::PathBuf, String> {
    if !safe_name(name) { return Err("bad add-on name".into()); }
    match addon_files(s).into_iter().find(|a| a.0 == name) {
        Some(a) if a.1 == want => Ok(Path::new(&s.models_dir).join("addons").join(name)),
        Some(a) => Err(format!("{name} is a {} add-on, not a {want} one", a.1)),
        None => Err(format!("no such add-on: {name}")),
    }
}

/// GET /api/addons
fn addons_list(s: &Server, c: &mut TcpStream) -> std::io::Result<()> {
    let items: Vec<String> = addon_files(s).iter().map(|(n, k, t, b)| format!("{{\"name\": {}, \"kind\": {}, \"title\": {}, \"bytes\": {b}}}", json::quote(n), json::quote(k), json::quote(t))).collect();
    send(c, "200 OK", "application/json", format!("{{\"addons\": [{}]}}", items.join(", ")).as_bytes())
}

/// POST /api/jobs/{id}/control {"op": "pause" | "resume" | "prompt" | "cfg", "text": "...", "value": 3.5}
/// The job takes the command at its next step boundary (pause holds it there until resume).
fn control(s: &Server, c: &mut TcpStream, id: &str, body: &[u8]) -> std::io::Result<()> {
    let v = match std::str::from_utf8(body).ok().and_then(|b| json::parse(b).ok()) { Some(v) => v, None => return json_err(c, "400 Bad Request", "body must be JSON") };
    let line = match v.get("op").and_then(json::Value::as_str) {
        Some("pause") => "pause".to_string(),
        Some("resume") => "resume".to_string(),
        Some("prompt") => {
            let t: String = v.get("text").and_then(json::Value::as_str).unwrap_or("").chars().map(|ch| if ch.is_control() { ' ' } else { ch }).collect();
            if t.trim().is_empty() || t.chars().count() > MAX_PROMPT { return json_err(c, "400 Bad Request", "prompt must be 1..2000 characters"); }
            format!("prompt {}", t.trim())
        }
        Some("cfg") => match v.get("value").and_then(json::Value::as_f64) {
            Some(g) if (0.0..=30.0).contains(&g) => format!("cfg {g}"),
            _ => return json_err(c, "400 Bad Request", "value must be a number from 0 to 30"),
        },
        _ => return json_err(c, "400 Bad Request", "op must be pause, resume, prompt or cfg"),
    };
    let job = s.job.lock().unwrap().clone();
    let Some(job) = job.filter(|j| j.lock().unwrap().id == id) else { return json_err(c, "404 Not Found", "no such job") };
    let mut j = job.lock().unwrap();
    if j.done { return json_err(c, "409 Conflict", "the job has ended"); }
    let sent = j.stdin.as_mut().map(|w| writeln!(w, "{line}").and_then(|_| w.flush()).is_ok()).unwrap_or(false);
    if !sent { return json_err(c, "409 Conflict", "the job cannot take commands"); }
    send(c, "200 OK", "application/json", b"{\"ok\": true}")
}

/// Read the child's stderr: plan lines become "log" events, `progress:` lines become progress events with an ETA.
fn watch(job: Arc<Mutex<Job>>, err: std::process::ChildStderr) {
    let t0 = Instant::now();
    let (mut est_total, mut est_decode, mut est_up) = (0f64, 0f64, 0f64);
    let mut denoise_start: Option<Instant> = None;
    let (mut paused_at, mut paused_total): (Option<Instant>, f64) = (None, 0.0);     // time held at a pause is not work
    let mut last = String::new();
    let push = |e: String| job.lock().unwrap().events.push(e);
    for line in BufReader::new(err).lines().map_while(Result::ok) {
        let l = line.trim();
        if let Some(p) = l.strip_prefix("progress: ") {
            let f: Vec<&str> = p.split_whitespace().collect();
            if f.first() == Some(&"estimate") {
                est_total = f.get(1).and_then(|x| x.parse().ok()).unwrap_or(0.0);
                est_decode = f.get(2).and_then(|x| x.parse().ok()).unwrap_or(0.0);
                est_up = f.get(4).and_then(|x| x.parse().ok()).unwrap_or(0.0);
                push(format!("{{\"type\": \"estimate\", \"total_s\": {est_total:.1}}}"));
                continue;
            }
            let (stage, frac) = (f.first().copied().unwrap_or(""), f.get(1).copied().unwrap_or("0/1"));
            if matches!(stage, "paused" | "resumed" | "prompt") {
                let step = frac.split_once('/').map_or("0", |x| x.0);
                if stage == "paused" { paused_at = Some(Instant::now()); }
                if stage == "resumed" { if let Some(t) = paused_at.take() { paused_total += t.elapsed().as_secs_f64(); } }
                push(format!("{{\"type\": \"{stage}\", \"step\": {}}}", step.parse::<u32>().unwrap_or(0)));
                continue;
            }
            if stage == "preview" {
                let (i, n) = frac.split_once('/').unwrap_or(("0", "1"));
                push(format!("{{\"type\": \"preview\", \"step\": {}, \"n\": {}}}", i.parse::<u32>().unwrap_or(0), n.parse::<u32>().unwrap_or(1)));
                continue;
            }
            let (i, n) = frac.split_once('/').map(|(a, b)| (a.parse::<f64>().unwrap_or(0.0), b.parse::<f64>().unwrap_or(1.0).max(1.0))).unwrap_or((0.0, 1.0));
            let el = (t0.elapsed().as_secs_f64() - paused_total).max(0.0);
            // ETA: measured time per denoising step for the steps left, then the plan's decode estimate
            let eta = match stage {
                "denoise" => {
                    let ds = *denoise_start.get_or_insert_with(Instant::now);
                    let done_s = (ds.elapsed().as_secs_f64() - paused_total).max(0.0);
                    if i > 0.0 { done_s / i * (n - i) + est_decode + est_up } else { (est_total - el).max(0.0) }
                }
                "decode" => est_decode * (n - i) / n + est_up,
                "upscale" => est_up * (n - i) / n,
                _ => (est_total - el).max(0.0),
            };
            push(format!("{{\"type\": \"progress\", \"stage\": {}, \"i\": {i}, \"n\": {n}, \"elapsed_s\": {el:.1}, \"eta_s\": {eta:.1}}}", json::quote(stage)));
        } else if !l.is_empty() {
            last = l.to_string();
            push(format!("{{\"type\": \"log\", \"line\": {}}}", json::quote(l)));
        }
    }
    let mut j = job.lock().unwrap();
    let ok = j.child.as_mut().and_then(|c| c.wait().ok()).map(|s| s.success()).unwrap_or(false) && j.png.exists();
    let secs = t0.elapsed().as_secs_f64();
    if ok {
        let side = j.png.with_extension("json");
        let _ = std::fs::write(&side, format!("{{{}, \"seconds\": {secs:.1}, \"created_ms\": {}}}", j.meta, now_ms()));
        let e = format!("{{\"type\": \"done\", {}, \"seconds\": {secs:.1}}}", j.meta);
        j.events.push(e);
    } else {
        let e = format!("{{\"type\": \"error\", \"message\": {}}}", json::quote(if last.is_empty() { "the image process failed" } else { &last }));
        j.events.push(e);
    }
    j.child = None;
    j.stdin = None;
    let _ = std::fs::remove_dir_all(&j.preview_dir);
    j.done = true;
}

/// Server-sent events of a job: every event so far, then new ones as they come, until the job ends.
fn events(s: &Server, c: &mut TcpStream, id: &str) -> std::io::Result<()> {
    let job = s.job.lock().unwrap().clone();
    let Some(job) = job.filter(|j| j.lock().unwrap().id == id) else { return json_err(c, "404 Not Found", "no such job") };
    write!(c, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n")?;
    let mut sent = 0;
    loop {
        let (new, done) = { let j = job.lock().unwrap(); (j.events[sent..].to_vec(), j.done) };
        for e in &new { write!(c, "data: {e}\n\n")?; }
        sent += new.len();
        c.flush()?;
        if done && new.is_empty() { return Ok(()); }
        std::thread::sleep(Duration::from_millis(150));
    }
}

fn gallery_list(s: &Server, c: &mut TcpStream) -> std::io::Result<()> {
    let mut items: Vec<(String, String)> = std::fs::read_dir(&s.gallery).map(|d| d.flatten().filter_map(|e| {
        let p = e.path();
        if p.extension().and_then(|x| x.to_str()) != Some("json") { return None; }
        let t = std::fs::read_to_string(&p).ok()?;
        json::parse(&t).ok()?;                                       // skip a damaged sidecar
        Some((p.file_name()?.to_string_lossy().into_owned(), t))
    }).collect()).unwrap_or_default();
    items.sort_by(|a, b| b.0.cmp(&a.0));                              // newest first (ids are timestamps)
    send(c, "200 OK", "application/json", format!("[{}]", items.into_iter().map(|x| x.1).collect::<Vec<_>>().join(", ")).as_bytes())
}

/// The built app (index.html for unknown paths, so client routes work). No path leaves the static folder.
fn static_file(s: &Server, c: &mut TcpStream, path: &str) -> std::io::Result<()> {
    let Some(root) = &s.static_dir else { return json_err(c, "404 Not Found", "no app folder (--static)") };
    let rel = path.trim_start_matches('/');
    let ok = rel.split('/').all(|seg| seg.is_empty() || safe_name(seg));
    let file = if ok && !rel.is_empty() && root.join(rel).is_file() { root.join(rel) } else { root.join("index.html") };
    let ct = match file.extension().and_then(|x| x.to_str()).unwrap_or("") {
        "html" => "text/html; charset=utf-8", "js" => "text/javascript", "css" => "text/css", "svg" => "image/svg+xml",
        "png" => "image/png", "woff2" => "font/woff2", "json" => "application/json", "ico" => "image/x-icon", _ => "application/octet-stream",
    };
    match std::fs::read(&file) { Ok(b) => send(c, "200 OK", ct, &b), Err(_) => json_err(c, "404 Not Found", "not found") }
}
