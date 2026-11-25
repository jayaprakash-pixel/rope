//! The C interface of librope (P5.1.9): `include/rope.h`. This module is the library's boundary to other languages and
//! the only place of the library where C pointers are read; everything behind it is the same safe code `rope.exe` runs.
//!
//! - Requests are JSON text (the same fields as the command line, see rope.h): the interface stays small and can grow without
//!   a new function.
//! - Strings go out into caller buffers with snprintf rules: the answer is cut to fit and the full length is returned.
//! - One image job at a time in a process (the engine keeps its settings, such as the buffer cache, per process). A second
//!   `rope_job_start` while a job runs returns NULL.
//! - Downloads are off unless the request says `"allow_download": true`.
#![allow(unsafe_code)]

use crate::{cli, json, models, scan, sd, weights};
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::thread::JoinHandle;

static BUSY: AtomicBool = AtomicBool::new(false);
static LAST_ERROR: Mutex<String> = Mutex::new(String::new());

fn set_error(e: &str) { if let Ok(mut g) = LAST_ERROR.lock() { *g = e.to_string(); } }

/// A C string argument as &str (None for NULL or text that is not UTF-8).
unsafe fn text<'a>(p: *const c_char) -> Option<&'a str> {
    if p.is_null() { return None; }
    // SAFETY: the caller passes a NUL-terminated string (documented in rope.h); it is only read here.
    unsafe { CStr::from_ptr(p) }.to_str().ok()
}

/// Write `s` into the caller's buffer (cut to fit, always NUL-terminated when cap > 0); return the full length, or -1 on error.
unsafe fn put(out: *mut c_char, cap: usize, r: Result<String, String>) -> c_int {
    match r {
        Ok(s) => {
            if !out.is_null() && cap > 0 {
                let n = s.len().min(cap - 1);
                // SAFETY: out is writable for cap bytes (documented in rope.h); n + 1 <= cap.
                unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(), out as *mut u8, n); *out.add(n) = 0; }
            }
            s.len() as c_int
        }
        Err(e) => { set_error(&e); -1 }
    }
}

/// Request JSON -> the argument list of the command line (`--model F --prompt P ...`).
fn args_from_request(req: &str) -> Result<Vec<String>, String> {
    let v = json::parse(req).map_err(|e| format!("request is not JSON: {e}"))?;
    let mut a: Vec<String> = Vec::new();
    for (key, flag) in [("model", "--model"), ("prompt", "--prompt"), ("quality", "--quality"), ("out", "--out"), ("models_dir", "--models-dir"),
                        ("catalog", "--catalog"), ("lora", "--lora"), ("upscale", "--upscale"), ("only", "--only")] {
        if let Some(s) = v.get(key).and_then(json::Value::as_str) { a.extend([flag.to_string(), s.to_string()]); }
    }
    for (key, flag) in [("seed", "--seed"), ("size", "--size"), ("steps", "--steps"), ("lora_scale", "--lora-scale"), ("ram_gib", "--ram-gib")] {
        if let Some(x) = v.get(key).and_then(json::Value::as_f64) { a.extend([flag.to_string(), if key == "lora_scale" || key == "ram_gib" { x.to_string() } else { (x as u64).to_string() }]); }
    }
    if v.get("no_gpu").and_then(json::Value::as_bool) == Some(true) { a.push("--no-gpu".into()); }
    a.extend(["--disk-mb".to_string(), "0".to_string()]);
    models::set_no_download(v.get("allow_download").and_then(json::Value::as_bool) != Some(true));
    Ok(a)
}

/// Version text of the library, e.g. "0.1.0".
#[no_mangle]
pub extern "C" fn rope_version() -> *const c_char { concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char }

/// The text of the last error of a failed call (any thread), snprintf rules.
#[no_mangle]
pub unsafe extern "C" fn rope_last_error(out: *mut c_char, cap: usize) -> c_int {
    let e = LAST_ERROR.lock().map(|g| g.clone()).unwrap_or_default();
    unsafe { put(out, cap, Ok(e)) }
}

/// This laptop as JSON (CPU, RAM, GPU, disk), same as `rope probe`. `dir`: a folder on the disk to measure (NULL = ".").
#[no_mangle]
pub unsafe extern "C" fn rope_probe_json(dir: *const c_char, out: *mut c_char, cap: usize) -> c_int {
    let d = unsafe { text(dir) }.unwrap_or(".");
    unsafe { put(out, cap, crate::ffi::probe_json(d, 0)) }
}

/// Every catalog model with its plan for this laptop, as JSON, same as `rope suggest --json`.
/// `request`: JSON with the optional fields models_dir, catalog, quality, size, only, upscale (true = plan the upscaler), no_gpu, ram_gib.
#[no_mangle]
pub unsafe extern "C" fn rope_plan_json(request: *const c_char, out: *mut c_char, cap: usize) -> c_int {
    let r = (|| {
        let req = unsafe { text(request) }.unwrap_or("{}");
        let v = json::parse(req).map_err(|e| format!("request is not JSON: {e}"))?;
        let mut a = args_from_request(req)?;
        if v.get("upscale").and_then(json::Value::as_bool) == Some(true) { a.push("--upscale".into()); }
        cli::suggest_json(&a)
    })();
    unsafe { put(out, cap, r) }
}

/// Outlier-injection scan of a model or add-on file, as JSON: {"tensors", "groups", "flagged": [names], "clean": bool}.
#[no_mangle]
pub unsafe extern "C" fn rope_scan_json(path: *const c_char, out: *mut c_char, cap: usize) -> c_int {
    let r = (|| {
        let p = unsafe { text(path) }.ok_or("path is NULL or not UTF-8")?;
        let w = weights::Weights::open(Path::new(p))?;
        let reps = scan::scan(&w)?;
        let flagged: Vec<String> = reps.iter().filter(|r| r.stats.flagged()).map(|r| json::quote(&r.name)).collect();
        Ok(format!("{{\"tensors\": {}, \"groups\": {}, \"flagged\": [{}], \"clean\": {}}}", reps.len(), reps.iter().map(|r| r.stats.groups).sum::<u64>(), flagged.join(", "), flagged.is_empty()))
    })();
    unsafe { put(out, cap, r) }
}

/// Progress callback: `stage` is "text", "denoise", "decode" or "upscale"; step `i` of `n`. Called from the job's thread.
pub type ProgressFn = extern "C" fn(stage: *const c_char, i: c_int, n: c_int, user: *mut c_void);

struct UserData(*mut c_void);
// SAFETY: the pointer is only handed back to the caller's own callback; what it points to is the caller's responsibility (rope.h).
unsafe impl Send for UserData {}
unsafe impl Sync for UserData {}

pub struct Job { handle: Option<JoinHandle<Result<(), String>>>, done: Option<Result<(), String>> }

impl Job {
    fn finish(&mut self) -> &Result<(), String> {
        if let Some(h) = self.handle.take() {
            let r = h.join().unwrap_or_else(|_| Err("the job thread panicked".into()));
            self.done = Some(r);
        }
        self.done.as_ref().unwrap()
    }
}

/// Start one image in a thread. `request`: JSON with prompt (required), model (a file; without it rope picks the best catalog model
/// for this laptop), quality, seed, size, steps, out (PNG path), models_dir, catalog, lora, lora_scale, upscale (add-on file paths),
/// no_gpu, ram_gib, allow_download. Returns NULL if the request is wrong or a job runs (see rope_last_error).
#[no_mangle]
pub unsafe extern "C" fn rope_job_start(request: *const c_char, cb: Option<ProgressFn>, user: *mut c_void) -> *mut Job {
    let args = match unsafe { text(request) }.ok_or("request is NULL".to_string()).and_then(args_from_request) { Ok(a) => a, Err(e) => { set_error(&e); return std::ptr::null_mut(); } };
    if !args.iter().any(|a| a == "--prompt") { set_error("the request has no prompt"); return std::ptr::null_mut(); }
    if BUSY.swap(true, Ordering::SeqCst) { set_error("a job is already running in this process"); return std::ptr::null_mut(); }
    sd::request_cancel(false);
    let ud = UserData(user);
    sd::set_progress_hook(cb.map(|f| -> sd::ProgressHook {
        Box::new(move |stage, i, n| { let _ = &ud; if let Ok(c) = CString::new(stage) { f(c.as_ptr(), i as c_int, n as c_int, ud.0); } })
    }));
    let h = std::thread::spawn(move || {
        let r = std::panic::catch_unwind(|| cli::image(&args)).unwrap_or_else(|_| Err("the engine panicked".into()));
        sd::set_progress_hook(None);
        BUSY.store(false, Ordering::SeqCst);
        r
    });
    Box::into_raw(Box::new(Job { handle: Some(h), done: None }))
}

/// Wait for the job to end. Returns 0 (image written), 1 (failed; `msg` has the reason) or 2 (cancelled). `msg` may be NULL.
#[no_mangle]
pub unsafe extern "C" fn rope_job_wait(job: *mut Job, msg: *mut c_char, cap: usize) -> c_int {
    // SAFETY: `job` came from rope_job_start and is not freed yet (rope.h).
    let Some(j) = (unsafe { job.as_mut() }) else { return 1 };
    let (code, text) = match j.finish() { Ok(()) => (0, String::new()), Err(e) if e == "cancelled" => (2, e.clone()), Err(e) => (1, e.clone()) };
    unsafe { put(msg, cap, Ok(text)) };
    code
}

/// Ask the job to stop at its next step (within one denoising step or one tile). Then call rope_job_wait.
#[no_mangle]
pub extern "C" fn rope_job_cancel(_job: *mut Job) { sd::request_cancel(true); }

/// Free the job (cancels and waits first if it still runs).
#[no_mangle]
pub unsafe extern "C" fn rope_job_free(job: *mut Job) {
    if job.is_null() { return; }
    // SAFETY: `job` came from rope_job_start and is freed once (rope.h).
    let mut j = unsafe { Box::from_raw(job) };
    if j.handle.is_some() { sd::request_cancel(true); }
    let _ = j.finish();
}
