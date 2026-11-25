//! Model manager: make sure the files a plan needs are present and verified.
//! Every file is checked against the sha256 in the catalog. A file that fails is renamed to `<name>.rejected`
//! and never used. Verified files get a marker (sha, size, mtime) so large models are not re-hashed every run.
#![forbid(unsafe_code)]

use crate::sha256;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone)]
pub struct FileSpec { pub role: String, pub name: String, pub repo: String, pub path: String, pub bytes: u64, pub sha256: String }

fn marker(dir: &Path, name: &str) -> PathBuf { dir.join(".rope").join(format!("{name}.verified")) }

/// (size, mtime in nanoseconds). Full precision: a same-size rewrite within the same second must still be noticed.
/// The marker is a speed cache against accidental change; `rope verify` re-hashes everything (someone who can
/// write the user's files could also forge a marker).
fn stamp(p: &Path) -> Option<(u64, u128)> {
    let m = fs::metadata(p).ok()?;
    let mt = m.modified().ok()?.duration_since(std::time::UNIX_EPOCH).ok()?.as_nanos();
    Some((m.len(), mt))
}

/// True if the file exists and a marker says it was verified with this sha, size and mtime.
pub fn is_verified(dir: &Path, f: &FileSpec) -> bool {
    let (Some((len, mt)), Ok(m)) = (stamp(&dir.join(&f.name)), fs::read_to_string(marker(dir, &f.name))) else { return false };
    m.trim() == format!("{} {} {}", f.sha256, len, mt)
}

/// Hash the file; on success write the marker, on failure move it aside as `<name>.rejected`.
pub fn verify(dir: &Path, f: &FileSpec, file: &Path) -> Result<(), String> {
    let got = sha256::of_reader(fs::File::open(file).map_err(|e| format!("{}: {e}", file.display()))?).map_err(|e| e.to_string())?;
    if got != f.sha256 {
        let bad = dir.join(format!("{}.rejected", f.name));
        let _ = fs::remove_file(&bad);
        fs::rename(file, &bad).map_err(|e| format!("could not quarantine {}: {e}", file.display()))?;
        return Err(format!("{}: sha256 mismatch (expected {}, got {}); moved to {}", f.name, f.sha256, got, bad.display()));
    }
    let final_path = dir.join(&f.name);
    if file != final_path { fs::rename(file, &final_path).map_err(|e| e.to_string())?; }
    let (len, mt) = stamp(&final_path).ok_or("cannot stat verified file")?;
    let mk = marker(dir, &f.name);
    fs::create_dir_all(mk.parent().unwrap()).map_err(|e| e.to_string())?;
    fs::write(mk, format!("{} {} {}", f.sha256, len, mt)).map_err(|e| e.to_string())
}

fn curl() -> PathBuf {
    let sys = PathBuf::from(std::env::var("SystemRoot").unwrap_or_else(|_| r"C:\Windows".into())).join("System32").join("curl.exe");
    if sys.exists() { sys } else { PathBuf::from("curl") }
}

/// First-use record of a file the user brings: its sha256 and the safety-scan verdict.
#[derive(Debug, Clone)]
pub struct Admission { pub sha256: String, pub scan: String, pub first_use: bool }

/// Admit a user's model file: sha256 + outlier-injection scan once, remembered in `<models_dir>/.rope/admitted/`
/// (the user's own folder is never written). Re-checked when size or mtime change. Suspicious files are refused
/// unless `allow_suspicious`.
pub fn admit(models_dir: &Path, file: &Path, w: &crate::weights::Weights, allow_suspicious: bool) -> Result<Admission, String> {
    let name = file.file_name().ok_or("bad model path")?.to_string_lossy().into_owned();
    let mk = models_dir.join(".rope").join("admitted").join(format!("{name}.admitted"));
    let (len, mt) = stamp(file).ok_or_else(|| format!("cannot stat {}", file.display()))?;
    let full = fs::canonicalize(file).map_err(|e| e.to_string())?.to_string_lossy().into_owned();
    if let Ok(m) = fs::read_to_string(&mk) {
        let f: Vec<&str> = m.trim().splitn(5, ' ').collect();
        if f.len() == 5 && f[1] == len.to_string() && f[2] == mt.to_string() && f[4] == full {
            return verdict(Admission { sha256: f[0].into(), scan: f[3].into(), first_use: false }, allow_suspicious);
        }
    }
    let sha = sha256::of_reader(fs::File::open(file).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let reps = crate::scan::scan(w)?;
    let scan = if reps.is_empty() { "n/a".to_string() } else if reps.iter().any(|r| r.stats.flagged()) { "suspicious".into() } else { "clean".into() };
    fs::create_dir_all(mk.parent().unwrap()).map_err(|e| e.to_string())?;
    fs::write(&mk, format!("{sha} {len} {mt} {scan} {full}")).map_err(|e| e.to_string())?;
    verdict(Admission { sha256: sha, scan, first_use: true }, allow_suspicious)
}

fn verdict(a: Admission, allow: bool) -> Result<Admission, String> {
    if a.scan == "suspicious" && !allow {
        return Err("SUSPICIOUS: this file shows the outlier-injection pattern (rope scan for details); refusing to run it. Use --allow-suspicious only if you trust its source.".into());
    }
    Ok(a)
}

#[derive(Debug, PartialEq, Eq)]
pub enum Outcome { AlreadyVerified, VerifiedExisting, Downloaded }

static NO_DOWNLOAD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// When on, `ensure` never downloads: a missing file is an error. The library turns this on unless the caller allows downloads.
pub fn set_no_download(on: bool) { NO_DOWNLOAD.store(on, std::sync::atomic::Ordering::SeqCst); }

/// Make one file present and verified. A tampered/corrupt existing file is rejected, then downloaded fresh.
pub fn ensure(dir: &Path, f: &FileSpec, log: &mut Vec<String>) -> Result<Outcome, String> {
    fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    if is_verified(dir, f) { return Ok(Outcome::AlreadyVerified); }
    let target = dir.join(&f.name);
    if target.exists() {
        match verify(dir, f, &target) {
            Ok(()) => return Ok(Outcome::VerifiedExisting),
            Err(e) => log.push(format!("REJECTED {e}")),
        }
    }
    if NO_DOWNLOAD.load(std::sync::atomic::Ordering::SeqCst) { return Err(format!("{} is missing and downloads are off (allow_download)", f.name)); }
    if let Some(parent) = target.parent() { fs::create_dir_all(parent).map_err(|e| e.to_string())?; }
    let part = dir.join(format!("{}.part", f.name));
    let url = format!("https://huggingface.co/{}/resolve/main/{}", f.repo, f.path);
    let st = Command::new(curl()).args(["-L", "--fail", "--silent", "--show-error", "--retry", "3", "-C", "-", "-o"]).arg(&part).arg(&url)
        .status().map_err(|e| format!("cannot run curl: {e}"))?;
    if !st.success() { return Err(format!("download failed for {} ({url})", f.name)); }
    let len = fs::metadata(&part).map_err(|e| e.to_string())?.len();
    if len != f.bytes { return Err(format!("{}: downloaded {len} bytes, catalog says {}", f.name, f.bytes)); }
    verify(dir, f, &part)?;
    Ok(Outcome::Downloaded)
}
