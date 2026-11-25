//! `rope selftest`: one command to run on any laptop and send the result back (P2.5.2).
//! It probes the machine, asks the planner what this laptop can run, makes one small real image with the test model
//! and compares the plan with what happened (time and peak memory). Result: a readable summary and `report.json`.
//! Each step runs as a child process of this same executable (like `rope serve`), so memory is measured per image.
//! Nothing is downloaded: if the model file is missing, the report says which file is needed and why no run happened.
#![forbid(unsafe_code)]

use crate::json;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// rope selftest [--out report.json] [--model-id sd-turbo] [--size N] [--models-dir models] [--catalog catalog/models.json]
///               [--timeout-s 900] [--cap-ws-gib X: hard RAM cap on the image process, to test a smaller laptop] [device overrides: --ram-gib X --no-gpu --disk-mb N ...]
/// Exit code: 0 = ran and the plan held, 1 = ran but the plan was off by more than 50%, 2 = no run (model missing or run failed).
pub fn selftest(args: &[String]) -> Result<i32, String> {
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let out = get("--out").unwrap_or_else(|| "report.json".into());
    let models_dir = get("--models-dir").unwrap_or_else(|| "models".into());
    let catalog = get("--catalog").unwrap_or_else(|| "catalog/models.json".into());
    let model_id = get("--model-id").unwrap_or_else(|| "sd-turbo".into());
    let size = get("--size");                        // default: the model's own size, where the planner is calibrated
    let cap_ws: Option<f64> = get("--cap-ws-gib").and_then(|s| s.parse().ok());
    let timeout = Duration::from_secs(get("--timeout-s").and_then(|s| s.parse().ok()).unwrap_or(900));
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    // device overrides pass through to every child (simulate a smaller laptop: --ram-gib 3.4 --no-gpu)
    let mut dev: Vec<String> = Vec::new();
    for k in ["--ram-gib", "--ram-free-gib", "--gflops", "--gflops-1t", "--disk-gbs", "--disk-mb"] {
        if let Some(v) = get(k) { dev.extend([k.to_string(), v]); }
    }
    if args.iter().any(|a| a == "--no-gpu") { dev.push("--no-gpu".into()); }
    let say = |s: &str| eprintln!("selftest: {s}");

    // 1. the machine
    say("reading this machine");
    let probe = child_json(&exe, &["probe".into(), "--dir".into(), models_dir.clone(), "--disk-mb".into(), get("--disk-mb").unwrap_or_else(|| "256".into())], &[])?;
    let probe_v = json::parse(&probe)?;
    let ram = probe_v.get("ram_usable_gib").and_then(json::Value::as_f64).unwrap_or(0.0);
    let path = probe_v.get("cpu_path").and_then(json::Value::as_str).unwrap_or("?").to_string();
    let gpu = probe_v.get("gpu").and_then(|g| g.get("name")).and_then(json::Value::as_str).map(str::to_string);
    let summary = machine_summary(&probe_v);

    // 2. what the planner says every catalog model does here
    say("planning the catalog models for this machine");
    let mut sa: Vec<String> = vec!["suggest".into(), "--json".into(), "--models-dir".into(), models_dir.clone(), "--catalog".into(), catalog.clone()];
    sa.extend(dev.iter().cloned());
    let suggest = child_json(&exe, &sa, &[])?;
    let suggest_v = json::parse(&suggest)?;
    let m = suggest_v.get("models").and_then(json::Value::as_arr).and_then(|ms| ms.iter().find(|m| m.get("id").and_then(json::Value::as_str) == Some(&model_id)))
        .ok_or_else(|| format!("model {model_id} is not in the catalog"))?;
    let file = m.get("file").and_then(json::Value::as_str).unwrap_or("").to_string();
    let on_disk = m.get("on_disk").and_then(json::Value::as_bool).unwrap_or(false);
    let runs = m.get("runs").and_then(json::Value::as_bool).unwrap_or(false);

    // 3. one real image
    let mut run = String::from("null");
    let mut code = 2;
    let verdict: String;
    if !runs {
        verdict = format!("the planner says {model_id} does not run on this machine: {}", m.get("why").and_then(json::Value::as_str).unwrap_or("no reason given"));
    } else if !on_disk {
        verdict = format!("the test model file {file} is not in {models_dir}; copy it there and run again (selftest does not download)");
    } else {
        say(&format!("making one image with {model_id} (the first run also checks the file for safety)"));
        let img = std::env::temp_dir().join(format!("rope-selftest-{}.png", std::process::id()));
        let mut ia: Vec<String> = vec!["image".into(), "--model".into(), Path::new(&models_dir).join(&file).to_string_lossy().into_owned(),
            "--prompt".into(), "a red bicycle leaning on a brick wall, photo".into(), "--quality".into(), "fast".into(),
            "--seed".into(), "1".into(), "--models-dir".into(), models_dir.clone(), "--catalog".into(), catalog.clone(), "--out".into(), img.to_string_lossy().into_owned(),
            "--no-download".into()];                                  // the test never uses the network
        if let Some(sz) = &size { ia.extend(["--size".to_string(), sz.clone()]); }
        ia.extend(dev.iter().cloned());
        let t0 = Instant::now();
        let r = child_run(&exe, &ia, &[("ROPE_PROGRESS", "1")], timeout, cap_ws);
        let wall = t0.elapsed().as_secs_f64();
        let _ = std::fs::remove_file(&img);
        match r {
            Err(e) => verdict = format!("the image run failed: {e}"),
            Ok((ok, stderr)) => {
                // the child's own lines: plan estimate (last "~N s estimated" wins: it includes the GPU placement),
                // engine total, first-use check, peak memory
                let num_before = |line: &str, marker: &str| line.split(marker).next().and_then(|h| h.split_whitespace().last()).and_then(|t| t.trim_start_matches('~').parse::<f64>().ok());
                let (mut est, mut total, mut first_use, mut peak_commit, mut peak_ws) = (None, None, None, None, None);
                let mut ran_size = String::from("null");
                for l in stderr.lines() {
                    if l.starts_with("plan: ") { if let Some(wh) = l.split_whitespace().nth(2).and_then(|t| t.trim_end_matches(',').split_once('x')) { ran_size = wh.0.to_string(); } }
                    if l.contains("s estimated") { est = num_before(l, " s estimated").or(est); }
                    if l.starts_with("image:") { total = l.rsplit(" total ").next().and_then(|t| t.split_whitespace().next()).and_then(|t| t.parse::<f64>().ok()).or(total); }
                    if let Some(rest) = l.strip_prefix("first use: sha256 + safety scan in ") { first_use = rest.split_whitespace().next().and_then(|t| t.parse::<f64>().ok()); }
                    if let Some(rest) = l.strip_prefix("memory: peak commit ") {
                        peak_commit = rest.split_whitespace().next().and_then(|t| t.parse::<f64>().ok());
                        peak_ws = rest.split("working set ").nth(1).and_then(|t| t.split_whitespace().next()).and_then(|t| t.parse::<f64>().ok());
                    }
                }
                let err_pct = match (est, total) { (Some(e), Some(t)) if t > 0.0 => Some(100.0 * (e - t) / t), _ => None };
                let f = |o: Option<f64>| o.map_or("null".to_string(), |v| format!("{v:.2}"));
                run = format!("{{\"model\": {}, \"size\": {ran_size}, \"exit_ok\": {ok}, \"wall_s\": {wall:.1}, \"engine_total_s\": {}, \"plan_estimate_s\": {}, \"plan_error_pct\": {}, \
                               \"first_use_check_s\": {}, \"peak_commit_gib\": {}, \"peak_working_set_gib\": {}}}",
                              json::quote(&model_id), f(total), f(est), f(err_pct), f(first_use), f(peak_commit), f(peak_ws));
                match (ok, err_pct) {
                    (false, _) => verdict = format!("the image run exited with an error: {}", stderr.lines().last().unwrap_or("")),
                    (true, Some(p)) if p.abs() <= 50.0 => { code = 0; verdict = format!("the plan held: estimated {:.1} s, measured {:.1} s ({p:+.0}%)", est.unwrap_or(0.0), total.unwrap_or(0.0)); }
                    (true, Some(p)) => { code = 1; verdict = format!("the plan was off by {p:+.0}% (estimated {:.1} s, measured {:.1} s): please send report.json back", est.unwrap_or(0.0), total.unwrap_or(0.0)); }
                    (true, None) => verdict = "the image was made but the plan or the time could not be read from its output".into(),
                }
            }
        }
    }

    let when = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let report = format!("{{\"selftest\": 1, \"rope\": {}, \"build\": {}, \"unix_time\": {when}, \"model_id\": {}, \"verdict\": {}, \"code\": {code},\n \"machine_summary\": {},\n \"network_used\": false,\n \"probe\": {probe},\n \"suggest\": {suggest},\n \"run\": {run}}}\n",
                         json::quote(crate::VERSION), json::quote(if crate::ffi::has_cuda() { "cpu+cuda" } else { "cpu" }), json::quote(&model_id), json::quote(&verdict), json::quote(&summary));
    json::parse(&report).map_err(|e| format!("internal: the report is not valid JSON: {e}"))?;
    std::fs::write(&out, &report).map_err(|e| format!("{out}: {e}"))?;
    println!("Rope selftest");
    let gpu_line = if args.iter().any(|a| a == "--no-gpu") { "no GPU (simulated with --no-gpu)".to_string() } else { gpu.clone().unwrap_or_else(|| "no GPU".into()) };
    println!("  machine   {summary}");
    println!("  this run  {ram:.1} GiB RAM usable, {gpu_line}, kernels {path}");
    println!("  model     {model_id} ({file})");
    println!("  result    {verdict}");
    println!("  report    {out}  (send this file back)");
    Ok(code)
}

/// One readable line about the machine, from the probe: maker and model, CPU, RAM, GPU, disk, power source.
fn machine_summary(p: &json::Value) -> String {
    let s = |k: &str| p.get(k).and_then(json::Value::as_str).unwrap_or("").trim().to_string();
    let f = |v: Option<&json::Value>, k: &str| v.and_then(|x| x.get(k)).and_then(json::Value::as_f64).unwrap_or(0.0);
    let gpu = match p.get("gpu").filter(|g| g.get("name").is_some()) {
        Some(g) => format!("{} ({:.0} GB VRAM)", g.get("name").and_then(json::Value::as_str).unwrap_or("?"), f(Some(g), "vram_total_gib")),
        None => "no NVIDIA GPU".to_string(),
    };
    let disk = p.get("disk");
    let ssd = match disk.and_then(|d| d.get("ssd")).and_then(json::Value::as_bool) { Some(true) => "SSD", Some(false) => "hard disk", None => "disk type unknown" };
    let bus = disk.and_then(|d| d.get("bus")).and_then(json::Value::as_str).unwrap_or("");
    let power = p.get("power");
    let ac = power.and_then(|x| x.get("on_ac")).and_then(json::Value::as_bool);
    let pct = power.and_then(|x| x.get("battery_percent")).and_then(json::Value::as_f64);
    let saver = power.and_then(|x| x.get("battery_saver")).and_then(json::Value::as_bool) == Some(true);
    let power_s = match ac { Some(true) => "on mains power".to_string(), Some(false) => format!("ON BATTERY{} (speed can be lower)", pct.map_or(String::new(), |v| format!(" {v:.0}%"))), None => "no battery".to_string() };
    format!("{}, {} | {} | {} cores / {} threads, kernels {} | {:.1} GiB RAM installed, {:.1} usable, {:.1} free | {} | {} {}, {:.0} GiB free of {:.0} | {}{}",
            s("machine"), s("os_name"), s("cpu"), f(Some(p), "cores"), f(Some(p), "threads"), s("cpu_path"),
            f(Some(p), "ram_installed_gib"), f(Some(p), "ram_usable_gib"), f(Some(p), "ram_free_gib"), gpu, bus, ssd,
            f(disk, "free_gib"), f(disk, "total_gib"), power_s, if saver { ", battery saver ON" } else { "" })
}

/// Run a sub-command of this executable and return its stdout, which must be JSON.
fn child_json(exe: &Path, args: &[String], env: &[(&str, &str)]) -> Result<String, String> {
    let o = Command::new(exe).args(args).envs(env.iter().copied()).stdin(Stdio::null()).stderr(Stdio::null()).output().map_err(|e| e.to_string())?;
    if !o.status.success() { return Err(format!("`rope {}` failed", args.first().map_or("", String::as_str))); }
    let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
    json::parse(&s).map_err(|e| format!("`rope {}` gave no JSON: {e}", args.first().map_or("", String::as_str)))?;
    Ok(s)
}

/// Run a sub-command with a time limit; (success, stderr). The child is killed when the limit is reached.
fn child_run(exe: &Path, args: &[String], env: &[(&str, &str)], limit: Duration, cap_ws_gib: Option<f64>) -> Result<(bool, String), String> {
    use std::io::Read;
    let mut ch = Command::new(exe).args(args).envs(env.iter().copied()).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::piped()).spawn().map_err(|e| e.to_string())?;
    if let Some(g) = cap_ws_gib {
        #[cfg(not(windows))]
        let _ = g;   // the working-set cap exists on Windows only
        #[cfg(windows)]
        { use std::os::windows::io::AsRawHandle; if !crate::ffi::cap_working_set(ch.as_raw_handle().cast(), (g * 1073741824.0) as usize) { let _ = ch.kill(); return Err("the OS refused the working-set cap".into()); } }
    }
    let mut err = ch.stderr.take().ok_or("no stderr")?;
    let reader = std::thread::spawn(move || { let mut s = String::new(); let _ = err.read_to_string(&mut s); s });
    let t0 = Instant::now();
    let status = loop {
        if let Some(s) = ch.try_wait().map_err(|e| e.to_string())? { break s; }
        if t0.elapsed() > limit { let _ = ch.kill(); let _ = ch.wait(); return Err(format!("no result within {} s", limit.as_secs())); }
        std::thread::sleep(Duration::from_millis(200));
    };
    Ok((status.success(), reader.join().unwrap_or_default()))
}
