//! P1.1.2: the probe must work on machines with no NVIDIA GPU and with no NVIDIA driver.
//! It never calls nvidia-smi; it reads the driver library directly, so both cases are simulated here.
use std::process::Command;

fn probe(env: &[(&str, &str)]) -> String {
    let mut c = Command::new(env!("CARGO_BIN_EXE_rope"));
    c.args(["probe", "--disk-mb", "0"]);
    for (k, v) in env { c.env(k, v); }
    let out = c.output().expect("rope probe did not start");
    assert!(out.status.success(), "rope probe failed: {}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8(out.stdout).expect("probe output is not UTF-8")
}

#[test]
fn gpu_hidden_reports_null() {
    let j = probe(&[("CUDA_VISIBLE_DEVICES", "-1")]);
    assert!(j.contains("\"gpu\": null"), "GPU should be hidden:\n{j}");
    assert!(j.contains("\"ram_usable_gib\""), "rest of the probe must still work:\n{j}");
}

#[test]
fn no_driver_reports_null() {
    let j = probe(&[("ROPE_NVCUDA_DLL", "rope_no_such_driver.dll")]);
    assert!(j.contains("\"gpu\": null"), "missing driver should give gpu=null:\n{j}");
    assert!(j.contains("\"avx2\""), "CPU fields must still be present:\n{j}");
}

#[test]
fn default_probe_runs() {
    let j = probe(&[]);
    assert!(j.trim_start().starts_with('{') && j.trim_end().ends_with('}'), "probe output is not a JSON object:\n{j}");
}
