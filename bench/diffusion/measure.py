"""Run one benchmark and record: wall time, sd.cpp phase times, peak RAM (working set) and commit,
GPU dedicated + shared (spill) memory of the process, system disk reads, minimum free RAM.
Stdlib only (runs with python -I).  Results: one JSON line per run in results.jsonl, child log in out/<label>.log

  python -I measure.py --label sdturbo_gpu [--profile 4gb] [--no-gpu] -- bin/sd-cuda/sd-cli.exe ...
  python -I measure.py --selfcheck
"""
import base64, json, os, re, subprocess, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memcap

HERE = os.path.dirname(os.path.abspath(__file__))
GB = 1 << 30
SD_TIMES = {  # sd.cpp log line -> field (seconds)
    "load_s": r"loading tensors completed, taking ([\d.]+)s",
    "text_enc_s": r"get_learned_condition completed, taking ([\d.]+)s",
    "sampling_s": r"generating (?:\d+ latent images|latent video) completed, taking ([\d.]+)s",
    "vae_s": r"decode_first_stage completed, taking ([\d.]+)s",
    "total_s": r"generate_(?:image|video) completed in ([\d.]+)s",
}

PS_SAMPLER = r"""
$p={pid}; $c=@("\GPU Process Memory(pid_$($p)_*)\Dedicated Usage","\GPU Process Memory(pid_$($p)_*)\Shared Usage",
 "\PhysicalDisk(_Total)\Disk Read Bytes/sec","\Memory\Available MBytes")
while (Get-Process -Id $p -ErrorAction SilentlyContinue) {{
  try {{ $s=(Get-Counter -Counter $c -ErrorAction SilentlyContinue).CounterSamples
    $d=($s|?{{$_.Path -like '*dedicated usage'}}|Measure-Object CookedValue -Sum).Sum
    $h=($s|?{{$_.Path -like '*shared usage'}}|Measure-Object CookedValue -Sum).Sum
    $r=($s|?{{$_.Path -like '*disk read bytes/sec'}}).CookedValue
    $a=($s|?{{$_.Path -like '*available mbytes'}}).CookedValue
    "$([DateTimeOffset]::Now.ToUnixTimeMilliseconds()) $([double]$d) $([double]$h) $([double]$r) $([double]$a)" }} catch {{}} }}
"""


def run(label, cmd, ws_gb=None, commit_gb=None, no_gpu=False, sample=True):
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    log_path = os.path.join(HERE, "out", f"{label}.log")
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        p, job = memcap.spawn(cmd, ws_gb, commit_gb, no_gpu, stdout=log, stderr=subprocess.STDOUT)
        samples = []
        if sample:
            enc = base64.b64encode(PS_SAMPLER.format(pid=p.pid).encode("utf-16-le")).decode()
            ps = subprocess.Popen(["powershell", "-NoProfile", "-EncodedCommand", enc],
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            th = threading.Thread(target=lambda: samples.extend(ps.stdout), daemon=True); th.start()
        rc = p.wait()
        wall = time.time() - t0
        m = memcap.mem_info(p)
        if sample:
            ps.wait(timeout=30); th.join(timeout=5)
    rows = []
    for line in samples:
        f = line.split()
        if len(f) == 5:
            try:
                rows.append([float(x or 0) for x in f])
            except ValueError:
                pass
    disk = sum(r[3] * (b[0] - r[0]) / 1000 for r, b in zip(rows, rows[1:]))  # rate x interval
    text = open(log_path, encoding="utf-8", errors="replace").read()
    res = {"label": label, "cmd": " ".join(cmd), "rc": rc, "wall_s": round(wall, 2),
           "profile": {"ws_gb": ws_gb, "commit_gb": commit_gb, "no_gpu": no_gpu},
           "peak_ws_gb": round(m.PeakWorkingSetSize / GB, 3), "peak_commit_gb": round(m.PeakPagefileUsage / GB, 3),
           "page_faults": m.PageFaultCount,
           "gpu_dedicated_peak_gb": round(max((r[1] for r in rows), default=0) / GB, 3),
           "gpu_shared_peak_gb": round(max((r[2] for r in rows), default=0) / GB, 3),
           "disk_read_gb": round(disk / GB, 2), "avail_ram_min_mb": min((r[4] for r in rows), default=None),
           "n_samples": len(rows), "when": time.strftime("%Y-%m-%d %H:%M:%S")}
    for k, rx in SD_TIMES.items():
        hit = re.findall(rx, text)
        if hit:
            res[k] = float(hit[-1])
    with open(os.path.join(HERE, "results.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(res) + "\n")
    return res


def selfcheck():
    s = run("_selfcheck_sampler6", [sys.executable, "-I", "-c", "import time; time.sleep(6)"])
    assert s["n_samples"] >= 2 and s["avail_ram_min_mb"], f"sampler: {s['n_samples']} samples"
    r = run("_selfcheck_sleep2", [sys.executable, "-I", "-c", "import time; b=bytearray(200<<20); time.sleep(2)"])
    assert r["rc"] == 0, r
    assert 1.7 <= r["wall_s"] <= 2.6, f"wall {r['wall_s']} not 2.0 +/- 0.3 (+0.3 spawn overhead)"
    assert r["peak_ws_gb"] > 0.15, f"peak RAM {r['peak_ws_gb']} GB should see the 200 MB buffer"
    print(f"selfcheck ok: wall {r['wall_s']} s, peak RAM {r['peak_ws_gb']} GB; sampler {s['n_samples']} samples in 6 s, "
          f"min free RAM {s['avail_ram_min_mb']:.0f} MB")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["--selfcheck"]:
        selfcheck(); sys.exit(0)
    label = None; ws = commit = None; no_gpu = False
    while a and a[0] != "--":
        x = a.pop(0)
        if x == "--label": label = a.pop(0)
        elif x == "--profile": ws, commit = memcap.PROFILES[a.pop(0)]
        elif x == "--ws-gb": ws = float(a.pop(0))
        elif x == "--commit-gb": commit = float(a.pop(0))
        elif x == "--no-gpu": no_gpu = True
    r = run(label, a[1:], ws, commit, no_gpu)
    print(json.dumps(r, indent=1))
    sys.exit(r["rc"])
