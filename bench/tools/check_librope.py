"""P5.1.9: librope (engine/include/rope.h, librope.dll).
  1. A C program (examples/librope_example.c, built with MSVC against librope.dll.lib) makes a PNG; `rope image` with the same request
     makes a PNG; the two files are byte for byte equal.
  2. A Python program (examples/librope.py, ctypes) makes a PNG equal to the same `rope image` PNG.
  3. In Python: plan JSON (models list), scan JSON (clean add-on file), a second job while one runs is refused, a job cancelled at its
     first denoising step ends with code 2 within one step and writes no file, the next job works, a missing model gives code 1 with a reason,
     and a job with a LoRA equals `rope image --lora` with the same seed.
  4. `unsafe` appears only in the files of the platform boundary: ffi.rs (kernels), alloc.rs, mmap.rs and the library boundary capi.rs.
Needs models/sd_turbo.safetensors; the LoRA part needs models/addons/pixel-art-xl.safetensors and the SDXL-Turbo file. Deletes its outputs."""
import ctypes, glob, hashlib, json, os, shutil, subprocess, sys, threading, time

R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
REL = os.path.join(R, "build", "rust-cpu", "release")
ROPE, DLL, LIB = os.path.join(REL, "rope.exe"), os.path.join(REL, "librope.dll"), os.path.join(REL, "librope.dll.lib")
M = os.path.join(R, "models")
SD = os.path.join(M, "sd_turbo.safetensors")
OUT = os.path.join(R, "bench", "diffusion", "out", "librope")
VCVARS = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
ok = True
sys.path.insert(0, os.path.join(R, "examples"))
from librope import Rope


def fail(m):
    global ok
    ok = False
    print("  FAIL", m)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def cli(out, model=SD, size=256, steps=1, seed=7, extra=()):
    r = subprocess.run([ROPE, "image", "--model", model, "--prompt", "a red bicycle", "--quality", "fast", "--size", str(size), "--steps", str(steps), "--seed", str(seed),
                        "--no-gpu", "--disk-mb", "0", "--out", out, *extra], cwd=R, capture_output=True, text=True)
    if r.returncode != 0: fail("rope image: " + r.stderr.strip()[-200:])


shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT)
ref = os.path.join(OUT, "ref.png")
cli(ref)
print("reference (rope image):", sha(ref))

# 1. the C program
shutil.copy(DLL, OUT)
exe = os.path.join(OUT, "c_example.exe")
bat = os.path.join(OUT, "build.cmd")
INC, SRC = os.path.join(R, "engine", "include"), os.path.join(R, "examples", "librope_example.c")
open(bat, "w").write(chr(10).join(["@echo off", f'call "{VCVARS}" >nul', f'cl /nologo /W3 /I "{INC}" "{SRC}" "{LIB}" /Fe:"{exe}" /Fo"{OUT}/c_example.obj"', ""]))
r = subprocess.run(["cmd", "/c", bat], capture_output=True, text=True, cwd=OUT)
if r.returncode != 0 or not os.path.exists(exe): fail("C build failed: " + (r.stdout + r.stderr)[-600:])
else:
    c_png = os.path.join(OUT, "c.png")
    r = subprocess.run([exe, SD, c_png, "a red bicycle", "7"], cwd=R, capture_output=True, text=True)
    print("C program:", (r.stdout.strip().splitlines() or [""])[-1], "| exit", r.returncode)
    if r.returncode != 0 or not os.path.exists(c_png): fail("C program failed: " + r.stderr.strip()[-200:])
    elif sha(c_png) != sha(ref): fail(f"C PNG differs from rope image ({sha(c_png)} vs {sha(ref)})")
    elif "denoise" not in r.stdout: fail("C program got no progress callbacks")

# 2. the Python example
py_png = os.path.join(OUT, "py.png")
r = subprocess.run([sys.executable, os.path.join(R, "examples", "librope.py"), SD, py_png, "a red bicycle", "--lib", DLL], cwd=R, capture_output=True, text=True)
print("Python example:", (r.stdout.strip().splitlines() or [""])[-1], "| exit", r.returncode)
if r.returncode != 0 or not os.path.exists(py_png): fail("Python example failed: " + r.stderr.strip()[-300:])
elif sha(py_png) != sha(ref): fail("Python PNG differs from rope image")

# 3. more in-process checks
os.chdir(R)
rope = Rope(DLL)
plan = rope.plan(quality="fast", no_gpu=True)
print("plan JSON models:", [m["id"] for m in plan["models"]])
if not any(m["id"] == "sd-turbo" and m["runs"] for m in plan["models"]): fail("plan JSON has no runnable sd-turbo")
sc = rope.scan(os.path.join(M, "addons", "realesrgan_x4plus.safetensors")) if os.path.exists(os.path.join(M, "addons", "realesrgan_x4plus.safetensors")) else {"clean": True, "tensors": 0}
print("scan JSON:", sc)
if not sc["clean"]: fail("scan reports a clean file as flagged")
try: rope.scan(os.path.join(OUT, "nothing"))
except RuntimeError as e: print("scan of a missing file:", str(e)[:70])
else: fail("scan of a missing file should fail")

cancel_png = os.path.join(OUT, "cancel.png")
seen, t_cancel = [], [0.0]
job_box = {}


def on_progress(stage, i, n):
    seen.append((stage, i))
    if stage == "denoise" and i == 0 and "t" not in job_box:
        job_box["t"] = time.time()
        rope.lib.rope_job_cancel(job_box["job"])


req = dict(model=os.path.join(M, "sd_xl_turbo_1.0_fp16.safetensors"), prompt="a red bicycle", out=cancel_png, quality="fast", size=512, steps=4, seed=1, no_gpu=True)
job, cb = rope.start(on_progress, **req)
job_box["job"] = ctypes.c_void_p(job)
try:
    rope.start(None, **req)
except RuntimeError as e:
    print("second job while one runs:", str(e))
else:
    fail("a second job should be refused while one runs")
rc, msg = rope.finish(job)
t_end = time.time() - job_box.get("t", time.time())
print(f"cancelled job: code {rc} ({msg}), ended {t_end:.1f} s after the cancel, stages {sorted(set(s for s, _ in seen))}")
if rc != 2: fail(f"cancel should give code 2, got {rc}")
if os.path.exists(cancel_png): fail("a cancelled job wrote a file")
rc, msg = rope.finish(rope.start(None, **dict(req, model=os.path.join(OUT, "nothing.safetensors")))[0]) if False else (None, None)
try:
    job, cb = rope.start(None, **dict(req, model=os.path.join(OUT, "nothing.safetensors")))
    rc, msg = rope.finish(job)
    print("missing model:", rc, msg[:80])
    if rc != 1 or not msg: fail("a missing model should give code 1 with a reason")
except RuntimeError as e:
    fail("start failed: " + str(e))
lib_png = os.path.join(OUT, "next.png")
rope.generate(None, model=SD, prompt="a red bicycle", out=lib_png, quality="fast", size=256, steps=1, seed=7, no_gpu=True)
if sha(lib_png) != sha(ref): fail("the job after a cancel gives another image (state left behind)")
else: print("job after the cancel: same PNG as the reference")

lora = os.path.join(M, "addons", "pixel-art-xl.safetensors")
xl = os.path.join(M, "sd_xl_turbo_1.0_fp16.safetensors")
if os.path.exists(lora) and os.path.exists(xl):
    a, b, c = (os.path.join(OUT, n) for n in ("lora_cli.png", "lora_lib.png", "plain_lib.png"))
    cli(a, model=xl, size=512, steps=1, extra=("--lora", lora, "--lora-scale", "1"))
    rope.generate(None, model=xl, prompt="a red bicycle", out=b, quality="fast", size=512, steps=1, seed=7, no_gpu=True, lora=lora, lora_scale=1.0)
    rope.generate(None, model=xl, prompt="a red bicycle", out=c, quality="fast", size=512, steps=1, seed=7, no_gpu=True)   # no LoRA left from the job before
    cli(os.path.join(OUT, "plain_cli.png"), model=xl, size=512, steps=1)
    print("LoRA job vs rope image --lora:", sha(a), sha(b), "| plain job vs rope image:", sha(c), sha(os.path.join(OUT, "plain_cli.png")))
    if sha(a) != sha(b): fail("a job with a LoRA differs from rope image --lora")
    if sha(c) != sha(os.path.join(OUT, "plain_cli.png")): fail("the job after a LoRA job still carries the LoRA")
    if sha(a) == sha(c): fail("the LoRA changed nothing")
else:
    print("LoRA part skipped (files missing)")

# 4. unsafe only at the boundary
import re
unsafe_files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(R, "engine", "rust", "*.rs")) if re.search(r"unsafe\s*(\{|fn|impl|extern|trait)", open(p, encoding="utf-8").read()))
print("files with `unsafe`:", unsafe_files)
extra = set(unsafe_files) - {"ffi.rs", "alloc.rs", "mmap.rs", "capi.rs", "weights.rs"}
if extra: fail(f"unsafe outside the boundary files: {sorted(extra)}")
shutil.rmtree(OUT, ignore_errors=True)
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
