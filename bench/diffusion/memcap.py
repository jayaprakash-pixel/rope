"""Run a command as if on a weaker Windows laptop, using a Job Object.

A real 4 GB laptop does not kill a program at 4 GB: Windows itself uses ~1.5 GB, the program gets
~2.5 GB of physical RAM, and the rest pages to the page file. So we set two limits:
  --ws-gb      hard cap on the process working set (physical RAM the program may hold)
  --commit-gb  cap on committed memory (RAM + page file the program may allocate)
Profiles: --profile 4gb  -> ws 2.5, commit 6.0   |   --profile 8gb -> ws 5.5, commit 12.0
Usage:  python -I memcap.py --profile 4gb [--no-gpu] -- cmd args...      python -I memcap.py --selfcheck
"""
import ctypes, os, subprocess, sys, time
from ctypes import wintypes as W

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
ntdll = ctypes.WinDLL("ntdll")
PROFILES = {"4gb": (2.5, 6.0), "8gb": (5.5, 12.0)}
GB = 1 << 30


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [(n, ctypes.c_ulonglong) for n in ("r", "w", "o", "rb", "wb", "ob")]


class BASIC(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", W.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", W.DWORD),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", W.DWORD), ("SchedulingClass", W.DWORD)]


class EXTENDED(ctypes.Structure):
    _fields_ = [("Basic", BASIC), ("Io", IO_COUNTERS), ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t)]


LIMIT_WORKINGSET, LIMIT_JOB_MEMORY, LIMIT_KILL_ON_CLOSE = 0x1, 0x200, 0x2000
CREATE_SUSPENDED = 0x4
HARDWS_MAX, HARDWS_MIN_OFF = 0x4, 0x2
k32.SetProcessWorkingSetSizeEx.argtypes = [W.HANDLE, ctypes.c_size_t, ctypes.c_size_t, W.DWORD]
k32.CreateJobObjectW.restype = W.HANDLE
k32.SetInformationJobObject.argtypes = [W.HANDLE, ctypes.c_int, ctypes.c_void_p, W.DWORD]
k32.AssignProcessToJobObject.argtypes = [W.HANDLE, W.HANDLE]
k32.QueryInformationJobObject.argtypes = [W.HANDLE, ctypes.c_int, ctypes.c_void_p, W.DWORD, ctypes.c_void_p]


def spawn(cmd, ws_gb=None, commit_gb=None, no_gpu=False, **popen_kw):
    """Start cmd inside a job with the given caps. Returns (Popen, job_handle)."""
    env = dict(popen_kw.pop("env", None) or os.environ)
    if no_gpu:
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    p = subprocess.Popen(cmd, creationflags=CREATE_SUSPENDED, env=env, **popen_kw)
    job = k32.CreateJobObjectW(None, None)
    info = EXTENDED()
    info.Basic.LimitFlags = LIMIT_KILL_ON_CLOSE
    if commit_gb:
        info.Basic.LimitFlags |= LIMIT_JOB_MEMORY
        info.JobMemoryLimit = int(commit_gb * GB)
    if not k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
        p.kill(); raise OSError(ctypes.get_last_error(), "SetInformationJobObject failed")
    if not k32.AssignProcessToJobObject(job, int(p._handle)):
        p.kill(); raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")
    if ws_gb:  # hard per-process working-set cap (a job WS limit needs an admin privilege; this does not)
        if not k32.SetProcessWorkingSetSizeEx(W.HANDLE(int(p._handle)), ctypes.c_size_t(16 << 20),
                                              ctypes.c_size_t(int(ws_gb * GB)), W.DWORD(HARDWS_MAX | HARDWS_MIN_OFF)):
            p.kill(); raise OSError(ctypes.get_last_error(), "SetProcessWorkingSetSizeEx failed")
    ntdll.NtResumeProcess(W.HANDLE(int(p._handle)))
    return p, job


def job_peak_commit(job):
    info = EXTENDED()
    k32.QueryInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info), None)
    return info.PeakJobMemoryUsed


def _child(gb):
    return [sys.executable, "-I", "-c",
            f"b=bytearray({int(gb * GB)})\nfor i in range(0,len(b),4096): b[i]=1\nimport time; time.sleep(1); print('ok')"]


class PMC(ctypes.Structure):
    _fields_ = [("cb", W.DWORD), ("PageFaultCount", W.DWORD)] + [(n, ctypes.c_size_t) for n in (
        "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
        "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]


def mem_info(p):
    """Working set / peak working set / commit of a live or exited Popen (handle still open)."""
    m = PMC(); m.cb = ctypes.sizeof(m)
    k32.K32GetProcessMemoryInfo(W.HANDLE(int(p._handle)), ctypes.byref(m), m.cb)
    return m


def _peak_ws(p):
    p.wait()
    return mem_info(p).PeakWorkingSetSize


def selfcheck():
    # 1. small child completes under the 4gb profile
    p, _ = spawn(_child(1.0), *PROFILES["4gb"], stdout=subprocess.PIPE)
    out = p.communicate()[0].decode().strip()
    assert p.returncode == 0 and out == "ok", ("1 GB child should finish", p.returncode, out)
    # 2. child above the commit cap is stopped (allocation fails)
    p, _ = spawn(_child(7.0), *PROFILES["4gb"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.communicate()
    assert p.returncode != 0, "7 GB child must fail under a 6 GB commit cap"
    # 3. working-set cap holds: 3.5 GB touched under a 2.5 GB WS cap -> completes, peak WS <= cap + 5%
    p, _ = spawn(_child(3.5), 2.5, 6.0, stdout=subprocess.PIPE)
    peak = _peak_ws(p)
    assert p.returncode == 0, "3.5 GB child should page, not die"
    assert peak <= 2.5 * GB * 1.05, f"peak working set {peak / GB:.2f} GB exceeds 2.5 GB cap"
    print(f"selfcheck ok (WS cap held: peak {peak / GB:.2f} GB under 2.50 GB cap)")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["--selfcheck"]:
        selfcheck(); sys.exit(0)
    ws = commit = None; no_gpu = False
    while a and a[0] != "--":
        x = a.pop(0)
        if x == "--profile": ws, commit = PROFILES[a.pop(0)]
        elif x == "--ws-gb": ws = float(a.pop(0))
        elif x == "--commit-gb": commit = float(a.pop(0))
        elif x == "--no-gpu": no_gpu = True
    p, _ = spawn(a[1:], ws, commit, no_gpu)
    sys.exit(p.wait())
