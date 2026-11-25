"""P5.1.2 check: Rope-Setup.exe installs for this user, the Start menu entry and the Apps entry exist, `rope app` serves the
app and makes a picture, and the uninstaller removes everything it added.

    python bench/tools/check_installer.py        (after installer/build-installer.ps1)

The install goes to a temp program folder and a temp data folder (ROPE_INSTALL_DIR / ROPE_DATA_DIR), so the real
installation (if any) is not touched. The Start menu entry and the Apps entry are the real ones: the check removes them.
No download: the models of this repo are hard-linked into the temp data folder (same disk, no copy); `rope get` then only
verifies the plan's files. Exit 1 on any failure.
"""
import json, os, shutil, subprocess, sys, tempfile, time, urllib.request, winreg

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SETUP = os.path.join(ROOT, "build", "installer", "Rope-Setup.exe")
KEY = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\Rope"
LNK = os.path.join(os.environ["APPDATA"], r"Microsoft\Windows\Start Menu\Programs\Rope.lnk")
PORT = 4110
fails = []

def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what)
    if not ok: fails.append(what)

def reg(name):
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, KEY) as k: return winreg.QueryValueEx(k, name)[0]
    except OSError: return None

def get(path, timeout=10):
    return urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=timeout)

def main():
    if reg("DisplayName") or os.path.exists(LNK):
        sys.exit("Rope is already installed for this user (Start menu or Apps entry): uninstall it first")
    tmp = tempfile.mkdtemp(prefix="rope-inst-")
    prog, data = os.path.join(tmp, "prog"), os.path.join(tmp, "data")
    # the models, hard-linked (no copy, no download)
    src = os.path.join(ROOT, "models")
    for d, dirs, files in os.walk(src):
        dirs[:] = [x for x in dirs if not x.startswith(".") and x != "addons"]
        for f in files:
            dst = os.path.join(data, "models", os.path.relpath(os.path.join(d, f), src))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.link(os.path.join(d, f), dst)
    env = dict(os.environ, ROPE_INSTALL_DIR=prog, ROPE_DATA_DIR=data, ROPE_INSTALL_GET="yes", ROPE_INSTALL_START="no")
    srv = None
    try:
        # 1. the setup file itself, as a double-click runs it (it extracts to temp and runs install.cmd)
        t0 = time.time()
        r = subprocess.run([SETUP], env=env, timeout=1800)
        check(r.returncode == 0, f"Rope-Setup.exe finished without a UAC prompt or error ({time.time() - t0:.0f} s, exit {r.returncode})")
        for f in ["rope.exe", "uninstall.cmd", r"catalog\models.json", r"app\index.html"]:
            check(os.path.exists(os.path.join(prog, f)), f"installed {f}")
        check(os.path.exists(LNK), "Start menu entry Rope.lnk")
        check(reg("DisplayName") == "Rope" and prog in (reg("UninstallString") or ""), "Apps entry with the uninstall command")
        # 2. the app, as the Start menu entry starts it
        srv = subprocess.Popen([os.path.join(prog, "rope.exe"), "app", "--port", str(PORT), "--no-browser"], env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        for _ in range(100):
            try: health = json.load(get("/api/health")); break
            except OSError: time.sleep(0.1)
        else: health = {}
        check(health.get("api") == 1, f"rope app answers on {PORT}: {health}")
        page = get("/").read().decode("utf-8", "replace")
        check("<div id=\"root\"" in page or "<script" in page, "the app page is served")
        out = os.path.join(tmp, "img.png")
        r = subprocess.run([sys.executable, os.path.join(ROOT, "examples", "client.py"), "a red bicycle", out, "--size", "256", "--port", str(PORT)],
                           capture_output=True, text=True, timeout=900)
        ok = r.returncode == 0 and os.path.exists(out) and open(out, "rb").read(8) == b"\x89PNG\r\n\x1a\n"
        check(ok, "a picture made through the installed app" + ("" if ok else f": {r.stdout[-300:]} {r.stderr[-300:]}"))
        pics = os.listdir(os.path.join(data, "gallery")) if os.path.isdir(os.path.join(data, "gallery")) else []
        check(any(p.endswith(".png") for p in pics), "the picture is in the data folder's gallery")
        # 3. a second start only opens the browser (no second server)
        r = subprocess.run([os.path.join(prog, "rope.exe"), "app", "--port", str(PORT), "--no-browser"], env=env, capture_output=True, text=True, timeout=30)
        check("already runs" in r.stdout, "a second start finds the running app")
    finally:
        if srv: srv.kill(); srv.wait()
    # 4. uninstall, as Settings - Apps runs it
    r = subprocess.run(["cmd.exe", "/c", os.path.join(prog, "uninstall.cmd")], env=dict(env, ROPE_UNINSTALL_DATA="yes"), timeout=120)
    time.sleep(5)
    check(not os.path.exists(LNK), "uninstall removed the Start menu entry")
    check(reg("DisplayName") is None, "uninstall removed the Apps entry")
    check(not os.path.exists(prog), "uninstall removed the program folder")
    check(not os.path.exists(data), "uninstall removed the data folder (answer: yes)")
    shutil.rmtree(tmp, ignore_errors=True)
    print("RESULT", "PASS" if not fails else f"FAIL ({len(fails)})")
    return 1 if fails else 0

if __name__ == "__main__":
    sys.exit(main())
