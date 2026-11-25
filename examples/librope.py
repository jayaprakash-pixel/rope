"""librope from Python with ctypes only (no package to install).

    python examples/librope.py MODEL_FILE OUT.png "prompt" [--lib build/rust-cpu/release/librope.dll] [--seed 7]
"""
import argparse, ctypes, json, os


class Rope:
    def __init__(self, path):
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(os.path.dirname(os.path.abspath(path)))
        L = self.lib = ctypes.CDLL(path)
        self.progress_t = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p)
        L.rope_version.restype = ctypes.c_char_p
        L.rope_job_start.restype = ctypes.c_void_p
        L.rope_job_start.argtypes = [ctypes.c_char_p, self.progress_t, ctypes.c_void_p]
        L.rope_job_wait.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
        L.rope_job_cancel.argtypes = [ctypes.c_void_p]
        L.rope_job_free.argtypes = [ctypes.c_void_p]

    def version(self):
        return self.lib.rope_version().decode()

    def error(self):
        buf = ctypes.create_string_buffer(1024)
        self.lib.rope_last_error(buf, len(buf))
        return buf.value.decode()

    def _json(self, fn, arg):
        cap = 1 << 16
        while True:                                     # snprintf rule: a result >= cap means the buffer was too small
            buf = ctypes.create_string_buffer(cap)
            n = fn(arg, buf, cap)
            if n < 0:
                raise RuntimeError(self.error())
            if n < cap:
                return json.loads(buf.value.decode())
            cap = n + 1

    def plan(self, **request):
        return self._json(self.lib.rope_plan_json, json.dumps(request).encode())

    def scan(self, path):
        return self._json(self.lib.rope_scan_json, path.encode())

    def start(self, on_progress=None, **request):
        """Start one image (request fields: engine/include/rope.h). Returns (job, callback); keep the callback alive until the job ends."""
        cb = self.progress_t(lambda stage, i, n, user: on_progress(stage.decode(), i, n)) if on_progress else self.progress_t()
        job = self.lib.rope_job_start(json.dumps(request).encode(), cb, None)
        if not job:
            raise RuntimeError(self.error())
        return job, cb

    def finish(self, job):
        """Wait for the job, free it. Returns 0 (done), 1 (failed) or 2 (cancelled) and the message."""
        msg = ctypes.create_string_buffer(1024)
        rc = self.lib.rope_job_wait(job, msg, len(msg))
        self.lib.rope_job_free(job)
        return rc, msg.value.decode()

    def generate(self, on_progress=None, **request):
        """Make one image and wait. Raises on failure or cancel."""
        job, cb = self.start(on_progress, **request)
        rc, msg = self.finish(job)
        if rc != 0:
            raise RuntimeError(("cancelled" if rc == 2 else "failed") + ": " + msg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("prompt")
    ap.add_argument("--lib", default=os.path.join(os.path.dirname(__file__), "..", "build", "rust-cpu", "release", "librope.dll"))
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rope = Rope(a.lib)
    print("librope", rope.version())
    rope.generate(lambda s, i, n: print(f"  {s} {i}/{n}"), model=a.model, prompt=a.prompt, out=a.out, quality="fast", size=256, steps=1, seed=a.seed, no_gpu=True)
    print("wrote", a.out)
