//! C ABI of the kernel library `ropek` (engine/src). Safe wrappers check arguments before calling C.
use std::ffi::{c_char, c_void, CString};

extern "C" {
    fn rope_wtype_supported(t: i32) -> i32;
    fn rope_wtype_block(t: i32) -> i32;
    fn rope_row_bytes(t: i32, k: i64) -> usize;
    fn rope_dequant_row(t: i32, src: *const c_void, dst: *mut f32, k: i64);
    fn rope_quantize_rows(t: i32, x: *const f32, dst: *mut c_void, rows: i64, k: i64) -> i32;
    fn rope_set_threads(n: i32);
    fn rope_run_tests(filter: *const c_char) -> i32;
    fn rope_c_has_cuda() -> i32;
    fn rope_c_probe(dir: *const c_char, disk_mb: i32) -> i32;
    fn rope_c_bench(m: i64, k: i64, n: i64, t: i32) -> i32;
    #[cfg(feature = "cuda")]
    fn rope_c_bench_cuda(m: i64, k: i64, n: i64, t: i32, mode: i32) -> i32;
    fn rope_matmul(x: *const f32, m: i64, k: i64, w: *const c_void, wtype: i32, n: i64, y: *mut f32);
    fn rope_attention_ex(q: *const f32, k: *const f32, v: *const f32, o: *mut f32, nq: i64, nk: i64, d: i32,
                         qs: i64, ks: i64, vs: i64, os: i64, scale: f32, causal: i32, bias: *const f32);
    fn rope_layernorm(x: *mut f32, m: i64, n: i64, g: *const f32, b: *const f32, eps: f32);
    fn rope_gelu_erf(x: *mut f32, n: i64);
    fn rope_gelu_tanh(x: *mut f32, n: i64);
    fn rope_add_rows(y: *mut f32, m: i64, n: i64, b: *const f32);
    fn rope_c_probe_json(dir: *const c_char, disk_mb: i32, out: *mut c_char, cap: usize) -> i32;
    fn rope_c_gemm_gflops() -> f64;
    fn rope_get_threads() -> i32;
    fn rope_silu(x: *mut f32, n: i64);
    fn rope_groupnorm(x: *mut f32, c: i64, hw: i64, groups: i32, g: *const f32, b: *const f32, eps: f32);
    fn rope_conv2d(inp: *const f32, c: i32, h: i32, w: i32, wt: *const c_void, wtype: i32, bias: *const f32,
                   o: i32, kh: i32, kw: i32, stride: i32, pad: i32, out: *mut f32);
}

// ---------------- profiling: count work per weight block (optionally without computing) ----------------

/// Work counter for a profiling run. Kernel calls add their FLOPs to the block that owns the weight they read
/// (found by address); attention, which reads no weight, goes to the last block seen. In `dry` mode the kernels
/// skip the arithmetic, so a whole model runs in seconds with exact shapes and real allocations.
pub struct Counter {
    pub dry: bool,
    ranges: Vec<(usize, usize, String)>,          // weight address ranges -> block, sorted by start
    pub flops: std::collections::HashMap<String, f64>,
    last: String,
    pub peak_private: u64,
    pub kernels: std::collections::BTreeMap<&'static str, (f64, f64)>,   // kind -> (seconds, FLOPs), timed runs only
    pub shapes: std::collections::HashMap<(usize, usize, usize, u32), (u32, f64)>,  // matmul (m,k,n,type) -> (calls, seconds)
}

static COUNTER: std::sync::Mutex<Option<Counter>> = std::sync::Mutex::new(None);

/// Start counting. `ranges`: (start address, length, block) of every weight tensor.
pub fn profile_begin(dry: bool, mut ranges: Vec<(usize, usize, String)>) {
    ranges.sort_by_key(|r| r.0);
    let ranges = ranges.into_iter().map(|(a, n, b)| (a, a + n, b)).collect();
    *COUNTER.lock().unwrap() = Some(Counter { dry, ranges, flops: Default::default(), last: "(none)".into(), peak_private: private_bytes(), kernels: Default::default(), shapes: Default::default() });
}

/// Stop counting and return the counts.
pub fn profile_end() -> Option<Counter> { COUNTER.lock().unwrap().take() }

/// Add work; returns true when the kernel must skip its arithmetic (dry run).
fn account(kind: &'static str, weight: Option<*const u8>, flops: f64) -> bool {
    let mut g = COUNTER.lock().unwrap();
    let Some(c) = g.as_mut() else { return false };
    c.kernels.entry(kind).or_default().1 += flops;
    if let Some(p) = weight {
        let a = p as usize;
        let i = c.ranges.partition_point(|r| r.0 <= a);
        if i > 0 && a < c.ranges[i - 1].1 { c.last = c.ranges[i - 1].2.clone(); }
    }
    *c.flops.entry(c.last.clone()).or_default() += flops;
    c.peak_private = c.peak_private.max(private_bytes());
    c.dry
}

/// After a kernel ran during a timed profiling run: add its wall time to its kind.
fn kernel_done(kind: &'static str, t0: std::time::Instant) {
    if let Some(c) = COUNTER.lock().unwrap().as_mut() { c.kernels.entry(kind).or_default().0 += t0.elapsed().as_secs_f64(); }
}

/// Ask the OS to read these mapped address ranges (start, length) into RAM with large concurrent reads
/// (PrefetchVirtualMemory, Windows 8+). A hint: failures are ignored, the data still faults in on use.
#[cfg(windows)]
pub fn prefetch(ranges: &[(usize, usize)]) {
    #[repr(C)]
    struct Entry { addr: *mut c_void, len: usize }
    extern "system" {
        fn GetCurrentProcess() -> *mut c_void;
        fn PrefetchVirtualMemory(h: *mut c_void, n: usize, e: *const Entry, flags: u32) -> i32;
    }
    let e: Vec<Entry> = ranges.iter().filter(|r| r.1 > 0).map(|&(a, n)| Entry { addr: a as *mut c_void, len: n }).collect();
    if e.is_empty() { return; }
    // SAFETY: the ranges are views of a live read-only file mapping; the call only reads page-table state.
    unsafe { PrefetchVirtualMemory(GetCurrentProcess(), e.len(), e.as_ptr(), 0); }
}
#[cfg(not(windows))]
pub fn prefetch(ranges: &[(usize, usize)]) {
    extern "C" { fn madvise(addr: *mut c_void, len: usize, advice: i32) -> i32; }
    const MADV_WILLNEED: i32 = 3;
    for &(a, n) in ranges {
        // SAFETY: the range is a view of a live file mapping; MADV_WILLNEED only starts reading ahead (a hint, errors ignored).
        // The start must be page aligned: round it down inside the mapping's first page.
        if n > 0 { let start = a & !4095; unsafe { madvise(start as *mut c_void, n + (a - start), MADV_WILLNEED); } }
    }
}

/// Read one byte of every 4 KB page in these mapped ranges, so the pages are mapped into this process before the
/// compute threads need them (a prefetched page still costs a soft fault per page on first touch).
pub fn touch_pages(ranges: &[(usize, usize)]) {
    let mut sum = 0u8;
    for &(a, n) in ranges {
        let mut p = a;
        // SAFETY: every address is inside a live read-only file mapping of at least n bytes.
        while p < a + n { sum = sum.wrapping_add(unsafe { std::ptr::read_volatile(p as *const u8) }); p += 4096; }
    }
    std::hint::black_box(sum);
}

/// (private bytes now, peak commit, peak working set) of this process.
#[cfg(windows)]
fn mem_counters() -> (u64, u64, u64) {
    #[repr(C)]
    struct Pmc { cb: u32, faults: u32, peak_ws: usize, ws: usize, qppp: usize, qpp: usize, qpnpp: usize, qnpp: usize, pagefile: usize, peak_pagefile: usize, private: usize }
    extern "system" {
        fn GetCurrentProcess() -> *mut c_void;
        fn K32GetProcessMemoryInfo(h: *mut c_void, p: *mut Pmc, cb: u32) -> i32;
    }
    let mut m = Pmc { cb: std::mem::size_of::<Pmc>() as u32, faults: 0, peak_ws: 0, ws: 0, qppp: 0, qpp: 0, qpnpp: 0, qnpp: 0, pagefile: 0, peak_pagefile: 0, private: 0 };
    // SAFETY: m is a correctly sized PROCESS_MEMORY_COUNTERS_EX; the call only writes into it.
    if unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut m, m.cb) } != 0 { (m.private as u64, m.peak_pagefile as u64, m.peak_ws as u64) } else { (0, 0, 0) }
}
/// Linux: private bytes = anonymous resident memory (RssAnon, what the program allocates; mapped model weights are
/// not in it). /proc has no peak of it, so a sampler thread records the highest value (every 10 ms).
#[cfg(not(windows))]
fn mem_counters() -> (u64, u64, u64) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static PEAK: AtomicU64 = AtomicU64::new(0);
    static START: std::sync::Once = std::sync::Once::new();
    fn read() -> (u64, u64) {
        let mut anon = 0u64; let mut hwm = 0u64;
        if let Ok(s) = std::fs::read_to_string("/proc/self/status") {
            for l in s.lines() {
                let kb = |k: &str| l.strip_prefix(k).and_then(|v| v.trim().trim_end_matches("kB").trim().parse::<u64>().ok()).map(|v| v * 1024);
                if let Some(v) = kb("RssAnon:") { anon = v; } else if let Some(v) = kb("VmHWM:") { hwm = v; }
            }
        }
        (anon, hwm)
    }
    START.call_once(|| { std::thread::spawn(|| loop { PEAK.fetch_max(read().0, Ordering::Relaxed); std::thread::sleep(std::time::Duration::from_millis(10)); }); });
    let (anon, hwm) = read();
    PEAK.fetch_max(anon, Ordering::Relaxed);
    (anon, PEAK.load(Ordering::Relaxed), hwm)
}

/// Private (committed) bytes of this process: activations and buffers; memory-mapped weights are not included.
pub fn private_bytes() -> u64 { mem_counters().0 }

/// (peak commit, peak working set) of this process so far, in bytes. Peak commit is the number the planner budgets.
pub fn peak_memory() -> (u64, u64) { let m = mem_counters(); (m.1, m.2) }

/// Hard cap on the working set (RAM the process may hold) of another process, to simulate a smaller laptop in tests.
/// `handle` is the child's process handle. Returns false when the OS refuses (or off Windows).
#[cfg(windows)]
pub fn cap_working_set(handle: *mut c_void, max_bytes: usize) -> bool {
    extern "system" { fn SetProcessWorkingSetSizeEx(h: *mut c_void, min: usize, max: usize, flags: u32) -> i32; }
    const HARDWS_MAX_ENABLE: u32 = 0x4;
    const HARDWS_MIN_DISABLE: u32 = 0x2;
    // SAFETY: the handle comes from a live std::process::Child; the call only changes that process's working-set limits.
    unsafe { SetProcessWorkingSetSizeEx(handle, 16 << 20, max_bytes, HARDWS_MAX_ENABLE | HARDWS_MIN_DISABLE) != 0 }
}
#[cfg(not(windows))]
pub fn cap_working_set(_handle: *mut c_void, _max_bytes: usize) -> bool { false }


// ---------------- GPU dispatch (G1): heavy ops of blocks the plan puts on the GPU run as CUDA kernels ----------------
// The block of a call is found from its weight's address, exactly like the profiler: model code never knows the device.
// G1 transfers per call: quantized weights are uploaded and converted to FP16 on the GPU, x goes up, y comes back.
// Any CUDA failure falls back to the CPU kernel, so a GPU problem can never produce a silently wrong result.

static GPU_RANGES: std::sync::Mutex<Vec<(usize, usize)>> = std::sync::Mutex::new(Vec::new());
/// Weights the plan keeps resident in VRAM (address ranges). Empty = any GPU weight may become resident.
static RESIDENT_RANGES: std::sync::Mutex<Vec<(usize, usize)>> = std::sync::Mutex::new(Vec::new());
/// The plan's GPU blocks and resident blocks, by block name (see profile::block_of).
static GPU_PLAN: std::sync::Mutex<Option<(std::collections::HashSet<String>, std::collections::HashSet<String>)>> = std::sync::Mutex::new(None);

/// Set the placement plan: blocks that run on the GPU and the subset kept resident. Weight files opened afterwards
/// register their matching address ranges (Weights::open -> gpu_register).
pub fn gpu_plan(gpu: std::collections::HashSet<String>, resident: std::collections::HashSet<String>) {
    *GPU_PLAN.lock().unwrap() = Some((gpu, resident));
}

/// Add a weight file's tensors to the GPU dispatch tables according to the plan (no plan: nothing happens).
pub fn gpu_register(tensors: &[(String, usize, usize)]) {
    let g = GPU_PLAN.lock().unwrap();
    let Some((gpu, res)) = g.as_ref() else { return };
    let mut on = GPU_RANGES.lock().unwrap();
    let mut keep = RESIDENT_RANGES.lock().unwrap();
    for (name, a, n) in tensors {
        let b = crate::profile::block_of(name);
        if gpu.contains(&b) { on.push((*a, a + n)); }
        if res.contains(&b) { keep.push((*a, a + n)); }
    }
    on.sort_by_key(|x| x.0);
    keep.sort_by_key(|x| x.0);
}

/// Remove every dispatch range inside [lo, hi) (a weight file was closed).
pub fn gpu_forget(lo: usize, hi: usize) {
    for t in [&GPU_RANGES, &RESIDENT_RANGES] { t.lock().unwrap().retain(|r| r.1 <= lo || r.0 >= hi); }
}

/// True when a placement plan exists and streams this GPU weight (not kept resident): block-ahead staging applies.
#[cfg(feature = "cuda")]
fn plan_streams(p: usize) -> bool {
    let g = RESIDENT_RANGES.lock().unwrap();
    if g.is_empty() || GPU_PLAN.lock().unwrap().is_none() { return false; }
    let i = g.partition_point(|r| r.0 <= p);
    !(i > 0 && p < g[i - 1].1)
}

#[cfg(feature = "cuda")]
fn want_resident(p: usize) -> bool {
    let g = RESIDENT_RANGES.lock().unwrap();
    if g.is_empty() { return true; }
    let i = g.partition_point(|r| r.0 <= p);
    i > 0 && p < g[i - 1].1
}

/// Device-resident activations (G3): the CUDA build has the real API; the CPU build has the same names, never used.
#[cfg(feature = "cuda")]
pub use cuda::{dev_add, dev_add_gated, dev_add_row_scalars, dev_alloc, dev_attention, dev_attention3, dev_conv2d, dev_copy2d, dev_download,
               dev_gelu_erf, dev_gelu_tanh, dev_groupnorm, dev_layernorm, dev_matmul, dev_modulate, dev_mul, dev_qk_norm_rope, dev_silu,
               dev_transpose, dev_upload, dev_upsample2x, DevBuf};
#[cfg(feature = "cuda")]
pub fn weight_on_gpu(w: &[u8]) -> bool { on_gpu(w.as_ptr()) }

/// VRAM manager statistics: (budget, peak used, resident hits, bytes uploaded, resident bytes).
#[cfg(feature = "cuda")]
pub fn gpu_stats() -> Option<(usize, usize, u64, u64, usize)> { cuda::stats() }
#[cfg(feature = "cuda")]
pub fn gpu_upload_seconds() -> f64 { cuda::upload_seconds() }
#[cfg(feature = "cuda")]
pub fn gpu_staged_bytes() -> u64 { cuda::staged_bytes() }
/// Block-ahead GPU staging of these weight ranges (no-op without a GPU plan or in the CPU build).
#[cfg(feature = "cuda")]
pub fn gpu_stage(ranges: Vec<(usize, usize)>) { cuda::stage(ranges) }
#[cfg(not(feature = "cuda"))]
pub fn gpu_stage(_ranges: Vec<(usize, usize)>) {}

/// Weights in these address ranges (start, length) run on the GPU. Empty = everything on the CPU.
pub fn gpu_set_ranges(mut r: Vec<(usize, usize)>) {
    r.sort_by_key(|x| x.0);
    *GPU_RANGES.lock().unwrap() = r.into_iter().map(|(a, n)| (a, a + n)).collect();
}

#[cfg(feature = "cuda")]
fn on_gpu(p: *const u8) -> bool {
    let g = GPU_RANGES.lock().unwrap();
    let a = p as usize;
    let i = g.partition_point(|r| r.0 <= a);
    i > 0 && a < g[i - 1].1
}
#[cfg(not(feature = "cuda"))]
fn on_gpu(_p: *const u8) -> bool { false }

/// Ops without a weight (attention) belong to the block of the last weight op, like the profiler's attribution:
/// they run on the GPU when that op did.
static LAST_ON_GPU: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[cfg(feature = "cuda")]
mod cuda {
    //! VRAM manager (G2a, P2.1.2). Owns every large device buffer:
    //!  - resident: raw (stored-form) weights kept in VRAM after their first upload, keyed by host address
    //!  - scratch:  grow-only x / y / FP32-weight / raw-staging / bias buffers (no cudaMalloc per call)
    //!  - pinned:   host staging buffer, so streamed weights upload at full PCIe speed
    //! Budget = free VRAM at start - GPU_RESERVE (or ROPE_GPU_VRAM_GIB). It is never exceeded: scratch growth
    //! evicts resident weights first; a weight that does not fit is streamed. So Windows never spills to shared memory.
    use std::collections::HashMap;
    use std::ffi::{c_char, c_void, CStr};
    use std::sync::Mutex;
    extern "C" {
        fn rope_cuda_alloc(n: usize) -> *mut c_void;
        fn rope_cuda_free(d: *mut c_void);
        fn rope_cuda_download(h: *mut c_void, d: *const c_void, n: usize);
        fn rope_cuda_sync();
        fn rope_cuda_last_error() -> *const c_char;
        fn rope_cuda_host_alloc(n: usize) -> *mut c_void;
        fn rope_cuda_free_bytes() -> usize;
        fn rope_cuda_dequant(d_raw: *const c_void, wtype: i32, n: i64, k: i64, mode: i32, d_out: *mut c_void);
        fn rope_cuda_gemm(dx: *const f32, m: i64, n: i64, k: i64, mode: i32, dw: *const c_void, dy: *mut f32);
        fn rope_cuda_conv2d_raw(din: *const f32, c: i32, h: i32, w: i32, o: i32, mode: i32, dw: *const c_void, dbias: *const f32,
                                kh: i32, kw: i32, stride: i32, pad: i32, dout: *mut f32);
        fn rope_cuda_attention(dq: *const f32, dk: *const f32, dv: *const f32, dout: *mut f32, nq: i64, nk: i64, d: i32,
                               qs: i64, ks: i64, vs: i64, os: i64, scale: f32);
        fn rope_cuda_modulate(dx: *const f32, dy: *mut f32, m: i64, n: i64, dshift: *const f32, dscale: *const f32, eps: f32);
        fn rope_cuda_add_gated(dx: *mut f32, dgate: *const f32, dy: *const f32, m: i64, n: i64);
        fn rope_cuda_add_rows(dy: *mut f32, db: *const f32, m: i64, n: i64);
        fn rope_cuda_qk_norm_rope(dbuf: *mut f32, rows: i64, rs: i64, d: i32, heads: i32, hd: i32, dqs: *const f32, dks: *const f32,
                                  dcos: *const f32, dsin: *const f32, row0: i64);
        fn rope_cuda_copy2d(dst: *mut c_void, dpitch: usize, src: *const c_void, spitch: usize, width: usize, rows: usize);
        fn rope_cuda_gelu_tanh(dx: *mut f32, n: i64);
        fn rope_cuda_gelu_erf(dx: *mut f32, n: i64);
        fn rope_cuda_silu(dx: *mut f32, n: i64);
        fn rope_cuda_layernorm(dx: *mut f32, m: i64, n: i64, dg: *const f32, db: *const f32, eps: f32);
        fn rope_cuda_groupnorm(dx: *mut f32, c: i64, hw: i64, groups: i32, dg: *const f32, db: *const f32, eps: f32);
        fn rope_cuda_transpose(dx: *const f32, dy: *mut f32, r: i64, c: i64);
        fn rope_cuda_add(dx: *mut f32, dy: *const f32, n: i64);
        fn rope_cuda_mul(dx: *mut f32, dy: *const f32, n: i64);
        fn rope_cuda_add_row_scalars(dx: *mut f32, ds: *const f32, r: i64, c: i64);
        fn rope_cuda_upsample2x(dx: *const f32, dy: *mut f32, ch: i64, h: i64, w: i64);
        fn rope_cuda_upload_async(d: *mut c_void, h: *const c_void, n: usize, stream: *mut c_void);
        fn rope_cuda_event_new() -> *mut c_void;
        fn rope_cuda_event_record(e: *mut c_void, stream: *mut c_void);
        fn rope_cuda_event_sync(e: *mut c_void);
        fn rope_cuda_compute_wait(e: *mut c_void);
        fn rope_cuda_copy_after_compute(e: *mut c_void, stream: *mut c_void);
        fn rope_cuda_new_stream() -> *mut c_void;
    }
    const GIB: f64 = 1073741824.0;
    const GPU_RESERVE: f64 = 0.4;     // CUDA context + kernel workspace + conv im2col scratch headroom
    /// Pinned (page-locked) staging size. Transfers go through it in chunks: locked RAM stays small on 4 GB laptops,
    /// and the driver never allocates its own staging in shared GPU memory.
    const PIN: usize = 32 << 20;
    /// Commit kept free when sizing the VRAM budget: on Windows (WDDM) every device allocation is charged to the
    /// process commit (measured: private bytes = 0.7 GB + dedicated VRAM in use).
    const COMMIT_MARGIN: f64 = 1.0;

    #[cfg(windows)]
    fn commit_avail_gib() -> f64 {
        #[repr(C)]
        struct Ms { len: u32, load: u32, phys: u64, avail_phys: u64, page: u64, avail_page: u64, virt: u64, avail_virt: u64, ext: u64 }
        extern "system" { fn GlobalMemoryStatusEx(m: *mut Ms) -> i32; }
        let mut m = Ms { len: std::mem::size_of::<Ms>() as u32, load: 0, phys: 0, avail_phys: 0, page: 0, avail_page: 0, virt: 0, avail_virt: 0, ext: 0 };
        // SAFETY: m is a correctly sized MEMORYSTATUSEX; the call only writes into it.
        if unsafe { GlobalMemoryStatusEx(&mut m) } != 0 { m.avail_page as f64 / GIB } else { f64::INFINITY }
    }
    #[cfg(not(windows))]
    fn commit_avail_gib() -> f64 { f64::INFINITY }

    /// GPU math: FP32 by default (FP16 converts activations to half too: 6.7e-2 error on FLUX); ROPE_GPU_MODE=f16.
    fn mode() -> i32 { if std::env::var("ROPE_GPU_MODE").as_deref() == Ok("f16") { 1 } else { 0 } }

    #[derive(Default)]
    struct Buf { p: usize, cap: usize }

    #[derive(Default)]
    pub struct Stats { pub resident_bytes: usize, pub resident_hits: u64, pub streamed_bytes: u64, pub peak_used: usize, pub budget: usize, pub put_s: f64 }

    #[derive(Default)]
    struct Vram {
        budget: usize, used: usize, init: bool,
        resident: HashMap<usize, (usize, usize)>,       // host address -> (device address, bytes)
        order: Vec<usize>,                              // admission order (eviction: newest first keeps early blocks)
        x: Buf, y: Buf, wf: Buf, raw: Buf, bias: Buf, ak: Buf, av: Buf,
        pool: HashMap<usize, Vec<usize>>,              // free device activation buffers by size (bytes)
        pool_bytes: usize,
        // upload arena (two pinned halves on the compute stream: uploads stay in program order with the kernels, so a
        // reused buffer is never overwritten early, and the CPU only waits when a half is reused before the GPU reached it)
        pin: Buf, pin2: Buf, ev: [usize; 2], half: usize, cursor: usize, dl: Buf,
        // block-ahead staging (G2b part 2): two device ring slots, staged weights (host -> (device, slot)), the staging thread
        ring: [Buf; 2], ring_next: usize, staged: HashMap<usize, (usize, usize)>, stage_thread: Option<std::thread::JoinHandle<()>>,
        st: Option<StageCtx>, staged_bytes: u64,
        stats: Stats,
    }
    /// Resources of the staging thread (its own pinned slots, events and stream; the per-ring-slot "staged" events and
    /// the compute fence). Plain addresses, so a thread can take a copy without touching the manager's lock.
    #[derive(Clone, Copy)]
    struct StageCtx { pin: [usize; 2], ev: [usize; 2], stream: usize, done: [usize; 2], fence: usize }

    // SAFETY: the raw addresses are only used under the mutex, from the single model thread.
    unsafe impl Send for Vram {}
    static V: Mutex<Option<Vram>> = Mutex::new(None);

    /// A GPU op could not get memory inside the VRAM budget and will run on the CPU instead: say so once per kind,
    /// so a slow silent fallback is visible.
    fn warn_once(kind: &'static str, n: usize, used: usize, budget: usize) {
        static SEEN: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());
        if let Ok(mut s) = SEEN.lock() { if !s.contains(&kind) {
            s.push(kind);
            eprintln!("rope: VRAM budget full ({:.2} of {:.2} GiB): a {:.0} MB {kind} buffer does not fit; that GPU op runs on the CPU",
                      used as f64 / GIB, budget as f64 / GIB, n as f64 / 1048576.0);
        } }
    }

    fn ok() -> bool {
        // SAFETY: returns NULL or a pointer to a static error string.
        let e = unsafe { rope_cuda_last_error() };
        if e.is_null() { return true; }
        eprintln!("rope: CUDA error, falling back to the CPU: {}", unsafe { CStr::from_ptr(e) }.to_string_lossy());
        false
    }

    impl Vram {
        fn start(&mut self) {
            if self.init { return; }
            self.init = true;
            let free = unsafe { rope_cuda_free_bytes() } as f64 / GIB;
            let gib = std::env::var("ROPE_GPU_VRAM_GIB").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(free - GPU_RESERVE);
            let commit = commit_avail_gib() - COMMIT_MARGIN;
            self.budget = (gib.min(free - GPU_RESERVE).min(commit).max(0.0) * GIB) as usize;
            self.stats.budget = self.budget;
        }
        fn evict_one(&mut self) -> bool {
            let Some(h) = self.order.pop() else { return false };
            if let Some((d, n)) = self.resident.remove(&h) { unsafe { rope_cuda_free(d as *mut c_void) }; self.used -= n; self.stats.resident_bytes -= n; }
            true
        }
        /// Grow a scratch buffer to at least n bytes (evicting resident weights if the budget requires it).
        fn grow(&mut self, which: fn(&mut Vram) -> &mut Buf, n: usize) -> Option<usize> {
            let cur = which(self).cap;
            if cur >= n { return Some(which(self).p); }
            while self.used - cur + n > self.budget { if !self.evict_one() { warn_once("scratch", n, self.used, self.budget); return None; } }
            let b = which(self);
            if b.p != 0 { unsafe { rope_cuda_free(b.p as *mut c_void) }; }
            let p = unsafe { rope_cuda_alloc(n) } as usize;
            let b = which(self);
            if p == 0 { b.p = 0; b.cap = 0; return None; }
            b.p = p; b.cap = n;
            self.used = self.used - cur + n;
            self.stats.peak_used = self.stats.peak_used.max(self.used);
            Some(p)
        }
        fn pinned(&mut self) -> Option<usize> {
            if self.pin.p == 0 { self.pin.p = unsafe { rope_cuda_host_alloc(PIN) } as usize; self.pin.cap = PIN; }
            (self.pin.p != 0).then_some(self.pin.p)
        }
        /// The two arena halves and their events (created on first use).
        fn arena(&mut self) -> Option<([usize; 2], [usize; 2])> {
            let a = self.pinned()?;
            if self.pin2.p == 0 { self.pin2.p = unsafe { rope_cuda_host_alloc(PIN) } as usize; self.pin2.cap = PIN; }
            if self.pin2.p == 0 { return None; }
            for e in &mut self.ev { if *e == 0 { *e = unsafe { rope_cuda_event_new() } as usize; } }
            self.ev.iter().all(|&e| e != 0).then_some(([a, self.pin2.p], self.ev))
        }
        /// Host -> device through the pinned arena, on the compute stream (in program order with the kernels). The
        /// source is copied synchronously, so it may be freed on return; small uploads pack into one half.
        fn put(&mut self, d: usize, h: *const u8, n: usize) -> Option<()> {
            let t0 = std::time::Instant::now();
            let (half, ev) = self.arena()?;
            let mut o = 0;
            while o < n {
                let c = (n - o).min(PIN);
                if self.cursor + c > PIN {
                    self.half ^= 1;
                    self.cursor = 0;
                    unsafe { rope_cuda_event_sync(ev[self.half] as *mut c_void) };   // the GPU has read this half's last copy
                }
                let src = half[self.half] + self.cursor;
                unsafe {
                    std::ptr::copy_nonoverlapping(h.add(o), src as *mut u8, c);
                    rope_cuda_upload_async((d + o) as *mut c_void, src as *const c_void, c, std::ptr::null_mut());
                    rope_cuda_event_record(ev[self.half] as *mut c_void, std::ptr::null_mut());
                }
                self.cursor += c.div_ceil(256) * 256;
                o += c;
            }
            self.stats.put_s += t0.elapsed().as_secs_f64();
            Some(())
        }
        fn stage_ctx(&mut self) -> Option<StageCtx> {
            if self.st.is_none() {
                let mk = || unsafe { rope_cuda_event_new() } as usize;
                let c = StageCtx {
                    pin: [unsafe { rope_cuda_host_alloc(PIN) } as usize, unsafe { rope_cuda_host_alloc(PIN) } as usize],
                    ev: [mk(), mk()], stream: unsafe { rope_cuda_new_stream() } as usize, done: [mk(), mk()], fence: mk(),
                };
                if c.pin.contains(&0) || c.ev.contains(&0) || c.done.contains(&0) || c.stream == 0 || c.fence == 0 { return None; }
                self.st = Some(c);
            }
            self.st
        }
        /// Ring slot s with at least n bytes (inside the budget; growing frees the old slot, which synchronizes).
        fn ring_slot(&mut self, s: usize, n: usize) -> Option<usize> {
            if self.ring[s].cap >= n { return Some(self.ring[s].p); }
            let cur = self.ring[s].cap;
            while self.used - cur + n > self.budget { if !self.evict_one() { warn_once("staging ring", n, self.used, self.budget); return None; } }
            if self.ring[s].p != 0 { unsafe { rope_cuda_free(self.ring[s].p as *mut c_void) }; }
            let p = unsafe { rope_cuda_alloc(n) } as usize;
            if p == 0 { self.ring[s] = Buf::default(); self.used -= cur; return None; }
            self.ring[s] = Buf { p, cap: n };
            self.used = self.used - cur + n;
            self.stats.peak_used = self.stats.peak_used.max(self.used);
            Some(p)
        }

        /// Device address of this weight's raw bytes: staged by the block-ahead ring, resident copy, a new resident
        /// copy if it fits the budget, else streamed through pinned memory into the raw scratch buffer.
        fn weight(&mut self, w: &[u8]) -> Option<usize> {
            let h = w.as_ptr() as usize;
            if let Some(&(d, s)) = self.staged.get(&h) {
                // the staging thread has queued (not necessarily finished) the copies once it has ended; kernels wait on the GPU
                if let Some(t) = self.stage_thread.take() { let _ = t.join(); }
                if let Some(c) = self.st { unsafe { rope_cuda_compute_wait(c.done[s] as *mut c_void) }; }
                return Some(d);
            }
            if let Some(&(d, _)) = self.resident.get(&h) { self.stats.resident_hits += 1; return Some(d); }
            let n = w.len();
            // resident only if the plan chose it (or no plan restricts it), and it fits next to the scratch in use
            if super::want_resident(h) && self.used + n <= self.budget {
                let d = unsafe { rope_cuda_alloc(n) } as usize;
                if d != 0 {
                    // through our pinned buffer: a copy from pageable (mapped) memory makes the driver allocate its own
                    // staging in shared GPU memory
                    if self.put(d, w.as_ptr(), n).is_none() { unsafe { rope_cuda_free(d as *mut c_void) }; return None; }
                    self.resident.insert(h, (d, n)); self.order.push(h);
                    self.used += n; self.stats.resident_bytes += n;
                    self.stats.peak_used = self.stats.peak_used.max(self.used);
                    self.stats.streamed_bytes += n as u64;     // first upload
                    return Some(d);
                }
            }
            let raw = self.grow(|v| &mut v.raw, n)?;
            self.put(raw, w.as_ptr(), n)?;
            self.stats.streamed_bytes += n as u64;
            Some(raw)
        }
        /// Device -> host through the pinned buffer (a download into pageable memory also uses driver staging).
        fn download(&mut self, h: &mut [f32], d: usize) -> Option<()> {
            // its own pinned buffer (never shared with the upload arena); the synchronous copy runs after queued kernels
            if self.dl.p == 0 { self.dl.p = unsafe { rope_cuda_host_alloc(PIN) } as usize; self.dl.cap = PIN; }
            let (n, pin) = (h.len() * 4, if self.dl.p != 0 { self.dl.p } else { return None });
            let mut o = 0;
            while o < n {
                let c = (n - o).min(PIN);
                unsafe {
                    rope_cuda_download(pin as *mut c_void, (d + o) as *const c_void, c);
                    std::ptr::copy_nonoverlapping(pin as *const u8, (h.as_mut_ptr() as *mut u8).add(o), c);
                }
                o += c;
            }
            Some(())
        }
        fn upload(&mut self, which: fn(&mut Vram) -> &mut Buf, h: &[f32]) -> Option<usize> {
            let n = h.len() * 4;
            let d = self.grow(which, n)?;
            self.put(d, h.as_ptr() as *const u8, n)?;
            Some(d)
        }
    }

    impl Vram {
        /// Device activation buffer of n bytes: a pooled one of the same size, else a new allocation inside the budget
        /// (evicting resident weights, then dropping pooled buffers, if needed).
        fn talloc(&mut self, n: usize) -> Option<usize> {
            if let Some(p) = self.pool.get_mut(&n).and_then(|v| v.pop()) { self.pool_bytes -= n; return Some(p); }
            while self.used + n > self.budget {
                if self.evict_one() { continue; }
                let Some((&sz, _)) = self.pool.iter().find(|(_, v)| !v.is_empty()) else {
                    warn_once("activation", n, self.used, self.budget);
                    return None;
                };
                let q = self.pool.get_mut(&sz).unwrap().pop().unwrap();
                unsafe { rope_cuda_free(q as *mut c_void) };
                self.used -= sz; self.pool_bytes -= sz;
            }
            let d = unsafe { rope_cuda_alloc(n) } as usize;
            if d == 0 { return None; }
            self.used += n;
            self.stats.peak_used = self.stats.peak_used.max(self.used);
            Some(d)
        }
        fn tfree(&mut self, p: usize, n: usize) { self.pool.entry(n).or_default().push(p); self.pool_bytes += n; }
    }

    /// A device buffer of f32 activations, returned to the manager's pool when dropped.
    pub struct DevBuf { pub p: usize, pub len: usize }
    impl Drop for DevBuf {
        fn drop(&mut self) { if let Ok(mut g) = V.lock() { if let Some(v) = g.as_mut() { v.tfree(self.p, self.len * 4); } } }
    }
    pub fn dev_alloc(len: usize) -> Option<DevBuf> { with(|v| v.talloc(len.max(1) * 4)).map(|p| DevBuf { p, len }) }
    pub fn dev_upload(h: &[f32]) -> Option<DevBuf> {
        let d = dev_alloc(h.len())?;
        with(|v| v.put(d.p, h.as_ptr() as *const u8, h.len() * 4))?;
        Some(d)
    }
    pub fn dev_download(d: &DevBuf, h: &mut [f32]) -> bool { with(|v| v.download(&mut h[..d.len], d.p)).is_some() && ok() }

    /// y = x W^T (+ bias) entirely on the device: no activation transfer.
    pub fn dev_matmul(x: &DevBuf, m: usize, k: usize, w: &[u8], wtype: u32, n: usize, bias: Option<&DevBuf>) -> Option<DevBuf> {
        let md = mode();
        let y = dev_alloc(m * n)?;
        with(|v| {
            // scratch first, weight last: a later allocation could evict a just-admitted resident weight
            let wf = v.grow(|v| &mut v.wf, n * k * if md == 1 { 2 } else { 4 })?;
            let raw = v.weight(w)?;
            unsafe {
                rope_cuda_dequant(raw as *const c_void, wtype as i32, n as i64, k as i64, md, wf as *mut c_void);
                rope_cuda_gemm(x.p as *const f32, m as i64, n as i64, k as i64, md, wf as *const c_void, y.p as *mut f32);
                if let Some(b) = bias { rope_cuda_add_rows(y.p as *mut f32, b.p as *const f32, m as i64, n as i64); }
            }
            Some(())
        })?;
        ok().then_some(y)
    }
    pub fn dev_modulate(x: &DevBuf, m: usize, n: usize, shift: &DevBuf, scale: &DevBuf) -> Option<DevBuf> {
        let y = dev_alloc(m * n)?;
        unsafe { rope_cuda_modulate(x.p as *const f32, y.p as *mut f32, m as i64, n as i64, shift.p as *const f32, scale.p as *const f32, 1e-6) };
        ok().then_some(y)
    }
    pub fn dev_add_gated(x: &DevBuf, gate: &DevBuf, y: &DevBuf, m: usize, n: usize) -> bool {
        unsafe { rope_cuda_add_gated(x.p as *mut f32, gate.p as *const f32, y.p as *const f32, m as i64, n as i64) };
        ok()
    }
    pub fn dev_gelu_tanh(x: &DevBuf) -> bool { unsafe { rope_cuda_gelu_tanh(x.p as *mut f32, x.len as i64) }; ok() }
    #[allow(clippy::too_many_arguments)]
    pub fn dev_qk_norm_rope(buf: &DevBuf, rows: usize, rs: usize, d: usize, heads: usize, hd: usize, qs: &DevBuf, ks: &DevBuf,
                            cos: &DevBuf, sin: &DevBuf, row0: usize) -> bool {
        unsafe { rope_cuda_qk_norm_rope(buf.p as *mut f32, rows as i64, rs as i64, d as i32, heads as i32, hd as i32, qs.p as *const f32,
                                        ks.p as *const f32, cos.p as *const f32, sin.p as *const f32, row0 as i64) };
        ok()
    }
    /// Multi-head attention over device buffers: q/k/v at float offsets qo/ko/vo of `src` rows (stride rs) -> [n, heads*hd].
    #[allow(clippy::too_many_arguments)]
    pub fn dev_attention(src: &DevBuf, n: usize, rs: usize, qo: usize, ko: usize, vo: usize, heads: usize, hd: usize, scale: f32) -> Option<DevBuf> {
        let d = heads * hd;
        let o = dev_alloc(n * d)?;
        unsafe {
            for h in 0..heads {
                let off = h * hd;
                rope_cuda_attention((src.p + 4 * (qo + off)) as *const f32, (src.p + 4 * (ko + off)) as *const f32, (src.p + 4 * (vo + off)) as *const f32,
                                    (o.p + 4 * off) as *mut f32, n as i64, n as i64, hd as i32, rs as i64, rs as i64, rs as i64, d as i64, scale);
            }
        }
        ok().then_some(o)
    }
    /// dst[r, dcol..dcol+w] = src[r, scol..scol+w] for r in 0..rows (row pitches in floats).
    #[allow(clippy::too_many_arguments)]
    pub fn dev_copy2d(dst: &DevBuf, dpitch: usize, dcol: usize, src: &DevBuf, spitch: usize, scol: usize, w: usize, rows: usize) -> bool {
        unsafe { rope_cuda_copy2d((dst.p + 4 * dcol) as *mut c_void, dpitch * 4, (src.p + 4 * scol) as *const c_void, spitch * 4, w * 4, rows) };
        ok()
    }

    pub fn dev_gelu_erf(x: &DevBuf) -> bool { unsafe { rope_cuda_gelu_erf(x.p as *mut f32, x.len as i64) }; ok() }
    pub fn dev_silu(x: &DevBuf) -> bool { unsafe { rope_cuda_silu(x.p as *mut f32, x.len as i64) }; ok() }
    /// LayerNorm over rows [m, n] with optional affine (g, b).
    pub fn dev_layernorm(x: &DevBuf, m: usize, n: usize, g: Option<&DevBuf>, b: Option<&DevBuf>, eps: f32) -> bool {
        let gp = g.map_or(std::ptr::null(), |g| g.p as *const f32);
        let bp = b.map_or(std::ptr::null(), |b| b.p as *const f32);
        unsafe { rope_cuda_layernorm(x.p as *mut f32, m as i64, n as i64, gp, bp, eps) };
        ok()
    }
    pub fn dev_groupnorm(x: &DevBuf, c: usize, hw: usize, groups: usize, g: &DevBuf, b: &DevBuf, eps: f32) -> bool {
        unsafe { rope_cuda_groupnorm(x.p as *mut f32, c as i64, hw as i64, groups as i32, g.p as *const f32, b.p as *const f32, eps) };
        ok()
    }
    pub fn dev_transpose(x: &DevBuf, r: usize, c: usize) -> Option<DevBuf> {
        let y = dev_alloc(r * c)?;
        unsafe { rope_cuda_transpose(x.p as *const f32, y.p as *mut f32, r as i64, c as i64) };
        ok().then_some(y)
    }
    pub fn dev_add(x: &DevBuf, y: &DevBuf) -> bool { unsafe { rope_cuda_add(x.p as *mut f32, y.p as *const f32, x.len as i64) }; ok() }
    pub fn dev_mul(x: &DevBuf, y: &DevBuf) -> bool { unsafe { rope_cuda_mul(x.p as *mut f32, y.p as *const f32, x.len as i64) }; ok() }
    pub fn dev_add_row_scalars(x: &DevBuf, s: &DevBuf, r: usize, c: usize) -> bool {
        unsafe { rope_cuda_add_row_scalars(x.p as *mut f32, s.p as *const f32, r as i64, c as i64) };
        ok()
    }
    pub fn dev_upsample2x(x: &DevBuf, ch: usize, h: usize, w: usize) -> Option<DevBuf> {
        let y = dev_alloc(ch * h * w * 4)?;
        unsafe { rope_cuda_upsample2x(x.p as *const f32, y.p as *mut f32, ch as i64, h as i64, w as i64) };
        ok().then_some(y)
    }
    /// conv2d with device input and output (weights through the VRAM manager: scratch first, weight last).
    #[allow(clippy::too_many_arguments)]
    pub fn dev_conv2d(x: &DevBuf, c: usize, h: usize, w: usize, wt: &[u8], wtype: u32, bias: Option<&DevBuf>,
                      o: usize, kh: usize, kw: usize, stride: usize, pad: usize) -> Option<(DevBuf, usize, usize)> {
        let (ho, wo) = ((h + 2 * pad - kh) / stride + 1, (w + 2 * pad - kw) / stride + 1);
        let md = mode();
        let ck = c * kh * kw;
        let y = dev_alloc(o * ho * wo)?;
        with(|v| {
            let wf = v.grow(|v| &mut v.wf, o * ck * if md == 1 { 2 } else { 4 })?;
            let raw = v.weight(wt)?;
            unsafe {
                rope_cuda_dequant(raw as *const c_void, wtype as i32, o as i64, ck as i64, md, wf as *mut c_void);
                rope_cuda_conv2d_raw(x.p as *const f32, c as i32, h as i32, w as i32, o as i32, md, wf as *const c_void,
                                     bias.map_or(std::ptr::null(), |b| b.p as *const f32), kh as i32, kw as i32, stride as i32, pad as i32, y.p as *mut f32);
            }
            Some(())
        })?;
        ok().then_some((y, ho, wo))
    }
    /// Multi-head attention with separate q [nq, d], k [nk, d], v [nk, d] device buffers -> [nq, d].
    pub fn dev_attention3(q: &DevBuf, k: &DevBuf, v: &DevBuf, nq: usize, nk: usize, heads: usize, hd: usize, scale: f32) -> Option<DevBuf> {
        let d = heads * hd;
        let o = dev_alloc(nq * d)?;
        unsafe {
            for h in 0..heads {
                let off = 4 * h * hd;
                rope_cuda_attention((q.p + off) as *const f32, (k.p + off) as *const f32, (v.p + off) as *const f32, (o.p + off) as *mut f32,
                                    nq as i64, nk as i64, hd as i32, d as i64, d as i64, d as i64, d as i64, scale);
            }
        }
        ok().then_some(o)
    }

    /// Stage the next block's streamed weights (host ranges) into the other ring slot while the current block computes.
    /// Called when a model prefetches its next block; the fence makes the copies wait only for the kernels queued so far
    /// (the block that used this slot), not for the current block, which is enqueued after this call.
    pub fn stage(ranges: Vec<(usize, usize)>) {
        // off by default: measured no gain on the dev laptop (the CPU already copies ahead on the compute stream);
        // ROPE_STAGE=1 turns it on (it may help where the PCIe transfer itself is a large share, e.g. slower links)
        if std::env::var("ROPE_STAGE").is_err() { return; }
        let todo: Vec<(usize, usize)> = ranges.into_iter().filter(|r| r.1 > 0 && super::on_gpu(r.0 as *const u8) && super::plan_streams(r.0)).collect();
        if todo.is_empty() { return; }
        let _ = with(|v| {
            if let Some(t) = v.stage_thread.take() { let _ = t.join(); }        // one staging thread at a time
            let c = v.stage_ctx()?;
            let s = v.ring_next;
            let align = |n: usize| n.div_ceil(256) * 256;
            let total: usize = todo.iter().map(|r| align(r.1)).sum();
            let base = v.ring_slot(s, total)?;
            v.ring_next ^= 1;
            unsafe { rope_cuda_copy_after_compute(c.fence as *mut c_void, c.stream as *mut c_void) };
            v.staged.retain(|_, e| e.1 != s);
            let mut jobs = Vec::with_capacity(todo.len());
            let mut off = 0;
            for (h, n) in todo { v.staged.insert(h, (base + off, s)); jobs.push((h, n, base + off)); off += align(n); }
            v.staged_bytes += total as u64;
            v.stage_thread = Some(std::thread::spawn(move || {
                let mut i = 0usize;
                for (h, n, d) in jobs {
                    let mut o = 0;
                    while o < n {
                        let ch = (n - o).min(PIN);
                        let k = i % 2;
                        // SAFETY: h..h+n is a live mapped weight (the Weights object joins its prefetchers before unmapping,
                        // and the model holds it during the forward pass); pinned slot k is free after its event.
                        unsafe {
                            rope_cuda_event_sync(c.ev[k] as *mut c_void);
                            std::ptr::copy_nonoverlapping((h + o) as *const u8, c.pin[k] as *mut u8, ch);
                            rope_cuda_upload_async((d + o) as *mut c_void, c.pin[k] as *const c_void, ch, c.stream as *mut c_void);
                            rope_cuda_event_record(c.ev[k] as *mut c_void, c.stream as *mut c_void);
                        }
                        o += ch; i += 1;
                    }
                }
                unsafe { rope_cuda_event_record(c.done[s] as *mut c_void, c.stream as *mut c_void) };
            }));
            Some(())
        });
    }
    /// Bytes staged by the block-ahead ring so far.
    pub fn staged_bytes() -> u64 { V.lock().ok().and_then(|g| g.as_ref().map(|v| v.staged_bytes)).unwrap_or(0) }

    /// Resident / streamed statistics of this process (for reports).
    pub fn stats() -> Option<(usize, usize, u64, u64, usize)> {
        V.lock().ok()?.as_ref().map(|v| (v.stats.budget, v.stats.peak_used, v.stats.resident_hits, v.stats.streamed_bytes, v.stats.resident_bytes))
    }
    /// Seconds spent copying host data to the device (mapped file -> pinned -> VRAM), all uploads.
    pub fn upload_seconds() -> f64 { V.lock().ok().and_then(|g| g.as_ref().map(|v| v.stats.put_s)).unwrap_or(0.0) }

    fn with<T>(f: impl FnOnce(&mut Vram) -> Option<T>) -> Option<T> {
        let mut g = V.lock().ok()?;
        let v = g.get_or_insert_with(Vram::default);
        v.start();
        f(v)
    }

    /// y[m,n] = x[m,k] W^T on the GPU. false = not done (caller runs the CPU kernel).
    pub fn matmul(x: &[f32], m: usize, k: usize, w: &[u8], wtype: u32, n: usize, y: &mut [f32]) -> bool {
        let md = mode();
        let done = with(|v| {
            // scratch first, weight last: a later allocation could evict a just-admitted resident weight
            let wf = v.grow(|v| &mut v.wf, n * k * if md == 1 { 2 } else { 4 })?;
            let dx = v.upload(|v| &mut v.x, &x[..m * k])?;
            let dy = v.grow(|v| &mut v.y, m * n * 4)?;
            let raw = v.weight(w)?;
            // SAFETY: every device buffer was sized above from m, k, n; y holds m*n floats (checked by ffi::matmul).
            unsafe {
                rope_cuda_dequant(raw as *const c_void, wtype as i32, n as i64, k as i64, md, wf as *mut c_void);
                rope_cuda_gemm(dx as *const f32, m as i64, n as i64, k as i64, md, wf as *const c_void, dy as *mut f32);
                rope_cuda_sync();
            }
            v.download(&mut y[..m * n], dy)
        });
        done.is_some() && ok()
    }

    /// All attention heads on the GPU (see ffi::attention_mh). false = not done.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_mh(q: &[f32], k: &[f32], v: &[f32], o: &mut [f32], nq: usize, nk: usize, heads: usize, hd: usize,
                        qs: usize, ks: usize, vs: usize, os: usize, scale: f32) -> bool {
        let d = heads * hd;
        let span = |rows: usize, stride: usize| (rows - 1) * stride + d;
        let qb = q.as_ptr() as usize;
        let inside = |p: &[f32], rows: usize, stride: usize| {
            let a = p.as_ptr() as usize;
            a >= qb && a + span(rows, stride) * 4 <= qb + q.len() * 4
        };
        let done = with(|vr| {
            // q's whole slice goes up once; k / v point into it when they lie inside (fused qkv), else go up apart
            let dq = vr.upload(|v| &mut v.x, q)?;
            let dk = if inside(k, nk, ks) { dq + (k.as_ptr() as usize - qb) } else { vr.upload(|v| &mut v.ak, &k[..span(nk, ks)])? };
            let dv = if inside(v, nk, vs) { dq + (v.as_ptr() as usize - qb) } else { vr.upload(|v| &mut v.av, &v[..span(nk, vs)])? };
            let olen = span(nq, os);
            let dout = vr.grow(|v| &mut v.y, olen * 4)?;
            // SAFETY: every head reads rows inside the uploaded spans and writes inside dout (olen floats).
            unsafe {
                for h in 0..heads {
                    let off = h * hd * 4;
                    rope_cuda_attention((dq + off) as *const f32, (dk + off) as *const f32, (dv + off) as *const f32, (dout + off) as *mut f32,
                                        nq as i64, nk as i64, hd as i32, qs as i64, ks as i64, vs as i64, os as i64, scale);
                }
                rope_cuda_sync();
            }
            vr.download(&mut o[..olen], dout)
        });
        done.is_some() && ok()
    }

    /// conv2d on the GPU (out already sized by the caller). false = not done.
    #[allow(clippy::too_many_arguments)]
    pub fn conv2d(inp: &[f32], c: usize, h: usize, w: usize, wt: &[u8], wtype: u32, bias: Option<&[f32]>,
                  o: usize, kh: usize, kw: usize, stride: usize, pad: usize, out: &mut [f32]) -> bool {
        let md = mode();
        let ck = c * kh * kw;
        let done = with(|v| {
            // scratch first, weight last: a later allocation could evict a just-admitted resident weight
            let wf = v.grow(|v| &mut v.wf, o * ck * if md == 1 { 2 } else { 4 })?;
            let di = v.upload(|v| &mut v.x, &inp[..c * h * w])?;
            let db = match bias { Some(b) => v.upload(|v| &mut v.bias, b)?, None => 0 };
            let dout = v.grow(|v| &mut v.y, out.len() * 4)?;
            let raw = v.weight(wt)?;
            // SAFETY: as in matmul; out holds o*ho*wo floats (resized by ffi::conv2d).
            unsafe {
                rope_cuda_dequant(raw as *const c_void, wtype as i32, o as i64, ck as i64, md, wf as *mut c_void);
                rope_cuda_conv2d_raw(di as *const f32, c as i32, h as i32, w as i32, o as i32, md, wf as *const c_void, db as *const f32,
                                     kh as i32, kw as i32, stride as i32, pad as i32, dout as *mut f32);
                rope_cuda_sync();
            }
            v.download(out, dout)
        });
        done.is_some() && ok()
    }
}

/// Probe the device and return the JSON text (dir: disk to measure; disk_mb 0 skips the speed test).
pub fn probe_json(dir: &str, disk_mb: i32) -> Result<String, String> {
    let d = CString::new(dir).map_err(|_| "directory contains NUL")?;
    let mut buf = vec![0u8; 64 << 10];
    // SAFETY: buf is writable for buf.len() bytes; the C side writes at most cap bytes incl. the NUL.
    let n = unsafe { rope_c_probe_json(d.as_ptr(), disk_mb, buf.as_mut_ptr() as *mut c_char, buf.len()) };
    if n < 0 { return Err("probe failed".into()); }
    buf.truncate(n as usize);
    String::from_utf8(buf).map_err(|_| "probe JSON is not UTF-8".into())
}

pub fn gemm_gflops() -> f64 { unsafe { rope_c_gemm_gflops() } }

/// GEMM speed with `n` threads; the previous thread count is restored.
pub fn gemm_gflops_threads(n: i32) -> f64 {
    let prev = unsafe { rope_get_threads() };
    set_threads(n);
    let g = gemm_gflops();
    set_threads(prev);
    g
}

pub fn silu(x: &mut [f32]) { unsafe { rope_silu(x.as_mut_ptr(), x.len() as i64) } }

pub fn groupnorm(x: &mut [f32], c: usize, hw: usize, groups: usize, g: &[f32], b: &[f32], eps: f32) {
    assert!(x.len() >= c * hw && g.len() >= c && b.len() >= c && groups > 0 && c % groups == 0, "groupnorm: bad sizes");
    // SAFETY: checked above.
    unsafe { rope_groupnorm(x.as_mut_ptr(), c as i64, hw as i64, groups as i32, g.as_ptr(), b.as_ptr(), eps) }
}

/// out[o, ho, wo] = conv2d(inp[c, h, w], W[o, c*kh*kw]) + bias. Returns (ho, wo).
#[allow(clippy::too_many_arguments)]
pub fn conv2d(inp: &[f32], c: usize, h: usize, w: usize, wt: &[u8], wtype: u32, bias: Option<&[f32]>,
              o: usize, kh: usize, kw: usize, stride: usize, pad: usize, out: &mut Vec<f32>) -> (usize, usize) {
    assert!(stride > 0 && h + 2 * pad >= kh && w + 2 * pad >= kw, "conv2d: bad geometry");
    let (ho, wo) = ((h + 2 * pad - kh) / stride + 1, (w + 2 * pad - kw) / stride + 1);
    let ck = c * kh * kw;
    assert!(wtype_supported(wtype) && ck as u64 % wtype_block(wtype) == 0, "conv2d: bad weight type");
    assert!(inp.len() >= c * h * w && wt.len() >= o * row_bytes(wtype, ck as u64) && bias.is_none_or(|b| b.len() >= o), "conv2d: buffer too small");
    out.resize(o * ho * wo, 0.0);
    if account("conv2d", Some(wt.as_ptr()), 2.0 * (o * ho * wo) as f64 * ck as f64) { return (ho, wo); }
    let t0 = std::time::Instant::now();
    #[cfg(feature = "cuda")]
    if on_gpu(wt.as_ptr()) && cuda::conv2d(inp, c, h, w, wt, wtype, bias, o, kh, kw, stride, pad, out) {
        kernel_done("conv2d_gpu", t0);
        return (ho, wo);
    }
    // SAFETY: sizes checked above; out holds o*ho*wo floats.
    unsafe { rope_conv2d(inp.as_ptr(), c as i32, h as i32, w as i32, wt.as_ptr() as *const c_void, wtype as i32,
                         bias.map_or(std::ptr::null(), |b| b.as_ptr()), o as i32, kh as i32, kw as i32, stride as i32, pad as i32, out.as_mut_ptr()) }
    kernel_done("conv2d", t0);
    if let Some(c) = COUNTER.lock().unwrap().as_mut() {   // conv shapes appear as (positions, C*kh*kw, O, 1000 + kh*10 + stride)
        if !c.dry { let e = c.shapes.entry((ho * wo, ck, o, 1000 + kh as u32 * 10 + stride as u32)).or_default(); e.0 += 1; e.1 += t0.elapsed().as_secs_f64(); }
    }
    (ho, wo)
}

// ---------------- kernel wrappers: every length is checked before C sees a pointer ----------------

/// y[m,n] = x[m,k] . W[n,k]^T, W given as raw rows in GGUF type `wtype`.
pub fn matmul(x: &[f32], m: usize, k: usize, w: &[u8], wtype: u32, n: usize, y: &mut [f32]) {
    assert!(wtype_supported(wtype) && k as u64 % wtype_block(wtype) == 0, "matmul: bad weight type / K");
    assert!(x.len() >= m * k && y.len() >= m * n && w.len() >= n * row_bytes(wtype, k as u64), "matmul: buffer too small");
    let gpu = m >= 4 && on_gpu(w.as_ptr());
    LAST_ON_GPU.store(gpu, std::sync::atomic::Ordering::Relaxed);
    let kind = if gpu { "matmul_gpu" } else if m < 4 { "gemv" } else { "matmul" };   // few-row calls are memory bound
    if account(kind, Some(w.as_ptr()), 2.0 * m as f64 * k as f64 * n as f64) { return; }
    let t0 = std::time::Instant::now();
    #[cfg(feature = "cuda")]
    if gpu && cuda::matmul(x, m, k, w, wtype, n, y) { kernel_done(kind, t0); return; }
    let kind = if m < 4 { "gemv" } else { "matmul" };          // CUDA fallback lands here
    // SAFETY: sizes checked above; the kernel reads x[m*k], w[n rows], writes y[m*n].
    unsafe { rope_matmul(x.as_ptr(), m as i64, k as i64, w.as_ptr() as *const c_void, wtype as i32, n as i64, y.as_mut_ptr()) }
    kernel_done(kind, t0);
    if let Some(c) = COUNTER.lock().unwrap().as_mut() {
        if !c.dry { let e = c.shapes.entry((m, k, n, wtype)).or_default(); e.0 += 1; e.1 += t0.elapsed().as_secs_f64(); }
    }
}

/// One attention head over strided rows: row i of q is q[i*qs .. i*qs+d] (same for k, v, o).
#[allow(clippy::too_many_arguments)]
pub fn attention(q: &[f32], k: &[f32], v: &[f32], o: &mut [f32], nq: usize, nk: usize, d: usize,
                 qs: usize, ks: usize, vs: usize, os: usize, scale: f32, causal: bool) {
    attention_bias(q, k, v, o, nq, nk, d, qs, ks, vs, os, scale, causal, None)
}

/// `attention` with an optional additive score bias [nq, nk] (row-major), added after scaling, before the softmax.
#[allow(clippy::too_many_arguments)]
pub fn attention_bias(q: &[f32], k: &[f32], v: &[f32], o: &mut [f32], nq: usize, nk: usize, d: usize,
                      qs: usize, ks: usize, vs: usize, os: usize, scale: f32, causal: bool, bias: Option<&[f32]>) {
    assert!(bias.map_or(true, |b| b.len() >= nq * nk), "attention: bias too small");
    let need = |rows: usize, stride: usize| if rows == 0 { 0 } else { (rows - 1) * stride + d };
    assert!(q.len() >= need(nq, qs) && k.len() >= need(nk, ks) && v.len() >= need(nk, vs) && o.len() >= need(nq, os), "attention: buffer too small");
    assert!(!causal || nq <= nk, "attention: causal needs nq <= nk");
    if account("attention", None, 4.0 * nq as f64 * nk as f64 * d as f64) { return; }
    let t0 = std::time::Instant::now();
    // SAFETY: every row access stays inside the checked lengths.
    unsafe { rope_attention_ex(q.as_ptr(), k.as_ptr(), v.as_ptr(), o.as_mut_ptr(), nq as i64, nk as i64, d as i32,
                               qs as i64, ks as i64, vs as i64, os as i64, scale, causal as i32,
                               bias.map_or(std::ptr::null(), |b| b.as_ptr())) }
    kernel_done("attention", t0);
}

/// All heads of a multi-head attention in one call: head h reads q/k/v columns [h*hd, (h+1)*hd) of rows with strides
/// qs/ks/vs and writes o likewise (os). On the GPU, q/k/v are uploaded once for all heads (once in total when k and v
/// lie inside q's buffer, as with a fused qkv projection), and o comes back once.
#[allow(clippy::too_many_arguments)]
pub fn attention_mh(q: &[f32], k: &[f32], v: &[f32], o: &mut [f32], nq: usize, nk: usize, heads: usize, hd: usize,
                    qs: usize, ks: usize, vs: usize, os: usize, scale: f32) {
    let d = heads * hd;
    let need = |rows: usize, stride: usize| if rows == 0 { 0 } else { (rows - 1) * stride + d };
    assert!(q.len() >= need(nq, qs) && k.len() >= need(nk, ks) && v.len() >= need(nk, vs) && o.len() >= need(nq, os), "attention_mh: buffer too small");
    #[cfg(feature = "cuda")]
    if LAST_ON_GPU.load(std::sync::atomic::Ordering::Relaxed) {
        let fl = 4.0 * nq as f64 * nk as f64 * d as f64;
        if account("attention_gpu", None, fl) { return; }
        let t0 = std::time::Instant::now();
        if cuda::attention_mh(q, k, v, o, nq, nk, heads, hd, qs, ks, vs, os, scale) { kernel_done("attention_gpu", t0); return; }
    }
    for h in 0..heads {
        attention(&q[h * hd..], &k[h * hd..], &v[h * hd..], &mut o[h * hd..], nq, nk, hd, qs, ks, vs, os, scale, false);
    }
}

pub fn layernorm(x: &mut [f32], m: usize, n: usize, g: Option<&[f32]>, b: Option<&[f32]>, eps: f32) {
    assert!(x.len() >= m * n && g.is_none_or(|g| g.len() >= n) && b.is_none_or(|b| b.len() >= n), "layernorm: buffer too small");
    let p = |o: Option<&[f32]>| o.map_or(std::ptr::null(), |s| s.as_ptr());
    // SAFETY: checked above; null gamma/beta mean 1/0.
    unsafe { rope_layernorm(x.as_mut_ptr(), m as i64, n as i64, p(g), p(b), eps) }
}

pub fn gelu_erf(x: &mut [f32]) { unsafe { rope_gelu_erf(x.as_mut_ptr(), x.len() as i64) } }
/// y[i, :] += b for every row of y (row length = b.len()); parallel C kernel.
pub fn add_rows(y: &mut [f32], b: &[f32]) {
    assert!(!b.is_empty() && y.len() % b.len() == 0, "add_rows: y is not a whole number of rows");
    // SAFETY: y holds y.len()/b.len() rows of b.len() floats.
    unsafe { rope_add_rows(y.as_mut_ptr(), (y.len() / b.len()) as i64, b.len() as i64, b.as_ptr()) }
}
/// GELU, tanh form (T5 v1.1 gated-GELU, FLUX MLPs); parallel C kernel.
pub fn gelu_tanh(x: &mut [f32]) { unsafe { rope_gelu_tanh(x.as_mut_ptr(), x.len() as i64) } }

pub fn has_cuda() -> bool { unsafe { rope_c_has_cuda() != 0 } }
pub fn set_threads(n: i32) { unsafe { rope_set_threads(n) } }
pub fn wtype_supported(t: u32) -> bool { t < 64 && unsafe { rope_wtype_supported(t as i32) } != 0 }
pub fn wtype_block(t: u32) -> u64 { unsafe { rope_wtype_block(t as i32) as u64 } }
pub fn row_bytes(t: u32, k: u64) -> usize { unsafe { rope_row_bytes(t as i32, k as i64) } }

/// Dequantize one row of `k` elements. Checks that `src` holds the whole row and `k` is a block multiple.
/// GGUF type ids rope can quantize to.
pub const RT_Q8_0: u32 = 8;
pub const RT_Q4_K: u32 = 12;

/// Quantize rows [rows, k] of f32 into GGUF type `t` (RT_Q8_0 or RT_Q4_K); parallel C kernel.
pub fn quantize_rows(t: u32, x: &[f32], rows: usize, k: usize) -> Result<Vec<u8>, String> {
    if (t != RT_Q8_0 && t != RT_Q4_K) || k as u64 % wtype_block(t) != 0 || x.len() < rows * k { return Err(format!("cannot quantize to type {t} with row length {k}")); }
    let mut out = vec![0u8; rows * row_bytes(t, k as u64)];
    // SAFETY: x holds rows*k floats and out holds rows rows of row_bytes(t, k), as checked above.
    let r = unsafe { rope_quantize_rows(t as i32, x.as_ptr(), out.as_mut_ptr() as *mut c_void, rows as i64, k as i64) };
    if r == 0 { Ok(out) } else { Err("quantize failed".into()) }
}

pub fn dequant_row(t: u32, src: &[u8], k: usize) -> Vec<f32> {
    assert!(wtype_supported(t) && k as u64 % wtype_block(t) == 0, "unsupported type or row length");
    assert!(src.len() >= row_bytes(t, k as u64), "row data shorter than its declared size");
    let mut out = vec![0f32; k];
    // SAFETY: src covers row_bytes(t, k) bytes and out holds k floats, as checked above.
    unsafe { rope_dequant_row(t as i32, src.as_ptr() as *const c_void, out.as_mut_ptr(), k as i64) };
    out
}

pub fn run_c_tests(filter: Option<&str>) -> i32 {
    let f = filter.map(|s| CString::new(s).unwrap());
    unsafe { rope_run_tests(f.as_ref().map_or(std::ptr::null(), |c| c.as_ptr())) }
}

pub fn probe(dir: &str, disk_mb: i32) -> i32 {
    let d = CString::new(dir).expect("directory contains NUL");
    unsafe { rope_c_probe(d.as_ptr(), disk_mb) }
}

pub fn bench(m: i64, k: i64, n: i64, t: i32) -> i32 { unsafe { rope_c_bench(m, k, n, t) } }

#[cfg(feature = "cuda")]
pub fn bench_cuda(m: i64, k: i64, n: i64, t: i32, mode: i32) -> i32 { unsafe { rope_c_bench_cuda(m, k, n, t, mode) } }
