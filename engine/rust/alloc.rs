//! Caching global allocator for large blocks. Model code allocates a fresh activation buffer per layer
//! (`vec![0.0; m * n]`); for large sizes Windows returns fresh pages each time, and the kernel threads then
//! page-fault and zero every 4 KB page on first write (measured: a 4096x320 matmul took 3.75 ms in the model vs
//! 2.0 ms with a reused buffer). Freed blocks >= 1 MB are kept (up to CAP bytes, 64 slots) and handed out again
//! for the same size and alignment; zeroed reuse is a memset, which is far cheaper than the page faults.
//! Everything else goes straight to the system allocator.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

const MIN: usize = 1 << 20;
/// Cache limit in bytes. A planner decision (it costs private memory, which is scarce on 4 GB laptops); 256 MB until set.
static CAP: AtomicUsize = AtomicUsize::new(256 << 20);

/// Set the cache limit (the planner calls this from the device's free RAM).
pub fn set_cap(bytes: usize) { CAP.store(bytes, Ordering::Relaxed); }
pub fn cap() -> usize { CAP.load(Ordering::Relaxed) }
const SLOTS: usize = 64;

/// Give every cached block back to the system. Called between pipeline stages: the next stage must not pay for the last stage's cache.
pub fn trim() { crate::GLOBAL.trim() }

impl Caching {
    fn trim(&self) {
    let blocks: Vec<(usize, usize, usize)> = match self.pool.lock() {
        Ok(mut p) => { p.total = 0; p.slot.iter_mut().filter(|s| s.0 != 0).map(|s| std::mem::replace(s, (0, 0, 0))).collect() }
        Err(_) => return,
    };
    for (size, align, addr) in blocks {
        // SAFETY: each block came from System with exactly this size and alignment and was removed from the pool above, so nobody else holds it.
        unsafe { System.dealloc(addr as *mut u8, Layout::from_size_align_unchecked(size, align)) };
    }
    }
}

struct Pool { slot: [(usize, usize, usize); SLOTS], total: usize }   // (size, align, address); size 0 = empty

pub struct Caching { pool: Mutex<Pool> }

impl Caching {
    pub const fn new() -> Self { Caching { pool: Mutex::new(Pool { slot: [(0, 0, 0); SLOTS], total: 0 }) } }

    fn take(&self, l: Layout) -> Option<*mut u8> {
        let mut p = self.pool.lock().ok()?;
        let i = p.slot.iter().position(|s| s.0 == l.size() && s.1 == l.align())?;
        let addr = p.slot[i].2;
        p.slot[i] = (0, 0, 0);
        p.total -= l.size();
        Some(addr as *mut u8)
    }
}

// SAFETY: blocks are only handed out once (removed from the pool under the lock) and always come from System
// with the same size and alignment they are reused for.
unsafe impl GlobalAlloc for Caching {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        if l.size() >= MIN { if let Some(p) = self.take(l) { return p; } }
        unsafe { System.alloc(l) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        if l.size() >= MIN {
            if let Some(p) = self.take(l) { unsafe { std::ptr::write_bytes(p, 0, l.size()) }; return p; }
        }
        unsafe { System.alloc_zeroed(l) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, l: Layout) {
        if l.size() >= MIN {
            if let Ok(mut p) = self.pool.lock() {
                if p.total + l.size() <= CAP.load(Ordering::Relaxed) {
                    if let Some(i) = p.slot.iter().position(|s| s.0 == 0) {
                        p.slot[i] = (l.size(), l.align(), ptr as usize);
                        p.total += l.size();
                        return;
                    }
                }
            }
        }
        unsafe { System.dealloc(ptr, l) }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn large_blocks_are_reused_and_zeroed() {
        let a = vec![7u8; 4 << 20];
        let p = a.as_ptr() as usize;
        drop(a);
        let b = vec![0u8; 4 << 20];                     // same size: comes back from the pool, zeroed
        assert_eq!(b.as_ptr() as usize, p);
        assert!(b.iter().all(|&v| v == 0));
    }
}
