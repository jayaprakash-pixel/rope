//! rope: run big image models on low-end laptops. Platform in safe Rust; math kernels in C/AVX2 and CUDA (engine/src), called through ffi.rs.
//! This is the library (`rlib` for Rust, `cdylib` for the C interface in `capi.rs`); `rope.exe` is a thin command line over it.
pub mod alloc;
pub mod capi;
pub mod cli;
pub mod detect;
pub mod ffi;
pub mod flux;
pub mod gguf_write;
pub mod json;
pub mod lora;
pub mod mmap;
pub mod modelfile;
pub mod models;
pub mod ops;
pub mod placement;
pub mod planner;
pub mod png;
pub mod profile;
pub mod scan;
pub mod sd;
pub mod selftest;
pub mod serve;
pub mod sha256;
pub mod t5;
pub mod tokenizer;
pub mod tokenizer_t5;
pub mod unicode_tables;
pub mod upscale;
pub mod weights;

/// Large activation buffers are reused instead of page-faulted in fresh every layer (see alloc.rs).
#[global_allocator]
static GLOBAL: alloc::Caching = alloc::Caching::new();

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
