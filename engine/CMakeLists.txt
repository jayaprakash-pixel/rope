# Builds the C/CUDA kernel library `ropek` (static). The Rust crate (Cargo.toml) drives this through build.rs
# and links it into rope.exe. Kernels, probe and C tests live here; the platform lives in rust/.
if(WIN32)
  cmake_minimum_required(VERSION 3.24)   # the CUDA architecture variable
else()
  cmake_minimum_required(VERSION 3.16)   # Ubuntu 20.04 and 22.04 ship 3.16 and 3.22 (no CUDA build on Linux yet)
endif()
project(rope C)

option(ROPE_CUDA "Build CUDA kernels (needs the CUDA toolkit)" OFF)

# Static C runtime: matches the static CUDA runtime and Rust's +crt-static; no VC++ redistributable needed.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)   # the Rust crate also builds librope as a shared library (Linux)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
# No global instruction-set flag: rope.exe must start on any x64 CPU. Only the kernel variant below uses AVX2 (P2.2.5).
if(MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:C>:/W4> $<$<COMPILE_LANGUAGE:C>:/utf-8>)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
  set(ROPE_AVX2_FLAGS /arch:AVX2)
  set(ROPE_SSE41_FLAGS "")
else()
  add_compile_options($<$<COMPILE_LANGUAGE:C>:-Wall> $<$<COMPILE_LANGUAGE:C>:-Wextra>)
  set(ROPE_AVX2_FLAGS -mavx2 -mfma -mf16c)
  set(ROPE_SSE41_FLAGS -msse4.1)
endif()

if(ROPE_CUDA)
  set(CMAKE_CUDA_ARCHITECTURES 61 75)  # Pascal (GTX 10xx / MX) + Turing (GTX 16xx); must precede enable_language
  enable_language(CUDA)
endif()

find_package(OpenMP REQUIRED)
# The CPU kernels, compiled once per instruction set (src/vf8.h); src/dispatch.c picks one at run time.
foreach(isa AVX2 SSE41)
  string(TOLOWER ${isa} lisa)
  add_library(ropek_${lisa} OBJECT src/quant.c src/kernels_cpu.c)
  target_include_directories(ropek_${lisa} PRIVATE src)
  target_compile_definitions(ropek_${lisa} PRIVATE ROPE_ISA_${isa}=1)
  target_compile_options(ropek_${lisa} PRIVATE ${ROPE_${isa}_FLAGS})
  target_link_libraries(ropek_${lisa} PRIVATE OpenMP::OpenMP_C)
endforeach()
add_library(ropek STATIC src/dispatch.c src/probe.c src/cli_c.c src/test.c src/test_kernels.c
            $<TARGET_OBJECTS:ropek_avx2> $<TARGET_OBJECTS:ropek_sse41>)
if(UNIX)
  target_sources(ropek PRIVATE src/probe_linux.c)   # the Linux hardware probe (probe.c holds the Windows one)
endif()
target_include_directories(ropek PRIVATE src)
target_link_libraries(ropek PRIVATE OpenMP::OpenMP_C)

if(ROPE_CUDA)
  target_sources(ropek PRIVATE src/cuda/kernels.cu src/test_cuda.c)
  target_compile_definitions(ropek PRIVATE ROPE_CUDA=1)
  target_compile_options(ropek PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>)
  set_target_properties(ropek PROPERTIES CUDA_RUNTIME_LIBRARY Static)
endif()
