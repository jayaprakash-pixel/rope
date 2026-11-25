// Builds the C/CUDA kernel library with CMake (the nvcc fixes live in this one place) and links it.
use std::{env, path::PathBuf, process::Command};

fn run(cmd: &mut Command) {
    let st = cmd.status().unwrap_or_else(|e| panic!("cannot run {cmd:?}: {e}"));
    assert!(st.success(), "command failed: {cmd:?}");
}

fn main() {
    let src = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(env::var("OUT_DIR").unwrap()).join("cmake");
    let cuda = env::var("CARGO_FEATURE_CUDA").is_ok();
    let windows = env::var("CARGO_CFG_TARGET_OS").map_or(false, |o| o == "windows");
    let mut cfg = Command::new("cmake");
    cfg.arg("-S").arg(&src).arg("-B").arg(&out).arg(format!("-DROPE_CUDA={}", if cuda { "ON" } else { "OFF" }));
    if windows { cfg.args(["-G", "Visual Studio 17 2022", "-A", "x64"]); } else { cfg.arg("-DCMAKE_BUILD_TYPE=Release"); }
    let mut toolkit = None;
    if cuda && !windows { panic!("the CUDA build is Windows-only for now"); }
    if cuda {
        // Newest installed toolkit, passed explicitly: works in shells opened before the CUDA install.
        let root = PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA");
        let mut v: Vec<_> = std::fs::read_dir(&root).expect("CUDA toolkit not found").flatten().map(|e| e.path()).collect();
        v.sort();
        let tk = v.pop().expect("CUDA toolkit not found");
        cfg.arg("-T").arg(format!("cuda={}", tk.display()));
        // nvcc fails silently when %TEMP% contains a space (e.g. C:\Users\First Last\...): use a space-free temp.
        let tmp = out.join("tmp");
        std::fs::create_dir_all(&tmp).unwrap();
        cfg.env("TEMP", &tmp).env("TMP", &tmp);
        toolkit = Some(tk);
    }
    run(&mut cfg);
    let mut b = Command::new("cmake");
    b.arg("--build").arg(&out).args(["--config", "Release", "--target", "ropek"]);
    if cuda { let tmp = out.join("tmp"); b.env("TEMP", &tmp).env("TMP", &tmp); }
    run(&mut b);

    println!("cargo:rustc-link-search=native={}", if windows { out.join("Release") } else { out.clone() }.display());
    println!("cargo:rustc-link-lib=static=ropek");
    if windows {
        println!("cargo:rustc-link-lib=dylib=advapi32");   // registry reads in the probe (Windows edition, machine model)
        println!("cargo:rustc-link-lib=dylib=dxgi");
        println!("cargo:rustc-link-lib=dylib=dxguid");
    } else {
        // OpenMP runtime: libgomp.so.1 (the shared library cannot take the static one). The kit ships a copy next to the program;
        // the rpath $ORIGIN lets the program find it there, on a laptop that has no libgomp.
        let gomp = Command::new("gcc").arg("-print-file-name=libgomp.a").output().ok().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default();
        if let Some(dir) = PathBuf::from(&gomp).parent() { println!("cargo:rustc-link-search=native={}", dir.display()); }
        println!("cargo:rustc-link-lib=dylib=gomp");
        println!("cargo:rustc-link-arg-bins=-Wl,-rpath,$ORIGIN");
        println!("cargo:rustc-link-lib=dylib=dl");
        println!("cargo:rustc-link-lib=dylib=pthread");
        println!("cargo:rustc-link-lib=dylib=m");
    }
    if let Some(tk) = toolkit {
        println!("cargo:rustc-link-search=native={}", tk.join("lib").join("x64").display());
        println!("cargo:rustc-link-lib=static=cudart_static");
    }
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=CMakeLists.txt");
}
