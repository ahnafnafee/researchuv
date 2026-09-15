//! Compile the CUDA kernels to PTX with `nvcc`. When no toolkit is present
//! the build still succeeds: an empty PTX file is written and the solver
//! reports "unavailable" at runtime.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let ptx_path = out.join("cg_kernels.ptx");
    let src = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("src/kernels.cu");
    println!("cargo:rerun-if-changed={}", src.display());
    println!("cargo:rustc-check-cfg=cfg(have_ptx)");

    let nvcc = env::var("NVCC")
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            // CUDA_PATH/CUDA_HOME first, then plain PATH lookup.
            ["CUDA_PATH", "CUDA_HOME"]
                .iter()
                .find_map(|k| env::var(k).ok())
                .map(|root| PathBuf::from(root).join("bin").join("nvcc.exe"))
                .filter(|p| p.exists())
        })
        .unwrap_or_else(|| PathBuf::from("nvcc"));

    // nvcc routes .cu files through the MSVC host preprocessor even for
    // PTX-only compilation, so cl.exe must be reachable. CUDA toolkits
    // support a bounded range of MSVC versions, so prefer the OLDEST
    // installed toolchain (the newest is most likely unsupported).
    let mut path = env::var_os("PATH").unwrap_or_default();
    if let Some(cl_dir) = find_cl_dir() {
        let mut p = cl_dir.clone().into_os_string();
        p.push(if cfg!(windows) { ";" } else { ":" });
        p.push(&path);
        path = p;
    }

    let result = Command::new(&nvcc)
        .env("PATH", &path)
        .arg("-ptx")
        // compute_80 PTX JITs forward-compatible on every newer GPU.
        .arg("-arch=compute_80")
        .arg("-o")
        .arg(&ptx_path)
        .arg(&src)
        .output();

    let ok = result
        .map(|o| {
            if !o.status.success() {
                eprintln!(
                    "researchuv-gpu: nvcc failed:\n{}",
                    String::from_utf8_lossy(&o.stderr)
                );
            }
            o.status.success() && ptx_path.exists()
        })
        .unwrap_or_else(|e| {
            eprintln!("researchuv-gpu: cannot run nvcc: {e}");
            false
        });

    if ok {
        println!("cargo:rustc-cfg=have_ptx");
    } else {
        eprintln!(
            "researchuv-gpu: building without GPU kernels — the solver will \
             report unavailable at runtime"
        );
        std::fs::write(&ptx_path, "").expect("write placeholder ptx");
    }
}

/// The directory holding the oldest installed cl.exe (Windows only).
fn find_cl_dir() -> Option<PathBuf> {
    if !cfg!(windows) {
        return None;
    }
    let vswhere = Path::new(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe");
    if !vswhere.exists() {
        return None;
    }
    let out = Command::new(vswhere)
        .args([
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-find",
            "VC/Tools/MSVC/*/bin/Hostx64/x64/cl.exe",
        ])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let cls: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|l| l.to_ascii_lowercase().ends_with("cl.exe"))
        .collect();
    // Sorted toolchains: the first is the oldest supported pair for nvcc.
    let oldest = *cls.first()?;
    PathBuf::from(oldest).parent().map(|p| p.to_path_buf())
}
