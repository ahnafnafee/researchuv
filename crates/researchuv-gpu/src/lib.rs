//! # researchuv-gpu — CUDA execution for ResearchUV
//!
//! A conjugate-gradient solver for the engine's SPD conformal systems,
//! running on the GPU through the CUDA driver API. The driver library
//! (`nvcuda.dll` / `libcuda.so.1`) is loaded at runtime — nothing links
//! against the toolkit — and the kernels are compiled to forward-compatible
//! PTX by the build script when `nvcc` is available. On machines without a
//! CUDA driver or toolkit, [`GpuSolver::new`] returns an unavailable solver
//! and callers fall back to the CPU path.
//!
//! | Module | Responsibility |
//! | --- | --- |
//! | [`ffi`] | Runtime-loaded driver entry points (the workspace's only `unsafe`) |
//! | [`solver`] | The safe CG wrapper over CSR systems |
//!
//! This crate intentionally does not carry `#![forbid(unsafe_code)]`: the
//! driver boundary requires it. Everything above [`solver`] stays safe.

pub mod ffi;
pub mod solver;

pub use solver::{CsrMatrix, GpuSolver};

#[cfg(test)]
mod tests {
    #[test]
    fn ptx_is_embedded_when_a_toolkit_was_present() {
        // With nvcc available (the CI/dev machine), the PTX must be real;
        // without one, this crate reports unavailable instead (tested in
        // solver).
        if cfg!(have_ptx) {
            assert!(!crate::solver::PTX_IS_EMPTY);
        }
    }
}
