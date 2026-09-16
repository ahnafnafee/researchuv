//! Test-only AMG introspection: apply the preconditioner to explicit
//! vectors and read the result back (exposed to `#[cfg(test)]` code).

use crate::amg::{self, DeviceAmg};
use crate::precond;
use crate::solver::{CsrMatrix, DeviceMem, GpuSolver};

/// Upload the hierarchy for `a` under the crate's own test access.
fn upload(gpu: &GpuSolver, a: &CsrMatrix) -> Option<(DeviceMem, DeviceAmg)> {
    let c = &gpu.cuda;
    let mut mem = DeviceMem::new(c);
    let amg = amg::build(&mut mem, c, a)?;
    Some((mem, amg))
}

/// Apply the AMG preconditioner to `r` and return `M⁻¹ r` on the host.
pub fn probe_apply(gpu: &GpuSolver, a: &CsrMatrix, r: &[f64]) -> Vec<f64> {
    let n = a.rows();
    let (_mem, mut amg) = match upload(gpu, a) {
        Some(x) => x,
        None => return vec![f64::NAN; n],
    };
    let c = &gpu.cuda;
    let mut mem = DeviceMem::new(c);
    let d_r = match mem.upload(c, r) {
        Some(p) => p,
        None => return vec![f64::NAN; n],
    };
    let d_z = match mem.alloc_f64(c, n) {
        Some(p) => p,
        None => return vec![f64::NAN; n],
    };
    if !amg::apply(gpu, &mut amg, 0, d_r, d_z) {
        return vec![f64::NAN; n];
    }
    let mut out = vec![0.0f64; n];
    // SAFETY: host buffer sized for the copy.
    unsafe {
        if (c.cuMemcpyDtoH)(out.as_mut_ptr().cast(), d_z, n * 8) != crate::ffi::CUDA_SUCCESS {
            return vec![f64::NAN; n];
        }
    }
    out
}

/// The preconditioner's action on the first `k` basis vectors.
pub fn probe_columns(gpu: &GpuSolver, a: &CsrMatrix, k: usize) -> Vec<Vec<f64>> {
    let n = a.rows();
    (0..k.min(n))
        .map(|i| {
            let mut e = vec![0.0f64; n];
            e[i] = 1.0;
            probe_apply(gpu, a, &e)
        })
        .collect()
}

/// Silence the unused warning for the precond import on some builds.
#[allow(dead_code)]
fn _unused(_: &precond::Precond) {}
