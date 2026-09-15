//! Preconditioners for the GPU conjugate-gradient solver.
//!
//! - [`Precond::Jacobi`] — diagonal scaling; one element-wise kernel apply,
//!   no extra synchronization. Helps whenever the diagonal varies (the
//!   engine's cotan-weight systems do).
//! - [`Precond::Ic0`] — incomplete Cholesky with zero fill-in (IC(0)) in
//!   natural ordering. The factorization sweep is sequential and runs once
//!   on the host (O(nnz·k) over the existing sparsity); the per-iteration
//!   triangular solves run on the GPU as **level-scheduled** kernel
//!   launches: rows are grouped into dependency levels and each level is
//!   solved by one fully parallel launch. For the 2-D mesh Laplacians the
//!   engine produces, the level count grows like the mesh diameter.

use crate::solver::DeviceMem;
use crate::ffi::Cuda;
use crate::CsrMatrix;

/// The preconditioner choice for [`crate::GpuSolver::cg_solve_precond`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precond {
    /// Unpreconditioned CG.
    None,
    /// Jacobi (inverse-diagonal) scaling.
    Jacobi,
    /// Incomplete Cholesky, zero fill-in, natural ordering.
    Ic0,
}

/// A factored IC(0) preconditioner ready for device upload.
#[derive(Clone, Debug)]
pub struct Ic0Factor {
    /// Lower factor L (CSR, each row's diagonal stored LAST).
    pub l_vals: Vec<f64>,
    pub l_cols: Vec<i32>,
    pub l_ptr: Vec<i32>,
    /// Transpose Lᵀ (CSR, diagonal last) for the backward sweep.
    pub lt_vals: Vec<f64>,
    pub lt_cols: Vec<i32>,
    pub lt_ptr: Vec<i32>,
    /// Forward levels: row ids grouped by dependency level.
    pub fwd_levels: Vec<Vec<i32>>,
    /// Backward levels over Lᵀ.
    pub bwd_levels: Vec<Vec<i32>>,
}

/// Factor `a` into L·Lᵀ with the sparsity of `a`'s lower triangle (IC(0),
/// natural row order). Pivot floors keep the factor real and positive on
/// near-singular systems.
pub fn factor_ic0(a: &CsrMatrix) -> Option<Ic0Factor> {
    let n = a.rows();
    // The strictly lower triangle of row i.
    fn lower(a: &CsrMatrix, i: usize) -> Vec<(usize, f64)> {
        let mut out = Vec::new();
        for (c, v) in a.row(i) {
            let cu = c as usize;
            if cu < i {
                out.push((cu, v));
            }
        }
        // Ascending column order: each L(i,k) must see the k' < k entries
        // of its own row, already computed.
        out.sort_unstable_by_key(|(c, _)| *c);
        out
    }
    // Build L rows with the diagonal last; keep row values addressable by
    // column for the dot products.
    let mut rows: Vec<std::collections::BTreeMap<usize, f64>> =
        vec![std::collections::BTreeMap::new(); n];
    let mut l_vals = Vec::new();
    let mut l_cols = Vec::new();
    let mut l_ptr = vec![0i32];
    let mut diag = vec![0.0f64; n];
    for i in 0..n {
        for (k, a_ik) in lower(a, i) {
            // L(i,k) = (A(i,k) − Σ_{j<k} L(i,j)·L(k,j)) / L(k,k)
            let mut s = a_ik;
            for (&j, &l_ij) in rows[i].range(..k) {
                if let Some(&l_kj) = rows[k].get(&j) {
                    s -= l_ij * l_kj;
                }
            }
            let l_kk = diag[k];
            if l_kk.abs() < 1e-300 {
                return None;
            }
            rows[i].insert(k, s / l_kk);
        }
        // L(i,i) = sqrt(A(i,i) − Σ L(i,k)²), floored positive.
        let a_ii = a.row_get(i, i as i32);
        if a_ii.abs() < 1e-300 {
            return None;
        }
        let mut sum = a_ii;
        for &v in rows[i].values() {
            sum -= v * v;
        }
        let floor = 1e-12 * a_ii.abs().max(1e-30);
        if sum < floor {
            sum = floor;
        }
        diag[i] = sum.sqrt();
        // Emit the row: strict lower entries (ascending), diagonal last.
        for (&c, &v) in rows[i].iter() {
            l_vals.push(v);
            l_cols.push(c as i32);
        }
        l_vals.push(diag[i]);
        l_cols.push(i as i32);
        l_ptr.push(l_vals.len() as i32);
    }
    // Transpose (diagonal last per row).
    let (lt_vals, lt_cols, lt_ptr) = transpose_lower(&l_vals, &l_cols, &l_ptr, n);
    // Dependency levels: level[i] = 1 + max(level[k]) over row deps. The
    // forward factor's deps point backward (col < row) so rows are layered
    // ascending; the transpose's deps point forward, so it must be layered
    // DESCENDING — otherwise every dependency is still at its default level.
    let levels_of = |vals: &[f64], cols: &[i32], ptr: &[i32], descending: bool| -> Vec<Vec<i32>> {
        let n = ptr.len() - 1;
        let mut level = vec![0i32; n];
        let mut out: Vec<Vec<i32>> = Vec::new();
        let order: Vec<usize> = if descending {
            (0..n).rev().collect()
        } else {
            (0..n).collect()
        };
        for i in order {
            let mut lv = 0i32;
            for k in ptr[i] as usize..ptr[i + 1] as usize - 1 {
                let dep = cols[k] as usize;
                lv = lv.max(level[dep] + 1);
            }
            level[i] = lv;
            while out.len() <= lv as usize {
                out.push(Vec::new());
            }
            out[lv as usize].push(i as i32);
        }
        let _ = vals;
        out
    };
    Some(Ic0Factor {
        fwd_levels: levels_of(&l_vals, &l_cols, &l_ptr, false),
        bwd_levels: levels_of(&lt_vals, &lt_cols, &lt_ptr, true),
        l_vals,
        l_cols,
        l_ptr,
        lt_vals,
        lt_cols,
        lt_ptr,
    })
}

/// Transpose a CSR matrix whose rows carry the diagonal last; the result's
/// rows also carry the diagonal last.
fn transpose_lower(
    vals: &[f64],
    cols: &[i32],
    ptr: &[i32],
    n: usize,
) -> (Vec<f64>, Vec<i32>, Vec<i32>) {
    let mut t: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for i in 0..n {
        for k in ptr[i] as usize..ptr[i + 1] as usize {
            let j = cols[k] as usize;
            let v = vals[k];
            if j == i {
                t[i].push((i, v)); // diagonal lands on its own row
            } else {
                t[j].push((i, v));
            }
        }
    }
    let mut out_vals = Vec::new();
    let mut out_cols = Vec::new();
    let mut out_ptr = vec![0i32];
    for (i, row) in t.iter().enumerate() {
        let mut entries: Vec<(usize, f64)> = row.clone();
        entries.sort_unstable_by_key(|(c, _)| *c);
        // Reorder: ascending, but the diagonal last.
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        let (diag_pos, _) = entries.iter().enumerate().find(|(_, (c, _))| *c == i).unwrap();
        let diag = entries.remove(diag_pos);
        for (c, v) in entries {
            out_vals.push(v);
            out_cols.push(c as i32);
        }
        out_vals.push(diag.1);
        out_cols.push(diag.0 as i32);
        out_ptr.push(out_vals.len() as i32);
    }
    (out_vals, out_cols, out_ptr)
}

/// Device-resident IC(0) preconditioner: both triangular factors and the
/// flattened per-level row tables for the two sweeps.
pub(crate) struct DeviceIc0 {
    pub l_vals: u64,
    pub l_cols: u64,
    pub l_ptr: u64,
    pub lt_vals: u64,
    pub lt_cols: u64,
    pub lt_ptr: u64,
    /// Device array of the forward-sweep row ids, grouped by level.
    pub d_fwd_rows: u64,
    /// Device array of the backward-sweep row ids, grouped by level.
    pub d_bwd_rows: u64,
    /// Host-side level offsets (launch geometry), one past-the-end entry.
    pub fwd_offsets: Vec<i32>,
    pub bwd_offsets: Vec<i32>,
    /// Device scratch vector for the forward sweep's intermediate y.
    pub scratch: u64,
}

/// Upload the IC(0) preconditioner for `a` to the device.
pub(crate) fn upload(
    mem: &mut DeviceMem,
    c: &Cuda,
    a: &CsrMatrix,
    kind: Precond,
) -> Option<DeviceIc0> {
    match kind {
        Precond::None | Precond::Jacobi => None,
        Precond::Ic0 => {
            let f = factor_ic0(a)?;
            let l_vals = mem.upload(c, &f.l_vals)?;
            let l_cols = mem.upload(c, &f.l_cols)?;
            let l_ptr = mem.upload(c, &f.l_ptr)?;
            let lt_vals = mem.upload(c, &f.lt_vals)?;
            let lt_cols = mem.upload(c, &f.lt_cols)?;
            let lt_ptr = mem.upload(c, &f.lt_ptr)?;
            let flatten = |levels: &[Vec<i32>]| -> (Vec<i32>, Vec<i32>) {
                let mut rows = Vec::new();
                let mut offsets = vec![0i32];
                for lvl in levels {
                    rows.extend_from_slice(lvl);
                    offsets.push(rows.len() as i32);
                }
                (rows, offsets)
            };
            let (fwd_rows, fwd_offsets) = flatten(&f.fwd_levels);
            let (bwd_rows, bwd_offsets) = flatten(&f.bwd_levels);
            let d_fwd_rows = mem.upload(c, &fwd_rows)?;
            let d_bwd_rows = mem.upload(c, &bwd_rows)?;
            let scratch = mem.alloc_f64(c, a.rows())?;
            Some(DeviceIc0 {
                l_vals,
                l_cols,
                l_ptr,
                lt_vals,
                lt_cols,
                lt_ptr,
                d_fwd_rows,
                d_bwd_rows,
                fwd_offsets,
                bwd_offsets,
                scratch,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diag_dominant_banded(n: usize, band: usize) -> CsrMatrix {
        let mut vals = Vec::new();
        let mut cols = Vec::new();
        let mut row_ptr = vec![0i32];
        for i in 0usize..n {
            for d in 1..=band {
                if i >= d {
                    vals.push(-0.25 / d as f64);
                    cols.push((i - d) as i32);
                }
            }
            // A varying diagonal — this is what makes Jacobi matter.
            vals.push(4.0 + ((i % 11) as f64) * 0.75);
            cols.push(i as i32);
            for d in 1..=band {
                if i + d < n {
                    vals.push(-0.25 / d as f64);
                    cols.push((i + d) as i32);
                }
            }
            row_ptr.push(vals.len() as i32);
        }
        CsrMatrix { vals, cols, row_ptr }
    }

    #[test]
    fn ic0_factor_reproduces_the_matrix_on_its_pattern() {
        let a = diag_dominant_banded(200, 3);
        let f = factor_ic0(&a).expect("factorization succeeds");
        // L·Lᵀ equals A on every stored entry of A (zero fill-in).
        let n = a.rows();
        let get = |i: usize, j: usize| -> f64 {
            // L(i,j): row i, column j (strict lower + diagonal-last).
            let end = f.l_ptr[i + 1] as usize;
            for k in f.l_ptr[i] as usize..end {
                if f.l_cols[k] as usize == j {
                    return f.l_vals[k];
                }
            }
            0.0
        };
        for i in 0..n {
            for (cj, a_ij) in a.row(i) {
                let j = cj as usize;
                let mut s = 0.0;
                for t in 0..n {
                    s += get(i, t) * get(j, t);
                }
                assert!(
                    (s - a_ij).abs() < 1e-9 * (1.0 + a_ij.abs()),
                    "({i},{j}): LLᵗ = {s}, A = {a_ij}"
                );
            }
        }
    }

    #[test]
    fn levels_cover_every_row_exactly_once() {
        let a = diag_dominant_banded(150, 2);
        let f = factor_ic0(&a).expect("factorization succeeds");
        for (levels, ptr, name) in [
            (&f.fwd_levels, &f.l_ptr, "fwd"),
            (&f.bwd_levels, &f.lt_ptr, "bwd"),
        ] {
            let mut seen = std::collections::BTreeSet::new();
            for lvl in levels {
                for &r in lvl {
                    assert!(seen.insert(r), "{name}: row {r} in two levels");
                }
            }
            assert_eq!(seen.len(), ptr.len() - 1, "{name}: every row leveled");
        }
    }

    #[test]
    fn ic0_host_solve_matches_device_apply() {
        use crate::solver::GpuSolver;
        let Some(gpu) = GpuSolver::new() else {
            eprintln!("skipping: no CUDA device/PTX available");
            return;
        };
        let a = diag_dominant_banded(400, 3);
        let n = a.rows();
        let r: Vec<f64> = (0..n).map(|i| ((i * 7919) % 101) as f64 / 101.0 - 0.5).collect();
        let f = factor_ic0(&a).expect("factor");
        // Host forward/backward over the diagonal-last CSR rows.
        let get = |vals: &[f64], cols: &[i32], ptr: &[i32], i: usize, j: i32| -> f64 {
            for k in ptr[i] as usize..ptr[i + 1] as usize {
                if cols[k] == j {
                    return vals[k];
                }
            }
            0.0
        };
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            let mut s = r[i];
            for k in f.l_ptr[i] as usize..f.l_ptr[i + 1] as usize - 1 {
                s -= f.l_vals[k] * y[f.l_cols[k] as usize];
            }
            y[i] = s / get(&f.l_vals, &f.l_cols, &f.l_ptr, i, i as i32);
        }
        let mut z = vec![0.0f64; n];
        for i in (0..n).rev() {
            let mut s = y[i];
            for k in f.lt_ptr[i] as usize..f.lt_ptr[i + 1] as usize - 1 {
                s -= f.lt_vals[k] * z[f.lt_cols[k] as usize];
            }
            z[i] = s / get(&f.lt_vals, &f.lt_cols, &f.lt_ptr, i, i as i32);
        }
        let dz = gpu.apply_ic0_for_test(&a, &r).expect("device apply");
        let max_diff = dz.iter().zip(z.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        eprintln!("host-vs-device IC0 apply max diff: {max_diff:e}");
        assert!(max_diff < 1e-9, "device apply diverges: {max_diff:e}");
    }
}
