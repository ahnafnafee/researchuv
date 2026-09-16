//! Aggregation algebraic multigrid — the AMG-class preconditioner.
//!
//! **Setup** (host, once per solve): pairwise matched aggregation — every
//! unmatched vertex joins its strongest unmatched neighbor (ties by smallest
//! index, scan in index order: deterministic); unmatched leftovers become
//! singletons. Coarse operators are Galerkin products `R·A·Rᵀ` over the
//! boolean restriction (one row per fine vertex marking its aggregate), so
//! every level stays SPD. The hierarchy is built until a level drops below
//! [`COARSEST`] unknowns; a level whose aggregation fails to shrink (an
//! edge-free/diagonal tail) is solved by its own Jacobi exact diagonal.
//!
//! **Apply** (device, one V-cycle): restrict the residual by per-aggregate
//! sums ([`crate`]'s `agg_sum` kernel — one thread per aggregate, members in
//! list order, no atomics), recurse, prolong by indexed gather, then one
//! damped-Jacobi post-smooth `e += ω·D⁻¹(r − A·e)` per level. The coarsest
//! level (≤ [`COARSEST`] unknowns) is copied to the host and solved exactly
//! by dense Cholesky. Every reduction is single-thread-per-row: the cycle is
//! bitwise deterministic.

use crate::ffi::{Cuda, CUDA_SUCCESS};
use crate::solver::{CsrMatrix, DeviceMem, GpuSolver, BLOCK};
use std::ffi::c_void;

/// Below this unknown count a level is solved exactly on the host.
pub const COARSEST: usize = 96;
/// Damped-Jacobi relaxation factor for the post-smooth.
pub const OMEGA: f64 = 2.0 / 3.0;

/// The number of top levels running the full K-form (two coarse sweeps
/// with CG steplengths). Below this the cycle takes its V(1,1) form: the
/// K-recursion costs 2^level visits per level, and at hierarchy depth ~18
/// the deepest levels are a few hundred unknowns visited tens of thousands
/// of times — launch-bound, not work-bound. Two K-levels carry nearly all
/// of the robustness benefit at ~4x the V-cycle cost.
pub const K_LEVELS: usize = 2;

/// One level of the hierarchy (all device buffers, sizes in elements).
pub(crate) struct AmgLevel {
    pub n: usize,
    pub n_coarse: usize,
    /// Level operator (CSR) and its inverse diagonal.
    pub d_vals: u64,
    pub d_cols: u64,
    pub d_ptr: u64,
    pub d_inv_diag: u64,
    /// Aggregate id per fine vertex (i32 × n).
    pub d_agg: u64,
    /// Aggregate member lists (CSR over aggregates).
    pub d_members: u64,
    pub d_m_ptr: u64,
    /// Restricted residual handed to the next level (n_coarse).
    pub d_rc: u64,
    /// This level's correction buffer (n; level 0 writes to the caller's z).
    pub d_e: u64,
    /// Per-level cycle scratch: `t = A·x`, the running residual `t2`, and
    /// the prolonged coarse correction `tmp` (the K-cycle's intermediate
    /// values must survive the nested recursive calls).
    pub d_t: u64,
    pub d_t2: u64,
    pub d_tmp: u64,
    /// Host copy of the operator for the exact coarse solve.
    pub host_op: Option<(Vec<f64>, Vec<i32>, Vec<i32>)>,
    /// Diagonal-only level (aggregation stalled): apply = D⁻¹r.
    pub diag_only: bool,
}

/// The uploaded hierarchy plus the reusable dot-partials scratch.
pub(crate) struct DeviceAmg {
    pub levels: Vec<AmgLevel>,
    pub omega: f64,
    /// Reusable dot-product partials (grown on demand by level size).
    pub partials: crate::solver::ScratchBuf,
}

/// Build and upload the hierarchy for `a`.
pub(crate) fn build(mem: &mut DeviceMem, c: &Cuda, a: &CsrMatrix) -> Option<DeviceAmg> {
    let mut levels: Vec<AmgLevel> = Vec::new();
    let mut cur = a.clone();
    let n_fine = cur.rows();
    if n_fine == 0 {
        return None;
    }
    loop {
        let n = cur.rows();
        let (agg, n_coarse) = aggregate(&cur);
        let diag_only = !agg.is_empty() && n_coarse >= n;
        let coarsest = n <= COARSEST || diag_only;
        let inv_diag: Vec<f64> = (0..n)
            .map(|i| {
                let d = cur.row_get(i, i as i32);
                if d.abs() < 1e-300 {
                    f64::NAN
                } else {
                    1.0 / d
                }
            })
            .collect();
        if inv_diag.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let d_vals = mem.upload(c, &cur.vals)?;
        let d_cols = mem.upload(c, &cur.cols)?;
        let d_ptr = mem.upload(c, &cur.row_ptr)?;
        let d_inv_diag = mem.upload(c, &inv_diag)?;
        let (d_agg, d_members, d_m_ptr, d_rc) = if coarsest {
            (0, 0, 0, 0)
        } else {
            // Member lists: aggregate-major, vertex index ascending.
            let mut counts = vec![0i32; n_coarse + 1];
            for &g in &agg {
                counts[g as usize + 1] += 1;
            }
            for k in 1..counts.len() {
                counts[k] += counts[k - 1];
            }
            let m_ptr = counts;
            let mut members = vec![0i32; n];
            let mut cursor = m_ptr.clone();
            for (v, &g) in agg.iter().enumerate() {
                let at = cursor[g as usize];
                cursor[g as usize] += 1;
                members[at as usize] = v as i32;
            }
            let d_agg = mem.upload(c, &agg)?;
            let d_members = mem.upload(c, &members)?;
            let d_m_ptr = mem.upload(c, &m_ptr)?;
            let d_rc = mem.alloc_f64(c, n_coarse)?;
            (d_agg, d_members, d_m_ptr, d_rc)
        };
        let d_e = mem.alloc_f64(c, n)?;
        let d_t = mem.alloc_f64(c, n)?;
        let d_t2 = mem.alloc_f64(c, n)?;
        let d_tmp = mem.alloc_f64(c, n)?;
        let host_op = if coarsest && !diag_only {
            Some((cur.vals.clone(), cur.cols.clone(), cur.row_ptr.clone()))
        } else {
            None
        };
        levels.push(AmgLevel {
            n,
            n_coarse,
            d_vals,
            d_cols,
            d_ptr,
            d_inv_diag,
            d_agg,
            d_members,
            d_m_ptr,
            d_rc,
            d_e,
            d_t,
            d_t2,
            d_tmp,
            host_op,
            diag_only,
        });
        if coarsest {
            break;
        }
        cur = galerkin(&cur, &agg, n_coarse)?;
        if levels.len() > 40 {
            return None; // pathological depth guard
        }
    }
    Some(DeviceAmg { levels, omega: OMEGA, partials: crate::solver::ScratchBuf::default() })
}

/// Pairwise matched aggregation: vertex order scan, strongest unmatched
/// neighbor (|A(v,u)| max, ties by smallest u).
fn aggregate(a: &CsrMatrix) -> (Vec<i32>, usize) {
    let n = a.rows();
    let mut agg = vec![-1i32; n];
    let mut next = 0i32;
    for v in 0..n {
        if agg[v] != -1 {
            continue;
        }
        // Candidates in ascending column order: the tie-break (first
        // strictly-strongest wins) must not depend on the caller's CSR
        // insertion order.
        let mut cands: Vec<(usize, f64)> = a
            .row(v)
            .filter(|(c, _)| *c as usize != v)
            .map(|(c, val)| (c as usize, val.abs()))
            .collect();
        cands.sort_unstable_by_key(|(u, _)| *u);
        let mut best: Option<(f64, usize)> = None;
        for (u, s) in cands {
            if agg[u] != -1 {
                continue;
            }
            match best {
                Some((bs, _)) if s <= bs => {}
                _ => best = Some((s, u)),
            }
        }
        agg[v] = next;
        if let Some((_, u)) = best {
            agg[u] = next;
        }
        next += 1;
    }
    (agg, next as usize)
}

/// Galerkin coarse operator `R·A·Rᵀ` (boolean restriction).
fn galerkin(a: &CsrMatrix, agg: &[i32], n_coarse: usize) -> Option<CsrMatrix> {
    use std::collections::BTreeMap;
    let mut rows: Vec<BTreeMap<i32, f64>> = vec![BTreeMap::new(); n_coarse];
    for i in 0..a.rows() {
        let gi = agg[i];
        for (cj, v) in a.row(i) {
            let gj = agg[cj as usize];
            *rows[gi as usize].entry(gj).or_insert(0.0) += v;
        }
    }
    let mut vals = Vec::new();
    let mut cols = Vec::new();
    let mut row_ptr = vec![0i32; n_coarse + 1];
    for (r, row) in rows.iter().enumerate() {
        for (&c, &v) in row {
            cols.push(c);
            vals.push(v);
        }
        row_ptr[r + 1] = vals.len() as i32;
    }
    Some(CsrMatrix { vals, cols, row_ptr })
}

/// A borrowed view of one level's device operands (copied to locals so the
/// K-cycle's recursive `&mut` calls don't fight the borrow checker).
struct LevelRef {
    n: usize,
    #[allow(dead_code)]
    n_coarse: usize,
    d_vals: u64,
    d_cols: u64,
    d_ptr: u64,
    d_inv_diag: u64,
    #[allow(dead_code)]
    d_agg: u64,
    #[allow(dead_code)]
    d_members: u64,
    #[allow(dead_code)]
    d_m_ptr: u64,
    #[allow(dead_code)]
    d_rc: u64,
    d_t: u64,
    d_t2: u64,
    d_tmp: u64,
    diag_only: bool,
    has_host_op: bool,
}

/// One K-cycle: `out = M⁻¹ r_in` at `lvl` (Notay's recursive
/// preconditioner). Returns false on driver failure.
///
/// Per level: pre-smooth from zero, then **two** coarse corrections, each
/// followed by a CG-style steplength `β = (r·r)/(r·A·y)` (guarded — a
/// non-positive denominator skips the update), then one post-smooth. The
/// steplengths are what make the recursion robust for unsmoothed
/// aggregation, where the plain V-cycle stagnates on near-singular systems;
/// the cost is one extra coarse sweep per level. Every reduction is
/// single-thread-per-row or block-deterministic: the cycle is bitwise
/// reproducible.
pub(crate) fn apply(s: &GpuSolver, amg: &mut DeviceAmg, lvl: usize, d_r_in: u64, d_out: u64) -> bool {
    let c = &s.cuda;
    #[allow(non_snake_case)]
    let L = {
        let l = &amg.levels[lvl];
        (
            l.n,
            l.n_coarse,
            l.d_vals,
            l.d_cols,
            l.d_ptr,
            l.d_inv_diag,
            l.d_agg,
            l.d_members,
            l.d_m_ptr,
            l.d_rc,
            l.d_e,
            l.d_t,
            l.d_t2,
            l.d_tmp,
            l.diag_only,
            l.host_op.is_some(),
        )
    };
    let (n, n_coarse, d_vals, d_cols, d_ptr, d_inv_diag, d_agg, d_members, d_m_ptr, d_rc, _d_e, d_t, d_t2, d_tmp, diag_only, has_host_op) = L;
    #[allow(non_snake_case)]
    let L = LevelRef {
        n,
        n_coarse,
        d_vals,
        d_cols,
        d_ptr,
        d_inv_diag,
        d_agg,
        d_members,
        d_m_ptr,
        d_rc,
        d_t,
        d_t2,
        d_tmp,
        diag_only,
        has_host_op,
    };
    let launch = |f: crate::ffi::CUfunction, n: usize, params: &mut [*mut c_void]| unsafe {
        (c.cuLaunchKernel)(
            f,
            n.div_ceil(BLOCK as usize) as u32,
            1,
            1,
            BLOCK,
            1,
            1,
            0,
            std::ptr::null_mut(),
            params.as_mut_ptr(),
            std::ptr::null_mut(),
        ) == CUDA_SUCCESS
    };
    let p_u64 = |v: &u64| v as *const u64 as *mut c_void;
    let p_i32 = |v: &i32| v as *const i32 as *mut c_void;
    let p_f64 = |v: &f64| v as *const f64 as *mut c_void;
    let (t, t2) = (L.d_t, L.d_t2);

    if L.diag_only {
        let n_i = L.n as i32;
        return launch(s.k_mul, L.n, &mut [p_u64(&L.d_inv_diag), p_u64(&d_r_in), p_u64(&d_out), p_i32(&n_i)]);
    }
    if L.has_host_op {
        return coarse_exact(s, lvl, &amg.levels[lvl], d_r_in, d_out);
    }
    let n_i = L.n as i32;
    let om = amg.omega;
    let m1 = -1.0f64;

    // A helper closure set over this level's fixed operands.
    let spmv_into = |dst: u64, src: u64| -> bool {
        launch(
            s.k_spmv,
            L.n,
            &mut [
                p_u64(&L.d_vals),
                p_u64(&L.d_cols),
                p_u64(&L.d_ptr),
                p_u64(&src),
                p_u64(&dst),
                p_i32(&n_i),
            ],
        )
    };

    // 1. Pre-smooth from zero: out = ω·D⁻¹ r.
    if !launch(s.k_fill0, L.n, &mut [p_u64(&d_out), p_i32(&n_i)]) {
        return false;
    }
    if !launch(
        s.k_mul,
        L.n,
        &mut [p_u64(&L.d_inv_diag), p_u64(&d_r_in), p_u64(&t), p_i32(&n_i)],
    ) {
        return false;
    }
    if !launch(s.k_axpy, L.n, &mut [p_f64(&om), p_u64(&t), p_u64(&d_out), p_i32(&n_i)]) {
        return false;
    }
    // 2. r1 = r − A·out → t2.
    if !spmv_into(t, d_out) {
        return false;
    }
    if !launch(s.k_copy, L.n, &mut [p_u64(&t2), p_u64(&d_r_in), p_i32(&n_i)]) {
        return false;
    }
    if !launch(s.k_axpy, L.n, &mut [p_f64(&m1), p_u64(&t), p_u64(&t2), p_i32(&n_i)]) {
        return false;
    }
    // 3-4. Coarse correction #1 with a CG steplength.
    let coarse_step = |amg: &mut DeviceAmg, resid: u64| -> bool {
        let n_agg = n_coarse as i32;
        if !launch(
            s.k_agg,
            n_coarse,
            &mut [
                p_u64(&resid),
                p_u64(&d_rc),
                p_u64(&d_members),
                p_u64(&d_m_ptr),
                p_i32(&n_agg),
            ],
        ) {
            return false;
        }
        let coarse_e = amg.levels[lvl + 1].d_e;
        if !apply(s, amg, lvl + 1, d_rc, coarse_e) {
            return false;
        }
        launch(
            s.k_gather,
            n,
            &mut [p_u64(&d_tmp), p_u64(&coarse_e), p_u64(&d_agg), p_i32(&n_i)],
        )
    };
    // The V(1,1) tail: one coarse correction, no steplength, post-smooth.
    if lvl >= K_LEVELS {
        if !coarse_step(amg, t2) {
            return false;
        }
        let one = 1.0f64;
        if !launch(s.k_axpy, L.n, &mut [p_f64(&one), p_u64(&L.d_tmp), p_u64(&d_out), p_i32(&n_i)]) {
            return false;
        }
        if !spmv_into(L.d_t, d_out) {
            return false;
        }
        if !launch(s.k_copy, L.n, &mut [p_u64(&L.d_t2), p_u64(&d_r_in), p_i32(&n_i)]) {
            return false;
        }
        if !launch(s.k_axpy, L.n, &mut [p_f64(&m1), p_u64(&L.d_t), p_u64(&L.d_t2), p_i32(&n_i)]) {
            return false;
        }
        if !launch(
            s.k_mul,
            L.n,
            &mut [p_u64(&L.d_inv_diag), p_u64(&L.d_t2), p_u64(&L.d_t), p_i32(&n_i)],
        ) {
            return false;
        }
        return launch(s.k_axpy, L.n, &mut [p_f64(&om), p_u64(&L.d_t), p_u64(&d_out), p_i32(&n_i)]);
    }

    if !coarse_step(amg, t2) {
        return false;
    }
    // denom = (r1·A y1) via t; numer = (r1·r1).
    if !spmv_into(t, L.d_tmp) {
        return false;
    }
    let (num1, den1) = match (s.dot_s(&mut amg.partials, t2, t2, L.n), s.dot_s(&mut amg.partials, t2, t, L.n)) {
        (Some(a), Some(b)) => (a, b),
        _ => return false,
    };
    if den1 > 1e-300 {
        let beta = num1 / den1;
        if !launch(s.k_axpy, L.n, &mut [p_f64(&beta), p_u64(&L.d_tmp), p_u64(&d_out), p_i32(&n_i)]) {
            return false;
        }
        // r2 = r1 − β·A y1 (reuse t2; t already holds A y1).
        if !launch(s.k_axpy, L.n, &mut [p_f64(&-beta), p_u64(&t), p_u64(&t2), p_i32(&n_i)]) {
            return false;
        }
    }
    // 5-6. Coarse correction #2 on the updated residual.
    if !coarse_step(amg, t2) {
        return false;
    }
    if !spmv_into(t, L.d_tmp) {
        return false;
    }
    let (num2, den2) = match (s.dot_s(&mut amg.partials, t2, t2, L.n), s.dot_s(&mut amg.partials, t2, t, L.n)) {
        (Some(a), Some(b)) => (a, b),
        _ => return false,
    };
    if den2 > 1e-300 {
        let beta = num2 / den2;
        if !launch(s.k_axpy, L.n, &mut [p_f64(&beta), p_u64(&L.d_tmp), p_u64(&d_out), p_i32(&n_i)]) {
            return false;
        }
    }
    // Notay's K-form ends here: NO post-smooth inside the K-level (the
    // steplength-conjugated coarse corrections close the level; adding a
    // smoothing sweep after them breaks the recursion's guaranteed
    // positivity — verified empirically: FCG + K-with-post-smooth
    // stagnates on Poisson where the plain form converges).
    true
}

/// Exact coarse solve on the host (dense Cholesky of the small operator).
#[allow(non_snake_case)]
pub(crate) fn coarse_exact(s: &GpuSolver, _lvl: usize, L: &AmgLevel, d_r: u64, d_out: u64) -> bool {
    let c = &s.cuda;
    let n = L.n;
    let (vals, cols, ptr) = L.host_op.as_ref().expect("coarsest carries its operator");
    let mut b = vec![0.0f64; n];
    // SAFETY: host buffer sized for the copy.
    unsafe {
        if (c.cuMemcpyDtoH)(b.as_mut_ptr().cast(), d_r, n * 8) != CUDA_SUCCESS {
            return false;
        }
    }
    // Dense assembly + Cholesky (n ≤ COARSEST = 96).
    let mut m = vec![0.0f64; n * n];
    for i in 0..n {
        for k in ptr[i] as usize..ptr[i + 1] as usize {
            m[i * n + cols[k] as usize] = vals[k];
        }
    }
    for j in 0..n {
        let mut d = m[j * n + j];
        for k in 0..j {
            d -= m[j * n + k] * m[j * n + k];
        }
        if d <= 1e-300 {
            return false;
        }
        m[j * n + j] = d.sqrt();
        for i in (j + 1)..n {
            let mut s_ = m[i * n + j];
            for k in 0..j {
                s_ -= m[i * n + k] * m[j * n + k];
            }
            m[i * n + j] = s_ / m[j * n + j];
        }
    }
    // Forward/backward substitution.
    for i in 0..n {
        let mut s_ = b[i];
        for k in 0..i {
            s_ -= m[i * n + k] * b[k];
        }
        b[i] = s_ / m[i * n + i];
    }
    for i in (0..n).rev() {
        let mut s_ = b[i];
        for k in (i + 1)..n {
            s_ -= m[k * n + i] * b[k];
        }
        b[i] = s_ / m[i * n + i];
    }
    // SAFETY: host buffer sized for the copy.
    unsafe { (c.cuMemcpyHtoD)(d_out, b.as_ptr().cast(), n * 8) == CUDA_SUCCESS }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairwise_aggregation_shrinks_and_covers() {
        // Path graph: matching pairs up vertices; every vertex aggregated.
        let n = 101;
        let mut vals = Vec::new();
        let mut cols = Vec::new();
        let mut row_ptr = vec![0i32];
        for i in 0..n {
            if i > 0 {
                vals.push(-1.0);
                cols.push((i - 1) as i32);
            }
            vals.push(2.0);
            cols.push(i as i32);
            if i + 1 < n {
                vals.push(-1.0);
                cols.push((i + 1) as i32);
            }
            row_ptr.push(vals.len() as i32);
        }
        let a = CsrMatrix { vals, cols, row_ptr };
        let (agg, n_coarse) = aggregate(&a);
        assert!(agg.iter().all(|&g| g >= 0), "every vertex aggregated");
        assert!(n_coarse <= (n + 1) / 2 + 1, "pairs: {n_coarse} aggregates for {n} vertices");
        assert!(n_coarse < n);
    }

    #[test]
    fn galerkin_keeps_symmetry_and_spd_scale() {
        let n = 40;
        let mut vals = Vec::new();
        let mut cols = Vec::new();
        let mut row_ptr = vec![0i32];
        for i in 0..n {
            if i > 0 {
                vals.push(-1.0);
                cols.push((i - 1) as i32);
            }
            vals.push(4.0);
            cols.push(i as i32);
            if i + 1 < n {
                vals.push(-1.0);
                cols.push((i + 1) as i32);
            }
            row_ptr.push(vals.len() as i32);
        }
        let a = CsrMatrix { vals, cols, row_ptr };
        let (agg, n_coarse) = aggregate(&a);
        let ac = galerkin(&a, &agg, n_coarse).expect("galerkin");
        assert_eq!(ac.rows(), n_coarse);
        // Symmetric: A_c(i,j) == A_c(j,i).
        for i in 0..n_coarse {
            for (cj, v) in ac.row(i) {
                let j = cj as usize;
                assert!((ac.row_get(j, i as i32) - v).abs() < 1e-12, "({i},{j}) asymmetric");
            }
        }
        // The Galerkin product preserves the quadratic form on the all-ones
        // vector (Rᵗ·1_c = 1_f and 1_cᵗ·R = 1_fᵗ), so the total entry sum is
        // invariant across levels — no mass is created or lost.
        let total = |m: &CsrMatrix| -> f64 { m.vals.iter().sum() };
        let tf = total(&a);
        let tc = total(&ac);
        assert!((tf - tc).abs() < 1e-9, "total {tc} vs {tf}");
    }
}
