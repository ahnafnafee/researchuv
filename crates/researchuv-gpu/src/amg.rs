//! Aggregation algebraic multigrid — the AMG-class preconditioner.
//!
//! **Setup** (host, once per solve): matched aggregation — every unmatched
//! vertex collects its strongest unmatched neighbors (ties by smallest
//! index, scan in index order: deterministic) until its aggregate is full.
//! [`Precond::Amg`] pairs (the K-cycle prefers small aggregates);
//! [`Precond::AmgSa`] targets [`SA_AGG`] members, because smoothed
//! aggregation wants the faster coarsening (4× on mesh graphs vs the pairs'
//! 2×) to keep the hierarchy shallow and its operators sparse. The transfer
//! between levels is a CSR prolongator `P` (one row per fine vertex): the
//! boolean tentative matrix `B` (one 1 per row), or the **smoothed
//! aggregation** form `P = (I − ωD⁻¹A)·B` — one damped-Jacobi sweep that
//! spreads each aggregate's column over its neighbors (each row truncated
//! to its [`P_MAX_ROW`] largest entries, own aggregate always kept, which
//! keeps `P` full column rank), making the coarse basis piecewise linear
//! instead of piecewise constant. Coarse operators are Galerkin products
//! `Pᵗ·A·P` of a full-rank prolongator, so every level stays SPD without
//! any operator surgery.
//!
//! **Coarse solve** (the "sparse coarse-solve pipeline"): the hierarchy is
//! built until a level drops below [`COARSEST`] unknowns; that level is
//! factored **exactly** once at setup — sparse Cholesky in a BFS
//! nested-dissection order ([`crate::precond::nd_order`] +
//! [`crate::precond::factor_exact`]) — and the triangular factors are
//! applied on the device by the level-scheduled `tri_level` sweeps (with
//! the elimination-order permutation round trip), no per-cycle host round
//! trip. A level whose aggregation fails to shrink (an edge-free/diagonal
//! tail) is solved by its own Jacobi diagonal.
//!
//! **Apply** (device, one cycle): restrict the residual with the CSR
//! restriction `R = Pᵗ` (the plain `spmv` kernel — one thread per coarse
//! row, members in ascending order: bitwise deterministic), recurse,
//! prolong with `P` the same way. The top [`K_LEVELS`] levels run
//! Notay's K-form (two coarse corrections with CG steplengths); below
//! that the cycle takes its V(1,1) form with a damped-Jacobi post-smooth
//! `e += ω·D⁻¹(r − A·e)` closing each level.
//!
//! [`Precond::Amg`]: crate::precond::Precond::Amg
//! [`Precond::AmgSa`]: crate::precond::Precond::AmgSa

use crate::ffi::{Cuda, CUDA_SUCCESS};
use crate::precond::Ic0Factor;
use crate::solver::{CsrMatrix, DeviceMem, GpuSolver, BLOCK};
use std::ffi::c_void;

/// Below this unknown count a level is factored exactly (sparse Cholesky)
/// and solved on the device. The sparse factorization replaces the former
/// dense host solve, so the cutoff is set by setup cost and launch depth,
/// not by per-iteration PCIe traffic.
pub const COARSEST: usize = 512;
/// Damping shared by the post-smooth and (for [`crate::precond::Precond::AmgSa`])
/// the prolongator smoothing sweep.
pub const OMEGA: f64 = 2.0 / 3.0;

/// The number of top levels running the full K-form (two coarse sweeps
/// with CG steplengths). Below this the cycle takes its V(1,1) form: the
/// K-recursion costs 2^level visits per level, and at hierarchy depth ~18
/// the deepest levels are a few hundred unknowns visited tens of thousands
/// of times — launch-bound, not work-bound. Two K-levels carry nearly all
/// of the robustness benefit at ~4x the V-cycle cost.
pub const K_LEVELS: usize = 2;

/// How a level bottoms out.
pub(crate) enum Coarse {
    /// Transfers to the next level exist; the cycle recurses.
    Deeper,
    /// Aggregation stalled: apply is the exact Jacobi diagonal `D⁻¹r`.
    Diag,
    /// Coarsest level: an exact sparse Cholesky applied on the device
    /// (with its elimination-order permutation).
    Sparse(crate::precond::DeviceCoarse),
}

/// One level of the hierarchy (all device buffers, sizes in elements).
pub(crate) struct AmgLevel {
    pub n: usize,
    pub n_coarse: usize,
    /// Level operator (CSR) and its inverse diagonal.
    pub d_vals: u64,
    pub d_cols: u64,
    pub d_ptr: u64,
    pub d_inv_diag: u64,
    /// Restriction `R = Pᵗ` (CSR, `n_coarse` rows).
    pub d_r_vals: u64,
    pub d_r_cols: u64,
    pub d_r_ptr: u64,
    /// Prolongation `P` (CSR, `n` rows).
    pub d_p_vals: u64,
    pub d_p_cols: u64,
    pub d_p_ptr: u64,
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
    pub coarse: Coarse,
}

/// The uploaded hierarchy plus the reusable dot-partials scratch.
pub(crate) struct DeviceAmg {
    pub levels: Vec<AmgLevel>,
    pub omega: f64,
    /// Reusable dot-product partials (grown on demand by level size).
    pub partials: crate::solver::ScratchBuf,
}

/// One host stage of [`hierarchy`]: the level operator, its prolongator to
/// the next level (when it recurses), and — on the coarsest level — the
/// exact sparse factor the device sweeps apply.
struct HostLevel {
    op: CsrMatrix,
    p: Option<CsrMatrix>,
    /// Exact factor of the coarsest operator with its elimination order
    /// (absent when the level is diagonal-only: its aggregation stalled).
    factor: Option<(Ic0Factor, Vec<i32>)>,
    diag_only: bool,
}

/// Build the host hierarchy for `a`: operators, prolongators, and the
/// coarsest level's exact factor. `smooth` selects the smoothed-aggregation
/// prolongator; `false` keeps the boolean tentative form.
fn hierarchy(a: &CsrMatrix, smooth: bool) -> Option<Vec<HostLevel>> {
    let mut levels: Vec<HostLevel> = Vec::new();
    let mut cur = a.clone();
    if cur.rows() == 0 {
        return None;
    }
    loop {
        let n = cur.rows();
        let (agg, n_coarse) = aggregate(&cur, if smooth { SA_AGG } else { 2 });
        let diag_only = n_coarse >= n;
        let coarsest = n <= COARSEST || diag_only;
        // Reject levels with a missing diagonal up front (the smoother
        // divides by it, and every sweep needs D⁻¹).
        if (0..n).any(|i| cur.row_get(i, i as i32).abs() < 1e-300) {
            return None;
        }
        let (p, factor) = if coarsest {
            let factor = if diag_only {
                None
            } else {
                let order = crate::precond::nd_order(&cur);
                Some((crate::precond::factor_exact(&cur, &order)?, order))
            };
            (None, factor)
        } else {
            (Some(prolongator(&cur, &agg, smooth)), None)
        };
        levels.push(HostLevel { op: cur.clone(), p, factor, diag_only });
        if coarsest {
            break;
        }
        cur = galerkin_p(&cur, levels.last().unwrap().p.as_ref().unwrap(), n_coarse);
        if levels.len() > 40 {
            return None; // pathological depth guard
        }
    }
    Some(levels)
}

/// Build and upload the hierarchy for `a` (`smooth` = smoothed aggregation).
pub(crate) fn build(mem: &mut DeviceMem, c: &Cuda, a: &CsrMatrix, smooth: bool) -> Option<DeviceAmg> {
    let host = hierarchy(a, smooth)?;
    let mut levels: Vec<AmgLevel> = Vec::with_capacity(host.len());
    for h in &host {
        let n = h.op.rows();
        let inv_diag: Vec<f64> = (0..n)
            .map(|i| {
                let d = h.op.row_get(i, i as i32);
                if d.abs() < 1e-300 {
                    f64::NAN
                } else {
                    1.0 / d
                }
            })
            .collect();
        let d_vals = mem.upload(c, &h.op.vals)?;
        let d_cols = mem.upload(c, &h.op.cols)?;
        let d_ptr = mem.upload(c, &h.op.row_ptr)?;
        let d_inv_diag = mem.upload(c, &inv_diag)?;
        // Transfers: R = Pᵗ (coarse rows) and P (fine rows), or nothing on
        // a level that bottoms out here.
        let (n_coarse, d_r, d_p, d_rc) = match &h.p {
            Some(p) => {
                let n_coarse = p.cols.iter().copied().max().map(|m| m as usize + 1).unwrap_or(0);
                let r = transpose_rect(p, n_coarse);
                let d_r_vals = mem.upload(c, &r.vals)?;
                let d_r_cols = mem.upload(c, &r.cols)?;
                let d_r_ptr = mem.upload(c, &r.row_ptr)?;
                let d_p_vals = mem.upload(c, &p.vals)?;
                let d_p_cols = mem.upload(c, &p.cols)?;
                let d_p_ptr = mem.upload(c, &p.row_ptr)?;
                let d_rc = mem.alloc_f64(c, n_coarse)?;
                (
                    n_coarse,
                    (d_r_vals, d_r_cols, d_r_ptr),
                    (d_p_vals, d_p_cols, d_p_ptr),
                    d_rc,
                )
            }
            None => (0, (0, 0, 0), (0, 0, 0), 0),
        };
        let d_e = mem.alloc_f64(c, n)?;
        let d_t = mem.alloc_f64(c, n)?;
        let d_t2 = mem.alloc_f64(c, n)?;
        let d_tmp = mem.alloc_f64(c, n)?;
        let coarse = if h.diag_only {
            Coarse::Diag
        } else if let Some((f, order)) = &h.factor {
            Coarse::Sparse(crate::precond::upload_coarse_factor(mem, c, f, order)?)
        } else {
            Coarse::Deeper
        };
        let ((d_r_vals, d_r_cols, d_r_ptr), (d_p_vals, d_p_cols, d_p_ptr)) = (d_r, d_p);
        levels.push(AmgLevel {
            n,
            n_coarse,
            d_vals,
            d_cols,
            d_ptr,
            d_inv_diag,
            d_r_vals,
            d_r_cols,
            d_r_ptr,
            d_p_vals,
            d_p_cols,
            d_p_ptr,
            d_rc,
            d_e,
            d_t,
            d_t2,
            d_tmp,
            coarse,
        });
    }
    Some(DeviceAmg { levels, omega: OMEGA, partials: crate::solver::ScratchBuf::default() })
}

/// Aggregate target for the smoothed hierarchy ([`aggregate`]).
const SA_AGG: usize = 4;

/// Matched aggregation, `max_members` vertices per aggregate: vertex order
/// scan, each unmatched vertex collects its strongest unmatched neighbors
/// (|A(v,u)| max, ties by smallest u) until the aggregate is full. The
/// unsmoothed hierarchy pairs (`max_members = 2`) — the K-cycle prefers
/// small aggregates; smoothed aggregation uses [`SA_AGG`] — bigger
/// aggregates coarsen faster (4× on mesh graphs instead of 2×), which is
/// what keeps the smoothed hierarchy shallow and its operators sparse.
fn aggregate(a: &CsrMatrix, max_members: usize) -> (Vec<i32>, usize) {
    let n = a.rows();
    let mut agg = vec![-1i32; n];
    let mut next = 0i32;
    for v in 0..n {
        if agg[v] != -1 {
            continue;
        }
        // Candidates in ascending column order: the tie-break (strongest
        // first, then smallest index) must not depend on the caller's CSR
        // insertion order.
        let mut cands: Vec<(usize, f64)> = a
            .row(v)
            .filter(|(c, _)| *c as usize != v)
            .map(|(c, val)| (c as usize, val.abs()))
            .collect();
        cands.sort_unstable_by_key(|(u, _)| *u);
        let mut unmatched: Vec<(f64, usize)> = cands
            .iter()
            .filter(|(u, _)| agg[*u] == -1)
            .map(|(u, s)| (*s, *u))
            .collect();
        unmatched.sort_by(|(s1, u1), (s2, u2)| {
            s2.partial_cmp(s1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(u1.cmp(u2))
        });
        let take = (max_members - 1).min(unmatched.len());
        let mut picks: Vec<usize> = unmatched[..take].iter().map(|&(_, u)| u).collect();
        picks.sort_unstable();
        agg[v] = next;
        for u in picks {
            agg[u] = next;
        }
        next += 1;
    }
    (agg, next as usize)
}

/// Cap on a smoothed prolongator row. One Jacobi sweep of a dense level's
/// operator would spread each row over every neighbor's aggregate, and the
/// Galerkin products then densify geometrically level over level. Keeping
/// each row's largest entries bounds that growth; the own-aggregate entry
/// always survives (it is what keeps P full column rank, so the Galerkin
/// products stay SPD).
const P_MAX_ROW: usize = 32;

/// The prolongator over `agg` (`n` fine rows, one column per aggregate).
/// With `smooth = false` this is the tentative boolean `B` (one 1 per
/// row). With `smooth = true`, one damped-Jacobi sweep of the tentative
/// columns: `P = (I − ω·D⁻¹A)·B`, i.e. row `v` carries `1 − ω·A(v,v)/D(v)`
/// in its own aggregate and `−ω·A(v,u)/D(v)` (summed over members `u`) in
/// every neighboring aggregate — truncated to its [`P_MAX_ROW`]
/// largest entries (own aggregate kept).
fn prolongator(a: &CsrMatrix, agg: &[i32], smooth: bool) -> CsrMatrix {
    let n = a.rows();
    let mut vals = Vec::new();
    let mut cols = Vec::new();
    let mut row_ptr = vec![0i32; n + 1];
    for v in 0..n {
        if !smooth {
            vals.push(1.0);
            cols.push(agg[v]);
        } else {
            let d = a.row_get(v, v as i32);
            let mut row: Vec<(i32, f64)> = Vec::new();
            for (u, val) in a.row(v) {
                row.push((agg[u as usize], -OMEGA * val / d));
            }
            row.push((agg[v], 1.0));
            let merged = merge_sorted(row);
            if merged.len() <= P_MAX_ROW {
                for (c, w) in merged {
                    cols.push(c);
                    vals.push(w);
                }
            } else {
                // (magnitude desc, column asc): deterministic top-k, with
                // the own-aggregate entry kept outside the budget.
                let mut order: Vec<usize> = (0..merged.len()).collect();
                order.sort_by(|&x, &y| {
                    let (cx, vx) = (merged[x].0, merged[x].1.abs());
                    let (cy, vy) = (merged[y].0, merged[y].1.abs());
                    vy.partial_cmp(&vx)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(cx.cmp(&cy))
                });
                let mut keep = vec![false; merged.len()];
                let mut budget = P_MAX_ROW - 1;
                for &k in &order {
                    if merged[k].0 == agg[v] {
                        keep[k] = true;
                    } else if budget > 0 {
                        keep[k] = true;
                        budget -= 1;
                    }
                }
                for (k, (c, w)) in merged.into_iter().enumerate() {
                    if keep[k] {
                        cols.push(c);
                        vals.push(w);
                    }
                }
            }
        }
        row_ptr[v + 1] = vals.len() as i32;
    }
    CsrMatrix { vals, cols, row_ptr }
}

/// Sort `(column, value)` pairs by column and sum duplicates (ascending
/// order throughout — the merged row is canonical and deterministic).
fn merge_sorted(mut ents: Vec<(i32, f64)>) -> Vec<(i32, f64)> {
    ents.sort_unstable_by_key(|(c, _)| *c);
    let mut out: Vec<(i32, f64)> = Vec::with_capacity(ents.len());
    for (c, w) in ents {
        match out.last_mut() {
            Some((lc, lw)) if *lc == c => *lw += w,
            _ => out.push((c, w)),
        }
    }
    out
}

/// Galerkin coarse operator `Pᵗ·A·P` in two phases: `W = A·P`, then
/// `A_c = Pᵗ·W` (rows of the transposed prolongator against rows of W).
/// Every phase sums merged-ascending rows, so the product is deterministic.
/// With a full-column-rank `P` (the prolongators here are — every aggregate
/// carries its members' own-aggregate entries) this is an exact Galerkin
/// product: SPD in, SPD out, level after level.
fn galerkin_p(a: &CsrMatrix, p: &CsrMatrix, n_coarse: usize) -> CsrMatrix {
    // W = A·P, one merged row per fine vertex.
    let n = a.rows();
    let mut w_rows: Vec<Vec<(i32, f64)>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut ents: Vec<(i32, f64)> = Vec::new();
        for (j, a_ij) in a.row(i) {
            let ju = j as usize;
            for k in p.row_ptr[ju] as usize..p.row_ptr[ju + 1] as usize {
                ents.push((p.cols[k], a_ij * p.vals[k]));
            }
        }
        w_rows.push(merge_sorted(ents));
    }
    let w_ptr: Vec<i32> = std::iter::once(0)
        .chain(w_rows.iter().scan(0, |acc, r| {
            *acc += r.len() as i32;
            Some(*acc)
        }))
        .collect();
    let w = CsrMatrix {
        vals: w_rows.iter().flatten().map(|(_, v)| *v).collect(),
        cols: w_rows.iter().flatten().map(|(c, _)| *c).collect(),
        row_ptr: w_ptr,
    };
    // A_c = R·W over the rows of R = Pᵗ.
    let r = transpose_rect(p, n_coarse);
    let mut vals = Vec::new();
    let mut cols = Vec::new();
    let mut row_ptr = vec![0i32; n_coarse + 1];
    for g in 0..n_coarse {
        let mut ents: Vec<(i32, f64)> = Vec::new();
        for (j, r_gj) in r.row(g) {
            let ju = j as usize;
            for k in w.row_ptr[ju] as usize..w.row_ptr[ju + 1] as usize {
                ents.push((w.cols[k], r_gj * w.vals[k]));
            }
        }
        for (c, v) in merge_sorted(ents) {
            cols.push(c);
            vals.push(v);
        }
        row_ptr[g + 1] = vals.len() as i32;
    }
    CsrMatrix { vals, cols, row_ptr }
}

/// Transpose a rectangular CSR matrix (`n_fine` rows × `n_coarse` columns)
/// into CSR over the coarse rows, fine indices ascending within each row.
fn transpose_rect(p: &CsrMatrix, n_coarse: usize) -> CsrMatrix {
    let mut counts = vec![0i32; n_coarse + 1];
    for &c in &p.cols {
        counts[c as usize + 1] += 1;
    }
    for k in 1..counts.len() {
        counts[k] += counts[k - 1];
    }
    let row_ptr = counts;
    let mut vals = vec![0.0f64; p.vals.len()];
    let mut cols = vec![0i32; p.cols.len()];
    let mut cursor = row_ptr.clone();
    for v in 0..p.rows() {
        for k in p.row_ptr[v] as usize..p.row_ptr[v + 1] as usize {
            let g = p.cols[k] as usize;
            let at = cursor[g];
            cursor[g] += 1;
            cols[at as usize] = v as i32;
            vals[at as usize] = p.vals[k];
        }
    }
    CsrMatrix { vals, cols, row_ptr }
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
    match &amg.levels[lvl].coarse {
        Coarse::Diag => {
            let n_i = amg.levels[lvl].n as i32;
            let d_inv_diag = amg.levels[lvl].d_inv_diag;
            // SAFETY: same launch contract as the cycle's other kernels.
            return unsafe {
                launch1(
                    c,
                    s.k_mul,
                    amg.levels[lvl].n,
                    &mut [p_u64(&d_inv_diag), p_u64(&d_r_in), p_u64(&d_out), p_i32(&n_i)],
                )
            };
        }
        Coarse::Sparse(f) => {
            let n = amg.levels[lvl].n;
            return coarse_sparse(s, f, d_r_in, d_out, n);
        }
        Coarse::Deeper => {}
    }
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
            l.d_r_vals,
            l.d_r_cols,
            l.d_r_ptr,
            l.d_p_vals,
            l.d_p_cols,
            l.d_p_ptr,
            l.d_rc,
            l.d_t,
            l.d_t2,
            l.d_tmp,
        )
    };
    let (n, n_coarse, d_vals, d_cols, d_ptr, d_inv_diag, d_r_vals, d_r_cols, d_r_ptr, d_p_vals, d_p_cols, d_p_ptr, d_rc, d_t, d_t2, d_tmp) = L;
    let (t, t2) = (d_t, d_t2);
    let launch = |f: crate::ffi::CUfunction, n: usize, params: &mut [*mut c_void]| unsafe {
        launch1(c, f, n, params)
    };
    let n_i = n as i32;
    let om = amg.omega;
    let m1 = -1.0f64;

    // A helper closure set over this level's fixed operands.
    let spmv_into = |vals: u64, cols: u64, ptr: u64, dst: u64, src: u64, rows: usize| -> bool {
        let rows_i = rows as i32;
        launch(
            s.k_spmv,
            rows,
            &mut [p_u64(&vals), p_u64(&cols), p_u64(&ptr), p_u64(&src), p_u64(&dst), p_i32(&rows_i)],
        )
    };

    // 1. Pre-smooth from zero: out = ω·D⁻¹ r.
    if !launch(s.k_fill0, n, &mut [p_u64(&d_out), p_i32(&n_i)]) {
        return false;
    }
    if !launch(
        s.k_mul,
        n,
        &mut [p_u64(&d_inv_diag), p_u64(&d_r_in), p_u64(&d_t), p_i32(&n_i)],
    ) {
        return false;
    }
    if !launch(s.k_axpy, n, &mut [p_f64(&om), p_u64(&d_t), p_u64(&d_out), p_i32(&n_i)]) {
        return false;
    }
    // 2. r1 = r − A·out → t2.
    if !spmv_into(d_vals, d_cols, d_ptr, d_t, d_out, n) {
        return false;
    }
    if !launch(s.k_copy, n, &mut [p_u64(&d_t2), p_u64(&d_r_in), p_i32(&n_i)]) {
        return false;
    }
    if !launch(s.k_axpy, n, &mut [p_f64(&m1), p_u64(&d_t), p_u64(&d_t2), p_i32(&n_i)]) {
        return false;
    }
    // 3-4. Coarse correction #1 with a CG steplength: rc = R·resid,
    // recurse, then tmp = P·e_c.
    let coarse_step = |amg: &mut DeviceAmg, resid: u64| -> bool {
        if !spmv_into(d_r_vals, d_r_cols, d_r_ptr, d_rc, resid, n_coarse) {
            return false;
        }
        let coarse_e = amg.levels[lvl + 1].d_e;
        if !apply(s, amg, lvl + 1, d_rc, coarse_e) {
            return false;
        }
        spmv_into(d_p_vals, d_p_cols, d_p_ptr, d_tmp, coarse_e, n)
    };
    // The V(1,1) tail: one coarse correction, no steplength, post-smooth.
    if lvl >= K_LEVELS {
        if !coarse_step(amg, t2) {
            return false;
        }
        let one = 1.0f64;
        if !launch(s.k_axpy, n, &mut [p_f64(&one), p_u64(&d_tmp), p_u64(&d_out), p_i32(&n_i)]) {
            return false;
        }
        if !spmv_into(d_vals, d_cols, d_ptr, d_t, d_out, n) {
            return false;
        }
        if !launch(s.k_copy, n, &mut [p_u64(&d_t2), p_u64(&d_r_in), p_i32(&n_i)]) {
            return false;
        }
        if !launch(s.k_axpy, n, &mut [p_f64(&m1), p_u64(&d_t), p_u64(&d_t2), p_i32(&n_i)]) {
            return false;
        }
        if !launch(
            s.k_mul,
            n,
            &mut [p_u64(&d_inv_diag), p_u64(&d_t2), p_u64(&d_t), p_i32(&n_i)],
        ) {
            return false;
        }
        return launch(s.k_axpy, n, &mut [p_f64(&om), p_u64(&d_t), p_u64(&d_out), p_i32(&n_i)]);
    }

    if !coarse_step(amg, t2) {
        return false;
    }
    // denom = (r1·A y1) via t; numer = (r1·r1).
    if !spmv_into(d_vals, d_cols, d_ptr, d_t, d_tmp, n) {
        return false;
    }
    let (num1, den1) = match (s.dot_s(&mut amg.partials, t2, t2, n), s.dot_s(&mut amg.partials, t2, t, n)) {
        (Some(a), Some(b)) => (a, b),
        _ => return false,
    };
    if den1 > 1e-300 {
        let beta = num1 / den1;
        if !launch(s.k_axpy, n, &mut [p_f64(&beta), p_u64(&d_tmp), p_u64(&d_out), p_i32(&n_i)]) {
            return false;
        }
        // r2 = r1 − β·A y1 (reuse t2; t already holds A y1).
        if !launch(s.k_axpy, n, &mut [p_f64(&-beta), p_u64(&t), p_u64(&t2), p_i32(&n_i)]) {
            return false;
        }
    }
    // 5-6. Coarse correction #2 on the updated residual.
    if !coarse_step(amg, t2) {
        return false;
    }
    if !spmv_into(d_vals, d_cols, d_ptr, d_t, d_tmp, n) {
        return false;
    }
    let (num2, den2) = match (s.dot_s(&mut amg.partials, t2, t2, n), s.dot_s(&mut amg.partials, t2, t, n)) {
        (Some(a), Some(b)) => (a, b),
        _ => return false,
    };
    if den2 > 1e-300 {
        let beta = num2 / den2;
        if !launch(s.k_axpy, n, &mut [p_f64(&beta), p_u64(&d_tmp), p_u64(&d_out), p_i32(&n_i)]) {
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

/// One grid-strided kernel launch over `n` elements (the cycle's launch
/// geometry — every kernel here maps one thread per row/element).
unsafe fn launch1(
    c: &Cuda,
    f: crate::ffi::CUfunction,
    n: usize,
    params: &mut [*mut c_void],
) -> bool {
    // SAFETY: same launch contract as cg_solve (1-D grid, BLOCK threads).
    unsafe {
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
    }
}

/// Parameter-pointer helpers (device address, i32 scalar, f64 scalar).
fn p_u64(v: &u64) -> *mut c_void {
    v as *const u64 as *mut c_void
}
fn p_i32(v: &i32) -> *mut c_void {
    v as *const i32 as *mut c_void
}
fn p_f64(v: &f64) -> *mut c_void {
    v as *const f64 as *mut c_void
}

/// The sparse coarse solve: gather the residual into the factor's
/// elimination order (`r[to_orig]`), run the two level-scheduled
/// `tri_level` sweeps (forward `L y = r_p` into the factor's scratch,
/// backward `Lᵗ x = y`), and scatter the solution back (`z[to_perm]`).
/// Levels launch in dependency order on the default stream (serialized),
/// so earlier levels are complete when the next begins — no host round
/// trip per cycle.
fn coarse_sparse(
    s: &GpuSolver,
    cf: &crate::precond::DeviceCoarse,
    d_r: u64,
    d_out: u64,
    n: usize,
) -> bool {
    let c = &s.cuda;
    let f = &cf.base;
    let n_i = n as i32;
    // rp = r[to_orig]: into the factor's order.
    // SAFETY: same launch contract as the cycle's other kernels.
    if !unsafe { launch1(
        c,
        s.k_gather,
        n,
        &mut [p_u64(&cf.d_rp), p_u64(&d_r), p_u64(&cf.d_to_orig), p_i32(&n_i)],
    ) } {
        return false;
    }
    let sweep = |tri: crate::ffi::CUfunction,
                 offs: &[i32],
                 rows: u64,
                 vals: u64,
                 cols: u64,
                 ptr: u64,
                 d_in: u64,
                 d_y: u64|
     -> bool {
        for l in 0..offs.len() - 1 {
            let count = (offs[l + 1] - offs[l]) as usize;
            if count == 0 {
                continue;
            }
            let rows_at = rows + (offs[l] as u64) * 4;
            let cnt = count as i32;
            // SAFETY: same launch contract as cg_solve's IC(0) sweeps.
            unsafe {
                if (c.cuLaunchKernel)(
                    tri,
                    count.div_ceil(BLOCK as usize) as u32,
                    1,
                    1,
                    BLOCK,
                    1,
                    1,
                    0,
                    std::ptr::null_mut(),
                    [p_u64(&vals), p_u64(&cols), p_u64(&ptr), p_u64(&rows_at), p_i32(&cnt), p_u64(&d_in), p_u64(&d_y)]
                        .as_mut_ptr(),
                    std::ptr::null_mut(),
                ) != CUDA_SUCCESS
                {
                    return false;
                }
            }
        }
        true
    };
    let ok = sweep(
        s.k_tri,
        &f.fwd_offsets,
        f.d_fwd_rows,
        f.l_vals,
        f.l_cols,
        f.l_ptr,
        cf.d_rp,
        f.scratch,
    ) && sweep(
        s.k_tri,
        &f.bwd_offsets,
        f.d_bwd_rows,
        f.lt_vals,
        f.lt_cols,
        f.lt_ptr,
        f.scratch,
        cf.d_zp,
    );
    // z = z_p[to_perm]: back to the coarse level's own indexing.
    // SAFETY: same launch contract as the cycle's other kernels.
    ok && unsafe {
        launch1(
            c,
            s.k_gather,
            n,
            &mut [p_u64(&d_out), p_u64(&cf.d_zp), p_u64(&cf.d_to_perm), p_i32(&n_i)],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path_laplacian(n: usize) -> CsrMatrix {
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
        CsrMatrix { vals, cols, row_ptr }
    }

    #[test]
    fn pairwise_aggregation_shrinks_and_covers() {
        // Path graph: matching pairs up vertices; every vertex aggregated.
        let n = 101;
        let a = path_laplacian(n);
        let (agg, n_coarse) = aggregate(&a, 2);
        assert!(agg.iter().all(|&g| g >= 0), "every vertex aggregated");
        assert!(n_coarse <= (n + 1) / 2 + 1, "pairs: {n_coarse} aggregates for {n} vertices");
        assert!(n_coarse < n);
    }

    #[test]
    fn smoothed_prolongator_matches_dense_reference() {
        // P = (I − ω·D⁻¹A)·B computed densely vs the CSR constructor.
        let a = path_laplacian(12);
        let n = a.rows();
        let (agg, n_coarse) = aggregate(&a, SA_AGG);
        let p = prolongator(&a, &agg, true);
        assert_eq!(p.rows(), n);
        for v in 0..n {
            let d = a.row_get(v, v as i32);
            for g in 0..n_coarse {
                let mut want = 0.0f64;
                if agg[v] as usize == g {
                    want += 1.0;
                }
                for (u, val) in a.row(v) {
                    if agg[u as usize] as usize == g {
                        want -= OMEGA * val / d;
                    }
                }
                let got = p.row_get(v, g as i32);
                assert!(
                    (got - want).abs() < 1e-12,
                    "P({v},{g}) = {got}, want {want}"
                );
            }
        }
        // The interior of the chain reaches into both neighboring
        // aggregates (the tentative B never does).
        let interior = p.row(6).count();
        assert!(interior >= 2, "smoothed row must widen: {interior} entries");
    }

    #[test]
    fn galerkin_matches_dense_triple_product() {
        // galerkin_p vs a dense Pᵗ·A·P on a smoothed transfer.
        let a = path_laplacian(40);
        let n = a.rows();
        let (agg, n_coarse) = aggregate(&a, SA_AGG);
        let p = prolongator(&a, &agg, true);
        let ac = galerkin_p(&a, &p, n_coarse);
        assert_eq!(ac.rows(), n_coarse);
        for gi in 0..n_coarse {
            for gj in 0..n_coarse {
                let mut want = 0.0f64;
                for i in 0..n {
                    for (j, a_ij) in a.row(i) {
                        want += p.row_get(i, gi as i32) * a_ij * p.row_get(j as usize, gj as i32);
                    }
                }
                let got = ac.row_get(gi, gj as i32);
                assert!(
                    (got - want).abs() < 1e-9 * (1.0 + want.abs()),
                    "Ac({gi},{gj}) = {got}, want {want}"
                );
            }
        }
        // Symmetric: A_c(i,j) == A_c(j,i).
        for i in 0..n_coarse {
            for (cj, v) in ac.row(i) {
                let j = cj as usize;
                assert!((ac.row_get(j, i as i32) - v).abs() < 1e-10, "({i},{j}) asymmetric");
            }
        }
    }

    #[test]
    fn boolean_galerkin_keeps_the_quadratic_form() {
        // For the tentative transfer, P·1_c = 1_f, so the total entry sum
        // is invariant across levels — no mass is created or lost.
        let a = path_laplacian(40);
        let (agg, n_coarse) = aggregate(&a, 2);
        let p = prolongator(&a, &agg, false);
        let ac = galerkin_p(&a, &p, n_coarse);
        let tf: f64 = a.vals.iter().sum();
        let tc: f64 = ac.vals.iter().sum();
        assert!((tf - tc).abs() < 1e-9, "total {tc} vs {tf}");
    }

    #[test]
    fn smoothed_hierarchy_stays_spd_and_bottoms_out() {
        // A 64×64 Poisson grid under smoothed aggregation: every level is
        // symmetric with a positive diagonal, and the hierarchy terminates
        // on an exactly-factorable coarsest level.
        let side = 64usize;
        let at = |r: usize, c: usize| r * side + c;
        let mut vals = Vec::new();
        let mut cols = Vec::new();
        let mut row_ptr = vec![0i32];
        for r in 0..side {
            for c in 0..side {
                let i = at(r, c);
                for (rr, cc) in [(r.wrapping_sub(1), c), (r, c.wrapping_sub(1))] {
                    if rr < side && cc < side && (rr != r || cc != c) {
                        vals.push(-1.0);
                        cols.push(at(rr, cc) as i32);
                    }
                }
                vals.push(4.0);
                cols.push(i as i32);
                for (rr, cc) in [(r + 1, c), (r, c + 1)] {
                    if rr < side && cc < side {
                        vals.push(-1.0);
                        cols.push(at(rr, cc) as i32);
                    }
                }
                row_ptr.push(vals.len() as i32);
            }
        }
        let a = CsrMatrix { vals, cols, row_ptr };
        let host = hierarchy(&a, true).expect("smoothed hierarchy builds");
        assert!(host.len() >= 3, "64² grid needs several levels: {}", host.len());
        assert!(host.len() <= 40, "depth guard");
        let last = host.last().unwrap();
        assert!(last.p.is_none(), "the last level carries no transfer");
        assert!(last.factor.is_some(), "the last level is exactly factored");
        assert!(last.op.rows() <= COARSEST);
        for (lvl, h) in host.iter().enumerate() {
            let op = &h.op;
            for i in 0..op.rows() {
                assert!(op.row_get(i, i as i32) > 0.0, "level {lvl} diagonal at {i}");
                for (cj, v) in op.row(i) {
                    let j = cj as usize;
                    if j != i {
                        assert!(
                            (op.row_get(j, i as i32) - v).abs() < 1e-8 * (1.0 + v.abs()),
                            "level {lvl} ({i},{j}) asymmetric"
                        );
                    }
                }
            }
        }
    }
}
