//! LSCM unfold — `CTaskUnfold` / `CIsomap`, mirroring the shared LS solver driver.
//!
//! Evidence (ALGORITHMS.md §3, constants from the decompile):
//! - Driver loop VA `0x14034CBB0`: re-assemble the system each iteration, direct solve
//!   (SuperLU `dgssv`, VA `0x1404E9080`), step-damped update with adaptive
//!   transient-release (a jumpy iteration halves the step, a stable one relaxes
//!   back toward 1.0), `maxIter` = 100 (literal in the body), divergence guard
//!   `maxIter * 10 < it`.
//! - `CIsomap::ComputeLs` (VA `0x140469240`): per-edge conformal weight = sum of
//!   the opposite-angle cotangents, clamped ≥ 0, normalized by the squared 3-D
//!   edge length. Degeneracy guard `EPS_DEGENERATE` = `1.1754943508222875e-38`
//!   (constant `0x140F8CCC8`).
//! - `CIsomap::ComputeArea` (VA `0x1404685A0`): `0.5 * Σ signed cross` (half factor
//!   `0.5` @ constant `0x140F84DC8`); `ComputeAreaGrad` (VA `0x140468920`) with the
//!   orientation sign flip.
//! - Border pass (VA `0x14046A090`, string `"Density"`): pin targets on
//!   perimeter/area-matched rectangles, density ∝ 3-D arc length.
//! - Blend: `A = #AngleDistanceMix` (default 1.0), `B = 1 - A` (`0x140F7BB78`).
//!
//! The default driver path pins the auxiliary slack variable (`M[nv][nv] = mixW = 1.0`,
//! `#KeepMetric = false`), so the conformal block is symmetric positive definite and
//! is solved by conjugate gradients — the same direct solve the driver performs,
//! expressed iteratively and dependency-free. With `#KeepMetric = true` the
//! area row is assembled (non-symmetric) and solved with the direct sparse LU.

use crate::lscm_eig::eig3;
use crate::segment::Chart;
use crate::sparse::Sparse;
use researchuv_core::model::SurfaceMesh;
use researchuv_math::Vec2;
use researchuv_math::Vec3;
use std::collections::BTreeMap;

// ---- constants taken verbatim from the decompile (ALGORITHMS.md §3.2 / §3.6) ----
/// `0x140F8CCC8` — ComputeLs degeneracy epsilon.
pub const EPS_DEGENERATE: f64 = 1.1754943508222875e-38;
/// `0x140F84DC8` — ComputeArea / ComputeAreaGrad half factor.
pub const AREA_HALF: f64 = 0.5;
/// `0x140F84D78` — target * 0.01 convergence threshold.
pub const CONV_THRESHOLD: f64 = 0.01;
/// `0x140F84DF8` — per-iteration error transient release.
pub const TRANS_THRESHOLD: f64 = 0.99;
/// Damped ratio ≤ this ⇒ stagnation (driver loop stop).
pub const STAGNATION: f64 = 0.01;
/// Literal `maxIter` in the `0x14034CBB0` body.
pub const DEFAULT_MAX_ITER: usize = 100;
/// Divergence guard: `maxIter * 10 < it`.
pub const DEFAULT_MAX_ITER_GUARD: usize = 10;
/// `#AngleDistanceMix` default (1.0).
pub const ANGLE_MIX: f64 = 1.0;
/// `#KeepMetric` default (false ⇒ no area row on the default path).
pub const KEEP_METRIC: bool = false;
/// Mix weight init (`0x140F7BB78`).
pub const MIX_W: f64 = 1.0;

/// Linear-solver backend for the unfold driver.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolverBackend {
    /// The direct sparse LU (default — the reference `dgssv` mirror).
    Cpu,
    /// CUDA conjugate gradients on the SPD free block
    /// ([`researchuv_gpu`]), falling back to the CPU solve when no device
    /// or PTX is available, the system is too small to be worth the
    /// transfer, or the solve fails. Identical converged result.
    Gpu,
}

impl Default for SolverBackend {
    fn default() -> Self {
        SolverBackend::Cpu
    }
}

/// Unfold driver options (documented task parameters).
#[derive(Clone, Copy, Debug)]
pub struct UnfoldOptions {
    pub max_iter: usize,
    /// `#AngleDistanceMix` — A; the blend is `B = 1 - A`.
    pub angle_mix: f64,
    /// `#KeepMetric` — enable the area-preservation row.
    pub keep_metric: bool,
    /// Target area (defaults to the chart's 3-D area when 0).
    pub target_area: f64,
    /// `mixW` — the slack-row weight (default 1.0).
    pub mix_w: f64,
    /// The linear-solver backend (default CPU).
    pub solver: SolverBackend,
}

impl Default for UnfoldOptions {
    fn default() -> Self {
        Self {
            max_iter: DEFAULT_MAX_ITER,
            angle_mix: ANGLE_MIX,
            keep_metric: KEEP_METRIC,
            target_area: 0.0,
            mix_w: MIX_W,
            solver: SolverBackend::default(),
        }
    }
}

/// Result of one chart unfold.
#[derive(Clone, Debug)]
pub struct UnfoldResult {
    /// 2-D coordinates in chart-local vertex order.
    pub uv: Vec<Vec2>,
    /// Iterations actually run.
    pub iters: usize,
    /// Accumulated per-iteration relative error.
    pub err_accum: f64,
    /// Effective target area used.
    pub target: f64,
    /// Chart 3-D area.
    pub area3d: f64,
}

/// `ComputeArea` — EXACT documented `CIsomap::ComputeArea` (VA `0x1404685A0`):
/// `0.5 * Σ (x1−x0)(y2−y0) − (x2−x0)(y1−y0)` over the triangles.
pub fn compute_area(uv: &[Vec2], tris: &[[u32; 3]]) -> f64 {
    let mut a = 0.0;
    for [i, j, k] in tris {
        let p0 = uv[*i as usize];
        let p1 = uv[*j as usize];
        let p2 = uv[*k as usize];
        a += (p1.u - p0.u) * (p2.v - p0.v) - (p2.u - p0.u) * (p1.v - p0.v);
    }
    a * AREA_HALF
}

/// `ComputeAreaGrad` — EXACT documented `CIsomap::ComputeAreaGrad` (VA `0x140468920`)
/// with the `0x8000…` orientation sign flip.
pub fn compute_area_grad(uv: &[Vec2], tris: &[[u32; 3]], nv: usize) -> Vec<Vec2> {
    let mut grad = vec![Vec2::new(0.0, 0.0); nv];
    for [a, b, c] in tris {
        let (x0, y0) = (uv[*a as usize].u, uv[*a as usize].v);
        let (x1, y1) = (uv[*b as usize].u, uv[*b as usize].v);
        let (x2, y2) = (uv[*c as usize].u, uv[*c as usize].v);
        let signed = AREA_HALF * (x0 * (y2 - y1) + x1 * (y0 - y2) + x2 * (y1 - y0));
        let sgn = if signed > 0.0 { 1.0 } else { -1.0 };
        grad[*a as usize].u += (y2 - y1) * AREA_HALF * sgn;
        grad[*a as usize].v += (x1 - x2) * AREA_HALF * sgn;
        grad[*b as usize].u += (y0 - y2) * AREA_HALF * sgn;
        grad[*b as usize].v += (x2 - x0) * AREA_HALF * sgn;
        grad[*c as usize].u += (y1 - y0) * AREA_HALF * sgn;
        grad[*c as usize].v += (x0 - x1) * AREA_HALF * sgn;
    }
    grad
}

/// PCA tangent-plane projection (two dominant covariance directions).
pub fn tangent_frame(p3: &[Vec3]) -> Vec<Vec2> {
    let n = p3.len();
    if n == 0 {
        return Vec::new();
    }
    let mut c = Vec3::new(0.0, 0.0, 0.0);
    for p in p3 {
        c = c + *p;
    }
    c = c / n as f64;
    let d: Vec<Vec3> = p3.iter().map(|p| *p - c).collect();
    let mut cm = [[0.0f64; 3]; 3];
    for v in &d {
        let vi = [v.x, v.y, v.z];
        for i in 0..3 {
            for j in 0..3 {
                cm[i][j] += vi[i] * vi[j];
            }
        }
    }
    let (w, v) = eig3(cm);
    // Sort eigenvalues descending; take the two largest directions.
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&a, &b| w[b].total_cmp(&w[a]));
    let (e1, e2) = (
        Vec3::new(v[0][idx[0]], v[1][idx[0]], v[2][idx[0]]),
        Vec3::new(v[0][idx[1]], v[1][idx[1]], v[2][idx[1]]),
    );
    d.iter()
        .map(|p| Vec2::new(p.dot(e1), p.dot(e2)))
        .collect()
}

/// Point on the perimeter of a centered `a × b` rectangle, `f ∈ [0,1)` from the
/// bottom-right corner walking clockwise (reference `_rect_point`).
pub fn rect_point(f: f64, a: f64, b: f64) -> Vec2 {
    let p2 = 2.0 * (a + b);
    let mut t = (f % 1.0) * p2;
    let (x0, y0) = (a / 2.0, -b / 2.0);
    if t < a {
        return Vec2::new(x0 - t, y0);
    }
    t -= a;
    if t < b {
        return Vec2::new(-a / 2.0, y0 + t);
    }
    t -= b;
    if t < a {
        return Vec2::new(-a / 2.0 + t, b / 2.0);
    }
    t -= a;
    Vec2::new(a / 2.0, b / 2.0 - t)
}

/// Greedy farthest-point anchors in 3-D (reference `_anchor_ids`): a
/// deterministic, well-separated set of `n` vertices used to pin a borderless
/// (closed) chart. Pinned to their tangent-frame positions these constrain the
/// affine null space.
pub fn anchor_ids(p3: &[Vec3], n: usize) -> Vec<usize> {
    let n3 = p3.len();
    if n3 <= n {
        return (0..n3).collect();
    }
    // Start at the min-x vertex (np.argmin convention: first occurrence).
    let min_x = p3.iter().map(|q| q.x).fold(f64::MAX, |m, x| m.min(x));
    let mut chosen: Vec<usize> = vec![p3.iter().position(|p| p.x == min_x).unwrap_or(0)];
    while chosen.len() < n {
        let mut d: Vec<f64> = (0..n3)
            .map(|i| {
                chosen
                    .iter()
                    .map(|&c| (p3[i] - p3[c]).len())
                    .fold(f64::MAX, |m, x| m.min(x))
            })
            .collect();
        for &c in &chosen {
            d[c] = -1.0;
        }
        // np.argmax convention: first occurrence of the maximum.
        let best = d
            .iter()
            .enumerate()
            .fold((0usize, f64::MIN), |(bi, bd), (i, x)| {
                if *x > bd {
                    (i, *x)
                } else {
                    (bi, bd)
                }
            })
            .0;
        chosen.push(best);
    }
    let mut s = chosen;
    s.sort_unstable();
    s
}

/// Per-undirected-edge LSCM conformal weight (reference `ew` assembly):
/// sum of opposite-angle cotangents over adjacent triangles, clamped ≥ 0,
/// with the `EPS_DEGENERATE` guard skipping (nearly) degenerate triangles.
pub fn edge_weights(p3: &[Vec3], tris: &[[u32; 3]]) -> BTreeMap<(usize, usize), f64> {
    let mut ew: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    for [i, j, k] in tris {
        let (i, j, k) = (*i as usize, *j as usize, *k as usize);
        for (e0, e1, o) in [(i, j, k), (j, k, i), (k, i, j)] {
            let a1 = p3[e1] - p3[o];
            let a2 = p3[e0] - p3[o];
            let la = a1.len();
            let lb = a2.len();
            if la <= EPS_DEGENERATE || lb <= EPS_DEGENERATE {
                continue;
            }
            let s = a1.cross_v(a2).len() / (la * lb);
            if s <= EPS_DEGENERATE {
                continue;
            }
            let cot = ((a1.dot(a2) / (la * lb)) / s).max(0.0);
            let key = (e0.min(e1), e0.max(e1));
            *ew.entry(key).or_insert(0.0) += cot;
        }
    }
    ew
}

/// Border-pin targets (documented border pass, VA `0x14046A090`, "Density"):
/// rectangle placement with density ∝ 3-D arc length. `loops` are the chart's
/// closed border loops (source vertex ids, first repeated at the end when closed).
/// Returns source-vertex → target-UV pins.
pub fn border_targets(mesh: &SurfaceMesh, chart: &Chart, loops: &[Vec<u32>]) -> BTreeMap<u32, Vec2> {
    let area3d = chart.area3d;
    // Arc-length fractions + perimeters per loop.
    let mut fracs: Vec<Vec<f64>> = Vec::new();
    let mut perims: Vec<f64> = Vec::new();
    for l in loops {
        let segs: Vec<f64> = l
            .windows(2)
            .map(|w| (mesh.positions[w[1] as usize] - mesh.positions[w[0] as usize]).len())
            .collect();
        let total = segs.iter().sum::<f64>().max(1e-30);
        let mut fr = vec![0.0f64];
        for s in &segs {
            fr.push(fr[fr.len() - 1] + s);
        }
        // Normalize by total; the last entry equals 1.0 for a closed loop.
        for f in fr.iter_mut() {
            *f /= total;
        }
        fracs.push(fr);
        perims.push(segs.iter().sum());
    }
    let n = loops.len();

    // Cylinder (annulus) special case: two end loops with similar perimeters.
    if n == 2 && (perims[0] - perims[1]).abs() <= 0.25 * perims.iter().cloned().fold(0.0f64, |m, p| m.max(p)) {
        let w = 0.5 * (perims[0] + perims[1]);
        let h = area3d / w.max(1e-30);
        // Strip the repeated closing vertex if present.
        let strip = |l: &[u32]| -> Vec<u32> {
            if l.first() == l.last() && l.len() > 1 {
                l[..l.len() - 1].to_vec()
            } else {
                l.to_vec()
            }
        };
        let l0u = strip(&loops[0]);
        let l1u = strip(&loops[1]);
        let f0 = &fracs[0][..l0u.len().min(fracs[0].len())];
        let mut pins = BTreeMap::new();
        for (k, &v) in l0u.iter().enumerate() {
            pins.insert(v, Vec2::new(-w / 2.0 + w * f0[k], h / 2.0));
        }
        // Match the second loop's x positions by 3-D proximity.
        for (_k1, &v1) in l1u.iter().enumerate() {
            let p1 = mesh.positions[v1 as usize];
            let mut best = 0usize;
            let mut best_d = f64::INFINITY;
            for (k0, &v0) in l0u.iter().enumerate() {
                let d = (mesh.positions[v0 as usize] - p1).len();
                if d < best_d {
                    best_d = d;
                    best = k0;
                }
            }
            pins.insert(v1, Vec2::new(-w / 2.0 + w * f0[best], -h / 2.0));
        }
        return pins;
    }

    // Rectangle sides.
    let sides: Vec<(f64, f64)> = if n == 1 {
        let (p, a) = (perims[0], area3d);
        if p * p >= 16.0 * a {
            let s_ = p / 2.0;
            let d = (s_ * s_ - 4.0 * a).max(0.0).sqrt();
            vec![((s_ + d) / 2.0, (s_ - d) / 2.0)]
        } else {
            let s = a.sqrt();
            vec![(s, s)]
        }
    } else {
        // General multi-loop: uniform similarity scale with the signed (alternating)
        // target-area sum equal to the chart 3-D area.
        let mut c = Vec3::new(0.0, 0.0, 0.0);
        let nv = chart.vertex_ids.len();
        for &v in &chart.vertex_ids {
            c = c + mesh.positions[v as usize];
        }
        c = c / nv.max(1) as f64;
        let dists: Vec<f64> = loops
            .iter()
            .map(|l| {
                let mut s = 0.0;
                for &v in l {
                    s += (mesh.positions[v as usize] - c).len();
                }
                s / l.len().max(1) as f64
            })
            .collect();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| dists[a].total_cmp(&dists[b]));
        let mut denom = 0.0f64;
        for (k, &li) in order.iter().enumerate() {
            let sign = if (n - 1 - k) % 2 == 0 { 1.0 } else { -1.0 };
            denom += sign * perims[li] * perims[li];
        }
        let s = 4.0 * (area3d / denom.abs().max(1e-30)).sqrt();
        let mut sides = vec![(0.0f64, 0.0f64); n];
        for &li in &order {
            let side = s * perims[li] / 4.0;
            sides[li] = (side, side);
        }
        sides
    };

    let mut pins = BTreeMap::new();
    for (li, l) in loops.iter().enumerate() {
        let (a, b) = sides[li];
        let fr = fracs[li].clone();
        // The border walk closes on itself (first == last): drop the repeat.
        let closed = l.first() == l.last() && l.len() > 1;
        let verts: &[u32] = if closed { &l[..l.len() - 1] } else { l.as_slice() };
        let frv: &[f64] = if closed { &fr[..l.len() - 1] } else { fr.as_slice() };
        // Relative winding: the `rect_point` perimeter walk is clockwise (negative
        // signed area). If the 3-D walk (measured in the tangent frame) winds the
        // other way, mirror the parameterization f -> (1 - f) % 1 so the 3-D→2-D
        // correspondence stays orientation-preserving.
        let p2loop = tangent_frame(
            &l.iter().map(|&v| mesh.positions[v as usize]).collect::<Vec<Vec3>>(),
        );
        let mut sa3 = 0.0f64;
        for i in 0..p2loop.len() {
            let j = (i + 1) % p2loop.len();
            sa3 += p2loop[i].u * p2loop[j].v - p2loop[i].v * p2loop[j].u;
        }
        sa3 *= 0.5;
        for (k, &v) in verts.iter().enumerate() {
            let f = if sa3 > 0.0 { (1.0 - frv[k]) % 1.0 } else { frv[k] };
            pins.insert(v, rect_point(f, a, b));
        }
    }
    pins
}

/// Conjugate-gradients solve of a symmetric positive-definite system.
/// Returns `None` on non-convergence within `max_iter` iterations.
pub fn cg_solve(matrix: &Sparse, rhs: &[f64], max_iter: usize) -> Option<Vec<f64>> {
    let n = matrix.n;
    let mut x = vec![0.0f64; n];
    let mut r: Vec<f64> = rhs.to_vec();
    let mut p = r.clone();
    let mut rz = r.iter().map(|v| v * v).sum::<f64>();
    let tol = (rhs.iter().map(|v| v * v).sum::<f64>()).sqrt() * 1e-12 + 1e-30;
    let tol2 = tol * tol;
    for _ in 0..max_iter {
        if rz < tol2 {
            break;
        }
        let ap: Vec<f64> = (0..n)
            .map(|i| {
                let mut s = 0.0;
                // matrix is stored row-major: sum over this row's nonzeros.
                for (j, &v) in matrix.row(i).iter() {
                    s += v * p[*j];
                }
                s
            })
            .collect();
        let p_ap = p.iter().zip(ap.iter()).map(|(a, b)| a * b).sum::<f64>();
        if p_ap.abs() < 1e-300 {
            return None;
        }
        let alpha = rz / p_ap;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        let rz_new = r.iter().map(|v| v * v).sum::<f64>();
        if rz_new < tol2 {
            break;
        }
        p = r.iter().zip(p.iter()).map(|(&r_, &p_)| r_ + (rz_new / rz) * p_).collect();
        rz = rz_new;
    }
    if !x.iter().all(|v| v.is_finite()) {
        return None;
    }
    Some(x)
}

/// Below this free-variable count the CPU LU wins (upload + JIT overhead).
const GPU_MIN_VARS: usize = 1024;

/// Solve the assembled system on the GPU: the SPD free block via CUDA
/// conjugate gradients (Jacobi-preconditioned — the engine's cotan-weight
/// diagonals vary enough for it to pay, and unlike IC(0)'s level-scheduled
/// triangular solves it has no launch-count risk on elongated charts), the
/// decoupled slack variable analytically. `None` when the GPU path is not
/// applicable (too small, `KeepMetric`'s non-symmetric rows, or no device) —
/// the caller falls back to the direct LU.
fn gpu_solve(
    matrix: &Sparse,
    rhs: &[f64],
    nfree: usize,
    mix_w: f64,
    max_iter: usize,
) -> Option<Vec<f64>> {
    let nvar_free = 2 * nfree;
    if nvar_free < GPU_MIN_VARS || nvar_free + 1 != rhs.len() || mix_w <= 0.0 {
        return None;
    }
    let (vals, cols, row_ptr) = matrix.to_csr(nvar_free);
    if vals.is_empty() {
        return None;
    }
    let a = researchuv_gpu::CsrMatrix { vals, cols, row_ptr };
    let gpu = researchuv_gpu::GpuSolver::global()?;
    // The engine default is Jacobi (measured best at chart sizes); the
    // RESEARCHUV_GPU_PRECOND env var switches the backend for benchmarking
    // and for very large / near-singular systems (amg | ic0color | ic0 |
    // none | jacobi).
    let pc = match std::env::var("RESEARCHUV_GPU_PRECOND").as_deref() {
        Ok("amg") => researchuv_gpu::Precond::Amg,
        Ok("ic0color") => researchuv_gpu::Precond::Ic0Color,
        Ok("ic0") => researchuv_gpu::Precond::Ic0,
        Ok("none") => researchuv_gpu::Precond::None,
        _ => researchuv_gpu::Precond::Jacobi,
    };
    let mut x = gpu
        .cg_solve_precond(&a, &rhs[..nvar_free], max_iter.max(1024) * 16, 1e-12, pc)?
        .0;
    // Slack row: M[aux][aux] = mix_w, rhs[aux] = mix_w ⇒ slack = 1.
    x.push(1.0);
    Some(x)
}

/// The unfold driver (mirror of `FUN_14034CBB0`).
///
/// Assembles the per-iteration system (conformal rows + slack row + pin rows),
/// solves it, and applies a step-damped update with adaptive transient release.
pub fn unfold_chart(mesh: &SurfaceMesh, chart: &Chart, opts: UnfoldOptions) -> UnfoldResult {
    let p3: Vec<Vec3> = chart.vertex_ids.iter().map(|&v| mesh.positions[v as usize]).collect();
    let nv = p3.len();
    let p2 = tangent_frame(&p3);
    let mut uv = p2.clone(); // seed from the projection (fresh unwrap)

    // Pins: border pass (documented "Density") or anchors for closed charts.
    // A closed chart has no border; the LSCM null space is the affine family
    // (6 DOF in 2-D), so a few well-separated anchors pinned to their tangent-
    // frame positions (an affine function of the 3-D points, hence in the null
    // space) make the system non-singular. #KeepMetric defaults to false, so no
    // area row unless asked.
    let pin_source: BTreeMap<u32, Vec2> = if chart.border_loops.is_empty() {
        let anchors = anchor_ids(&p3, 4);
        anchors.iter().map(|&i| (chart.vertex_ids[i], p2[i])).collect()
    } else {
        border_targets(mesh, chart, &chart.border_loops)
    };
    // Local pin ids (ascending source-id order); dedupe (a vertex on two loops
    // keeps the target of its smallest source id — matches the reference's
    // `np.unique(pin_ids, return_index=True)` over the source-sorted array).
    let mut pin_ids: Vec<usize> = Vec::new();
    let mut pin_uv: Vec<Vec2> = Vec::new();
    for (&v, &t) in &pin_source {
        if let Some(li) = chart.local_of(v) {
            if !pin_ids.contains(&li) {
                pin_ids.push(li);
                pin_uv.push(t);
            }
        }
    }

    // Edge weights (conformal).
    let ew = edge_weights(&p3, &chart.tris);
    let mut free = vec![true; nv];
    for &pi in &pin_ids {
        free[pi] = false;
    }
    // Quick pinned-vertex target lookup (for the RHS elimination below).
    let pin_lookup: BTreeMap<usize, Vec2> =
        pin_ids.iter().zip(pin_uv.iter()).map(|(&i, &t)| (i, t)).collect();

    // Per-vertex area weight (signmask(areaWeight) column).
    let areas = crate::segment::face_areas3d(mesh);
    let mut area_w = vec![0.0f64; nv];
    for (t, &fi) in chart.tris.iter().zip(&chart.face_ids) {
        for &vv in t {
            area_w[vv as usize] += areas[fi];
        }
    }
    let wsum = area_w.iter().sum::<f64>().max(1e-30);
    for w in area_w.iter_mut() {
        *w /= wsum;
    }
    let flip = if compute_area(&uv, &chart.tris) >= 0.0 { 1.0 } else { -1.0 };

    let target = if opts.target_area > 0.0 { opts.target_area } else { chart.area3d };

    let mut err_accum = 1.0;
    let mut step = 1.0;
    let mut prev_err = f64::INFINITY;
    let mut iters_done = 0;
    let max_iter = if opts.max_iter > 0 { opts.max_iter } else { DEFAULT_MAX_ITER };

    // Free-vertex index map (Dirichlet pins are eliminated analytically): the
    // reference driver solves the full system directly (spsolve / SuperLU), and
    // the pin rows make it non-symmetric — so CG on the full matrix would not
    // converge. Eliminating the pins (moving their known values to the RHS)
    // leaves the symmetric positive-definite free block, solved here by the same
    // direct sparse LU as the original's `dgssv` (VA `0x1404E9080`).
    let mut free_pos = vec![usize::MAX; nv];
    let mut free_ids: Vec<usize> = Vec::with_capacity(nv);
    for i in 0..nv {
        if free[i] {
            free_pos[i] = free_ids.len();
            free_ids.push(i);
        }
    }
    let nfree = free_ids.len();
    let nvar = 2 * nfree + 1; // free coords + slack variable
    let aux = 2 * nfree;

    for it in 0..max_iter {
        iters_done = it + 1;
        let mix_a = opts.angle_mix;
        let mix_b = 1.0 - mix_a; // documented blend: B = 1.0 - A (0x140F7BB78)
        let mut row: Vec<usize> = Vec::new();
        let mut col: Vec<usize> = Vec::new();
        let mut val: Vec<f64> = Vec::new();
        let mut rhs = vec![0.0f64; nvar];
        // Conformal rows (free vertices only):
        //   sum_j c_ij (u_a - u_j) = B * sum_j c_ij t_ij,  c_ij = w_ij/|p_i-p_j|^2.
        // Pinned neighbours contribute their known target to the RHS.
        for (&(i, j), &w) in &ew {
            if w <= EPS_DEGENERATE {
                continue;
            }
            let d3 = p3[j] - p3[i];
            let c = w / d3.dot(d3).max(EPS_DEGENERATE);
            let d = p2[j] - p2[i];
            for (aa, bb, dv) in [(i, j, d), (j, i, -d)] {
                if !free[aa] {
                    continue;
                }
                let ra = 2 * free_pos[aa];
                // Mirror the reference's `row += [2a, 2a+1, 2a, 2a+1]`,
                // `col += [2a, 2a+1, 2b, 2b+1]`: the x-row couples to bb's x and the
                // y-row couples to bb's y. Pinned neighbours (not in the free block)
                // are marked with the usize::MAX sentinel and dropped at assembly.
                let (bbx, bby) = if free[bb] {
                    let col = 2 * free_pos[bb];
                    (col, col + 1)
                } else {
                    (usize::MAX, usize::MAX)
                };
                row.extend([ra, ra + 1, ra, ra + 1]);
                col.extend([ra, ra + 1, bbx, bby]);
                val.extend([c, c, -c, -c]);
                if free[bb] {
                    rhs[ra] += mix_b * c * dv.u;
                    rhs[ra + 1] += mix_b * c * dv.v;
                } else {
                    // -c * u_bb = rhs  ->  rhs += c * u_bb (pinned value).
                    let pin = pin_lookup[&bb];
                    rhs[ra] += mix_b * c * dv.u + c * pin.u;
                    rhs[ra + 1] += mix_b * c * dv.v + c * pin.v;
                }
            }
        }
        // Slack row: area row (#KeepMetric) or the default mixW pin.
        if opts.keep_metric {
            let area_now = compute_area(&uv, &chart.tris);
            let g = compute_area_grad(&uv, &chart.tris, nv);
            for i in 0..nv {
                if free[i] {
                    let ri = 2 * free_pos[i];
                    row.extend([ri, ri + 1]);
                    col.extend([aux, aux]);
                    val.extend([flip * area_w[i], flip * area_w[i]]);
                    row.extend([aux, aux]);
                    col.extend([ri, ri + 1]);
                    val.extend([g[i].u, g[i].v]);
                }
            }
            row.push(aux);
            col.push(aux);
            val.push(area_now - target);
            rhs[aux] = 2.0 * target;
        } else {
            row.push(aux);
            col.push(aux);
            val.push(opts.mix_w);
            rhs[aux] = opts.mix_w;
        }

        // Solve the (symmetric on the free block) free-variable system with the
        // direct sparse LU — the faithful mirror of the original's SuperLU solve —
        // or the CUDA conjugate-gradient backend on the SPD free block (the
        // KeepMetric area rows are non-symmetric: CPU only).
        let mut m = Sparse::new(nvar);
        for (rr, (&r, &c)) in row.iter().zip(col.iter()).enumerate() {
            if c != usize::MAX {
                m.add(r, c, val[rr]);
            }
        }
        let sol = if opts.solver == SolverBackend::Gpu && !opts.keep_metric {
            gpu_solve(&m, &rhs, nfree, opts.mix_w, max_iter).or_else(|| m.solve(&rhs))
        } else {
            m.solve(&rhs)
        };
        let sol = match sol {
            Some(s) => s,
            None => break,
        };
        let mut new_uv = uv.clone();
        for &i in &free_ids {
            let pos = free_pos[i];
            new_uv[i] = Vec2::new(sol[2 * pos], sol[2 * pos + 1]);
        }
        // Dirichlet: pinned vertices sit at their targets (the reference's full
        // system returns the pin value in the pin rows, so `new_uv` jumps them to
        // the target on the first iteration).
        for (&i, &t) in pin_ids.iter().zip(pin_uv.iter()) {
            new_uv[i] = t;
        }
        let dv: Vec<Vec2> = (0..nv).map(|i| new_uv[i] - uv[i]).collect();
        let norm_dv: f64 = dv.iter().map(|d| d.u * d.u + d.v * d.v).sum::<f64>().sqrt();
        let norm_uv: f64 = uv.iter().map(|u| u.u * u.u + u.v * u.v).sum::<f64>().sqrt();
        let err = norm_dv / norm_uv.max(1e-30);
        let applied = err * step;
        // Adaptive damping (transient release).
        if prev_err > 1e-30 && err > 2.0 * prev_err {
            step = (step * 0.5).max(0.05);
        } else if it > 3 {
            step = (step * 1.2).min(1.0);
        }
        if applied < 1e-8 || (prev_err > 1e-30 && it != 0 && applied / prev_err <= STAGNATION) {
            break; // converged / stagnation
        }
        for i in 0..nv {
            uv[i] = uv[i] + dv[i] * step;
        }
        prev_err = applied;
        err_accum += err;
        if max_iter * DEFAULT_MAX_ITER_GUARD < it {
            break; // divergence guard (maxIter*10)
        }
    }
    UnfoldResult {
        uv,
        iters: iters_done,
        err_accum,
        target,
        area3d: chart.area3d,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_backend_matches_the_cpu_solve() {
        // Skipped automatically on machines without CUDA.
        if researchuv_gpu::GpuSolver::new().is_none() {
            eprintln!("skipping: no CUDA device/PTX available");
            return;
        }
        let (p, f) = crate::meshgen::grid(32, 32);
        let mesh = crate::weld::weld(p, f, 1e-12);
        let (charts, _) = crate::segment::segment(&mesh, 30.0);
        let ch = &charts[0];
        let cpu = unfold_chart(&mesh, ch, UnfoldOptions::default());
        let mut gpu_opts = UnfoldOptions::default();
        gpu_opts.solver = SolverBackend::Gpu;
        let gpu = unfold_chart(&mesh, ch, gpu_opts);
        assert_eq!(cpu.uv.len(), gpu.uv.len());
        let max_diff = cpu
            .uv
            .iter()
            .zip(gpu.uv.iter())
            .map(|(a, b)| (*a - *b).len())
            .fold(0.0f64, f64::max);
        let scale = cpu.uv.iter().map(|p| p.len()).fold(0.0f64, f64::max).max(1e-30);
        assert!(
            max_diff / scale < 1e-6,
            "GPU/CPU solutions diverge: {max_diff} over scale {scale}"
        );
    }
}
