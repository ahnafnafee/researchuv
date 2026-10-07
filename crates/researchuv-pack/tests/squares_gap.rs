//! Optimality-gap benchmark for the equal-square packing subtype.
//!
//! Protocol: `n` unit-square islands are packed into the unit target at a
//! *fixed uniform scale* `σ = 1/s` (margin 0, touching allowed — the exact
//! regime of the mathematical problem), and the layout is accepted only when
//! the packer placed every island with a clean polygon-level validation.
//! Binary search over the container side `s` finds the smallest side the
//! packer can realize, which is then compared against the certified bounds
//! for `s(n)` from [`researchuv_pack::sqbounds`]:
//!
//! - **Floor tripwire** — the achieved side must never fall below the
//!   certified lower bound. If it does, the packer produced a layout the
//!   mathematics rules out, which means an overlap slipped through
//!   validation (or the bound data is wrong). Either way, failing this test
//!   is a soundness alarm, not a benchmark regression.
//! - **Grid floor** — the achieved side must not exceed the trivial `⌈√n⌉`
//!   grid; a packer that cannot even reach the grid construction is broken.
//! - **Tilt experiment** — for the counts whose proven optima need tilted
//!   squares (5, 10, 11), the run is repeated with
//! [`researchuv_pack::PackParams::tilt_candidates`] enabled. The CPU exact
//! planner collides on axis-aligned bounding boxes, so tilted gains are
//! expected only on the polygon-footprint rasterizer path; the table this
//! test prints (run with `--nocapture`) records what actually happens.

use researchuv_math::Vec2;
use researchuv_pack::cert::PackingCertificate;
use researchuv_pack::sqbounds;
use researchuv_pack::{pack, Island, PackParams, ScaleMode};

fn unit_squares(n: u32) -> Vec<Island> {
    (0..n)
        .map(|i| {
            let mut isl = Island::from_polygon(vec![
                Vec2::new(0.0, 0.0),
                Vec2::new(1.0, 0.0),
                Vec2::new(1.0, 1.0),
                Vec2::new(0.0, 1.0),
            ]);
            isl.verts[0].u += i as f64 * 1e-9; // keep outlines distinct
            isl
        })
        .collect()
}

fn params_at_side(s: f64, tilt: bool, raster_resolution: u32) -> PackParams {
    PackParams {
        scale_mode: ScaleMode::FixedScale,
        scale: 1.0 / s,
        margin: 0.0,
        seed: 0,
        tilt_candidates: tilt,
        raster_resolution,
        ..PackParams::default()
    }
}

/// Run one pack at container side `s`; `Some(achieved_box_side)` when every
/// island placed with a clean layout. Disjointness is checked at the
/// *polygon* level with a small tolerance (touching allowed — the regime of
/// the mathematical problem). The packer's own `validation.overlapping`
/// flags touching islands by design (`lock_overlapping` semantics), so it
/// is not used here; a tolerance-based separating-axis test on the convex
/// outlines is exact for the (rotated) square islands of this benchmark.
/// The achieved side is measured from the placed boxes (the rasterizer
/// quantizes anchors to the grid, so the box side can slightly exceed `s`).
fn layout_at(
    islands: &[Island],
    params: &PackParams,
) -> Option<(f64, researchuv_pack::PackResult)> {
    let mut run = islands.to_vec();
    let r = pack(&mut run, params);
    let ok = r.non_packed.is_empty()
        && r.placed.iter().all(|p| p.is_some())
        && r.validation.outside.is_empty()
        && r.validation.self_intersecting.is_empty();
    if !ok {
        return None;
    }
    let outlines: Vec<Vec<Vec2>> = r
        .placed
        .iter()
        .enumerate()
        .map(|(i, t)| t.as_ref().unwrap().transform_poly(&islands[i].verts))
        .collect();
    for i in 0..outlines.len() {
        for j in (i + 1)..outlines.len() {
            if !convex_disjoint_tol(&outlines[i], &outlines[j], 1e-9) {
                return None;
            }
        }
    }
    // Unit-square islands at uniform scale σ occupy σ×σ boxes in the unit
    // target, so the equivalent container side is 1/σ. The largest placed
    // box is the effective scale.
    let scale_eff = r
        .placed
        .iter()
        .flatten()
        .map(|t| t.box_.max_extent())
        .fold(f64::MIN, f64::max);
    Some((1.0 / scale_eff, r))
}

/// Separating-axis disjointness for convex outlines with tolerance `tol`:
/// separated when some edge-normal axis leaves a gap of at least `−tol`
/// (i.e. penetration below `tol` counts as touching, not overlap).
fn convex_disjoint_tol(a: &[Vec2], b: &[Vec2], tol: f64) -> bool {
    for poly in [a, b] {
        for k in 0..poly.len() {
            let p = poly[k];
            let q = poly[(k + 1) % poly.len()];
            // Edge normal (not normalized; projections scale with it).
            let n = (q.v - p.v, -(q.u - p.u));
            let mut amin = f64::INFINITY;
            let mut amax = f64::NEG_INFINITY;
            for v in a.iter() {
                let d = n.0 * v.u + n.1 * v.v;
                amin = amin.min(d);
                amax = amax.max(d);
            }
            let mut bmin = f64::INFINITY;
            let mut bmax = f64::NEG_INFINITY;
            for v in b.iter() {
                let d = n.0 * v.u + n.1 * v.v;
                bmin = bmin.min(d);
                bmax = bmax.max(d);
            }
            if bmin - amax >= -tol || amin - bmax >= -tol {
                return true;
            }
        }
    }
    false
}

/// Smallest container side the packer can realize for `n` unit squares
/// (binary search on the side; feasibility is monotone in `s`).
fn achieved_side(n: u32, tilt: bool, raster_resolution: u32, iters: u32) -> f64 {
    let islands = unit_squares(n);
    let grid = (n as f64).sqrt().ceil(); // always feasible by construction
    let mut lo = 0.5f64; // far below any s(n) here — infeasible
    let mut hi = grid + 1.0; // comfortably feasible
    for _ in 0..iters {
        let mid = 0.5 * (lo + hi);
        if layout_at(&islands, &params_at_side(mid, tilt, raster_resolution)).is_some() {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    hi
}

fn check_against_bounds(n: u32, side: f64, label: &str, grid_slack: f64) {
    let b = sqbounds::bound(n);
    // Floor tripwire: never below the certified lower bound.
    assert!(
        side >= b.lower - 1e-6,
        "{label} n={n}: achieved side {side:.6} is below the certified floor {:.9} ({}) \
         — an overlap must have escaped validation",
        b.lower,
        b.source
    );
    // Grid floor: never worse than the trivial grid construction (plus the
    // given slack — the rasterizer's cell quantization inflates it).
    let grid = (n as f64).sqrt().ceil();
    assert!(
        side <= grid + grid_slack,
        "{label} n={n}: achieved side {side:.6} is worse than the trivial grid {grid} (+{grid_slack})"
    );
    println!(
        "{label} n={n:2}: achieved {side:9.6} | best known {upper:9.6} | certified floor {lower:9.6} \
         | gap to floor {pct:6.2}% | {src}",
        upper = b.upper,
        lower = b.lower,
        pct = (side / b.lower - 1.0) * 100.0,
        src = b.source,
    );
}

#[test]
fn exact_planner_respects_certified_floors() {
    for n in [2u32, 5, 10, 11, 12, 13] {
        let side = achieved_side(n, false, 0, 28);
        check_against_bounds(n, side, "cpu90", 1e-6);
    }
}

#[test]
fn tilt_candidates_do_not_break_soundness() {
    // The counts whose proven optima are tilted packings. The exact planner
    // collides on bounding boxes, so tilts cannot nest diamonds; they must
    // at least never make the result worse than a small search-perturbation
    // slack.
    for n in [5u32, 10, 11] {
        let s90 = achieved_side(n, false, 0, 28);
        let stilt = achieved_side(n, true, 0, 28);
        check_against_bounds(n, stilt, "tilt ", 1e-6);
        assert!(
            stilt <= s90 + 0.02,
            "n={n}: tilt candidates worsened the side: {stilt:.6} vs {s90:.6}"
        );
        println!(
            "             n={n:2}: tilt delta {d:+.6} ({p:+.2}%)",
            d = stilt - s90,
            p = (stilt / s90 - 1.0) * 100.0
        );
    }
}

#[test]
fn raster_path_respects_certified_floors() {
    // The rasterizer collides on polygon footprints (the occupancy grid),
    // so this is the path where tilted packings could in principle beat the
    // axis-aligned grid. Requires a CUDA device; skipped otherwise.
    // The grid-floor check uses the *quantized* grid bound: at resolution R
    // a k-per-row grid needs k boxes of ⌈σR⌉ cells plus the occupancy
    // grid's one-texel dilation floor between neighbors, so the achievable
    // side is R/⌊(R−k+1)/k⌋, slightly above the continuous ⌈√n⌉.
    if researchuv_gpu::GpuSolver::global().is_none() {
        eprintln!("skipping: no CUDA device/PTX available");
        return;
    }
    const R: f64 = 256.0;
    for n in [5u32, 10, 11] {
        let k = (n as f64).sqrt().ceil();
        let cells_per_box = ((R - k + 1.0) / k).floor();
        let quantized_grid = R / cells_per_box;
        let slack = (quantized_grid - k).max(0.0) + 1e-6;
        let side = achieved_side(n, false, 256, 24);
        check_against_bounds(n, side, "raster", slack);
        let stilt = achieved_side(n, true, 256, 24);
        check_against_bounds(n, stilt, "rtilt", slack);
        println!(
            "             n={n:2}: raster tilt delta {d:+.6} ({p:+.2}%)",
            d = stilt - side,
            p = (stilt / side - 1.0) * 100.0
        );
    }
}

#[test]
fn achieved_layouts_carry_replayable_certificates() {
    // The n=11 layout: capture, replay, and reject a mutant. This pins the
    // benchmark protocol to independently checkable artifacts (the squares
    // project's certificate discipline). A small margin keeps the replay
    // strict: the packer's validation flags touching islands by design, so
    // a marginless layout would not replay cleanly.
    let n = 11u32;
    let islands = unit_squares(n);
    let mut params = PackParams {
        scale_mode: ScaleMode::FixedScale,
        scale: 1.0 / 4.05, // 16 grid slots for 11 squares, comfortable
        margin: 0.001,
        ..PackParams::default()
    };
    params.tilt_candidates = false;
    let mut run = islands.clone();
    let result = pack(&mut run, &params);
    assert_eq!(result.retcode, researchuv_pack::UvpmRetcode::Success);
    assert!(result.validation.overlapping.is_empty());
    let cert = PackingCertificate::from_pack(&islands, &params, &result)
        .expect("full placement inside the target");
    cert.replay(&islands, &params).expect("certificate replays");
    let mut mutant = cert.clone();
    mutant.entries[3].tx += 0.5; // shove into a neighbor
    assert!(mutant.replay(&islands, &params).is_err(), "mutant must fail");
    println!("n={n}: certificate replays cleanly and rejects a mutant");
}
