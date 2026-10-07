//! Equal-squares optimality-gap benchmark — a `harness = false` binary that
//! runs the protocol from `tests/squares_gap.rs` over the full curated
//! fixture set and prints one table. Run with:
//!
//! ```sh
//! cargo bench -p researchuv-pack --locked
//! ```
//!
//! For each `n`: the smallest container side the packer realizes for `n`
//! unit squares (binary search over fixed-uniform-scale packs, margin 0),
//! against the certified lower bound and best-known upper bound for `s(n)`
//! from the squares project. Columns:
//!
//! - `cpu90` — exact planner, 90° rotation ladder (deterministic baseline);
//! - `tilt` — exact planner + `tilt_candidates` (near-square tilt ladder);
//! - `raster` — the GPU occupancy-grid placement path, when a CUDA device
//!   is present (polygon-footprint collision; the one path where tilted
//!   equal-square packings could in principle beat the axis-aligned grid).
//!
//! Timings are wall-clock medians over `ROUNDS` runs (default 3; override
//! with `RESEARCHUV_BENCH_ROUNDS`).

use researchuv_math::Vec2;
use researchuv_pack::sqbounds;
use researchuv_pack::{pack, Island, PackParams, ScaleMode};
use std::time::Instant;

const ITERS: u32 = 22;

fn unit_squares(n: u32) -> Vec<Island> {
    (0..n)
        .map(|i| {
            let mut isl = Island::from_polygon(vec![
                Vec2::new(0.0, 0.0),
                Vec2::new(1.0, 0.0),
                Vec2::new(1.0, 1.0),
                Vec2::new(0.0, 1.0),
            ]);
            isl.verts[0].u += i as f64 * 1e-9;
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

fn feasible(islands: &[Island], params: &PackParams) -> bool {
    let mut run = islands.to_vec();
    let r = pack(&mut run, params);
    let ok = r.non_packed.is_empty()
        && r.placed.iter().all(|p| p.is_some())
        && r.validation.outside.is_empty()
        && r.validation.self_intersecting.is_empty();
    if !ok {
        return false;
    }
    // Polygon-level disjointness with a touching tolerance (the packer's own
    // overlap flags treat touching as overlap by design). SAT on the convex
    // outlines; see tests/squares_gap.rs for the rationale.
    let outlines: Vec<Vec<Vec2>> = r
        .placed
        .iter()
        .enumerate()
        .map(|(i, t)| t.as_ref().unwrap().transform_poly(&islands[i].verts))
        .collect();
    for i in 0..outlines.len() {
        for j in (i + 1)..outlines.len() {
            if !convex_disjoint_tol(&outlines[i], &outlines[j], 1e-9) {
                return false;
            }
        }
    }
    true
}

/// Separating-axis disjointness for convex outlines with tolerance `tol`
/// (penetration below `tol` counts as touching).
fn convex_disjoint_tol(a: &[Vec2], b: &[Vec2], tol: f64) -> bool {
    for poly in [a, b] {
        for k in 0..poly.len() {
            let p = poly[k];
            let q = poly[(k + 1) % poly.len()];
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

fn achieved_side(n: u32, tilt: bool, raster_resolution: u32) -> f64 {
    let islands = unit_squares(n);
    let grid = (n as f64).sqrt().ceil();
    let mut lo = 0.5f64;
    let mut hi = grid + 1.0;
    for _ in 0..ITERS {
        let mid = 0.5 * (lo + hi);
        if feasible(&islands, &params_at_side(mid, tilt, raster_resolution)) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    hi
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal));
    v[v.len() / 2]
}

fn main() {
    let rounds: usize = std::env::var("RESEARCHUV_BENCH_ROUNDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let gpu = researchuv_gpu::GpuSolver::global().is_some();
    if !gpu {
        eprintln!("note: no CUDA device/PTX available — the raster column is omitted");
    }

    println!(
        "equal-squares optimality gap (packer-achieved side vs certified s(n) bounds)\n"
    );
    println!(
        "{n:>3} {cert_floor:>12} {best:>12} {exact:>7} {cpu90:>10} {tilt:>10} {raster:>10} {gap:>8} {util:>7} {ceil:>7} {ms:>8}",
        n = "n",
        cert_floor = "floor (cert)",
        best = "best known",
        exact = "optimal",
        cpu90 = "cpu90",
        tilt = "tilt",
        raster = if gpu { "raster" } else { "-" },
        gap = "gap%",
        util = "util",
        ceil = "ceil",
        ms = "ms/run",
    );

    for &n in sqbounds::BENCHMARK_N {
        let b = sqbounds::bound(n);
        let mut t90 = Vec::with_capacity(rounds);
        let mut s90 = 0.0;
        for _ in 0..rounds {
            let t = Instant::now();
            s90 = achieved_side(n, false, 0);
            t90.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        let stilt = achieved_side(n, true, 0);
        let (srast, srtilt, trast) = if gpu {
            let mut tv = Vec::with_capacity(rounds);
            let mut s = 0.0;
            for _ in 0..rounds {
                let t = Instant::now();
                s = achieved_side(n, false, 256);
                tv.push(t.elapsed().as_secs_f64() * 1000.0);
            }
            let st = achieved_side(n, true, 256);
            (s, st, median(tv))
        } else {
            (f64::NAN, f64::NAN, f64::NAN)
        };
        let util = n as f64 / (s90 * s90);
        let ceil = sqbounds::certified_max_utilization(n);
        println!(
            "{n:>3} {floor:>12.6} {best:>12.6} {exact:>7} {cpu90:>10.6} {tilt:>10.6} {raster:>10.6} {gap:>8.2} {util:>7.3} {ceil:>7.3} {ms:>8.0}",
            floor = b.lower,
            best = b.upper,
            exact = if b.exact { "yes" } else { "open" },
            cpu90 = s90,
            tilt = stilt,
            raster = srast,
            gap = (s90 / b.lower - 1.0) * 100.0,
            util = util,
            ceil = ceil,
            ms = median(t90),
        );
        if gpu && !srast.is_nan() {
            eprintln!(
                "      n={n}: raster {srast:.6} ({trast:.0} ms/run), raster+tilt {srtilt:.6} ({d:+.2}%)",
                d = (srtilt / srast - 1.0) * 100.0,
            );
        }
    }

    println!(
        "\ncolumns: floor = certified lower bound on s(n); best = best-known upper bound;\n\
         cpu90 = exact planner (90° ladder); tilt = +tilt_candidates; raster = GPU occupancy grid;\n\
         gap% = cpu90 over the certified floor; util = n / cpu90²; ceil = certified utilization\n\
         ceiling n / floor². Sources: researchuv-pack/src/sqbounds.rs."
    );
}
