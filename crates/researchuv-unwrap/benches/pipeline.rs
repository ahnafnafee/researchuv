//! Pipeline stage benchmarks — a `harness = false` binary that times each
//! pipeline stage over the built-in fixtures (plus the larger `grid` and
//! `cylinder` fixtures) and prints a table. Run with:
//!
//! ```sh
//! cargo bench -p researchuv-unwrap --locked
//! ```
//!
//! Timings are wall-clock medians over `ROUNDS` runs (default 3; override
//! with `RESEARCHUV_BENCH_ROUNDS`). The runner is dependency-free on
//! purpose: `std::time::Instant` and a stable fixture set keep results
//! comparable across machines and commits.

use researchuv_unwrap::meshgen;
use researchuv_unwrap::pipeline::{run, Packer, PipelineOptions};
use std::time::Instant;

fn fixtures() -> Vec<(&'static str, (Vec<researchuv_math::Vec3>, Vec<[u32; 3]>))> {
    vec![
        ("cube 6", meshgen::cube(6)),
        ("sphere 48x24", meshgen::uv_sphere(48, 24)),
        ("torus 64x40", meshgen::torus(2.0, 0.7, 64, 40)),
        ("annulus 64x40", meshgen::torus_annulus(2.0, 0.7, 64, 40)),
        ("grid 64", meshgen::grid(64, 64)),
        ("cylinder 64x32", meshgen::cylinder(1.0, 2.0, 64, 32)),
    ]
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

fn time_stage<R>(rounds: usize, mut f: impl FnMut() -> R) -> (f64, R) {
    let mut times = Vec::with_capacity(rounds);
    let mut last = None;
    for _ in 0..rounds {
        let t = Instant::now();
        last = Some(f());
        times.push(t.elapsed().as_secs_f64());
    }
    (median(times), last.unwrap())
}

fn main() {
    let rounds: usize = std::env::var("RESEARCHUV_BENCH_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3)
        .max(1);
    println!(
        "researchuv pipeline benchmarks ({rounds} rounds, median, release)\n"
    );
    println!(
        "{:<16} {:>8} {:>8} {:>9} {:>9} {:>12} {:>12}",
        "fixture", "verts", "tris", "weld+seg", "unfold", "full(shelf)", "full(islands)"
    );
    for (name, (p, f)) in fixtures() {
        let nv = p.len();
        let nf = f.len();
        // Stage 1: weld + segment.
        let (t_ws, charts) = time_stage(rounds, || {
            let mesh = researchuv_unwrap::weld::weld(p.clone(), f.clone(), 1e-12);
            let (charts, _) = researchuv_unwrap::segment::segment(&mesh, 30.0);
            (mesh, charts)
        });
        // Stage 2: unfold each chart.
        let (mesh, charts) = charts;
        let (t_unfold, _) = time_stage(rounds, || {
            let mut n = 0usize;
            for ch in &charts {
                let r = researchuv_unwrap::lscm::unfold_chart(&mesh, ch, Default::default());
                n += r.iters;
            }
            n
        });
        // Full runs with each packer (measured end to end — no derived
        // subtractions; compare against the stage columns above).
        let mut shelf_opts = PipelineOptions::default();
        shelf_opts.packer = Packer::Shelf;
        let (t_full_shelf, _) = time_stage(rounds, || {
            run(p.clone(), f.clone(), &shelf_opts).map(|r| r.placed.len()).unwrap_or(0)
        });
        let mut islands_opts = PipelineOptions::default();
        islands_opts.packer = Packer::Islands;
        let (t_full_islands, _) = time_stage(rounds, || {
            run(p.clone(), f.clone(), &islands_opts).map(|r| r.placed.len()).unwrap_or(0)
        });
        println!(
            "{:<16} {:>8} {:>8} {:>8.1}s {:>8.1}s {:>11.1}s {:>11.1}s",
            name, nv, nf, t_ws, t_unfold, t_full_shelf, t_full_islands
        );
    }
}
