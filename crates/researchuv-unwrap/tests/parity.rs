//! End-to-end regression tests for the unfolding pipeline.
//!
//! Stored baseline values with the default sharp-edge threshold of 30°:
//!
//! | mesh                     | welded verts/tris | charts | chart verts | area3d   | conformal (mean/max)   | areaRatio | flips | pack scale |
//! |--------------------------|-------------------|--------|-------------|----------|------------------------|-----------|-------|------------|
//! | cube 6×6×6               | 218 / 432         | 6      | 49          | 1.0000   | 1.000 / 1.000          | 1.000     | 0     | 0.262144   |
//! | uv-sphere 48×24          | 1036 / 2068       | 1      | 1036        | 12.5184  | 338.888 / 72584.138    | 0.015     | 0     | 0.800000   |
//! | torus 64×40 annulus      | 2560 / 4992       | 1      | 2560        | 54.2578  | 5.289 / 532.334        | 1.495     | 0     | 0.134218   |
//!
//! The Rust driver is a faithful port of the same mathematics (same constants,
//! same iteration/damping rules, same pin placement), so the *structural*
//! invariants are asserted exactly and the *solver-sensitive* numbers (worst
//! singular-value ratios, which sit at the sphere's antipodal singularity) are
//! asserted in bands.

use researchuv_unwrap::meshgen;
use researchuv_unwrap::pipeline::{run, PipelineOptions};

fn default_opts() -> PipelineOptions {
    PipelineOptions::default()
}

/// Cube: six developable face patches → six charts that are (up to solver
/// noise) exact affine squares: conformal ratio ≈ 1, area ratio ≈ 1, no flips.
#[test]
fn cube_produces_six_exact_affine_charts() {
    let (p, f) = meshgen::cube(6);
    let res = run(p, f, &default_opts()).expect("pipeline run");

    // Welded manifold: 8 corners + 60 edge verts + 150 interior = 218.
    assert_eq!(res.mesh.positions.len(), 218);
    assert_eq!(res.mesh.faces.len(), 432);
    res.mesh.invariant_test().expect("welded cube invariants");

    assert_eq!(res.charts.len(), 6, "cube must segment into 6 charts");
    for (i, cr) in res.charts.iter().enumerate() {
        assert_eq!(cr.chart.vertex_ids.len(), 49, "chart {i} vertex count");
        assert!((cr.area3d - 1.0).abs() < 1e-9, "chart {i} area3d = {}", cr.area3d);
        assert_eq!(cr.metrics.flips, 0, "chart {i} flips");
        assert!(
            (cr.metrics.conformal_mean - 1.0).abs() < 0.02,
            "chart {i} conformal mean {} (developable face must be flat)",
            cr.metrics.conformal_mean
        );
        assert!(
            cr.metrics.conformal_max < 1.05,
            "chart {i} conformal max {} (developable face must be flat)",
            cr.metrics.conformal_max
        );
        assert!(
            (cr.metrics.area_ratio_mean - 1.0).abs() < 0.10,
            "chart {i} area ratio mean {}",
            cr.metrics.area_ratio_mean
        );
        // Reference rect = 1.000 × 1.000.
        assert!((cr.ext.u - 1.0).abs() < 0.05, "chart {i} ext.u = {}", cr.ext.u);
        assert!((cr.ext.v - 1.0).abs() < 0.05, "chart {i} ext.v = {}", cr.ext.v);
    }
    // Six 1×1 charts → scale 0.8^6 = 0.262144 (reference value).
    assert!((res.scale - 0.262144).abs() < 0.001, "pack scale {}", res.scale);
    assert!(res.placed.iter().all(|pl| pl.is_some()), "all 6 charts placed");
    // Final multi-mesh: one island per chart, UVs inside the unit square.
    assert_eq!(res.multi.islands.len(), 6);
    for isl in &res.multi.islands {
        for uv in &isl.uv {
            assert!((0.0..=1.0).contains(&uv.u) && (0.0..=1.0).contains(&uv.v));
        }
    }
}

/// UV-sphere: a single closed (borderless) chart; LSCM is conformal, so the
/// mean distortion is high but finite and the solve never flips a triangle.
/// Reference: conformal 338.888 / 72584.138, areaRatio 0.015 (area collapses
/// at the antipode).
#[test]
fn sphere_is_one_closed_chart_with_finite_distortion() {
    let (p, f) = meshgen::uv_sphere(48, 24);
    let res = run(p, f, &default_opts()).expect("pipeline run");

    assert_eq!(res.mesh.positions.len(), 1036);
    assert_eq!(res.mesh.faces.len(), 2068);
    assert_eq!(res.charts.len(), 1, "sphere is one closed chart");
    let cr = &res.charts[0];
    assert_eq!(cr.chart.vertex_ids.len(), 1036);
    assert!(cr.chart.is_borderless(), "closed sphere chart has no border");
    assert!((cr.area3d - 12.5184).abs() < 0.01, "sphere area3d {}", cr.area3d);
    assert_eq!(cr.metrics.flips, 0);
    assert!(cr.metrics.conformal_mean.is_finite());
    assert!(cr.metrics.conformal_max.is_finite());
    // Reference mean 338.888 (band: the port's CG solve tracks the direct solve).
    assert!(
        (100.0..2000.0).contains(&cr.metrics.conformal_mean),
        "sphere conformal mean {}",
        cr.metrics.conformal_mean
    );
    // Reference areaRatio 0.015 → the map collapses area at the antipode.
    assert!(
        (0.001..0.2).contains(&cr.metrics.area_ratio_mean),
        "sphere area ratio {}",
        cr.metrics.area_ratio_mean
    );
    assert!(cr.iters >= 1 && cr.iters <= 100, "iters {}", cr.iters);
    for uv in &cr.uv {
        assert!(uv.u.is_finite() && uv.v.is_finite());
    }
    // Single chart extent ≈ (1.059, 1.121) → scale 0.8.
    assert!((res.scale - 0.8).abs() < 1e-12, "pack scale {}", res.scale);
    assert!(res.placed[0].is_some());
}

/// Torus annulus: removing the inner-equator ring band leaves a cylinder with
/// two real border loops → exactly one chart with two borders (the reference's
/// signature result), unfolded by the cylinder (annulus) border pass.
#[test]
fn torus_annulus_is_one_chart_with_two_borders() {
    let (p, f) = meshgen::torus_annulus(2.0, 0.7, 64, 40);
    let res = run(p, f, &default_opts()).expect("pipeline run");

    assert_eq!(res.mesh.positions.len(), 2560);
    assert_eq!(res.mesh.faces.len(), 4992);
    assert_eq!(res.charts.len(), 1, "annulus is one chart");
    let cr = &res.charts[0];
    assert_eq!(cr.chart.vertex_ids.len(), 2560);
    assert_eq!(
        cr.chart.border_loops.len(),
        2,
        "annulus has exactly 2 border loops"
    );
    assert!((cr.area3d - 54.2578).abs() < 0.05, "annulus area3d {}", cr.area3d);
    assert_eq!(cr.metrics.flips, 0);
    // Reference conformal 5.289 / 532.334, areaRatio 1.495.
    assert!(
        (1.0..50.0).contains(&cr.metrics.conformal_mean),
        "annulus conformal mean {}",
        cr.metrics.conformal_mean
    );
    assert!(
        (0.5..5.0).contains(&cr.metrics.area_ratio_mean),
        "annulus area ratio {}",
        cr.metrics.area_ratio_mean
    );
    assert!(cr.iters >= 1 && cr.iters <= 100, "iters {}", cr.iters);
    for uv in &cr.uv {
        assert!(uv.u.is_finite() && uv.v.is_finite());
    }
    // Reference rect ≈ 8.064 × 6.623 → scale 0.8^9 ≈ 0.1342.
    assert!(
        (0.10..0.20).contains(&res.scale),
        "annulus pack scale {}",
        res.scale
    );
    assert!(res.placed[0].is_some());
    // The final island carries the two borders (local indices).
    assert_eq!(res.multi.islands[0].border.len(), 2 * 64);
}

/// Charting/segmentation robustness: the full closed torus (no band removed)
/// is also a single borderless chart with zero cut edges at the 30° threshold.
#[test]
fn closed_torus_is_one_borderless_chart() {
    let (p, f) = meshgen::torus(2.0, 0.7, 64, 40);
    let res = run(p, f, &default_opts()).expect("pipeline run");
    assert_eq!(res.mesh.faces.len(), 2 * 64 * 40);
    assert_eq!(res.charts.len(), 1);
    assert!(res.charts[0].chart.is_borderless());
    assert!(res.charts[0].chart.border.is_empty());
    assert_eq!(res.charts[0].metrics.flips, 0);
    for uv in &res.charts[0].uv {
        assert!(uv.u.is_finite() && uv.v.is_finite());
    }
}
