//! Distortion-driven re-cutting — automatically splits charts that fold.
//!
//! Rectangle-pinned least-squares unfolds fold over themselves when a chart
//! is more curved than its pinned border can hold (the slit annulus needs to
//! fit a 17-unit outer circumference into an 8.2-unit pinned strip — exactly
//! cancelling signed areas are the fingerprint). The reference pipeline
//! leaves those folds in the atlas; the validator reports them.
//!
//! This controller closes the loop: unfold → measure winding-inconsistent
//! triangles ([`crate::metrics::winding_flips`]) → for every chart that folds
//! beyond the tolerance, add a *separating* cut and re-segment:
//!
//! - a **closed** chart gets the geodesic seam tree first (it has no border
//!   to cross, and anchor-pinned closed charts are the worst folders);
//! - an **open** chart gets a border-crossing chord ([`crate::seam::
//!   border_crossing_path`]) — the geodesic from its border through the
//!   interior to the farthest border vertex. Chords separate the chart into
//!   two face components, and narrower charts pin closer to their true
//!   developable width, so the fold fraction falls round over round.
//!
//! The loop stops when no chart exceeds the fold tolerance, when the split
//! budget is spent, or when no fresh cut exists. Charts that stay clean are
//! never touched.

use crate::metrics::winding_flips;
use crate::seam::{border_crossing_path, seam_cut_edges, SeamCutOptions};
use crate::segment::{build_charts, Chart, EdgeKey};
use crate::lscm::UnfoldOptions;
use researchuv_core::model::SurfaceMesh;
use std::collections::BTreeSet;

/// Distortion-driven re-cutting options.
#[derive(Clone, Copy, Debug)]
pub struct RecutOptions {
    /// Re-cut charts that fold. Off by default (the reference pipeline keeps
    /// its charts; the parity baselines record that behavior).
    pub enable: bool,
    /// Controller rounds (unfold → measure → cut every folding chart).
    pub max_rounds: usize,
    /// Total number of new cut edges the controller may add.
    pub max_splits: usize,
    /// Fold tolerance: re-cut a chart when more than this fraction of its
    /// triangles are winding-inconsistent.
    pub fold_fraction: f64,
    /// Guard: stop if the chart count explodes (pathological inputs).
    pub max_charts: usize,
}

impl Default for RecutOptions {
    fn default() -> Self {
        Self {
            enable: false,
            max_rounds: 6,
            max_splits: 48,
            fold_fraction: 0.02,
            max_charts: 64,
        }
    }
}

/// One controller round's verdict for a chart.
#[derive(Clone, Copy, Debug)]
pub struct ChartFolds {
    /// Winding-inconsistent triangles after unfolding.
    pub folds: usize,
    /// `folds / triangles` — compared against the tolerance.
    pub fraction: f64,
}

/// Measure the fold state of every chart (one unfold each).
pub fn chart_folds(mesh: &SurfaceMesh, charts: &[Chart], unfold: UnfoldOptions) -> Vec<ChartFolds> {
    charts
        .iter()
        .map(|ch| {
            let res = crate::lscm::unfold_chart(mesh, ch, unfold);
            let folds = winding_flips(&res.uv, &ch.tris);
            ChartFolds { folds, fraction: folds as f64 / ch.tris.len().max(1) as f64 }
        })
        .collect()
}

/// The cut edges that separate one chart: its seam tree when closed, its
/// border-crossing chord when open. Empty when no cut exists.
fn separating_cut(mesh: &SurfaceMesh, chart: &Chart, seam: SeamCutOptions) -> BTreeSet<EdgeKey> {
    if chart.is_borderless() {
        seam_cut_edges(mesh, chart, seam.branch_count.max(1))
    } else {
        let mut cut = BTreeSet::new();
        for w in border_crossing_path(mesh, chart).windows(2) {
            cut.insert(crate::segment::edge_key(w[0], w[1]));
        }
        cut
    }
}

/// The controller loop. Returns the re-cut chart list, the grown cut set,
/// and the number of edges actually added.
pub fn recut_folding_charts(
    mesh: &SurfaceMesh,
    charts: Vec<Chart>,
    cut: &BTreeSet<EdgeKey>,
    opts: RecutOptions,
    unfold: UnfoldOptions,
    seam: SeamCutOptions,
) -> (Vec<Chart>, BTreeSet<EdgeKey>, usize) {
    let mut all = cut.clone();
    let mut charts = charts;
    let mut added = 0usize;
    for _ in 0..opts.max_rounds {
        if added >= opts.max_splits {
            break;
        }
        let folds = chart_folds(mesh, &charts, unfold);
        let folding: Vec<usize> = folds
            .iter()
            .enumerate()
            .filter(|(_, f)| f.fraction > opts.fold_fraction)
            .map(|(i, _)| i)
            .collect();
        if folding.is_empty() {
            break;
        }
        // Cut every folding chart this round (worst first; ties by index).
        let mut order = folding;
        order.sort_by_key(|&i| std::cmp::Reverse(folds[i].folds));
        let mut grew = false;
        for i in order {
            if added >= opts.max_splits {
                break;
            }
            for e in separating_cut(mesh, &charts[i], seam) {
                if all.insert(e) {
                    added += 1;
                    grew = true;
                }
            }
        }
        if !grew {
            break; // no fresh cut exists — the charts cannot be split further
        }
        charts = build_charts(mesh, &all);
        if charts.len() > opts.max_charts {
            break;
        }
    }
    (charts, all, added)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segment::segment;
    use researchuv_math::Vec2;

    #[test]
    fn winding_flips_counts_local_inversions() {
        let uv = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(0.5, 0.0),
            Vec2::new(0.0, 0.5),
            Vec2::new(0.5, 0.5),
        ];
        let tris = vec![[0u32, 1, 2], [1, 2, 3]];
        assert_eq!(winding_flips(&uv, &tris), 1, "second triangle is CW against CCW majority");
        // Uniformly reversed winding is consistent, not folded.
        let tris2 = vec![[0u32, 2, 1], [1, 2, 3]];
        assert_eq!(winding_flips(&uv, &tris2), 0);
        assert_eq!(winding_flips(&uv, &[]), 0);
    }

    #[test]
    fn chord_cuts_split_the_annulus_strip() {
        // The slit-torus annulus folds under its rectangle pins; the
        // controller's chord cuts must split it and reduce the fold fraction.
        let (p, f) = crate::meshgen::torus_annulus(2.0, 0.7, 24, 16);
        let mesh = crate::weld::weld(p, f, 1e-12);
        let (charts, cut) = segment(&mesh, 30.0);
        assert_eq!(charts.len(), 1);
        let before = chart_folds(&mesh, &charts, UnfoldOptions::default());
        assert!(
            before[0].fraction > 0.02,
            "the fixture must actually fold ({}) — otherwise the test is vacuous",
            before[0].fraction
        );

        let (charts2, _cut2, added) = recut_folding_charts(
            &mesh,
            charts.clone(),
            &cut,
            RecutOptions { enable: true, ..RecutOptions::default() },
            UnfoldOptions::default(),
            SeamCutOptions::default(),
        );
        assert!(added > 0, "the controller must have cut something");
        assert!(
            charts2.len() > 1,
            "chord cuts must split the strip (got {} charts)",
            charts2.len()
        );
        let after = chart_folds(&mesh, &charts2, UnfoldOptions::default());
        let total = |cs: &[Chart]| cs.iter().map(|c| c.tris.len()).sum::<usize>();
        let after_frac =
            after.iter().map(|c| c.folds).sum::<usize>() as f64 / total(&charts2) as f64;
        let before_frac =
            before.iter().map(|c| c.folds).sum::<usize>() as f64 / total(&charts) as f64;
        assert!(
            after_frac < before_frac,
            "fold fraction must improve: {after_frac:.3} vs {before_frac:.3}"
        );
    }

    #[test]
    fn closed_charts_open_then_split() {
        // The closed torus: round 1 opens it with the seam tree, later
        // rounds chord-cut the opened chart; folds must fall overall.
        let (p, f) = crate::meshgen::torus(2.0, 0.7, 24, 16);
        let mesh = crate::weld::weld(p, f, 1e-12);
        let (charts, cut) = segment(&mesh, 30.0);
        let before = chart_folds(&mesh, &charts, UnfoldOptions::default());
        assert!(
            before[0].fraction > 0.02,
            "the closed torus must fold ({})",
            before[0].fraction
        );

        let (charts2, _cut2, added) = recut_folding_charts(
            &mesh,
            charts.clone(),
            &cut,
            RecutOptions { enable: true, max_rounds: 8, ..RecutOptions::default() },
            UnfoldOptions::default(),
            SeamCutOptions::default(),
        );
        assert!(added > 0);
        let after = chart_folds(&mesh, &charts2, UnfoldOptions::default());
        let total = |cs: &[Chart]| cs.iter().map(|c| c.tris.len()).sum::<usize>();
        let after_frac =
            after.iter().map(|c| c.folds).sum::<usize>() as f64 / total(&charts2) as f64;
        let before_frac =
            before.iter().map(|c| c.folds).sum::<usize>() as f64 / total(&charts) as f64;
        assert!(
            after_frac < before_frac,
            "fold fraction must improve: {after_frac:.3} vs {before_frac:.3}"
        );
    }

    #[test]
    fn clean_charts_are_left_alone() {
        // A flat grid does not fold: the controller adds no cuts.
        let (p, f) = crate::meshgen::grid(6, 6);
        let mesh = crate::weld::weld(p, f, 1e-12);
        let (charts, cut) = segment(&mesh, 30.0);
        let (charts2, _cut2, added) = recut_folding_charts(
            &mesh,
            charts.clone(),
            &cut,
            RecutOptions { enable: true, ..RecutOptions::default() },
            UnfoldOptions::default(),
            SeamCutOptions::default(),
        );
        assert_eq!(added, 0);
        assert_eq!(charts2.len(), charts.len());
    }
}
