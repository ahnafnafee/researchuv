//! The unwrap pipeline — the reference `run()` driver as a library:
//!
//! import → **weld** (`CTaskWeld`) → **segment/cut** (`CTaskCut` at
//! `Auto.SharpEdges.AngleMin`, optionally + a seam cut for closed charts) →
//! **unfold** each chart (`CTaskUnfold` / `CIsomap`, shared LS driver) →
//! **rectangularize** (`NGeoTopo::Mesh::Rectangularize`) → **pack** into the
//! unit square (the shelf `CFinalPack` or the configurable island packer) →
//! export as a [`MultiMesh`] of islands with per-chart distortion metrics and
//! validation reports.

use crate::atlas;
use crate::lscm::UnfoldOptions;
use crate::recut::RecutOptions;
use crate::metrics::{rectangularize, chart_distortion, ChartMetrics};
use crate::pack::{pack_charts, PackRect, Placed};
use crate::seam::{self, SeamCutOptions};
use crate::segment::{segment, Chart};
use crate::validate;
use crate::weld::weld;
use researchuv_core::model::{Island, MultiMesh, SurfaceMesh};
use researchuv_math::{Vec2, Vec3};

/// Final-atlas packing engine selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Packer {
    /// Greedy shelf packer with uniform ×0.8 scale-down retries (the historic
    /// reference `CFinalPack` behavior — the parity baselines).
    Shelf,
    /// The configurable island packer ([`researchuv_pack`]): rotation, flips,
    /// margins, scale modes, target boxes, validation, … see
    /// [`PipelineOptions::island_pack`].
    Islands,
}

impl Default for Packer {
    fn default() -> Self {
        Packer::Shelf
    }
}

/// Pipeline options (documented task parameters).
#[derive(Clone, Debug)]
pub struct PipelineOptions {
    /// Position weld tolerance (0 = no weld).
    pub weld_tol: f64,
    /// `Vars.AutoSelect.SharpEdges.Angle` — cut edges with dihedral above this (deg).
    pub angle_min_deg: f64,
    /// Seam cut for closed (borderless) charts — opens them along a
    /// geodesic-diameter path so the unfold pins a real border.
    pub seam_cut: SeamCutOptions,
    /// Distortion-driven re-cutting — splits charts that fold beyond the
    /// tolerance with border-crossing chords.
    pub recut: RecutOptions,
    /// Worker threads for the per-chart unfold stage (0 = all cores, 1 =
    /// serial). Results are identical regardless of the count.
    pub threads: u32,
    /// Unfold driver options.
    pub unfold: UnfoldOptions,
    /// Shelf-packing gutter (unit-square fraction); ignored by [`Packer::Islands`].
    pub padding: f64,
    /// Shelf-packing retries (each ×0.8 scale-down); ignored by [`Packer::Islands`].
    pub pack_max_iters: usize,
    /// The final-atlas packing engine.
    pub packer: Packer,
    /// Island-packer options (margins, rotation, scale mode, target box, …)
    /// used when `packer` is [`Packer::Islands`]. `margin`/pixel margins
    /// replace `padding`.
    pub island_pack: researchuv_pack::PackParams,
}

impl Default for PipelineOptions {
    fn default() -> Self {
        Self {
            weld_tol: 1e-12,
            angle_min_deg: 30.0,
            seam_cut: SeamCutOptions::default(),
            recut: RecutOptions::default(),
            threads: 0,
            unfold: UnfoldOptions::default(),
            padding: 0.01,
            pack_max_iters: 200,
            packer: Packer::default(),
            island_pack: researchuv_pack::PackParams::default(),
        }
    }
}

/// Per-chart pipeline output.
#[derive(Clone, Debug)]
pub struct ChartResult {
    /// The chart (face/vertex topology in source-mesh indices).
    pub chart: Chart,
    /// Unfolded 2-D coordinates in chart-local vertex order (pre-rectangularize).
    pub uv: Vec<Vec2>,
    /// Rectangularized (normalized `[0,1]²`-ish) coordinates, chart-local.
    pub rectified: Vec<Vec2>,
    /// 2-D extent of the pre-rectangularize unfold (the pack rectangle).
    pub ext: Vec2,
    /// Distortion metrics (evaluated on the pre-rectangularize UVs).
    pub metrics: ChartMetrics,
    /// Driver iterations actually run.
    pub iters: usize,
    /// Chart 3-D area.
    pub area3d: f64,
}

/// Whole-pipeline output.
#[derive(Clone, Debug)]
pub struct PipelineResult {
    /// The welded source mesh.
    pub mesh: SurfaceMesh,
    /// Per-chart results (input chart order).
    pub charts: Vec<ChartResult>,
    /// Uniform pack scale applied to every island (shelf packer); for the
    /// island packer, the largest per-island placement scale.
    pub scale: f64,
    /// Island placements in the unit square.
    pub placed: Vec<Option<Placed>>,
    /// Final multi-mesh: source + one island per chart with packed UVs.
    pub multi: MultiMesh,
    /// The island packer's full result contract when [`Packer::Islands`] ran
    /// (transforms, groups, validation, retcode).
    pub island_pack: Option<researchuv_pack::PackResult>,
    /// Input/topology findings for the welded mesh.
    pub mesh_report: validate::Report,
    /// Atlas findings (unplaced islands, UVs outside `[0,1]²`, overlaps).
    pub atlas_report: validate::Report,
}

/// Run the full unwrap pipeline on a raw triangle soup.
///
/// Returns `Err` only for inputs the pipeline cannot process at all (no
/// faces, out-of-range indices, non-finite positions); everything else is
/// reported through [`PipelineResult::mesh_report`] and
/// [`PipelineResult::atlas_report`].
pub fn run(
    positions: Vec<Vec3>,
    faces: Vec<[u32; 3]>,
    opts: &PipelineOptions,
) -> Result<PipelineResult, PipelineError> {
    let input_report = validate::validate_raw(&positions, &faces);
    if input_report.has_errors() {
        return Err(PipelineError::InvalidInput(input_report));
    }
    // 1. Weld (CTaskWeld / FilterEdges).
    let mesh = weld(positions, faces, opts.weld_tol);
    let mesh_report = validate::validate_mesh(&mesh);
    // 2. Segment: cut sharp edges, build charts (CTaskCut + face components);
    //    optionally open closed charts with a seam cut.
    let (mut charts, _cut) = segment(&mesh, opts.angle_min_deg);
    if opts.seam_cut.enable {
        let (cut_charts, _cut_all) =
            seam::cut_closed_charts(&mesh, &charts, &_cut, opts.seam_cut);
        charts = cut_charts;
    }
    // 2b. Distortion-driven re-cutting: split charts that fold.
    if opts.recut.enable {
        let (recut_charts, _cut_all, _added) = crate::recut::recut_folding_charts(
            &mesh,
            charts,
            &_cut,
            opts.recut,
            opts.unfold,
            opts.seam_cut,
        );
        charts = recut_charts;
    }
    // 3. Unfold + rectangularize each chart. Charts are independent, so the
    //    stage runs on the parallel executor (results in input order —
    //    identical to the serial run for any worker count).
    let unfold_stage = |ch: Chart| -> (ChartResult, PackRect) {
        let res = crate::lscm::unfold_chart(&mesh, &ch, opts.unfold);
        // Border vertices in chart-local indices.
        let bnd_local: Vec<usize> =
            ch.border.iter().filter_map(|&v| ch.local_of(v)).collect();
        let (q, ext) = rectangularize(&res.uv, &bnd_local);
        let met = chart_distortion(&mesh, &ch, &res.uv);
        let rect = PackRect { w: ext.u.max(1e-6), h: ext.v.max(1e-6) };
        (
            ChartResult {
                chart: ch,
                uv: res.uv.clone(),
                rectified: q,
                ext,
                metrics: met,
                iters: res.iters,
                area3d: res.area3d,
            },
            rect,
        )
    };
    let workers = researchuv_core::exec::worker_count(opts.threads);
    let per_chart: Vec<(ChartResult, PackRect)> =
        researchuv_core::exec::par_map(workers, charts, unfold_stage);
    let mut results: Vec<ChartResult> = Vec::with_capacity(per_chart.len());
    let mut rects: Vec<PackRect> = Vec::with_capacity(per_chart.len());
    for (cr, rect) in per_chart {
        results.push(cr);
        rects.push(rect);
    }
    // 4. Pack into the unit square, then assemble the final per-vertex UVs.
    let (scale, placed, island_pack, final_uv) = match opts.packer {
        Packer::Shelf => {
            let (scale, placed) = pack_charts(&rects, opts.padding, opts.pack_max_iters);
            let final_uv: Vec<Vec<Vec2>> = results
                .iter()
                .enumerate()
                .map(|(ci, cr)| shelf_map(cr, placed[ci]))
                .collect();
            (scale, placed, None, final_uv)
        }
        Packer::Islands => {
            let rings: Vec<atlas::ChartRings> = results
                .iter()
                .map(|cr| atlas::chart_rings(&cr.chart, &cr.rectified, cr.ext))
                .collect();
            let (transforms, pack_result, top_scale) =
                atlas::pack_outlines(rings, &opts.island_pack);
            let placed: Vec<Option<Placed>> = transforms.iter().map(|t| t.map(|t| atlas::placed_of(&t))).collect();
            let final_uv: Vec<Vec<Vec2>> = results
                .iter()
                .enumerate()
                .map(|(ci, cr)| match transforms[ci] {
                    Some(t) => cr
                        .rectified
                        .iter()
                        .map(|&q| t.apply(Vec2::new(q.u * cr.ext.u.max(1e-30), q.v * cr.ext.v.max(1e-30))))
                        .collect(),
                    None => shelf_map(cr, None),
                })
                .collect();
            (top_scale, placed, Some(pack_result), final_uv)
        }
    };
    // 5. Assemble the islands (export step).
    let mut islands: Vec<Island> = Vec::with_capacity(results.len());
    for (ci, cr) in results.iter().enumerate() {
        let ch = &cr.chart;
        let source_vertex_ids = ch.vertex_ids.clone();
        // Border as LOCAL island indices (the island contract).
        let border: Vec<u32> = ch.border.iter().filter_map(|&v| ch.local_of(v).map(|i| i as u32)).collect();
        let positions: Vec<Vec3> =
            ch.vertex_ids.iter().map(|&v| mesh.positions[v as usize]).collect();
        islands.push(Island {
            positions,
            uv: final_uv[ci].clone(),
            tris: ch.tris.clone(),
            source_vertex_ids,
            border,
        });
    }
    let mut atlas_report = validate::validate_atlas(&islands, &placed);
    // Merge the island packer's polygon-level findings (overlapping /
    // outside-target outlines, self-intersections) into the atlas report so
    // callers see one complete picture.
    if let Some(pack) = &island_pack {
        let v = &pack.validation;
        for &i in &v.overlapping {
            atlas_report.push(
                validate::Severity::Error,
                "OverlappingIslands",
                format!("island {i} overlaps another island (polygon-level check)"),
            );
        }
        for &i in &v.outside {
            atlas_report.push(
                validate::Severity::Error,
                "UvOutsideUnitSquare",
                format!("island {i} is outside the target box"),
            );
        }
        for &i in &v.self_intersecting {
            atlas_report.push(
                validate::Severity::Warning,
                "SelfIntersectingOutline",
                format!("island {i} has a self-intersecting outline"),
            );
        }
    }
    let mut multi = MultiMesh::new(mesh.clone());
    multi.islands = islands;
    Ok(PipelineResult {
        mesh,
        charts: results,
        scale,
        placed,
        multi,
        island_pack,
        mesh_report,
        atlas_report,
    })
}

/// The shelf path's chart-UV mapping: re-normalize the rectified chart to its
/// own largest extent, then map into the placed box (reference `run()`
/// final-UV assembly). `None` placement falls back to the bottom-left
/// half-square.
fn shelf_map(cr: &ChartResult, pl: Option<Placed>) -> Vec<Vec2> {
    let pl = pl.unwrap_or(Placed { x: 0.0, y: 0.0, w: 0.5, h: 0.5 });
    let mut min = cr.rectified[0];
    let mut max = cr.rectified[0];
    for p in &cr.rectified {
        min = Vec2::new(min.u.min(p.u), min.v.min(p.v));
        max = Vec2::new(max.u.max(p.u), max.v.max(p.v));
    }
    let rng = (max.u - min.u).max(max.v - min.v).max(1e-30);
    cr.rectified
        .iter()
        .map(|&t0| {
            let t = t0 - min;
            Vec2::new(pl.x + t.u / rng * pl.w, pl.y + t.v / rng * pl.h)
        })
        .collect()
}

/// Pipeline failure conditions.
#[derive(Clone, Debug)]
pub enum PipelineError {
    /// Malformed input — carries the raw-input validation findings
    /// (no faces, out-of-range indices, non-finite positions).
    InvalidInput(validate::Report),
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineError::InvalidInput(rep) => {
                write!(f, "invalid mesh input:")?;
                for e in rep.errors() {
                    write!(f, " [{}] {}", e.code, e.detail)?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for PipelineError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thread_count_does_not_change_results() {
        let (p, f) = crate::meshgen::cube(6);
        let mut one = PipelineOptions::default();
        one.threads = 1;
        let mut many = PipelineOptions::default();
        many.threads = 8;
        let a = run(p.clone(), f.clone(), &one).unwrap();
        let b = run(p, f, &many).unwrap();
        assert_eq!(a.charts.len(), b.charts.len());
        for (x, y) in a.charts.iter().zip(b.charts.iter()) {
            assert_eq!(x.uv.len(), y.uv.len());
            assert_eq!(x.uv, y.uv, "UVs must be bitwise identical across worker counts");
            assert_eq!(x.rectified, y.rectified);
        }
        assert_eq!(a.placed.len(), b.placed.len());
    }

    #[test]
    fn recut_reduces_folds_in_the_pipeline_result() {
        let (p, f) = crate::meshgen::torus_annulus(2.0, 0.7, 24, 16);
        let plain = run(p.clone(), f.clone(), &PipelineOptions::default()).unwrap();
        let mut o = PipelineOptions::default();
        o.recut.enable = true;
        let cut = run(p, f, &o).unwrap();
        let folds =
            |r: &PipelineResult| r.charts.iter().map(|c| c.metrics.folds).sum::<usize>();
        assert!(
            folds(&cut) < folds(&plain),
            "folds {} vs {}",
            folds(&cut),
            folds(&plain)
        );
        assert!(cut.charts.len() > plain.charts.len(), "the strip was split");
        assert!(cut.placed.iter().all(|p| p.is_some()), "all split charts placed");
    }

    #[test]
    fn invalid_input_is_rejected_with_findings() {
        let err = run(
            vec![Vec3::new(0.0, 0.0, 0.0)],
            vec![[0, 1, 2]],
            &PipelineOptions::default(),
        )
        .unwrap_err();
        match err {
            PipelineError::InvalidInput(rep) => {
                assert_eq!(rep.errors().next().unwrap().code, "IndexOutOfRange");
            }
        }
    }

    #[test]
    fn empty_input_is_rejected() {
        assert!(run(vec![], vec![], &PipelineOptions::default()).is_err());
    }

    #[test]
    fn island_packer_produces_a_valid_cube_atlas() {
        let (p, f) = crate::meshgen::cube(4);
        let mut opts = PipelineOptions::default();
        opts.packer = Packer::Islands;
        let res = run(p, f, &opts).expect("islands packer run");
        assert_eq!(res.charts.len(), 6);
        assert!(res.placed.iter().all(|p| p.is_some()), "all six faces placed");
        assert!(res.island_pack.is_some());
        assert!(!res.atlas_report.has_errors(), "{:?}", res.atlas_report.findings);
        for isl in &res.multi.islands {
            for uv in &isl.uv {
                assert!((0.0..=1.0).contains(&uv.u) && (0.0..=1.0).contains(&uv.v));
            }
        }
    }

    #[test]
    fn seam_cut_recovers_area_on_the_sphere() {
        let (p, f) = crate::meshgen::uv_sphere(24, 12);
        let mut plain = PipelineOptions::default();
        let closed = run(p.clone(), f.clone(), &plain).expect("closed run");
        assert_eq!(closed.charts.len(), 1);
        assert!(closed.charts[0].chart.is_borderless());
        let mut cut = plain.clone();
        cut.seam_cut.enable = true;
        let opened = run(p, f, &cut).expect("seam-cut run");
        assert_eq!(opened.charts.len(), 1, "the cut opens, not splits, the sphere");
        assert!(!opened.charts[0].chart.is_borderless());
        let (c, k) = (&closed.charts[0].metrics, &opened.charts[0].metrics);
        // The free-boundary solve collapses area on closed charts (ratio ≪ 1);
        // the seam tree lets the border pins preserve it (ratio ≈ 1).
        assert!(
            k.area_ratio_mean > c.area_ratio_mean * 10.0,
            "area ratio cut {} vs closed {}",
            k.area_ratio_mean,
            c.area_ratio_mean
        );
        assert!(k.area_ratio_mean > 0.8 && k.area_ratio_mean < 2.0);
        // Angular error stays finite and bounded (the pin pass trades a
        // little conformality for the recovered area).
        assert!(k.conformal_mean.is_finite() && k.conformal_mean < 1000.0);
        assert_eq!(k.flips, 0);
        assert!(!opened.atlas_report.has_errors(), "{:?}", opened.atlas_report.findings);
    }
}
