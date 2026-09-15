//! The unwrap pipeline — the reference `run()` driver as a library:
//!
//! import → **weld** (`CTaskWeld`) → **segment/cut** (`CTaskCut` at
//! `Auto.SharpEdges.AngleMin`) → **unfold** each chart (`CTaskUnfold` / `CIsomap`,
//! shared LS driver) → **rectangularize** (`NGeoTopo::Mesh::Rectangularize`) →
//! **pack** into the unit square (`CFinalPack` / GPUPack seam) → export as a
//! [`MultiMesh`] of islands with per-chart distortion metrics.

use crate::lscm::UnfoldOptions;
use crate::metrics::{rectangularize, chart_distortion, ChartMetrics};
use crate::pack::{pack_charts, PackRect, Placed};
use crate::segment::{segment, Chart};
use crate::weld::weld;
use researchuv_core::model::{Island, MultiMesh, SurfaceMesh};
use researchuv_math::{Vec2, Vec3};

/// Pipeline options (documented task parameters).
#[derive(Clone, Copy, Debug)]
pub struct PipelineOptions {
    /// Position weld tolerance (0 = no weld).
    pub weld_tol: f64,
    /// `Vars.AutoSelect.SharpEdges.Angle` — cut edges with dihedral above this (deg).
    pub angle_min_deg: f64,
    /// Unfold driver options.
    pub unfold: UnfoldOptions,
    /// Packing gutter (unit-square fraction).
    pub padding: f64,
    /// Packing retries (each ×0.8 scale-down).
    pub pack_max_iters: usize,
}

impl Default for PipelineOptions {
    fn default() -> Self {
        Self {
            weld_tol: 1e-12,
            angle_min_deg: 30.0,
            unfold: UnfoldOptions::default(),
            padding: 0.01,
            pack_max_iters: 200,
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
    /// Uniform pack scale applied to every island.
    pub scale: f64,
    /// Island placements in the unit square.
    pub placed: Vec<Option<Placed>>,
    /// Final multi-mesh: source + one island per chart with packed UVs.
    pub multi: MultiMesh,
}

/// Run the full unwrap pipeline on a raw triangle soup.
pub fn run(positions: Vec<Vec3>, faces: Vec<[u32; 3]>, opts: &PipelineOptions) -> PipelineResult {
    // 1. Weld (CTaskWeld / FilterEdges).
    let mesh = weld(positions, faces, opts.weld_tol);
    // 2. Segment: cut sharp edges, build charts (CTaskCut + face components).
    let (mut charts, _cut) = segment(&mesh, opts.angle_min_deg);
    // 3. Unfold + rectangularize each chart.
    let mut results: Vec<ChartResult> = Vec::with_capacity(charts.len());
    let mut rects: Vec<PackRect> = Vec::with_capacity(charts.len());
    for ch in charts.drain(..) {
        let res = crate::lscm::unfold_chart(&mesh, &ch, opts.unfold);
        // Border vertices in chart-local indices.
        let bnd_local: Vec<usize> = ch
            .border
            .iter()
            .filter_map(|&v| ch.local_of(v))
            .collect();
        let (q, ext) = rectangularize(&res.uv, &bnd_local);
        let met = chart_distortion(&mesh, &ch, &res.uv);
        rects.push(PackRect {
            w: ext.u.max(1e-6),
            h: ext.v.max(1e-6),
        });
        results.push(ChartResult {
            chart: ch,
            uv: res.uv.clone(),
            rectified: q,
            ext,
            metrics: met,
            iters: res.iters,
            area3d: res.area3d,
        });
    }
    // 4. Pack into the unit square (CFinalPack / GPUPack seam).
    let (scale, placed) = pack_charts(&rects, opts.padding, opts.pack_max_iters);
    // 5. Assemble the final per-vertex UVs into islands (export step).
    let mut islands: Vec<Island> = Vec::with_capacity(results.len());
    for (ci, cr) in results.iter().enumerate() {
        let pl = placed[ci].unwrap_or(Placed { x: 0.0, y: 0.0, w: 0.5, h: 0.5 });
        let ch = &cr.chart;
        // Re-normalize the rectified chart to its own largest extent, then map
        // into the placed box (reference `run()` final-UV assembly).
        let mut min = cr.rectified[0];
        let mut max = cr.rectified[0];
        for p in &cr.rectified {
            min = Vec2::new(min.u.min(p.u), min.v.min(p.v));
            max = Vec2::new(max.u.max(p.u), max.v.max(p.v));
        }
        let rng = (max.u - min.u).max(max.v - min.v).max(1e-30);
        let uv: Vec<Vec2> = cr
            .rectified
            .iter()
            .map(|&t0| {
                let t = t0 - min;
                Vec2::new(pl.x + t.u / rng * pl.w, pl.y + t.v / rng * pl.h)
            })
            .collect();
        let source_vertex_ids = ch.vertex_ids.clone();
        // Border as LOCAL island indices (the island contract).
        let border: Vec<u32> = ch.border.iter().filter_map(|&v| ch.local_of(v).map(|i| i as u32)).collect();
        let positions: Vec<Vec3> =
            ch.vertex_ids.iter().map(|&v| mesh.positions[v as usize]).collect();
        islands.push(Island {
            positions,
            uv,
            tris: ch.tris.clone(),
            source_vertex_ids,
            border,
        });
    }
    let mut multi = MultiMesh::new(mesh.clone());
    multi.islands = islands;
    PipelineResult {
        mesh,
        charts: results,
        scale,
        placed,
        multi,
    }
}
