//! UV unwrapping from triangle meshes to packed islands.
//!
//! The pipeline welds coincident vertices, segments charts at sharp edges
//! (optionally seam-cutting closed charts open), unfolds each chart, measures
//! distortion, and packs the results into the unit square with either the
//! shelf packer or the configurable [`researchuv_pack`] island packer.
//!
//! | Module | Responsibility |
//! | --- | --- |
//! | [`mod@weld`] | Vertex welding and degenerate-face filtering |
//! | [`mod@segment`] | Edge cuts, connected charts, and boundary loops |
//! | [`seam`] | Geodesic seam cuts that open closed charts |
//! | [`lscm`] | Least-squares conformal unfolding and border constraints |
//! | [`sparse`] | Sparse systems and numerical solvers |
//! | [`metrics`] | Distortion measurements and chart normalization |
//! | [`pack`] | CPU shelf packing |
//! | [`atlas`] | The bridge to the `researchuv-pack` island packer |
//! | [`validate`] | Malformed-mesh and atlas validation reports |
//! | [`pipeline`] | End-to-end library entry point |
//! | [`meshgen`] | Procedural meshes for examples and regression tests |

#![forbid(unsafe_code)]

pub mod atlas;
pub mod lscm;
pub mod lscm_eig;
pub mod meshgen;
pub mod metrics;
pub mod pack;
pub mod pipeline;
pub mod seam;
pub mod segment;
pub mod sparse;
pub mod validate;
pub mod weld;

pub use atlas::{chart_rings, convex_hull, pack_outlines, ChartRings};
pub use lscm::{
    anchor_ids, border_targets, cg_solve, compute_area, compute_area_grad, edge_weights,
    rect_point, tangent_frame, unfold_chart, UnfoldOptions, UnfoldResult,
    ANGLE_MIX, CONV_THRESHOLD, DEFAULT_MAX_ITER, DEFAULT_MAX_ITER_GUARD, EPS_DEGENERATE,
    KEEP_METRIC, MIX_W, STAGNATION, TRANS_THRESHOLD,
};
pub use metrics::{chart_distortion, rectangularize, ChartMetrics};
pub use pack::{pack_charts, PackBackend, PackRect, Placed};
pub use pipeline::{run, ChartResult, Packer, PipelineError, PipelineOptions, PipelineResult};
pub use seam::{cut_closed_charts, seam_path, SeamCutOptions};
pub use segment::{
    build_charts, chart_border_loops, edge_faces, edge_key, face_areas3d, face_dihedrals,
    merge_charts, mesh_boundary_edges, segment, Chart, EdgeKey,
};
pub use sparse::Sparse;
pub use validate::{validate_atlas, validate_mesh, validate_raw, Finding, Report, Severity};
pub use weld::{filter_edge_keeps, weld};
