//! UV unwrapping from triangle meshes to packed islands.
//!
//! The pipeline welds coincident vertices, segments charts at sharp edges,
//! unfolds each chart, measures distortion, and packs the results into a unit square.
//!
//! | Module | Responsibility |
//! | --- | --- |
//! | [`mod@weld`] | Vertex welding and degenerate-face filtering |
//! | [`mod@segment`] | Edge cuts, connected charts, and boundary loops |
//! | [`lscm`] | Least-squares conformal unfolding and border constraints |
//! | [`sparse`] | Sparse systems and numerical solvers |
//! | [`metrics`] | Distortion measurements and chart normalization |
//! | [`pack`] | CPU shelf packing |
//! | [`pipeline`] | End-to-end library entry point |
//! | [`meshgen`] | Procedural meshes for examples and regression tests |

#![forbid(unsafe_code)]

pub mod lscm;
pub mod lscm_eig;
pub mod meshgen;
pub mod metrics;
pub mod pack;
pub mod pipeline;
pub mod segment;
pub mod sparse;
pub mod weld;

pub use lscm::{
    anchor_ids, border_targets, cg_solve, compute_area, compute_area_grad, edge_weights,
    rect_point, tangent_frame, unfold_chart, UnfoldOptions, UnfoldResult,
    ANGLE_MIX, CONV_THRESHOLD, DEFAULT_MAX_ITER, DEFAULT_MAX_ITER_GUARD, EPS_DEGENERATE,
    KEEP_METRIC, MIX_W, STAGNATION, TRANS_THRESHOLD,
};
pub use metrics::{chart_distortion, rectangularize, ChartMetrics};
pub use pack::{pack_charts, PackBackend, PackRect, Placed};
pub use pipeline::{run, ChartResult, PipelineOptions, PipelineResult};
pub use segment::{
    build_charts, chart_border_loops, edge_faces, edge_key, face_areas3d, face_dihedrals,
    merge_charts, mesh_boundary_edges, segment, Chart, EdgeKey,
};
pub use sparse::Sparse;
pub use weld::{filter_edge_keeps, weld};
