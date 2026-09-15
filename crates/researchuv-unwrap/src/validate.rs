//! Input and output validation — malformed-mesh checks before the pipeline
//! runs and atlas checks on its result.
//!
//! [`validate_mesh`] inspects a raw triangle soup or a welded [`SurfaceMesh`]
//! for structural problems (out-of-range indices, non-finite coordinates,
//! degenerate faces, non-manifold edges, isolated vertices). Errors are
//! conditions the pipeline cannot recover from; warnings describe inputs the
//! welder/filter can still process but the caller should know about.
//!
//! [`validate_atlas`] inspects a finished pipeline output: unplaced islands,
//! UVs outside the unit square, flipped triangles, and overlapping placements.

use crate::pack::Placed;
use researchuv_core::model::{Island, SurfaceMesh};
use researchuv_math::Vec3;

/// Severity of a validation finding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Severity {
    /// The pipeline cannot produce a sound result.
    Error,
    /// The result is usable but the condition deserves attention.
    Warning,
}

impl Severity {
    pub fn as_str(self) -> &'static str {
        match self {
            Severity::Error => "error",
            Severity::Warning => "warning",
        }
    }
}

/// One validation finding.
#[derive(Clone, Debug)]
pub struct Finding {
    pub severity: Severity,
    pub code: &'static str,
    pub detail: String,
}

/// A validation report: findings plus counts, ordered errors-then-warnings.
#[derive(Clone, Debug, Default)]
pub struct Report {
    pub findings: Vec<Finding>,
}

impl Report {
    /// Append one finding (used by the pipeline to merge the island packer's
    /// polygon-level findings into the atlas report).
    pub fn push(&mut self, severity: Severity, code: &'static str, detail: impl Into<String>) {
        self.findings.push(Finding {
            severity,
            code,
            detail: detail.into(),
        });
    }

    /// Findings with `Error` severity.
    pub fn errors(&self) -> impl Iterator<Item = &Finding> {
        self.findings.iter().filter(|f| f.severity == Severity::Error)
    }

    /// Findings with `Warning` severity.
    pub fn warnings(&self) -> impl Iterator<Item = &Finding> {
        self.findings.iter().filter(|f| f.severity == Severity::Warning)
    }

    pub fn is_ok(&self) -> bool {
        self.findings.is_empty()
    }

    pub fn has_errors(&self) -> bool {
        self.errors().next().is_some()
    }
}

/// Validate raw positions/faces (pre-weld) for conditions the pipeline cannot
/// recover from, plus warnings it can.
pub fn validate_raw(positions: &[Vec3], faces: &[[u32; 3]]) -> Report {
    let mut rep = Report::default();
    if positions.is_empty() {
        rep.push(Severity::Error, "NoVertices", "the mesh has no vertices");
        return rep;
    }
    if faces.is_empty() {
        rep.push(Severity::Error, "NoFaces", "the mesh has no faces");
        return rep;
    }
    let non_finite = positions
        .iter()
        .filter(|p| !p.x.is_finite() || !p.y.is_finite() || !p.z.is_finite())
        .count();
    if non_finite > 0 {
        rep.push(
            Severity::Error,
            "NonFinitePositions",
            format!("{non_finite} vertex positions are NaN or infinite"),
        );
    }
    let mut oob = 0usize;
    for f in faces {
        for &v in f {
            if v as usize >= positions.len() {
                oob += 1;
                break;
            }
        }
    }
    if oob > 0 {
        rep.push(
            Severity::Error,
            "IndexOutOfRange",
            format!("{oob} faces reference a vertex beyond the position array"),
        );
        // Index errors make the remaining checks meaningless.
        return rep;
    }
    let degenerate = faces
        .iter()
        .filter(|[a, b, c]| a == b || b == c || a == c)
        .count();
    if degenerate > 0 {
        rep.push(
            Severity::Warning,
            "DegenerateFaces",
            format!("{degenerate} faces repeat a vertex and will be dropped by the welder"),
        );
    }
    let zero_area = faces
        .iter()
        .filter(|[a, b, c]| {
            let e1 = positions[*b as usize] - positions[*a as usize];
            let e2 = positions[*c as usize] - positions[*a as usize];
            e1.cross_v(e2).len() < 1e-30
        })
        .count();
    if zero_area > 0 {
        rep.push(
            Severity::Warning,
            "ZeroAreaFaces",
            format!("{zero_area} faces have (near-)zero area"),
        );
    }
    rep
}

/// Validate a welded [`SurfaceMesh`] (topology-level checks).
pub fn validate_mesh(mesh: &SurfaceMesh) -> Report {
    let mut rep = validate_raw(&mesh.positions, &mesh.faces);
    if rep.has_errors() {
        return rep;
    }
    if let Err(e) = mesh.invariant_test() {
        rep.push(Severity::Error, "BrokenHalfedges", e);
        return rep;
    }
    // Edge → face multiplicity.
    let mut edge_count: std::collections::BTreeMap<(u32, u32), usize> =
        std::collections::BTreeMap::new();
    for [a, b, c] in &mesh.faces {
        for (x, y) in [(*a, *b), (*b, *c), (*c, *a)] {
            *edge_count
                .entry(crate::segment::edge_key(x, y))
                .or_insert(0) += 1;
        }
    }
    let non_manifold = edge_count.values().filter(|&&n| n > 2).count();
    if non_manifold > 0 {
        rep.push(
            Severity::Warning,
            "NonManifoldEdges",
            format!("{non_manifold} edges are shared by more than two faces (twin linking keeps the first pair)"),
        );
    }
    let boundary = edge_count.values().filter(|&&n| n == 1).count();
    if boundary > 0 {
        rep.push(
            Severity::Warning,
            "OpenBoundary",
            format!("{boundary} boundary edges (an open surface; treated as mesh boundary by segmentation)"),
        );
    }
    let mut referenced = vec![false; mesh.positions.len()];
    for f in &mesh.faces {
        for &v in f {
            referenced[v as usize] = true;
        }
    }
    let isolated = referenced.iter().filter(|&&r| !r).count();
    if isolated > 0 {
        rep.push(
            Severity::Warning,
            "IsolatedVertices",
            format!("{isolated} vertices are referenced by no face"),
        );
    }
    rep
}

/// Validate a finished atlas (islands + placements).
pub fn validate_atlas(islands: &[Island], placed: &[Option<Placed>]) -> Report {
    let mut rep = Report::default();
    let unplaced: Vec<usize> = placed
        .iter()
        .enumerate()
        .filter(|(_, p)| p.is_none())
        .map(|(i, _)| i)
        .collect();
    if !unplaced.is_empty() {
        rep.push(
            Severity::Error,
            "UnplacedIslands",
            format!(
                "{} island(s) could not be placed ({})",
                unplaced.len(),
                unplaced
                    .iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        );
    }
    for (i, isl) in islands.iter().enumerate() {
        let mut outside = 0usize;
        for uv in &isl.uv {
            if !uv.u.is_finite() || !uv.v.is_finite() {
                rep.push(
                    Severity::Error,
                    "NonFiniteUv",
                    format!("island {i} has non-finite UV coordinates"),
                );
                outside = usize::MAX;
                break;
            }
            if !(0.0..=1.0).contains(&uv.u) || !(0.0..=1.0).contains(&uv.v) {
                outside += 1;
            }
        }
        if outside > 0 && outside != usize::MAX {
            rep.push(
                Severity::Error,
                "UvOutsideUnitSquare",
                format!("island {i} has {outside} UV coordinates outside [0,1]²"),
            );
        }
        // Flipped triangles: UV winding inconsistent with the island's
        // dominant winding. LSCM's null space includes reflections, so a
        // uniformly CW island is a valid orientation choice — only triangles
        // that disagree with the island majority (local inversions) are flips.
        let dominant = {
            let mut sum = 0.0f64;
            for [a, b, c] in &isl.tris {
                let u = isl.uv[*a as usize];
                let v = isl.uv[*b as usize];
                let w = isl.uv[*c as usize];
                sum += (v.u - u.u) * (w.v - u.v) - (w.u - u.u) * (v.v - u.v);
            }
            if sum < 0.0 { -1.0 } else { 1.0 }
        };
        let flips = isl
            .tris
            .iter()
            .filter(|[a, b, c]| {
                let u = isl.uv[*a as usize];
                let v = isl.uv[*b as usize];
                let w = isl.uv[*c as usize];
                ((v.u - u.u) * (w.v - u.v) - (w.u - u.u) * (v.v - u.v)) * dominant <= 0.0
            })
            .count();
        if flips > 0 {
            rep.push(
                Severity::Warning,
                "FlippedTriangles",
                format!("island {i} has {flips} flipped or degenerate UV triangles"),
            );
        }
    }
    // Pairwise overlap of the placed boxes (the atlas-level sanity check; the
    // island packer also performs polygon-level validation).
    for i in 0..placed.len() {
        let Some(a) = &placed[i] else { continue };
        for j in (i + 1)..placed.len() {
            let Some(b) = &placed[j] else { continue };
            let sep_x = a.x + a.w <= b.x + 1e-9 || b.x + b.w <= a.x + 1e-9;
            let sep_y = a.y + a.h <= b.y + 1e-9 || b.y + b.h <= a.y + 1e-9;
            if !(sep_x || sep_y) {
                rep.push(
                    Severity::Error,
                    "OverlappingIslands",
                    format!("islands {i} and {j} overlap in the atlas"),
                );
            }
        }
    }
    rep
}

#[cfg(test)]
mod tests {
    use super::*;
    use researchuv_math::{Vec2, Vec3};

    fn tri_mesh() -> SurfaceMesh {
        SurfaceMesh::from_triangles(
            vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            vec![[0, 1, 2]],
        )
    }

    #[test]
    fn empty_input_is_an_error() {
        let rep = validate_raw(&[], &[]);
        assert!(rep.has_errors());
        let codes: Vec<&str> = rep.errors().map(|f| f.code).collect();
        assert!(codes.contains(&"NoVertices"));
    }

    #[test]
    fn out_of_range_index_is_an_error() {
        let p = vec![Vec3::new(0.0, 0.0, 0.0)];
        let rep = validate_raw(&p, &[[0, 1, 2]]);
        assert_eq!(rep.errors().next().unwrap().code, "IndexOutOfRange");
    }

    #[test]
    fn nan_position_is_an_error() {
        let p = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(f64::NAN, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let rep = validate_raw(&p, &[[0, 1, 2]]);
        assert_eq!(rep.errors().next().unwrap().code, "NonFinitePositions");
    }

    #[test]
    fn clean_closed_mesh_reports_nothing() {
        let p = vec![
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, 0.0),
        ];
        let f: Vec<[u32; 3]> = vec![[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]];
        let m = SurfaceMesh::from_triangles(p, f);
        let rep = validate_mesh(&m);
        assert!(rep.is_ok(), "{:?}", rep.findings);
    }

    #[test]
    fn open_surface_warns_about_boundary() {
        let rep = validate_mesh(&tri_mesh());
        // from_triangles (no weld) → single triangle has 3 boundary edges.
        let codes: Vec<&str> = rep.warnings().map(|f| f.code).collect();
        assert!(codes.contains(&"OpenBoundary"));
    }

    #[test]
    fn isolated_vertices_are_reported() {
        let mut m = tri_mesh();
        m.positions.push(Vec3::new(9.0, 9.0, 9.0));
        let rep = validate_mesh(&m);
        let codes: Vec<&str> = rep.warnings().map(|f| f.code).collect();
        assert!(codes.contains(&"IsolatedVertices"));
    }

    #[test]
    fn unplaced_and_overlapping_islands_are_errors() {
        let isl = Island {
            positions: vec![Vec3::new(0.0, 0.0, 0.0); 3],
            uv: vec![Vec2::new(0.1, 0.1), Vec2::new(0.4, 0.1), Vec2::new(0.1, 0.4)],
            tris: vec![[0, 1, 2]],
            source_vertex_ids: vec![0, 1, 2],
            border: vec![],
        };
        let placed = vec![
            Some(Placed { x: 0.0, y: 0.0, w: 0.5, h: 0.5 }),
            None,
        ];
        let rep = validate_atlas(&[isl.clone(), isl], &placed);
        let codes: Vec<&str> = rep.errors().map(|f| f.code).collect();
        assert!(codes.contains(&"UnplacedIslands"));
    }

    #[test]
    fn overlapping_boxes_are_detected() {
        let isl = Island {
            positions: vec![Vec3::new(0.0, 0.0, 0.0); 3],
            uv: vec![Vec2::new(0.0, 0.0), Vec2::new(0.5, 0.0), Vec2::new(0.0, 0.5)],
            tris: vec![[0, 1, 2]],
            source_vertex_ids: vec![0, 1, 2],
            border: vec![],
        };
        let placed = vec![
            Some(Placed { x: 0.0, y: 0.0, w: 0.6, h: 0.6 }),
            Some(Placed { x: 0.5, y: 0.0, w: 0.5, h: 0.5 }),
        ];
        let rep = validate_atlas(&[isl.clone(), isl], &placed);
        assert_eq!(rep.errors().next().unwrap().code, "OverlappingIslands");
    }

    #[test]
    fn flipped_uv_triangles_warn() {
        // A square of two triangles: one consistent with the dominant
        // winding, one inverted against it (a genuine local flip). A
        // uniformly CW island is NOT flagged (a valid LSCM reflection).
        let uv = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(0.5, 0.0),
            Vec2::new(0.0, 0.5),
            Vec2::new(0.5, 0.5),
        ];
        let isl = Island {
            positions: vec![Vec3::new(0.0, 0.0, 0.0); 4],
            uv,
            tris: vec![[0, 1, 2], [1, 2, 3]], // second is CW against CCW majority
            source_vertex_ids: vec![0, 1, 2, 3],
            border: vec![],
        };
        let rep = validate_atlas(&[isl], &[Some(Placed { x: 0.0, y: 0.0, w: 0.5, h: 0.5 })]);
        assert_eq!(rep.warnings().next().unwrap().code, "FlippedTriangles");
        // Uniformly reversed winding: not a flip.
        let isl2 = Island {
            positions: vec![Vec3::new(0.0, 0.0, 0.0); 3],
            uv: vec![Vec2::new(0.0, 0.0), Vec2::new(0.5, 0.0), Vec2::new(0.25, 0.5)],
            tris: vec![[0, 2, 1]], // whole island CW — consistent
            source_vertex_ids: vec![0, 1, 2],
            border: vec![],
        };
        let rep2 = validate_atlas(&[isl2], &[Some(Placed { x: 0.0, y: 0.0, w: 0.5, h: 0.5 })]);
        assert!(rep2.warnings().count() == 0, "{:?}", rep2.findings);
    }
}
