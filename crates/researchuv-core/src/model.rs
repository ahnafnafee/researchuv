//! Half-edge mesh topology and UV island data.
//!
//! The source surface stores directed edges so segmentation can traverse face
//! adjacency. Each half-edge exposes its origin, destination, next, and previous
//! indices; islands retain the connection between source vertices and UVs.

use researchuv_math::{Vec2, Vec3};
use std::collections::HashMap;

/// `PRIMTYPES` — primitive kinds tracked by the topology (`CMultiMesh::InvariantTest`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrimType {
    Point,
    Line,
    Triangle,
    Quad,
    Polygon,
}

/// `IWS` — Interactive Working Set: the selection/editing mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkingSet {
    Vertices,
    Edges,
    Faces,
    Islands,
}

/// A directed half-edge — the `CEdge`/`CTri` edge record.
#[derive(Clone, Copy, Debug)]
pub struct Halfedge {
    /// `CEdge::OrigID()` — tail vertex index.
    pub orig: u32,
    /// `CEdge::DestID()` — head vertex index.
    pub dest: u32,
    /// Twin (opposite) half-edge, if the edge is manifold (shared by two faces).
    pub twin: Option<u32>,
    /// Face this half-edge bounds.
    pub face: u32,
    /// `CEdge::NextID()` — next half-edge around the face.
    pub next: u32,
    /// `CEdge::PrevID()` — previous half-edge around the face.
    pub prev: u32,
}

impl Halfedge {
    #[inline]
    pub fn orig_id(&self) -> u32 {
        self.orig
    }
    #[inline]
    pub fn dest_id(&self) -> u32 {
        self.dest
    }
    #[inline]
    pub fn next_id(&self) -> u32 {
        self.next
    }
    #[inline]
    pub fn prev_id(&self) -> u32 {
        self.prev
    }
}

/// A single connected triangle mesh — `CMesh` (the source model). Charts/islands are produced by
/// the segmentation stage in `researchuv-unwrap` from this structure's face adjacency.
#[derive(Clone, Debug, Default)]
pub struct SurfaceMesh {
    /// Vertex pool — `CVert[]` positions.
    pub positions: Vec<Vec3>,
    /// Face pool — `CTri/CPoly[]` as vertex-index triples into [`SurfaceMesh::positions`].
    pub faces: Vec<[u32; 3]>,
    /// Half-edge pool — `CEdge[]`, three per face, laid out as `h = 3*face + k`.
    pub halfedges: Vec<Halfedge>,
}

impl SurfaceMesh {
    /// Build a manifold half-edge structure from positions + triangles ("winged construction").
    /// Non-manifold edges (a vertex pair shared by >2 faces) keep the first twin pair only.
    pub fn from_triangles(positions: Vec<Vec3>, faces: Vec<[u32; 3]>) -> Self {
        let nh = faces.len() * 3;
        let mut halfedges = vec![
            Halfedge { orig: 0, dest: 0, twin: None, face: 0, next: 0, prev: 0 };
            nh
        ];
        // First pass: fill directed edges + per-face cyclic next/prev.
        for (f, [a, b, c]) in faces.iter().enumerate() {
            let tri = [*a, *b, *c];
            let base = 3 * f;
            for k in 0..3 {
                let h = base + k;
                halfedges[h].orig = tri[k];
                halfedges[h].dest = tri[(k + 1) % 3];
                halfedges[h].face = f as u32;
                halfedges[h].next = (base + (k + 1) % 3) as u32;
                halfedges[h].prev = (base + (k + 2) % 3) as u32;
            }
        }
        // Second pass: link twins via an undirected-edge index.
        let mut seen: HashMap<(u32, u32), u32> = HashMap::with_capacity(nh / 2);
        for h in 0..nh {
            let key = (halfedges[h].orig.min(halfedges[h].dest), halfedges[h].orig.max(halfedges[h].dest));
            match seen.get(&key) {
                Some(&other) => {
                    halfedges[h].twin = Some(other as u32);
                    halfedges[other as usize].twin = Some(h as u32);
                }
                None => {
                    seen.insert(key, h as u32);
                }
            }
        }
        Self { positions, faces, halfedges }
    }

    /// The three half-edges bounding face `f` (as `CTri::DoGetVertID`/`SetEdgeID` implies).
    #[inline]
    pub fn halfedges_of_face(&self, f: usize) -> [u32; 3] {
        [3 * f as u32, 3 * f as u32 + 1, 3 * f as u32 + 2]
    }

    /// Unit face normal for face `f` (same `cross(a−b, c−b)` convention as [`researchuv_math::compute_geo_tri2`]).
    pub fn face_normal(&self, f: usize) -> Vec3 {
        let [a, b, c] = self.faces[f];
        Vec3::cross(self.positions[a as usize], self.positions[b as usize], self.positions[c as usize])
    }

    /// Pairs of faces sharing a manifold edge — the input to seam selection / charting.
    /// Yields `(faceA, faceB)` once per shared edge (deduplicated over twins).
    pub fn adjacent_faces(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for (i, he) in self.halfedges.iter().enumerate() {
            if let Some(t) = he.twin {
                // Emit each adjacency once (only when this is the "lower" half-edge).
                if (i as u32) < t {
                    let a = he.face as usize;
                    let b = self.halfedges[t as usize].face as usize;
                    out.push((a, b));
                }
            }
        }
        out
    }

    /// `CMultiMesh::InvariantTest` — structural self-consistency check.
    pub fn invariant_test(&self) -> Result<(), String> {
        if self.halfedges.len() != self.faces.len() * 3 {
            return Err(format!("halfedge count {} != 3*faces {}", self.halfedges.len(), self.faces.len()));
        }
        for (i, he) in self.halfedges.iter().enumerate() {
            if self.halfedges[he.next as usize].prev as usize != i {
                return Err(format!("halfedge {i}: next/prev not reciprocal"));
            }
            if let Some(t) = he.twin {
                if self.halfedges[t as usize].twin != Some(i as u32) {
                    return Err(format!("halfedge {i}: twin not symmetric"));
                }
            }
        }
        Ok(())
    }
}

/// An unwrapped island — a `CSubMesh` carrying the UV coordinate set produced by the pipeline.
///
/// The source mesh is split into islands by seam cutting (`CTaskCut`); each island is flattened
/// to 2-D by the LSCM solve (`CIsomap` / `CTaskUnfold`), then optionally optimized
/// (`CTaskOptimize`) and constrained (`CTaskConstrain`) before packing (`CTaskPack`).
///
/// All index sets here are local to the island (into `positions`/`uv`); `source_vertex_ids`
/// maps back to the source mesh so UVs can be written back onto the imported surface.
#[derive(Clone, Debug)]
pub struct Island {
    /// 3-D positions of the island's vertices (subset of the source mesh).
    pub positions: Vec<Vec3>,
    /// 2-D UV coordinates, one per vertex in `positions` (zeros until unfolded).
    pub uv: Vec<Vec2>,
    /// Triangle connectivity — indices into `positions`/`uv`.
    pub tris: Vec<[u32; 3]>,
    /// Source-mesh vertex index for each local vertex (identity mapping on import).
    pub source_vertex_ids: Vec<u32>,
    /// Border vertex indices in boundary-loop order (empty ⇒ closed/borderless chart,
    /// e.g. the annulus component of the torus test case).
    pub border: Vec<u32>,
}

impl Island {
    /// Borderless ⇔ closed chart (no boundary edges).
    pub fn is_borderless(&self) -> bool {
        self.border.is_empty()
    }

    /// Signed UV area of all triangles (positive when the chart's winding is consistent).
    pub fn uv_area(&self) -> f64 {
        let mut a = 0.0;
        for [i, j, k] in &self.tris {
            let u = self.uv[*i as usize];
            let v = self.uv[*j as usize];
            let w = self.uv[*k as usize];
            a += (v.u - u.u) * (w.v - u.v) - (w.u - u.u) * (v.v - u.v);
        }
        a * 0.5
    }

    /// 3-D area of all triangles (for scale/area-ratio metrics).
    pub fn space_area(&self) -> f64 {
        let mut a = 0.0;
        for [i, j, k] in &self.tris {
            a += Vec3::cross(
                self.positions[*i as usize],
                self.positions[*j as usize],
                self.positions[*k as usize],
            )
            .len()
            * 0.5;
        }
        a
    }
}

/// `CMultiMesh` — the top-level multi-mesh document: the imported (welded) source surface
/// plus the islands the unwrap pipeline has produced. `CSubMesh` = island; `CMesh` = source.
#[derive(Clone, Debug)]
pub struct MultiMesh {
    /// The imported, welded source mesh.
    pub source: SurfaceMesh,
    /// Unwrapped islands produced by the pipeline.
    pub islands: Vec<Island>,
}

impl MultiMesh {
    pub fn new(source: SurfaceMesh) -> Self {
        Self {
            source,
            islands: Vec::new(),
        }
    }

    /// `CMultiMesh::InvariantTest` — source plus every island must be structurally sound.
    pub fn invariant_test(&self) -> Result<(), String> {
        self.source.invariant_test()?;
        for (i, isl) in self.islands.iter().enumerate() {
            if isl.positions.len() != isl.uv.len() {
                return Err(format!("island {i}: uv count {} != positions {}", isl.uv.len(), isl.positions.len()));
            }
            for [a, b, c] in &isl.tris {
                if *a as usize >= isl.positions.len() || *b as usize >= isl.positions.len() || *c as usize >= isl.positions.len() {
                    return Err(format!("island {i}: triangle index out of range"));
                }
            }
            if let Some(last) = isl.border.last() {
                if *last as usize >= isl.positions.len() {
                    return Err(format!("island {i}: border index out of range"));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tetra() -> SurfaceMesh {
        let p = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.5, 1.0, 0.0),
            Vec3::new(0.5, 0.5, 1.0),
        ];
        let f = vec![[0u32, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]];
        SurfaceMesh::from_triangles(p, f)
    }

    #[test]
    fn halfedge_invariant_holds_for_closed_tetrahedron() {
        let m = tetra();
        assert!(m.invariant_test().is_ok());
        // A closed tetrahedron has 6 edges → 6 twin pairs → 12 directed half-edges.
        assert_eq!(m.halfedges.len(), 12);
        assert!(m.halfedges.iter().all(|h| h.twin.is_some()));
    }

    #[test]
    fn adjacency_deduplicated() {
        let m = tetra();
        // A tetrahedron has 6 edges, each shared by exactly two faces.
        assert_eq!(m.adjacent_faces().len(), 6);
    }
}
