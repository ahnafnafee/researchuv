//! Segmentation / charting — `CTaskCut` + `CQuasiDevelopable::MergeChart`.
//!
//! Evidence:
//! - Seam selection: `CTaskCut` cuts edges whose face dihedral exceeds
//!   `Auto.SharpEdges.AngleMin` (`Vars.AutoSelect.SharpEdges.Angle`, default 30°)
//!   (ALGORITHMS.md §1.3, `face_dihedrals` in the reference).
//! - Charting: charts are the face components over non-cut edges
//!   (reference `build_charts`, BFS). Chart merging (`CQuasiDevelopable::MergeChart`,
//!   union-find, VA `0x1404C24B0`) is exposed as [`merge_charts`] for the
//!   quasi-developable pass.
//! - Border loops: the "keep-going-straight" boundary walk (reference
//!   `chart_border_loops`) — the input to the border-pin "Density" pass.

use researchuv_core::model::SurfaceMesh;
use researchuv_math::Vec3;
use std::collections::{BTreeMap, BTreeSet};

/// Undirected edge key (min, max vertex ids).
pub type EdgeKey = (u32, u32);

#[inline]
pub fn edge_key(a: u32, b: u32) -> EdgeKey {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

/// One chart (face component) produced by segmentation.
#[derive(Clone, Debug)]
pub struct Chart {
    /// Source-mesh face indices in this chart.
    pub face_ids: Vec<usize>,
    /// Source-mesh vertex indices (sorted) used by this chart.
    pub vertex_ids: Vec<u32>,
    /// Triangle connectivity in local (0..vertex_ids.len()) indices.
    pub tris: Vec<[u32; 3]>,
    /// Border vertex ids (sorted): vertices on cut or mesh-boundary edges.
    pub border: Vec<u32>,
    /// Ordered closed boundary loops (source vertex ids; the walk closes on itself,
    /// so the first id is repeated as the last when the loop is closed).
    pub border_loops: Vec<Vec<u32>>,
    /// Total 3-D triangle area of the chart.
    pub area3d: f64,
}

impl Chart {
    /// Borderless ⇔ closed chart (no boundary edges at all).
    pub fn is_borderless(&self) -> bool {
        self.border_loops.is_empty()
    }

    /// Local index of a source vertex id (or None if not in the chart).
    pub fn local_of(&self, v: u32) -> Option<usize> {
        self.vertex_ids.binary_search(&v).ok()
    }
}

/// Per-face 3-D area (0.5 · ‖(b−a)×(c−a)‖ — the reference `face_areas3d`).
pub fn face_areas3d(mesh: &SurfaceMesh) -> Vec<f64> {
    mesh.faces
        .iter()
        .map(|[a, b, c]| {
            let e1 = mesh.positions[*b as usize] - mesh.positions[*a as usize];
            let e2 = mesh.positions[*c as usize] - mesh.positions[*a as usize];
            0.5 * e1.cross_v(e2).len()
        })
        .collect()
}

/// Per-undirected-edge face list (edge → the faces that carry it).
pub fn edge_faces(mesh: &SurfaceMesh) -> BTreeMap<EdgeKey, Vec<usize>> {
    let mut m: BTreeMap<EdgeKey, Vec<usize>> = BTreeMap::new();
    for (fi, [a, b, c]) in mesh.faces.iter().enumerate() {
        for (x, y) in [(*a, *b), (*b, *c), (*c, *a)] {
            m.entry(edge_key(x, y)).or_default().push(fi);
        }
    }
    m
}

/// Dihedral angle per undirected edge (radians) — the SharpEdges input.
/// Boundary edges (1 face) → 0.0; non-manifold (>2 faces) → π.
pub fn face_dihedrals(mesh: &SurfaceMesh) -> BTreeMap<EdgeKey, f64> {
    let ef = edge_faces(mesh);
    ef.iter()
        .map(|(e, fys)| {
            let ang = match fys.len() {
                1 => 0.0,
                2 => {
                    let n1 = face_normal_unit(mesh, fys[0]);
                    let n2 = face_normal_unit(mesh, fys[1]);
                    let d = n1.dot(n2).clamp(-1.0, 1.0);
                    d.acos()
                }
                _ => std::f64::consts::PI,
            };
            (*e, ang)
        })
        .collect()
}

fn face_normal_unit(mesh: &SurfaceMesh, fi: usize) -> Vec3 {
    let [a, b, c] = mesh.faces[fi];
    let n = Vec3::cross(
        mesh.positions[a as usize],
        mesh.positions[b as usize],
        mesh.positions[c as usize],
    );
    let l = n.len();
    if l < 1e-30 {
        Vec3::new(0.0, 0.0, 1.0)
    } else {
        n / l
    }
}

/// Mesh-boundary edges: edges carried by exactly one face.
pub fn mesh_boundary_edges(mesh: &SurfaceMesh) -> BTreeSet<EdgeKey> {
    edge_faces(mesh)
        .iter()
        .filter(|(_, fys)| fys.len() == 1)
        .map(|(e, _)| *e)
        .collect()
}

/// Segment the mesh at `angle_min_deg`: cut edges with dihedral above the
/// threshold, then build the face-component charts.
pub fn segment(mesh: &SurfaceMesh, angle_min_deg: f64) -> (Vec<Chart>, BTreeSet<EdgeKey>) {
    let th = angle_min_deg.to_radians();
    let dihs = face_dihedrals(mesh);
    let cut: BTreeSet<EdgeKey> = dihs
        .iter()
        .filter(|(_, d)| **d > th)
        .map(|(e, _)| *e)
        .collect();
    let charts = build_charts(mesh, &cut);
    (charts, cut)
}

/// Build the face-component charts over non-cut edges (reference `build_charts`).
pub fn build_charts(mesh: &SurfaceMesh, cut: &BTreeSet<EdgeKey>) -> Vec<Chart> {
    let ef = edge_faces(mesh);
    let mbnd = mesh_boundary_edges(mesh);
    let areas = face_areas3d(mesh);
    let nf = mesh.faces.len();
    let mut seen = vec![false; nf];
    let mut charts = Vec::new();
    for fi in 0..nf {
        if seen[fi] {
            continue;
        }
        // BFS over non-cut edges.
        let mut stack = vec![fi];
        seen[fi] = true;
        let mut comp: Vec<usize> = Vec::new();
        while let Some(f) = stack.pop() {
            comp.push(f);
            let [a, b, c] = mesh.faces[f];
            for (x, y) in [(a, b), (b, c), (c, a)] {
                let key = edge_key(x, y);
                if cut.contains(&key) {
                    continue;
                }
                if let Some(gs) = ef.get(&key) {
                    for &g in gs {
                        if !seen[g] {
                            seen[g] = true;
                            stack.push(g);
                        }
                    }
                }
            }
        }
        // Vertex set (sorted) + local triangles.
        let mut vset: BTreeSet<u32> = BTreeSet::new();
        for &f in &comp {
            for &v in &mesh.faces[f] {
                vset.insert(v);
            }
        }
        let vertex_ids: Vec<u32> = vset.iter().copied().collect();
        let vid: BTreeMap<u32, u32> =
            vertex_ids.iter().enumerate().map(|(i, &v)| (v, i as u32)).collect();
        let tris: Vec<[u32; 3]> = comp
            .iter()
            .map(|&f| {
                let [a, b, c] = mesh.faces[f];
                [
                    *vid.get(&a).unwrap(),
                    *vid.get(&b).unwrap(),
                    *vid.get(&c).unwrap(),
                ]
            })
            .collect();
        // Border vertices: on cut or mesh-boundary edges.
        let mut bnd: BTreeSet<u32> = BTreeSet::new();
        for &f in &comp {
            let [a, b, c] = mesh.faces[f];
            for (x, y) in [(a, b), (b, c), (c, a)] {
                let key = edge_key(x, y);
                if cut.contains(&key) || mbnd.contains(&key) {
                    bnd.insert(x);
                    bnd.insert(y);
                }
            }
        }
        let border: Vec<u32> = bnd.iter().copied().collect();
        // Border loops (keep-going-straight walk).
        let border_loops = chart_border_loops(mesh, &comp, cut, &mbnd, &vid);
        let area3d: f64 = comp.iter().map(|&f| areas[f]).sum();
        charts.push(Chart {
            face_ids: comp,
            vertex_ids,
            tris,
            border,
            border_loops,
            area3d,
        });
    }
    charts
}

/// Keep-going-straight boundary walk (reference `chart_border_loops`).
/// Returns closed loops (first id repeated at the end) of ≥ 4 vertices.
pub fn chart_border_loops(
    mesh: &SurfaceMesh,
    comp: &[usize],
    cut: &BTreeSet<EdgeKey>,
    mbnd: &BTreeSet<EdgeKey>,
    vid: &BTreeMap<u32, u32>,
) -> Vec<Vec<u32>> {
    // Boundary edge set restricted to the chart's vertices.
    let mut bnd_edges: BTreeSet<EdgeKey> = BTreeSet::new();
    for &f in comp {
        let [a, b, c] = mesh.faces[f];
        for (x, y) in [(a, b), (b, c), (c, a)] {
            let key = edge_key(x, y);
            if cut.contains(&key) || mbnd.contains(&key) {
                bnd_edges.insert(key);
            }
        }
    }
    if bnd_edges.is_empty() {
        return Vec::new();
    }
    // Adjacency over chart-local vertices (keep ids as source ids for determinism).
    let mut adj: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for &(a, b) in &bnd_edges {
        if vid.contains_key(&a) && vid.contains_key(&b) {
            adj.entry(a).or_default().push(b);
            adj.entry(b).or_default().push(a);
        }
    }
    let mut used: BTreeSet<EdgeKey> = BTreeSet::new();
    let mut loops: Vec<Vec<u32>> = Vec::new();
    let starts: Vec<u32> = adj.keys().copied().collect();
    for start in starts {
        // Start only if it has an unused incident boundary edge.
        let has_unused = adj
            .get(&start)
            .map(|ns| {
                ns.iter().any(|&n| !used.contains(&edge_key(start, n)))
            })
            .unwrap_or(false);
        if !has_unused {
            continue;
        }
        let mut loop_v = vec![start];
        let mut cur = start;
        // First step: smallest candidate (deterministic tie-break).
        let mut cands: Vec<u32> = adj.get(&start).cloned().unwrap_or_default();
        cands.sort_unstable();
        if cands.is_empty() {
            continue;
        }
        let first = cands[0];
        used.insert(edge_key(cur, first));
        let mut prev = Some(cur);
        cur = first;
        loop_v.push(cur);
        'walk: loop {
            let nxts: Vec<u32> = adj
                .get(&cur)
                .map(|ns| {
                    ns.iter()
                        .copied()
                        .filter(|&n| n != prev.unwrap_or(u32::MAX))
                        .filter(|&n| !used.contains(&edge_key(cur, n)))
                        .collect()
                })
                .unwrap_or_default();
            if nxts.is_empty() {
                break;
            }
            // Keep going straight: maximize dot with the incoming direction.
            let d_in = mesh.positions[cur as usize] - mesh.positions[prev.unwrap_or(cur) as usize];
            let ln = d_in.len();
            let (next, best_dot) = if ln < 1e-30 {
                (nxts[0], None)
            } else {
                let din = d_in / ln;
                let mut best: Option<(f64, u32)> = None;
                for &n in &nxts {
                    let dot = (mesh.positions[n as usize] - mesh.positions[cur as usize]).dot(din);
                    match best {
                        Some((bd, _)) if dot <= bd => {}
                        _ => best = Some((dot, n)),
                    }
                }
                let (bd, n) = best.unwrap_or((0.0, nxts[0]));
                (n, Some(bd))
            };
            let _ = best_dot;
            used.insert(edge_key(cur, next));
            let p = prev;
            prev = Some(cur);
            cur = next;
            loop_v.push(cur);
            if cur == start {
                break 'walk;
            }
            let _ = p;
        }
        if loop_v.len() >= 4 {
            loops.push(loop_v);
        }
    }
    loops
}

/// `CQuasiDevelopable::MergeChart` (union-find, VA `0x1404C24B0`) — merge charts
/// whose face normals agree within `cos_tol` (quasi-developable: they flatten to
/// the same tangent plane family). Returns a new chart list.
pub fn merge_charts(mesh: &SurfaceMesh, charts: Vec<Chart>, cos_tol: f64) -> Vec<Chart> {
    if charts.len() <= 1 {
        return charts;
    }
    // Per-chart average normal (area-weighted).
    let normals: Vec<Vec3> = charts
        .iter()
        .map(|ch| {
            let areas = face_areas3d(mesh);
            let mut n = Vec3::new(0.0, 0.0, 0.0);
            for &fi in &ch.face_ids {
                let [a, b, c] = mesh.faces[fi];
                n = n + Vec3::cross(mesh.positions[a as usize], mesh.positions[b as usize], mesh.positions[c as usize]) * areas[fi];
            }
            let l = n.len();
            if l < 1e-30 {
                Vec3::new(0.0, 0.0, 1.0)
            } else {
                n / l
            }
        })
        .collect();
    // Union-find.
    let mut parent: Vec<usize> = (0..charts.len()).collect();
    fn find(p: &mut Vec<usize>, mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    for i in 0..charts.len() {
        for j in (i + 1)..charts.len() {
            let d = normals[i].dot(normals[j]).abs();
            if d >= cos_tol {
                let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
                if ri != rj {
                    parent[rj] = ri;
                }
            }
        }
    }
    // Merge chart data per root.
    let mut by_root: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for i in 0..charts.len() {
        by_root.entry(find(&mut parent, i)).or_default().push(i);
    }
    let mut merged: Vec<Chart> = Vec::new();
    for (_, ids) in by_root {
        let mut faces: Vec<usize> = Vec::new();
        let mut vset: BTreeSet<u32> = BTreeSet::new();
        for &ci in &ids {
            faces.extend(charts[ci].face_ids.iter().copied());
            vset.extend(charts[ci].vertex_ids.iter().copied());
        }
        faces.sort_unstable();
        let vertex_ids: Vec<u32> = vset.iter().copied().collect();
        let vid: BTreeMap<u32, u32> =
            vertex_ids.iter().enumerate().map(|(i, &v)| (v, i as u32)).collect();
        let tris: Vec<[u32; 3]> = faces
            .iter()
            .map(|&f| {
                let [a, b, c] = mesh.faces[f];
                [
                    *vid.get(&a).unwrap(),
                    *vid.get(&b).unwrap(),
                    *vid.get(&c).unwrap(),
                ]
            })
            .collect();
        let border: Vec<u32> = {
            let mut s: BTreeSet<u32> = BTreeSet::new();
            for &ci in &ids {
                s.extend(charts[ci].border.iter().copied());
            }
            s.into_iter().collect()
        };
        let border_loops: Vec<Vec<u32>> = ids
            .iter()
            .flat_map(|&ci| charts[ci].border_loops.iter().cloned())
            .collect();
        let area3d: f64 = ids.iter().map(|&ci| charts[ci].area3d).sum();
        merged.push(Chart {
            face_ids: faces,
            vertex_ids,
            tris,
            border,
            border_loops,
            area3d,
        });
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 2×2 grid of unit squares in the xy-plane (16 tris, 9 verts) — flat, no cuts.
    fn flat_grid() -> SurfaceMesh {
        let mut p: Vec<Vec3> = Vec::new();
        for j in 0..3 {
            for i in 0..3 {
                p.push(Vec3::new(i as f64, j as f64, 0.0));
            }
        }
        let mut f: Vec<[u32; 3]> = Vec::new();
        for j in 0..2 {
            for i in 0..2 {
                let a = (j * 3 + i) as u32;
                let b = a + 1;
                let c = a + 3;
                let d = c + 1;
                f.push([a, b, d]);
                f.push([a, d, c]);
            }
        }
        SurfaceMesh::from_triangles(p, f)
    }

    #[test]
    fn flat_grid_is_one_chart_no_border() {
        let m = flat_grid();
        let (charts, cut) = segment(&m, 30.0);
        assert!(cut.is_empty());
        assert_eq!(charts.len(), 1);
        assert_eq!(charts[0].face_ids.len(), 8);
        assert_eq!(charts[0].vertex_ids.len(), 9);
        // A single open sheet: the outer 8 perimeter vertices form one border loop;
        // `is_borderless` means a *closed* chart (no boundary edges at all).
        assert_eq!(charts[0].border.len(), 8);
        assert_eq!(charts[0].border_loops.len(), 1);
        assert_eq!(charts[0].border_loops[0].len(), 9); // 8 verts + closing repeat
    }

    #[test]
    fn open_flat_grid_has_one_border_loop() {
        // Single open square (4 tris? no — 2 tris) with a real mesh boundary.
        let p = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let f = vec![[0u32, 1, 2]];
        let m = SurfaceMesh::from_triangles(p, f);
        let (charts, _) = segment(&m, 30.0);
        assert_eq!(charts.len(), 1);
        assert_eq!(charts[0].border_loops.len(), 1, "open triangle has one border loop");
        assert_eq!(charts[0].border_loops[0].len(), 4); // closed: 3 verts + repeat
        assert_eq!(charts[0].border_loops[0][0], charts[0].border_loops[0][3]);
        assert!(!charts[0].is_borderless(), "an open sheet has a border");
    }

    #[test]
    fn closed_sphere_like_chart_is_borderless() {
        // Two triangles sharing the opposite edges form a closed 2-vertex "edge loop"
        // (a degenerate sphere) — no boundary edges at all ⇒ borderless chart.
        // Use a tetrahedron: 4 faces, every edge shared by exactly 2 faces.
        let p = vec![
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, 0.0),
        ];
        let f: Vec<[u32; 3]> = vec![[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]];
        let m = SurfaceMesh::from_triangles(p, f);
        // A regular tetrahedron's face-normal angle is 180° − 54.7356° = 125.26°,
        // so any threshold above it cuts no edge and the closed surface stays
        // one chart.
        let (charts, cut) = segment(&m, 130.0);
        assert!(cut.is_empty());
        assert_eq!(charts.len(), 1);
        assert!(charts[0].is_borderless(), "closed tetrahedron has no border");
        assert_eq!(charts[0].border.len(), 0);
    }
}
