//! Seam selection for closed surfaces — extra cuts that open borderless
//! charts so the unfold driver can pin a real border instead of sparse
//! anchors.
//!
//! A closed chart (sphere, torus, any surface kept whole by the sharp-edge
//! threshold) has no boundary loop: [`crate::lscm`] falls back to four
//! tangent-frame anchors, and the free-boundary conformal solve collapses
//! area (the sphere parity fixture reports a mean area ratio of 0.015 — 98%
//! of the texture space vanishes). Cutting the surface open gives the chart
//! a boundary, and the documented border "Density" pin pass then drives the
//! solve: area ratio recovers to ~1.2–1.6 and the angular error drops with
//! it.
//!
//! One *slit* is not enough: a slit of length `2L` pinned onto a rectangle
//! can enclose at most `L²/π` of area (isoperimetric bound), so a single
//! geodesic-diameter path leaves the pin system over-constrained. The seam is
//! therefore a **tree**: a geodesic-diameter path (two Dijkstra sweeps, the
//! classic farthest-point 2-approximation) plus `branch_count` branches, each
//! grown from the vertex farthest from the current seam back to the seam
//! (multi-source Dijkstra). A tree seam turns a sphere into a disk (one
//! border loop) and a torus into an annulus-like chart the two-loop border
//! pass handles natively.

use crate::segment::{build_charts, Chart, EdgeKey};
use researchuv_core::model::SurfaceMesh;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};

/// Seam-cut options (documented task parameters).
#[derive(Clone, Copy, Debug)]
pub struct SeamCutOptions {
    /// Cut borderless charts open along a seam tree.
    ///
    /// Off by default: the reference pipeline keeps closed charts whole
    /// (anchor pins), and the parity fixtures record that behavior.
    pub enable: bool,
    /// Cut only charts with at least this many faces (small closed charts —
    /// e.g. a tetrahedron — keep their anchors).
    pub min_faces: usize,
    /// Extra seam branches beyond the geodesic-diameter path. `0` is a
    /// single slit; `1`–`2` branches relieve the isoperimetric
    /// over-constraint of the border pins (see the module docs).
    pub branch_count: usize,
}

impl Default for SeamCutOptions {
    fn default() -> Self {
        Self { enable: false, min_faces: 8, branch_count: 1 }
    }
}

/// A (distance, vertex) pair for the Dijkstra heap; inverted so the binary
/// heap — a max-heap — pops the smallest distance first. Vertex ids break
/// ties for determinism.
#[derive(Clone, Copy, PartialEq)]
struct HeapItem(f64, u32);
impl Eq for HeapItem {}
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .0
            .partial_cmp(&self.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| self.1.cmp(&other.1))
    }
}

/// Multi-source Dijkstra over the chart's vertex graph (edge weight = 3-D
/// edge length) from `seeds`. Returns `(dist, prev)` keyed by source vertex
/// id; unreachable vertices keep `f64::INFINITY`.
fn dijkstra(
    adj: &BTreeMap<u32, Vec<(u32, f64)>>,
    seeds: &[u32],
) -> (BTreeMap<u32, f64>, BTreeMap<u32, u32>) {
    let mut dist: BTreeMap<u32, f64> = BTreeMap::new();
    let mut prev: BTreeMap<u32, u32> = BTreeMap::new();
    let mut heap = BinaryHeap::new();
    for &s in seeds {
        dist.insert(s, 0.0);
        heap.push(HeapItem(0.0, s));
    }
    while let Some(HeapItem(d, v)) = heap.pop() {
        match dist.get(&v) {
            Some(&seen) if seen < d - 1e-12 => continue,
            _ => {}
        }
        if let Some(nbrs) = adj.get(&v) {
            for &(w, len) in nbrs {
                let nd = d + len;
                let better = match dist.get(&w) {
                    Some(&dw) => nd < dw - 1e-12,
                    None => true,
                };
                if better {
                    dist.insert(w, nd);
                    prev.insert(w, v);
                    heap.push(HeapItem(nd, w));
                }
            }
        }
    }
    (dist, prev)
}

/// The reachable vertex with the largest finite distance, excluding the
/// vertices in `skip`. Deterministic: the smallest vertex id wins ties.
fn farthest_excluding(dist: &BTreeMap<u32, f64>, skip: &BTreeSet<u32>) -> Option<u32> {
    let mut best: Option<(u32, f64)> = None;
    for (&v, &d) in dist {
        if !d.is_finite() || skip.contains(&v) {
            continue;
        }
        best = match best {
            Some((bv, bd)) => {
                if d > bd || (d == bd && v < bv) {
                    Some((v, d))
                } else {
                    Some((bv, bd))
                }
            }
            None => Some((v, d)),
        };
    }
    best.map(|(v, _)| v)
}

/// Walk the predecessor chain from `target` back to its seed; returns
/// `[seed … target]`.
fn reconstruct(prev: &BTreeMap<u32, u32>, target: u32) -> Vec<u32> {
    let mut path = vec![target];
    let mut cur = target;
    while let Some(&p) = prev.get(&cur) {
        path.push(p);
        cur = p;
    }
    path.reverse();
    path
}

/// Undirected vertex adjacency with 3-D edge lengths over the chart's faces.
fn chart_adjacency(mesh: &SurfaceMesh, chart: &Chart) -> BTreeMap<u32, Vec<(u32, f64)>> {
    let mut edges: BTreeSet<EdgeKey> = BTreeSet::new();
    for &fi in &chart.face_ids {
        let [a, b, c] = mesh.faces[fi];
        edges.insert(crate::segment::edge_key(a, b));
        edges.insert(crate::segment::edge_key(b, c));
        edges.insert(crate::segment::edge_key(c, a));
    }
    let mut adj: BTreeMap<u32, Vec<(u32, f64)>> = BTreeMap::new();
    for (a, b) in edges {
        let len = (mesh.positions[a as usize] - mesh.positions[b as usize]).len();
        adj.entry(a).or_default().push((b, len));
        adj.entry(b).or_default().push((a, len));
    }
    adj
}

/// The seam tree for one chart as a list of vertex paths (each `[seed … end]`
/// over source vertex ids; empty when the chart is too small or degenerate).
///
/// The first path is the geodesic diameter (seed → A, A → B); each following
/// path branches from the vertex farthest away back to the current tree.
pub fn seam_paths(mesh: &SurfaceMesh, chart: &Chart, branch_count: usize) -> Vec<Vec<u32>> {
    if chart.face_ids.len() < 2 || chart.vertex_ids.is_empty() {
        return Vec::new();
    }
    let adj = chart_adjacency(mesh, chart);
    // Seed: min-x vertex (the anchor_ids convention — first occurrence).
    let &start = chart
        .vertex_ids
        .iter()
        .min_by(|&&a, &&b| {
            let pa = mesh.positions[a as usize];
            let pb = mesh.positions[b as usize];
            pa.x
                .partial_cmp(&pb.x)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        })
        .unwrap();
    let mut paths = Vec::new();
    let mut tree: BTreeSet<u32> = BTreeSet::new();
    // Geodesic-diameter trunk.
    let (dist0, _) = dijkstra(&adj, &[start]);
    let Some(a) = farthest_excluding(&dist0, &tree) else {
        return Vec::new();
    };
    let (dist1, prev) = dijkstra(&adj, &[a]);
    let Some(b) = farthest_excluding(&dist1, &tree) else {
        return Vec::new();
    };
    if a == b {
        return Vec::new();
    }
    let trunk = reconstruct(&prev, b);
    tree.extend(trunk.iter().copied());
    paths.push(trunk);
    // Branches from the farthest uncovered vertex back to the tree.
    for _ in 0..branch_count {
        let seeds: Vec<u32> = tree.iter().copied().collect();
        let (dt, pt) = dijkstra(&adj, &seeds);
        match farthest_excluding(&dt, &tree) {
            Some(f) => {
                let branch = reconstruct(&pt, f);
                if branch.len() < 2 {
                    break;
                }
                tree.extend(branch.iter().copied());
                paths.push(branch);
            }
            None => break,
        }
    }
    paths
}

/// The seam tree's cut-edge set for one chart.
pub fn seam_cut_edges(mesh: &SurfaceMesh, chart: &Chart, branch_count: usize) -> BTreeSet<EdgeKey> {
    let mut cut = BTreeSet::new();
    for path in seam_paths(mesh, chart, branch_count) {
        for w in path.windows(2) {
            cut.insert(crate::segment::edge_key(w[0], w[1]));
        }
    }
    cut
}

/// The single geodesic-diameter slit (the trunk only) — kept for callers that
/// want the minimal cut. Returns `[A … B]` over source vertex ids.
pub fn seam_path(mesh: &SurfaceMesh, chart: &Chart) -> Vec<u32> {
    seam_paths(mesh, chart, 0).into_iter().next().unwrap_or_default()
}

/// A border-crossing chord: the geodesic diameter *between border vertices* —
/// sweep 1 picks the min-x border vertex `A`, sweep 2 finds the farthest
/// border vertex `B` from it, and the predecessor chain is the path. On
/// annulus-like charts the farthest border vertex from one rim lies across
/// the chart on the other rim, so the chord crosses the interior and cutting
/// it separates the chart into two face components — narrower charts pin
/// closer to their true developable width, which is what reduces folding.
/// Returns `[A … B]` over source vertex ids (empty when no crossing exists).
pub fn border_crossing_path(mesh: &SurfaceMesh, chart: &Chart) -> Vec<u32> {
    if chart.border.is_empty() || chart.face_ids.len() < 2 {
        return Vec::new();
    }
    let adj = chart_adjacency(mesh, chart);
    // A: min-x border vertex (the anchor_ids convention — first occurrence).
    let &a = chart
        .border
        .iter()
        .min_by(|&&x, &&y| {
            let px = mesh.positions[x as usize];
            let py = mesh.positions[y as usize];
            px.x.partial_cmp(&py.x).unwrap_or(std::cmp::Ordering::Equal).then(x.cmp(&y))
        })
        .unwrap();
    let (dist, prev) = dijkstra(&adj, &[a]);
    // B: the farthest border vertex from A (smallest id on ties).
    let mut best: Option<(u32, f64)> = None;
    for &v in &chart.border {
        if v == a {
            continue;
        }
        match dist.get(&v) {
            Some(&d) if d.is_finite() => match best {
                Some((_, bd)) if d < bd => {}
                Some((bv, bd)) if d == bd => best = Some((bv.min(v), bd)),
                _ => best = Some((v, d)),
            },
            _ => {}
        }
    }
    match best {
        Some((target, d)) if d > 1e-12 => reconstruct(&prev, target),
        _ => Vec::new(),
    }
}

/// Extend `cut` with seam trees for every borderless chart and rebuild the
/// charts over the combined cut set. Charts below `opts.min_faces` keep their
/// anchors (no cut). Returns the new `(charts, cut)`; the cut set only grows.
pub fn cut_closed_charts(
    mesh: &SurfaceMesh,
    charts: &[Chart],
    cut: &BTreeSet<EdgeKey>,
    opts: SeamCutOptions,
) -> (Vec<Chart>, BTreeSet<EdgeKey>) {
    let mut all = cut.clone();
    for ch in charts {
        if !ch.is_borderless() || ch.face_ids.len() < opts.min_faces {
            continue;
        }
        for e in seam_cut_edges(mesh, ch, opts.branch_count) {
            all.insert(e);
        }
    }
    if all.len() == cut.len() {
        return (charts.to_vec(), all);
    }
    (build_charts(mesh, &all), all)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segment::segment;
    use researchuv_math::Vec3;

    fn tetra() -> SurfaceMesh {
        let p = vec![
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, 0.0),
        ];
        let f: Vec<[u32; 3]> = vec![[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]];
        SurfaceMesh::from_triangles(p, f)
    }

    #[test]
    fn small_closed_chart_is_not_cut() {
        let m = tetra();
        let (charts, cut) = segment(&m, 130.0);
        let opts = SeamCutOptions { enable: true, ..Default::default() };
        let (charts2, cut2) = cut_closed_charts(&m, &charts, &cut, opts);
        assert!(cut2.is_empty(), "4-face tetrahedron is below min_faces");
        assert_eq!(charts2.len(), 1);
        assert!(charts2[0].is_borderless());
    }

    #[test]
    fn closed_sphere_gets_a_border_after_the_cut() {
        // A small welded UV sphere: closed, and large enough for the seam tree
        // to open a real (>= 4-vertex) boundary loop.
        let (p, f) = crate::meshgen::uv_sphere(8, 6);
        let m = crate::weld::weld(p, f, 1e-12);
        let (charts, cut) = segment(&m, 180.0);
        assert_eq!(charts.len(), 1);
        assert!(charts[0].is_borderless());
        let opts = SeamCutOptions { enable: true, min_faces: 4, ..Default::default() };
        let (charts2, cut2) = cut_closed_charts(&m, &charts, &cut, opts);
        assert!(!cut2.is_empty(), "a seam was cut");
        // Cutting a tree on a sphere-like surface opens a disk: exactly one
        // chart, now with a border.
        assert_eq!(charts2.len(), 1);
        assert!(!charts2[0].is_borderless(), "opened chart has a border");
        assert_eq!(charts2[0].border_loops.len(), 1);
        // Every seam path visits distinct vertices (a simple path).
        for path in seam_paths(&m, &charts[0], opts.branch_count) {
            let uniq: BTreeSet<u32> = path.iter().copied().collect();
            assert_eq!(uniq.len(), path.len());
        }
        // The trunk alone spans at least 3 vertices.
        assert!(seam_path(&m, &charts[0]).len() >= 3);
    }

    #[test]
    fn branches_only_grow_the_cut() {
        let m = tetra();
        let (charts, _) = segment(&m, 130.0);
        let c0 = seam_cut_edges(&m, &charts[0], 0);
        let c2 = seam_cut_edges(&m, &charts[0], 2);
        assert!(c2.len() >= c0.len());
    }
}
