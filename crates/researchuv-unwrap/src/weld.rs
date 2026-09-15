//! `CTaskWeld` — weld coincident vertices of an imported triangle soup.
//!
//! Evidence: `CTaskWeld::FilterEdges` at VA `0x1403758B0` (ALGORITHMS.md §1.1).
//! The engine welds vertices by position proximity (optionally by angle,
//! `Vars.AutoSelect.ReWeld*`), then filters out degenerate edges that the merge
//! produced (zero-length / duplicate). The position-weld path (rounding to a
//! tolerance grid) mirrors the reference pipeline's `weld()` and is what the
//! test meshes (cube with per-face vertex grids) require.

use researchuv_core::model::SurfaceMesh;
use researchuv_math::Vec3;
use std::collections::HashMap;

/// Weld `positions`/`faces` into a manifold-consistent [`SurfaceMesh`].
///
/// `tol` is the position tolerance: vertices within `tol` (implemented as a
/// rounding grid, as in the reference) are merged. Faces that collapse to fewer
/// than 3 distinct vertices after welding are dropped (`FilterEdges`).
pub fn weld(positions: Vec<Vec3>, faces: Vec<[u32; 3]>, tol: f64) -> SurfaceMesh {
    let n = positions.len();
    if tol <= 0.0 {
        return SurfaceMesh::from_triangles(positions, faces);
    }
    // Positional weld: canonical key = rounded coordinates.
    let inv = 1.0 / tol;
    let mut canon: HashMap<(i64, i64, i64), u32> = HashMap::new();
    let mut remap = vec![0u32; n];
    let mut welded: Vec<Vec3> = Vec::new();
    for i in 0..n {
        let p = positions[i];
        // `round` to the nearest grid node (matches the reference `np.round`).
        let key = (
            (p.x * inv).round() as i64,
            (p.y * inv).round() as i64,
            (p.z * inv).round() as i64,
        );
        let c = match canon.get(&key) {
            Some(&c) => c,
            None => {
                let c = welded.len() as u32;
                canon.insert(key, c);
                welded.push(positions[i]);
                c
            }
        };
        remap[i] = c;
    }
    // Re-map faces; drop degenerate (collapsed) ones.
    let mut new_faces: Vec<[u32; 3]> = Vec::with_capacity(faces.len());
    for [a, b, c] in &faces {
        let f = [remap[*a as usize], remap[*b as usize], remap[*c as usize]];
        if f[0] != f[1] && f[1] != f[2] && f[0] != f[2] {
            new_faces.push(f);
        }
    }
    SurfaceMesh::from_triangles(welded, new_faces)
}

/// `CTaskWeld::FilterEdges` — the face-removal predicate: true when the welded
/// face still has 3 distinct vertices (i.e. is kept).
pub fn filter_edge_keeps(a: u32, b: u32, c: u32) -> bool {
    a != b && b != c && a != c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn welds_a_cube_face_grid_to_a_shared_manifold() {
        // Two adjacent unit squares in the xy-plane, given as separate vertex
        // grids sharing the seam at x=1 (duplicated columns must be welded).
        let mut p: Vec<Vec3> = Vec::new();
        let mut f: Vec<[u32; 3]> = Vec::new();
        for quad in 0..2 {
            let ox = quad as f64;
            let base = p.len() as u32;
            for j in 0..2 {
                for i in 0..2 {
                    p.push(Vec3::new(ox + i as f64, j as f64, 0.0));
                }
            }
            // 4 corner indices: (0,0),(1,0),(0,1),(1,1)
            let a = base;
            let b = base + 1;
            let c = base + 2;
            let d = base + 3;
            f.push([a, b, d]);
            f.push([a, d, c]);
        }
        let m = weld(p, f, 1e-6);
        // The two 2×2 vertex grids share the seam column at x=1 (2 duplicated
        // vertices); welding gives the union: a 3×2 grid = 6 unique vertices,
        // and all 4 triangles survive.
        assert_eq!(m.positions.len(), 6);
        assert_eq!(m.faces.len(), 4);
        assert!(m.invariant_test().is_ok());
    }

    #[test]
    fn degenerate_faces_are_dropped() {
        let p = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.5, 0.5, 0.0),
        ];
        // One valid triangle, one degenerate (two identical vertices).
        let f = vec![[0, 1, 2], [0, 0, 3]];
        let m = weld(p, f, 1e-12);
        assert_eq!(m.faces.len(), 1);
    }

    #[test]
    fn zero_tolerance_is_identity() {
        let p = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let f = vec![[0, 1, 2]];
        let m = weld(p, f, 0.0);
        assert_eq!(m.positions.len(), 3);
        assert_eq!(m.faces.len(), 1);
    }
}
