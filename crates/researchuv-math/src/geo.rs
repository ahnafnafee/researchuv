//! Per-triangle geometric factors for the unfolding solver.
//!
//! Factors combine the unit face normal, squared cross-product magnitude,
//! and edge-vector components used to assemble tangent-frame contributions.

use crate::vec::{Vec2, Vec3};

/// One chart's per-triangle geometric factors.
///
/// The engine stores this as 6 doubles per triangle — unit normal `(nx, ny, nz)`, the area
/// weight `length²/2` (1/2 factor @ const VA `0x140F84DC8`), and the projected edge vectors.
/// The fields contain `[nx, ny, nz, area, ac.x, ac.y, ac.z, ab.x]` for the
/// tangent-frame contribution used by the unfolding solver.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct CGeoTri2 {
    /// Unit face normal `(nx, ny, nz)`.
    pub n: Vec3,
    /// Area weight `length² / 2` (1/2 factor @ const `0x140F84DC8`).
    pub area: f64,
    /// Projected edge vector `c − a`.
    pub ac: Vec3,
    /// The `x` component of `b − a` consumed by the reference driver's tangent-frame RHS pull.
    pub ab_x: f64,
}

/// Compute per-triangle geometric factors for triangles `tris` over positions `pos`.
///
/// Uses the following per-triangle construction:
/// `cr = cross(pa−pb, pc−pb)`, `l2 = |cr|²`, `n = cr/√l2`, factors
/// `[n.x, n.y, n.z, 0.5·l2, ac.x, ac.y, ac.z, ab.x]` with `ac = pc−pa`, `ab = pb−pa`.
pub fn compute_geo_tri2(pos: &[Vec3], tris: &[[u32; 3]]) -> Vec<CGeoTri2> {
    tris.iter()
        .map(|[a, b, c]| {
            let (pa, pb, pc) = (pos[*a as usize], pos[*b as usize], pos[*c as usize]);
            let cr = Vec3::cross(pa, pb, pc); // cross(pa−pb, pc−pb)
            let l2 = cr.dot(cr);
            let n = cr / l2.sqrt();
            let ac = pc - pa;
            let ab = pb - pa;
            CGeoTri2 { n, area: 0.5 * l2, ac, ab_x: ab.x }
        })
        .collect()
}

/// The reference driver's tangent-frame RHS pull for one chart (ALGORITHMS.md §2.4):
/// `rhs[i] = mix_w · (N.x·ac.x + N.y·ab.x)` where `N` is the chart's single face normal.
pub fn rhs_pull(geo: &[CGeoTri2], chart_normal: Vec3, mix_w: f64) -> Vec<Vec2> {
    geo.iter()
        .map(|g| Vec2::new(mix_w * (chart_normal.x * g.ac.x + chart_normal.y * g.ab_x), 0.0))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geo_tri2_unit_right_triangle() {
        let pos = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let tris = [[0u32, 1, 2]];
        let g = compute_geo_tri2(&pos, &tris);
        assert_eq!(g.len(), 1);
        // cross(pa−pb, pc−pb) with pa at the origin is −z for this winding; unit length.
        assert!((g[0].n.z + 1.0).abs() < 1e-12);
        assert!((g[0].n.len() - 1.0).abs() < 1e-12);
        assert!((g[0].area - 0.5).abs() < 1e-12); // 0.5 * |cross|², |cross|=1
    }
}
