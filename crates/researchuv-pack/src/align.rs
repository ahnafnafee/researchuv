//! Alignment — `pixel_perfect_align_target` (snap island corners/centers to the
//! pixel grid) and `orient_to_3d` / `match_3d_axis` (rotate an island's UV so a
//! 3-D axis maps to a UV axis).
//!
//! Evidence: engine strings `pixel_perfect_align_target`,
//! `pixel_perfect_vert_align`, `match_3d_axis`, `orient_to_3d`; addon
//! `UVPM4_PixelPerfectAlignProps` (target Corner/Center, vert align
//! None/BboxCorners/Bbox/BorderEdges/All) and `UVPM4_OrientTo3dProps`
//! (prim_3d_axis Z → prim_uv_axis Y, sec_3d_axis X → sec_uv_axis X,
//! prim_sec_bias 80, space Local).

use crate::params::{
    CoordSpace, OrientTo3dParams, PackParams, PixelPerfectAlignTarget,
    PixelPerfectVertAlignMode, UvpmAxis,
};
use crate::poly;
use researchuv_math::Vec2;

/// Snap a placed island's reference point to the pixel grid.
///
/// `tex_size` is the texture pixel size; the grid step is `1.0 / tex_size` UV
/// units. Returns the translation (Δu, Δv) to apply.
pub fn align_offset(
    anchor: Vec2,
    target: PixelPerfectAlignTarget,
    tex_size: f32,
) -> Vec2 {
    let tex = (tex_size as f64).max(1.0);
    let step = 1.0 / tex;
    let snapped = match target {
        PixelPerfectAlignTarget::Corner => {
            // Snap the min corner (the island's reference corner) to the grid.
            Vec2::new((anchor.u / step).round() * step, (anchor.v / step).round() * step)
        }
        PixelPerfectAlignTarget::Center => {
            Vec2::new(
                ((anchor.u + 0.5) / step).round() * step - 0.5,
                ((anchor.v + 0.5) / step).round() * step - 0.5,
            )
        }
    };
    Vec2::new(snapped.u - anchor.u, snapped.v - anchor.v)
}

/// The pixel grid step for a texture size.
pub fn pixel_step(tex_size: f32) -> f64 {
    1.0 / (tex_size as f64).max(1.0)
}

/// Vertex alignment (the `pixel_perfect_vert_align` modes): returns the
/// per-vertex snapped coordinates.
pub fn align_vertices(
    poly_: &[Vec2],
    mode: PixelPerfectVertAlignMode,
    tex_size: f32,
) -> Vec<Vec2> {
    match mode {
        PixelPerfectVertAlignMode::None => poly_.to_vec(),
        PixelPerfectVertAlignMode::BboxCorners => poly::snap_bbox_corners(poly_, pixel_step(tex_size)),
        PixelPerfectVertAlignMode::Bbox => poly::snap_bbox_border(poly_, pixel_step(tex_size)),
        PixelPerfectVertAlignMode::BorderEdges | PixelPerfectVertAlignMode::All => {
            poly::snap_to_grid(poly_, pixel_step(tex_size))
        }
    }
}

/// The pixel-perfect offset for a placed island under `params`.
pub fn pixel_perfect_offset(island_anchor: Vec2, params: &PackParams) -> Vec2 {
    if !params.pixel_perfect_align {
        return Vec2::new(0.0, 0.0);
    }
    align_offset(
        island_anchor,
        params.pixel_perfect_align_target,
        params.pixel_margin_tex_size as f32,
    )
}

/// Orient-to-3-D rotation of an island outline (the `orient_to_3d` option):
/// find the 2-D rotation (around the island center) that maps the island's
/// dominant 3-D axis to the dominant UV axis.
///
/// Method: fit the island's UV→3-D mapping `d3 ≈ A·d_uv` (3×2 least squares over
/// the correlated vertex pairs), take the UV principal direction (max-variance
/// eigenvector of the outline), map it to 3-D through `A`, and rotate the
/// outline so that direction points along the UV axis paired with it in
/// `params` (`prim_3d_axis → prim_uv_axis`, `sec_3d_axis → sec_uv_axis`;
/// `prim_sec_bias` picks the preferred pair when the 3-D direction is
/// ambiguous).
pub fn orient_to_3d_rotation(island: &crate::island::Island, params: &OrientTo3dParams) -> Option<f64> {
    if !params.enable {
        return None;
    }
    let verts3d = island.verts3d.as_ref()?;
    if verts3d.len() < 3 || island.verts.len() < 3 || verts3d.len() != island.verts.len() {
        return None;
    }
    let (cu, cv) = centroid(&island.verts);
    let n = verts3d.len() as f64;
    let (cx, cy, cz) = (
        verts3d.iter().map(|p| p.x).sum::<f64>() / n,
        verts3d.iter().map(|p| p.y).sum::<f64>() / n,
        verts3d.iter().map(|p| p.z).sum::<f64>() / n,
    );
    // Normal equations for A (2 UV cols) per 3-D axis:
    // [S00 S01; S01 S11] [a_k0; a_k1] = [t0k; t1k].
    let (mut s00, mut s01, mut s11) = (0.0f64, 0.0f64, 0.0f64);
    let (mut t0, mut t1) = ([0.0f64; 3], [0.0f64; 3]);
    for (i, &pu) in island.verts.iter().enumerate() {
        let p3 = verts3d[i];
        let (du, dv) = (pu.u - cu, pu.v - cv);
        let (dx, dy, dz) = (p3.x - cx, p3.y - cy, p3.z - cz);
        s00 += du * du;
        s01 += du * dv;
        s11 += dv * dv;
        t0[0] += du * dx;
        t0[1] += du * dy;
        t0[2] += du * dz;
        t1[0] += dv * dx;
        t1[1] += dv * dy;
        t1[2] += dv * dz;
    }
    let det = s00 * s11 - s01 * s01;
    if det.abs() < 1e-18 {
        return None;
    }
    // Cramer's rule: a_k0 = (t0k·S11 − t1k·S01)/det, a_k1 = (S00·t1k − S01·t0k)/det.
    let a = [
        [(t0[0] * s11 - t1[0] * s01) / det, (s00 * t1[0] - s01 * t0[0]) / det],
        [(t0[1] * s11 - t1[1] * s01) / det, (s00 * t1[1] - s01 * t0[1]) / det],
        [(t0[2] * s11 - t1[2] * s01) / det, (s00 * t1[2] - s01 * t0[2]) / det],
    ];
    // UV principal direction and its 3-D image.
    let principal_uv = principal_angle(s00, s01, s11);
    let (cp, sp) = (principal_uv.cos(), principal_uv.sin());
    let (mx, my, mz) = (
        a[0][0] * cp + a[0][1] * sp,
        a[1][0] * cp + a[1][1] * sp,
        a[2][0] * cp + a[2][1] * sp,
    );
    let dom = dominant_axis(mx, my, mz);
    // Auto-correct the secondary 3-D axis away from the primary (the addon's
    // `_update_orient_3d_axes`; the UV axes may repeat).
    let prim3 = params.prim_3d_axis.unsigned();
    let sec3 = if params.sec_3d_axis.unsigned() == prim3 {
        next_axis(prim3)
    } else {
        params.sec_3d_axis.unsigned()
    };
    let prim_uv = params.prim_uv_axis;
    let sec_uv = params.sec_uv_axis;
    // Which UV axis should the principal direction point at?
    let target_axis = if dom == prim3 {
        prim_uv
    } else if dom == sec3 {
        sec_uv
    } else {
        // Ambiguous (near-square 3-D): prim_sec_bias ≥ 45 (of 0..90) prefers
        // the primary pair, otherwise the secondary pair.
        if params.prim_sec_bias >= 45.0 {
            prim_uv
        } else {
            sec_uv
        }
    };
    let target = match target_axis {
        UvpmAxis::None => return None, // no UV constraint
        UvpmAxis::X | UvpmAxis::Z => 0.0,
        UvpmAxis::Y => std::f64::consts::FRAC_PI_2,
        UvpmAxis::NegX | UvpmAxis::NegZ => std::f64::consts::PI,
        UvpmAxis::NegY => -std::f64::consts::FRAC_PI_2,
    };
    wrap_pi(target - principal_uv)
}

/// The next unsigned axis in X → Y → Z → X order (the addon's secondary-axis
/// auto-correction cycle over the positive axes).
fn next_axis(a: UvpmAxis) -> UvpmAxis {
    match a {
        UvpmAxis::X => UvpmAxis::Y,
        UvpmAxis::Y => UvpmAxis::Z,
        _ => UvpmAxis::X,
    }
}

/// Principal direction angle of a 2×2 symmetric matrix with entries
/// (suu, suv; suv, svv) — the max-variance eigenvector.
fn principal_angle(suu: f64, suv: f64, svv: f64) -> f64 {
    0.5 * (2.0 * suv).atan2(suu - svv)
}

/// The dominant coordinate axis of a 3-D direction.
fn dominant_axis(x: f64, y: f64, z: f64) -> UvpmAxis {
    let (ax, ay, az) = (x.abs(), y.abs(), z.abs());
    if ax >= ay && ax >= az {
        UvpmAxis::X
    } else if ay >= az {
        UvpmAxis::Y
    } else {
        UvpmAxis::Z
    }
}

/// Wrap an angle to (−π, π].
fn wrap_pi(a: f64) -> Option<f64> {
    let pi = std::f64::consts::PI;
    let mut a = a;
    while a > pi {
        a -= 2.0 * pi;
    }
    while a <= -pi {
        a += 2.0 * pi;
    }
    Some(a)
}

/// Centroid of a 2-D point set.
fn centroid(pts: &[Vec2]) -> (f64, f64) {
    let n = pts.len() as f64;
    (
        pts.iter().map(|p| p.u).sum::<f64>() / n,
        pts.iter().map(|p| p.v).sum::<f64>() / n,
    )
}

/// Match a 3-D axis to a UV axis (the `match_3d_axis` similarity option):
/// the rotation that aligns the island's 3-D `axis` projection onto the +X UV
/// axis.
pub fn match_3d_axis_rotation(
    island: &crate::island::Island,
    axis: UvpmAxis,
    space: CoordSpace,
) -> Option<f64> {
    let _ = space; // Local vs Global does not change the per-island rotation.
    if axis == UvpmAxis::None {
        return None; // match_3d_axis = None: don't match
    }
    let verts3d = island.verts3d.as_ref()?;
    if verts3d.len() < 2 || verts3d.len() != island.verts.len() {
        return None;
    }
    // Drop the matched 3-D axis → 2-D projection of the 3-D cloud (a negative
    // axis mirrors the projection; the principal-direction rotation is the
    // same, and `flipping_enable` covers mirroring).
    let n = verts3d.len() as f64;
    let (cx, cy, cz) = (
        verts3d.iter().map(|p| p.x).sum::<f64>() / n,
        verts3d.iter().map(|p| p.y).sum::<f64>() / n,
        verts3d.iter().map(|p| p.z).sum::<f64>() / n,
    );
    let comps = |p: &researchuv_math::Vec3| match axis.unsigned() {
        UvpmAxis::None => (0.0, 0.0),
        UvpmAxis::X => (p.y - cy, p.z - cz),
        UvpmAxis::Y => (p.x - cx, p.z - cz),
        UvpmAxis::Z => (p.x - cx, p.y - cy),
        _ => (p.x - cx, p.y - cy),
    };
    let mut suu = 0.0f64;
    let mut suv = 0.0f64;
    let mut svv = 0.0f64;
    for p in verts3d.iter() {
        let (du, dv) = comps(p);
        suu += du * du;
        suv += du * dv;
        svv += dv * dv;
    }
    let principal = principal_angle(suu, suv, svv);
    // Rotate the outline so that the projected principal direction lies on +u.
    wrap_pi(-principal)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::island::Island;
    use researchuv_math::Vec3;

    #[test]
    fn corner_snap_to_grid() {
        let tex = 1024.0f32;
        let step = pixel_step(tex);
        let anchor = Vec2::new(0.3001, 0.6999);
        let off = align_offset(anchor, PixelPerfectAlignTarget::Corner, tex);
        let snapped = anchor + off;
        assert!((snapped.u / step).round() - (snapped.u / step) < 1e-9);
        assert!((snapped.v / step).round() - (snapped.v / step) < 1e-9);
        assert!(off.u.abs() < step && off.v.abs() < step);
    }

    #[test]
    fn center_snap_to_grid() {
        let tex = 64.0f32;
        let anchor = Vec2::new(0.37, 0.12);
        let off = align_offset(anchor, PixelPerfectAlignTarget::Center, tex);
        let snapped = anchor + off;
        // Center snapped to the grid: center % step ≈ 0.
        let step = 1.0 / 64.0;
        assert!((snapped.u / step).round() - (snapped.u / step) < 1e-9);
    }

    #[test]
    fn vertex_align_modes() {
        let q = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(0.11, 0.0),
            Vec2::new(0.11, 0.09),
            Vec2::new(0.0, 0.09),
        ];
        let all = align_vertices(&q, PixelPerfectVertAlignMode::All, 100.0);
        for p in &all {
            assert!((p.u * 100.0).round() - p.u * 100.0 < 1e-9);
            assert!((p.v * 100.0).round() - p.v * 100.0 < 1e-9);
        }
        let none = align_vertices(&q, PixelPerfectVertAlignMode::None, 100.0);
        assert_eq!(none, q);
    }

    #[test]
    fn orient_square_to_x() {
        // A square with verts3d elongated in X; prim_uv_axis = X.
        let isl = Island {
            verts: vec![
                Vec2::new(0.0, 0.0),
                Vec2::new(1.0, 0.0),
                Vec2::new(1.0, 1.0),
                Vec2::new(0.0, 1.0),
            ],
            verts3d: Some(vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(5.0, 0.0, 0.0),
                Vec3::new(5.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ]),
            ..Island::from_polygon(vec![
                Vec2::new(0.0, 0.0),
                Vec2::new(1.0, 0.0),
            ])
        };
        let p = OrientTo3dParams {
            enable: true,
            prim_uv_axis: UvpmAxis::X,
            ..OrientTo3dParams::default()
        };
        let rot = orient_to_3d_rotation(&isl, &p);
        assert!(rot.is_some());
        // The square's UV principal axis is X (elongated along u), so the
        // rotation should be near 0 (±π ambiguity).
        let r = rot.unwrap();
        assert!(r.abs() < 0.01 || (r.abs() - std::f64::consts::PI).abs() < 0.01);
    }

    #[test]
    fn match_3d_axis_returns_rotation() {
        let isl = Island {
            verts: vec![
                Vec2::new(0.0, 0.0),
                Vec2::new(1.0, 0.0),
                Vec2::new(1.0, 1.0),
                Vec2::new(0.0, 1.0),
            ],
            verts3d: Some(vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(1.0, -1.0, 0.0),
            ]),
            ..Island::from_polygon(vec![Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0)])
        };
        let r = match_3d_axis_rotation(&isl, UvpmAxis::Z, CoordSpace::Local);
        assert!(r.is_some());
    }
}
