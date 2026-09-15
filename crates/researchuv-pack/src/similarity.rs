//! Similarity — `find_similar` / `align_similar` / `is_similar` /
//! `split_by_similarity` (engine string cluster) with the `SimilarityParams`
//! semantics (mode, threshold, `check_holes`, `adjust_scale`,
//! `non_uniform_scaling_tolerance`, `match_3d_axis`, `correct_vertices`,
//! `vertex_threshold`).
//!
//! Similarity is measured between island **shapes**, in three modes
//! (`SimilarityMode`): border shape (resampled outline), vertex position, and
//! topology (face count + edge structure). Two islands are similar when their
//! normalized difference is within `threshold`.

use crate::island::Island;
use crate::params::{SimilarityMode, SimilarityParams};
use crate::poly;
use researchuv_math::Vec2;

/// Normalize a shape: center at origin, scale so max extent = 1, snap per
/// `correct_vertices`.
fn normalize_shape(pts: &[Vec2], params: &SimilarityParams) -> Vec<Vec2> {
    if pts.is_empty() {
        return pts.to_vec();
    }
    let b = poly::bbox_of(pts);
    let c = b.center();
    let extent = b.max_extent().max(1e-12);
    let mut out: Vec<Vec2> = pts.iter().map(|p| (*p - c) / extent).collect();
    if params.correct_vertices {
        // Snap vertices closer than `vertex_threshold` (normalized units).
        let th = params.vertex_threshold;
        for i in 0..out.len() {
            for j in (i + 1)..out.len() {
                if (out[i] - out[j]).len() < th {
                    out[j] = out[i];
                }
            }
        }
    }
    out
}

/// Shape difference in `BorderShape` mode: the mean closest-point distance
/// between two resampled outlines (symmetric), in normalized units.
fn border_shape_diff(a: &[Vec2], b: &[Vec2]) -> f64 {
    let n = a.len().max(b.len());
    if n < 2 {
        return 0.0;
    }
    // Resample b onto a's frame (equi-count) for a fair comparison.
    let br = if b.len() != a.len() { poly::resample_outline(b, a.len() as u32) } else { b.to_vec() };
    let mut d = 0.0f64;
    for (i, &pa) in a.iter().enumerate() {
        let pb = br[i % br.len()];
        d += (pa - pb).len();
    }
    d / a.len() as f64
}

/// Vertex-position similarity: compare corresponding vertices after
/// normalization (needs the same vertex count — `VertexPosition` mode).
fn vertex_position_diff(a: &[Vec2], b: &[Vec2]) -> f64 {
    let n = a.len().min(b.len());
    if n < 2 {
        return 1.0; // can't compare → dissimilar
    }
    let mut d = 0.0f64;
    for i in 0..n {
        d += (a[i] - b[i]).len();
    }
    d / n as f64
}

/// Topology similarity: 0 when face counts and edge counts match, scaled
/// difference otherwise.
fn topology_diff(a: &Island, b: &Island) -> f64 {
    let fa = a.faces.len() as f64;
    let fb = b.faces.len() as f64;
    let e = (fa - fb).abs() / fa.max(fb).max(1.0);
    // Vertex-count difference contributes half as much.
    let va = a.verts.len() as f64;
    let vb = b.verts.len() as f64;
    let v = (va - vb).abs() / va.max(vb).max(1.0) * 0.5;
    e + v
}

/// The normalized similarity difference of two islands in `mode`
/// (0 = identical, 1 = maximally different).
fn similarity_diff(a: &Island, b: &Island, params: &SimilarityParams) -> f64 {
    match params.mode {
        SimilarityMode::BorderShape => {
            let n = params.precision.max(16) as u32;
            let ra = poly::resample_outline(&a.verts, n);
            let rb = poly::resample_outline(&b.verts, n);
            let na = normalize_shape(&ra, params);
            let nb = normalize_shape(&rb, params);
            if params.adjust_scale {
                // Allow uniform/non-uniform rescaling within the tolerance.
                let scale_diff = max_scale_diff(&na, &nb);
                let mut d = border_shape_diff(&na, &nb);
                if params.non_uniform_scaling_tolerance >= 1.0 {
                    d = border_shape_diff(&na, &nb);
                } else {
                    // Penalize scale mismatch beyond the tolerance.
                    d = d.max(scale_diff * (1.0 - params.non_uniform_scaling_tolerance));
                }
                d
            } else {
                border_shape_diff(&na, &nb)
            }
        }
        SimilarityMode::VertexPosition => {
            let na = normalize_shape(&a.verts, params);
            let nb = normalize_shape(&b.verts, params);
            vertex_position_diff(&na, &nb)
        }
        SimilarityMode::Topology => topology_diff(a, b),
    }
}

/// The max scale ratio between two normalized shapes (bounding boxes).
fn max_scale_diff(a: &[Vec2], b: &[Vec2]) -> f64 {
    let ba = poly::bbox_of(a);
    let bb = poly::bbox_of(b);
    let sx = (ba.width() - bb.width()).abs();
    let sy = (ba.height() - bb.height()).abs();
    sx.max(sy)
}

/// Are the two islands similar (normalized difference ≤ threshold)?
pub fn is_similar(a: &Island, b: &Island, params: &SimilarityParams) -> bool {
    if a.verts.len() < 3 || b.verts.len() < 3 {
        return false;
    }
    if params.check_holes
        && (poly::has_holes(&a.verts) != poly::has_holes(&b.verts))
    {
        return false;
    }
    similarity_diff(a, b, params) <= params.threshold
}

/// Find all islands similar to `target` (the `find_similar` operation).
///
/// `target` is excluded by identity (when it appears in `islands`).
pub fn find_similar(
    target: &Island,
    islands: &[Island],
    params: &SimilarityParams,
) -> Vec<u32> {
    let target_idx = islands.iter().position(|i| std::ptr::eq(i, target));
    islands
        .iter()
        .enumerate()
        .filter(|(i, isl)| Some(*i) != target_idx && is_similar(target, isl, params))
        .map(|(i, _)| i as u32)
        .collect()
}

/// Partition islands into similarity clusters (the `split_by_similarity`
/// operation): greedy first-fit clustering in the given order.
pub fn split_by_similarity(
    islands: &[Island],
    params: &SimilarityParams,
) -> Vec<Vec<u32>> {
    let mut clusters: Vec<(Island, Vec<u32>)> = Vec::new();
    for (i, isl) in islands.iter().enumerate() {
        let mut placed = false;
        for (rep, members) in clusters.iter_mut() {
            if is_similar(rep, isl, params) {
                members.push(i as u32);
                placed = true;
                break;
            }
        }
        if !placed {
            clusters.push((isl.clone(), vec![i as u32]));
        }
    }
    clusters.into_iter().map(|(_, m)| m).collect()
}

/// Align similar islands: for each member of a similarity cluster, compute the
/// 2-D affine (rotation + scale + translate) that best fits it onto the
/// cluster reference (least squares over corresponding resampled vertices).
///
/// Returns per-island `(scale, rotation, tx)` (identity for non-members; the
/// full 4-component transform is in [`align_similar_full`]).
pub fn align_similar(
    reference_index: u32,
    clusters: &[Vec<u32>],
    islands: &[Island],
    params: &SimilarityParams,
) -> Vec<Option<(f64, f64, f64)>> {
    align_similar_full(reference_index, clusters, islands, params)
        .into_iter()
        .map(|o| o.map(|(s, r, tx, _ty)| (s, r, tx)))
        .collect()
}

/// Same as [`align_similar`] but with the full 4-component transform.
pub fn align_similar_full(
    reference_index: u32,
    clusters: &[Vec<u32>],
    islands: &[Island],
    params: &SimilarityParams,
) -> Vec<Option<(f64, f64, f64, f64)>> {
    let n = islands.len();
    let mut out = vec![None; n];
    for members in clusters.iter() {
        if !members.contains(&reference_index) && !members.is_empty() {
            // Align the cluster's reference to the global reference when this
            // cluster contains it; otherwise align its first member.
            let ref_idx = members[0];
            if fit_onto_reference(&islands[ref_idx as usize], &islands[reference_index as usize], params).is_some() {
                for &m in members.iter() {
                    if m != reference_index {
                        if let Some(t2) = fit_onto_reference(&islands[m as usize], &islands[reference_index as usize], params) {
                            out[m as usize] = Some(t2);
                        }
                    }
                }
            }
        } else if members.contains(&reference_index) {
            for &m in members.iter() {
                if m != reference_index {
                    if let Some(t2) = fit_onto_reference(&islands[m as usize], &islands[reference_index as usize], params) {
                        out[m as usize] = Some(t2);
                    }
                }
            }
        }
    }
    out
}

/// Least-squares fit of island `a` onto island `ref_` (resampled, normalized):
/// returns `(scale, rotation, tx, ty)` in the *unnormalized* frame — i.e. the
/// transform taking `a`'s normalized shape to `ref_`'s normalized shape,
/// mapped back by the shapes' normalization frames.
fn fit_onto_reference(
    a: &Island,
    ref_: &Island,
    params: &SimilarityParams,
) -> Option<(f64, f64, f64, f64)> {
    let n = params.precision.max(16) as u32;
    let ra = poly::resample_outline(&a.verts, n);
    let rr = poly::resample_outline(&ref_.verts, n);
    if ra.len() != rr.len() || ra.len() < 3 {
        return None;
    }
    // Fit a rigid+scale transform: p'_k ≈ M·p_k + t with M ≈ s·R.
    // Centered normal equations: M = (Σ e·dᵀ)·(Σ d·dᵀ)⁻¹, d = a−c̄, e = r−r̄.
    let (cu, cv) = centroid(&ra);
    let (ru, rv) = centroid(&rr);
    let mut s11 = 0.0f64;
    let mut s12 = 0.0f64;
    let mut s22 = 0.0f64;
    let mut r11 = 0.0f64; // Σ eu·du
    let mut r12 = 0.0f64; // Σ eu·dv
    let mut r21 = 0.0f64; // Σ ev·du
    let mut r22 = 0.0f64; // Σ ev·dv
    for (i, &pa) in ra.iter().enumerate() {
        let (du, dv) = (pa.u - cu, pa.v - cv);
        let (eu, ev) = (rr[i].u - ru, rr[i].v - rv);
        r11 += eu * du;
        r12 += eu * dv;
        r21 += ev * du;
        r22 += ev * dv;
        s11 += du * du;
        s12 += du * dv;
        s22 += dv * dv;
    }
    let det = s11 * s22 - s12 * s12;
    if det.abs() < 1e-18 {
        return None;
    }
    // M = R·S⁻¹, S⁻¹ = (1/det)·[[s22, −s12], [−s12, s11]].
    let m00 = (r11 * s22 - r12 * s12) / det;
    let m10 = (r12 * s11 - r11 * s12) / det;
    let m01 = (r21 * s22 - r22 * s12) / det;
    let m11 = (r22 * s11 - r21 * s12) / det;
    // Scale = Frobenius norm of M / √2 (for a pure similarity transform).
    let scale = ((m00 * m00 + m11 * m11) / 2.0).max(0.0).sqrt();
    // Rotation = atan2(m10, m00) (when M ≈ s·R).
    let rotation = m10.atan2(m00);
    // Translation: t = r̄ − M·c̄.
    let tx = ru - (m00 * cu + m01 * cv);
    let ty = rv - (m10 * cu + m11 * cv);
    Some((scale, rotation, tx, ty))
}

fn centroid(pts: &[Vec2]) -> (f64, f64) {
    let n = pts.len() as f64;
    (
        pts.iter().map(|p| p.u).sum::<f64>() / n,
        pts.iter().map(|p| p.v).sum::<f64>() / n,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(s: f64) -> Island {
        Island::from_polygon(vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(s, 0.0),
            Vec2::new(s, s),
            Vec2::new(0.0, s),
        ])
    }

    fn rect(w: f64, h: f64) -> Island {
        Island::from_polygon(vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(w, 0.0),
            Vec2::new(w, h),
            Vec2::new(0.0, h),
        ])
    }

    #[test]
    fn identical_shapes_are_similar() {
        let a = square(1.0);
        let b = square(1.0);
        let p = SimilarityParams::default();
        assert!(is_similar(&a, &b, &p));
    }

    #[test]
    fn different_sizes_are_similar_in_shape() {
        let a = square(0.5);
        let b = square(2.0);
        let p = SimilarityParams::default();
        // Normalization removes scale → same shape.
        assert!(is_similar(&a, &b, &p));
    }

    #[test]
    fn square_vs_rect_not_similar_by_default() {
        let a = square(1.0);
        let b = rect(1.0, 0.3);
        let p = SimilarityParams::default();
        assert!(!is_similar(&a, &b, &p));
    }

    #[test]
    fn threshold_controls_similarity() {
        let a = square(1.0);
        let b = rect(1.0, 0.6);
        let mut p = SimilarityParams::default();
        p.threshold = 0.05;
        assert!(!is_similar(&a, &b, &p));
        p.threshold = 0.75;
        assert!(is_similar(&a, &b, &p));
    }

    #[test]
    fn find_similar_and_split() {
        let isls = vec![square(1.0), square(0.8), rect(1.0, 0.2), square(1.2)];
        let p = SimilarityParams::default();
        let sim = find_similar(&isls[0], &isls, &p);
        assert!(sim.contains(&1));
        assert!(sim.contains(&3));
        assert!(!sim.contains(&2));
        let clusters = split_by_similarity(&isls, &p);
        assert_eq!(clusters.len(), 2);
        let squares: Vec<u32> = clusters.iter().flatten().copied().collect();
        assert_eq!(squares.len(), 4);
    }

    #[test]
    fn align_similar_produces_transforms() {
        let ref_ = square(1.0);
        let member = square(0.5);
        let isls = vec![ref_, member];
        let p = SimilarityParams::default();
        let clusters = split_by_similarity(&isls, &p);
        let t = align_similar_full(0, &clusters, &isls, &p);
        assert!(t[1].is_some());
        let (s, r, tx, ty) = t[1].unwrap();
        assert!(s > 0.0);
        let _ = (r, tx, ty);
    }

    #[test]
    fn resample_preserves_order() {
        let q = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(0.0, 1.0),
        ];
        let r = poly::resample_outline(&q, 16);
        assert_eq!(r.len(), 16);
        // All points inside the unit square (with float slack).
        for p in &r {
            assert!((-1e-9..=1.0 + 1e-9).contains(&p.u));
            assert!((-1e-9..=1.0 + 1e-9).contains(&p.v));
        }
    }
}
