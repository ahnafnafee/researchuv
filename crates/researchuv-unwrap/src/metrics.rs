//! Distortion metrics + rectangularize — `CTaskOptimize` diagnostics and the
//! `NGeoTopo::Mesh::Rectangularize` approximation (reference `chart_distortion` /
//! `rectangularize`).
//!
//! Evidence: ALGORITHMS.md §4. The rectangularize pass fits the chart into the
//! minimum-area border rectangle using a 72-step angle ladder (the exe's
//! "nice-number" table), then verticalizes: boundary points are snapped onto
//! the rectangle's edges.

use crate::segment::Chart;
use researchuv_core::model::SurfaceMesh;
use researchuv_math::Vec2;

/// Per-triangle distortion metrics for an unfolded chart.
#[derive(Clone, Copy, Debug)]
pub struct ChartMetrics {
    /// Mean singular-value ratio s₁/s₂ of the per-triangle Jacobian (1.0 = conformal).
    pub conformal_mean: f64,
    /// Worst (max) singular-value ratio.
    pub conformal_max: f64,
    /// Mean |UV area| / |3-D area| per triangle (1.0 = area-preserving).
    pub area_ratio_mean: f64,
    /// Triangles whose 3-D projection was degenerate (counted as flips).
    pub flips: usize,
}

/// Singular values of a 2×2 matrix `m` (returns (s₁, s₂), s₁ ≥ s₂ ≥ 0).
fn svd2(m: [[f64; 2]; 2]) -> (f64, f64) {
    // G = M^T M (symmetric); eigenvalues of G, then sqrt.
    let a = m[0][0] * m[0][0] + m[1][0] * m[1][0];
    let b = m[0][0] * m[0][1] + m[1][0] * m[1][1];
    let c = m[0][1] * m[0][1] + m[1][1] * m[1][1];
    let tr = a + c;
    let d = 0.5 * (a - c);
    let disc = (d * d + b * b).max(0.0).sqrt();
    let l1 = 0.5 * tr + disc;
    let l2 = (0.5 * tr - disc).max(0.0);
    (l1.max(0.0).sqrt(), l2.sqrt())
}

/// Per-triangle conformal + area-ratio metrics (reference `chart_distortion`).
pub fn chart_distortion(mesh: &SurfaceMesh, chart: &Chart, uv_local: &[Vec2]) -> ChartMetrics {
    let mut conformal: Vec<f64> = Vec::new();
    let mut area_ratios: Vec<f64> = Vec::new();
    let mut flips = 0usize;
    for (t, &fi) in chart.tris.iter().zip(&chart.face_ids) {
        let [a, b, c] = mesh.faces[fi];
        let (a3, b3, c3) = (
            mesh.positions[a as usize],
            mesh.positions[b as usize],
            mesh.positions[c as usize],
        );
        let (u0, u1, u2) = (
            uv_local[t[0] as usize],
            uv_local[t[1] as usize],
            uv_local[t[2] as usize],
        );
        let e1 = b3 - a3;
        let e2 = c3 - a3;
        let nrm = e1.cross_v(e2);
        let ln = nrm.len();
        if ln < 1e-30 {
            continue;
        }
        let n = nrm / ln;
        let t1 = e1 / e1.len().max(1e-30);
        let t2 = n.cross_v(t1);
        // P (2x2): the reference builds `P = [[e1·t1, e2·t1],[e1·t2, e2·t2]]`
        // (rows = tangent basis (t1,t2), cols = 3-D edges (e1,e2)): it maps edge
        // coefficients (a,b) to the 2-D tangent-plane coords of a·e1 + b·e2.
        // `F = U @ P^-1` is therefore the UV Jacobian w.r.t. tangent-plane coords.
        // (Inverting the transpose couples the wrong axis pairs and over-states the
        // conformal ratio — e.g. a pure rotation reads as φ².)
        let e1t1 = e1.dot(t1);
        let e2t1 = e2.dot(t1);
        let e1t2 = e1.dot(t2);
        let e2t2 = e2.dot(t2);
        let det = e1t1 * e2t2 - e2t1 * e1t2;
        if det.abs() < 1e-30 {
            flips += 1;
            continue;
        }
        // inv(P) for P = [[e1t1, e2t1],[e1t2, e2t2]].
        let inv_p00 = e2t2 / det;
        let inv_p01 = -e2t1 / det;
        let inv_p10 = -e1t2 / det;
        let inv_p11 = e1t1 / det;
        let du1 = u1 - u0;
        let du2 = u2 - u0;
        // F = U @ inv(P), U's columns are (u1-u0, u2-u0).
        let f = [[
            du1.u * inv_p00 + du2.u * inv_p10,
            du1.u * inv_p01 + du2.u * inv_p11,
        ], [
            du1.v * inv_p00 + du2.v * inv_p10,
            du1.v * inv_p01 + du2.v * inv_p11,
        ]];
        let (s1, s2) = svd2(f);
        conformal.push(s1 / s2.max(1e-30));
        let uv_area = 0.5 * (du1.u * du2.v - du2.u * du1.v).abs();
        area_ratios.push(uv_area / (0.5 * ln));
    }
    ChartMetrics {
        conformal_mean: conformal.iter().sum::<f64>() / conformal.len().max(1) as f64,
        conformal_max: conformal.iter().cloned().fold(0.0f64, |m, x| m.max(x)),
        area_ratio_mean: area_ratios.iter().sum::<f64>() / area_ratios.len().max(1) as f64,
        flips,
    }
}

/// Rectangularize (reference `rectangularize`): fit into the minimum-area border
/// rectangle (72-step angle ladder), then verticalize — snap boundary points
/// onto the rectangle. Returns the normalized `[0,1]²` UVs and the 2-D extent
/// before normalization.
pub fn rectangularize(uv: &[Vec2], bnd_ids: &[usize]) -> (Vec<Vec2>, Vec2) {
    if bnd_ids.is_empty() {
        let mut min = uv[0];
        let mut max = uv[0];
        for p in uv {
            min = Vec2::new(min.u.min(p.u), min.v.min(p.v));
            max = Vec2::new(max.u.max(p.u), max.v.max(p.v));
        }
        let ext = max - min;
        let q: Vec<Vec2> = uv
            .iter()
            .map(|p| {
                let q = *p - min;
                Vec2::new(
                    q.u / ext.u.max(1e-30),
                    q.v / ext.v.max(1e-30),
                )
            })
            .collect();
        return (q, ext);
    }
    // 72-step angle ladder (np.linspace(0, π, 72) in the reference).
    let mut best: Option<(f64, f64, f64, f64)> = None; // (ang, cos, sin, score)
    let n_steps = 72;
    for k in 0..n_steps {
        let ang = std::f64::consts::PI * k as f64 / (n_steps - 1) as f64;
        let (cos, sin) = (ang.cos(), ang.sin());
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        let mut minv = f64::INFINITY;
        let mut maxv = f64::NEG_INFINITY;
        for &bi in bnd_ids {
            let p = uv[bi];
            // q = R p, R = [[cos, -sin], [sin, cos]] (the reference rotates the
            // row vectors with `bnd @ R.T`, which is R acting on column vectors).
            let qu = cos * p.u + sin * p.v;
            let qv = -sin * p.u + cos * p.v;
            min = min.min(qu);
            max = max.max(qu);
            minv = minv.min(qv);
            maxv = maxv.max(qv);
        }
        let ext0 = max - min;
        let ext1 = maxv - minv;
        // Reference score: sqrt(Σ ext·(ext + 1e-30)) — minimum-area rectangle.
        let score = (ext0 * (ext0 + 1e-30) + ext1 * (ext1 + 1e-30)).sqrt();
        match best {
            Some((_, _, _, bs)) if score >= bs => {}
            _ => best = Some((ang, cos, sin, score)),
        }
    }
    let (_, cos, sin, _) = best.unwrap_or((0.0, 1.0, 0.0, f64::INFINITY));
    // Rotate the whole chart.
    let mut q: Vec<Vec2> = uv
        .iter()
        .map(|p| Vec2::new(cos * p.u + sin * p.v, -sin * p.u + cos * p.v))
        .collect();
    let mut min = q[0];
    let mut max = q[0];
    for p in &q {
        min = Vec2::new(min.u.min(p.u), min.v.min(p.v));
        max = Vec2::new(max.u.max(p.u), max.v.max(p.v));
    }
    let ext0 = max - min;
    for p in q.iter_mut() {
        let t = *p - min;
        *p = Vec2::new(t.u / ext0.u.max(1e-30), t.v / ext0.v.max(1e-30));
    }
    // Verticalize: snap boundary points onto the nearest rectangle edge.
    for &bi in bnd_ids {
        let p = q[bi];
        let dx = p.u.min(1.0 - p.u);
        let dy = p.v.min(1.0 - p.v);
        if dx <= dy {
            q[bi] = Vec2::new(if p.u < 0.5 { 0.0 } else { 1.0 }, p.v);
        } else {
            q[bi] = Vec2::new(p.u, if p.v < 0.5 { 0.0 } else { 1.0 });
        }
    }
    (q, ext0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_unfold_of_a_unit_triangle_is_conformal() {
        // A single triangle, UV == its 3-D xy coordinates → exact (conformal ratio 1).
        let p = vec![
            researchuv_math::Vec3::new(0.0, 0.0, 0.0),
            researchuv_math::Vec3::new(1.0, 0.0, 0.0),
            researchuv_math::Vec3::new(0.0, 1.0, 0.0),
        ];
        let m = SurfaceMesh::from_triangles(p, vec![[0u32, 1, 2]]);
        let (charts, _) = crate::segment::segment(&m, 30.0);
        let chart = &charts[0];
        let uv: Vec<Vec2> = chart
            .vertex_ids
            .iter()
            .map(|&v| {
                let p3 = m.positions[v as usize];
                Vec2::new(p3.x, p3.y)
            })
            .collect();
        let met = chart_distortion(&m, chart, &uv);
        assert!((met.conformal_mean - 1.0).abs() < 1e-9, "conformal = {}", met.conformal_mean);
        assert_eq!(met.flips, 0);
    }

    #[test]
    fn rectangularize_open_square_fits_the_unit_square() {
        // A unit square given as two triangles, already axis-aligned.
        let uv = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 1.0),
            Vec2::new(1.0, 1.0),
        ];
        let bnd = vec![0usize, 1, 2, 3];
        let (q, ext) = rectangularize(&uv, &bnd);
        assert!((ext.u - 1.0).abs() < 1e-9);
        assert!((ext.v - 1.0).abs() < 1e-9);
        // All four points are on the unit-square perimeter.
        for p in &q {
            let on_border = p.u.abs() < 1e-9
                || (1.0 - p.u).abs() < 1e-9
                || p.v.abs() < 1e-9
                || (1.0 - p.v).abs() < 1e-9;
            assert!(on_border, "point ({}, {}) not on border", p.u, p.v);
        }
    }
}
