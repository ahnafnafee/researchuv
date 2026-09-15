//! Polygon operations — the CPU analog of the engine's Boost.Geometry overlay
//! tests (`has_self_intersections`, `check_holes`, `ProcessIntersections`) and
//! the CDT-based free-space intersection checks
//! (`uvpm_core/algorithms/CDT/Triangulation.hpp`, `insertEdgeIteration`).
//!
//! Overlap semantics (from `UvpmOverlapDetectionMode`):
//!
//! - **`AnyPart`** (default): the *polygons* intersect — any part of one island
//!   over any part of another counts.
//! - **`Exact`**: the *bounding boxes* overlap.

use researchuv_math::Vec2;
use crate::box2::Box2;

/// Signed area of a polygon (shoelace). CCW = positive.
pub fn signed_area(poly: &[Vec2]) -> f64 {
    if poly.len() < 3 {
        return 0.0;
    }
    let n = poly.len();
    let mut a = 0.0f64;
    for i in 0..n {
        let p0 = poly[i];
        let p1 = poly[(i + 1) % n];
        a += p0.u * p1.v - p1.u * p0.v;
    }
    0.5 * a
}

/// Absolute area.
pub fn area(poly: &[Vec2]) -> f64 {
    signed_area(poly).abs()
}

/// Bounding box of a polygon.
pub fn bbox_of(poly: &[Vec2]) -> Box2 {
    debug_assert!(!poly.is_empty());
    let mut min = poly[0];
    let mut max = poly[0];
    for &p in poly.iter() {
        min = Vec2::new(min.u.min(p.u), min.v.min(p.v));
        max = Vec2::new(max.u.max(p.u), max.v.max(p.v));
    }
    Box2::new(min, max)
}

/// Resample a polygon to `n` equi-arc-length points (starting at the vertex
/// closest to the bbox min corner) for shape comparison.
pub fn resample_outline(poly: &[Vec2], n: u32) -> Vec<Vec2> {
    let n = n.max(4);
    if poly.len() < 3 {
        return poly.to_vec();
    }
    // Total perimeter.
    let mut perim = 0.0f64;
    let m = poly.len();
    for i in 0..m {
        let a = poly[i];
        let b = poly[(i + 1) % m];
        perim += (b - a).len();
    }
    if perim <= 0.0 {
        return vec![poly[0]; n as usize];
    }
    // Start at the vertex nearest the bbox min corner.
    let b = bbox_of(poly);
    let mut start = 0usize;
    let mut bd = f64::INFINITY;
    for (i, &p) in poly.iter().enumerate() {
        let d = (p - b.min).len();
        if d < bd {
            bd = d;
            start = i;
        }
    }
    let step = perim / n as f64;
    let mut out = Vec::with_capacity(n as usize);
    let mut dist = 0.0f64;
    let mut i = start;
    let mut next = 0.0f64;
    for _ in 0..n {
        // Walk the edges until `next` lies on the current edge.
        loop {
            let l = edge_len(poly[i], poly[(i + 1) % m]);
            if next <= dist + l || l <= 1e-15 {
                break;
            }
            dist += l;
            i = (i + 1) % m;
        }
        let a = poly[i];
        let b = poly[(i + 1) % m];
        let l = edge_len(a, b);
        let t = if l > 1e-15 { ((next - dist) / l).min(1.0).max(0.0) } else { 0.0 };
        out.push(a + (b - a) * t);
        next += step;
    }
    while out.len() < n as usize {
        out.push(poly[0]);
    }
    out.truncate(n as usize);
    out
}

#[inline]
fn edge_len(from: Vec2, to: Vec2) -> f64 {
    (to - from).len()
}

/// Ray-cast point-in-polygon (inclusive of the boundary within `eps`).
pub fn point_in_poly(p: Vec2, poly: &[Vec2], eps: f64) -> bool {
    if poly.len() < 3 {
        return false;
    }
    let n = poly.len();
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let a = poly[i];
        let b = poly[j];
        // Standard even-odd ray cast along +u.
        if (a.v > p.v) != (b.v > p.v) {
            let t = (p.v - a.v) / (b.v - a.v);
            let xu = a.u + t * (b.u - a.u);
            if p.u <= xu + eps && p.u >= xu - eps {
                return true;
            }
            if p.u < xu {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

/// Orientation of the triple (a, b, c): +1 CCW, −1 CW, 0 collinear.
#[inline]
fn orient(a: Vec2, b: Vec2, c: Vec2) -> i32 {
    let v = (b.u - a.u) * (c.v - a.v) - (b.v - a.v) * (c.u - a.u);
    if v > 0.0 {
        1
    } else if v < 0.0 {
        -1
    } else {
        0
    }
}

/// On-segment test (with collinearity).
#[inline]
fn on_seg(a: Vec2, b: Vec2, p: Vec2) -> bool {
    p.u >= a.u.min(b.u) && p.u <= a.u.max(b.u) && p.v >= a.v.min(b.v) && p.v <= a.v.max(b.v)
}

/// Do segments (p1→p2) and (q1→q2) intersect (touching counts)?
pub fn segs_intersect(p1: Vec2, p2: Vec2, q1: Vec2, q2: Vec2) -> bool {
    let d1 = orient(q1, q2, p1);
    let d2 = orient(q1, q2, p2);
    let d3 = orient(p1, p2, q1);
    let d4 = orient(p1, p2, q2);
    if d1 != d2 && d3 != d4 {
        return true;
    }
    if d1 == 0 && on_seg(q1, q2, p1) {
        return true;
    }
    if d2 == 0 && on_seg(q1, q2, p2) {
        return true;
    }
    if d3 == 0 && on_seg(p1, p2, q1) {
        return true;
    }
    if d4 == 0 && on_seg(p1, p2, q2) {
        return true;
    }
    false
}

/// Polygon-polygon intersection (`AnyPart` overlap test).
///
/// True when any edges cross, or one polygon has a vertex inside the other.
pub fn polys_intersect(a: &[Vec2], b: &[Vec2]) -> bool {
    if a.len() < 3 || b.len() < 3 {
        return false;
    }
    // Fast reject on bbox.
    if bbox_of(a).intersect(&bbox_of(b)).is_none() {
        return false;
    }
    let na = a.len();
    let nb = b.len();
    for i in 0..na {
        let p1 = a[i];
        let p2 = a[(i + 1) % na];
        // Vertex containment (both directions).
        if point_in_poly(p1, b, 0.0) || point_in_poly(p2, b, 0.0) {
            return true;
        }
        for j in 0..nb {
            let q1 = b[j];
            let q2 = b[(j + 1) % nb];
            if segs_intersect(p1, p2, q1, q2) {
                return true;
            }
        }
    }
    false
}

/// Bounding-box overlap (`Exact` overlap test), with `eps` slack.
pub fn boxes_overlap(a: &Box2, b: &Box2, eps: f64) -> bool {
    !(a.max.u + eps < b.min.u
        || b.max.u + eps < a.min.u
        || a.max.v + eps < b.min.v
        || b.max.v + eps < a.min.v)
}

/// Overlap test per the detection mode.
pub fn overlap(a: &[Vec2], b: &[Vec2], mode: crate::params::OverlapDetectionMode, eps: f64) -> bool {
    match mode {
        crate::params::OverlapDetectionMode::AnyPart => polys_intersect(a, b),
        crate::params::OverlapDetectionMode::Exact => boxes_overlap(&bbox_of(a), &bbox_of(b), eps),
    }
}

/// Does the polygon self-intersect (non-simple contour)?
pub fn self_intersects(poly: &[Vec2]) -> bool {
    if poly.len() < 4 {
        return false;
    }
    let n = poly.len();
    for i in 0..n {
        let p1 = poly[i];
        let p2 = poly[(i + 1) % n];
        for j in (i + 2)..n {
            // Skip adjacent edges (i+1 and wrap i-1).
            if j == i + 1 {
                continue;
            }
            let q1 = poly[j];
            let q2 = poly[(j + 1) % n];
            // Also skip the wrap-adjacent pair (i, n-1).
            if j == n - 1 && i == 0 {
                continue;
            }
            if segs_intersect(p1, p2, q1, q2) {
                // Collinear shared vertices (touching, not crossing) are OK:
                // require a *proper* crossing or a vertex-in-segment interior.
                let proper = proper_cross(p1, p2, q1, q2);
                if proper {
                    return true;
                }
            }
        }
    }
    false
}

/// Proper crossing (interior of both segments cross).
fn proper_cross(p1: Vec2, p2: Vec2, q1: Vec2, q2: Vec2) -> bool {
    let d1 = orient(q1, q2, p1);
    let d2 = orient(q1, q2, p2);
    let d3 = orient(p1, p2, q1);
    let d4 = orient(p1, p2, q2);
    d1 != d2 && d3 != d4
}

/// Does the polygon have holes? For a single-contour island, a hole manifests
/// as a self-intersecting (non-simple) outline (`check_holes`).
pub fn has_holes(poly: &[Vec2]) -> bool {
    self_intersects(poly)
}

/// Snap a polygon to a grid of `step` (pixel-perfect alignment).
pub fn snap_to_grid(poly: &[Vec2], step: f64) -> Vec<Vec2> {
    if step <= 0.0 {
        return poly.to_vec();
    }
    poly.iter().map(|&p| Vec2::new((p.u / step).round() * step, (p.v / step).round() * step)).collect()
}

/// Snap only the vertices that lie on (or near) the polygon's bounding-box
/// border (the `Bbox` vertex-alignment mode).
pub fn snap_bbox_border(poly: &[Vec2], step: f64) -> Vec<Vec2> {
    if step <= 0.0 {
        return poly.to_vec();
    }
    let b = bbox_of(poly);
    let tol = step * 0.5;
    poly.iter()
        .map(|&p| {
            let on_border = (p.u - b.min.u).abs() < tol
                || (p.u - b.max.u).abs() < tol
                || (p.v - b.min.v).abs() < tol
                || (p.v - b.max.v).abs() < tol;
            if on_border {
                Vec2::new((p.u / step).round() * step, (p.v / step).round() * step)
            } else {
                p
            }
        })
        .collect()
}

/// Snap the 4 vertices nearest the bbox corners (the `BboxCorners` mode).
pub fn snap_bbox_corners(poly: &[Vec2], step: f64) -> Vec<Vec2> {
    if step <= 0.0 {
        return poly.to_vec();
    }
    let b = bbox_of(poly);
    let corners = [
        b.min,
        Vec2::new(b.max.u, b.min.v),
        b.max,
        Vec2::new(b.min.u, b.max.v),
    ];
    let mut out = poly.to_vec();
    for c in corners {
        // Find the vertex nearest to corner c and snap it.
        let mut best = 0usize;
        let mut bd = f64::INFINITY;
        for (i, &p) in out.iter().enumerate() {
            let d = ((p.u - c.u).powi(2) + (p.v - c.v).powi(2)).sqrt();
            if d < bd {
                bd = d;
                best = i;
            }
        }
        out[best] = Vec2::new((out[best].u / step).round() * step, (out[best].v / step).round() * step);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(x: f64, y: f64, s: f64) -> Vec<Vec2> {
        vec![
            Vec2::new(x, y),
            Vec2::new(x + s, y),
            Vec2::new(x + s, y + s),
            Vec2::new(x, y + s),
        ]
    }

    #[test]
    fn area_and_bbox() {
        let q = square(1.0, 2.0, 0.5);
        assert!((area(&q) - 0.25).abs() < 1e-12);
        let b = bbox_of(&q);
        assert_eq!(b.min, Vec2::new(1.0, 2.0));
        assert_eq!(b.max, Vec2::new(1.5, 2.5));
    }

    #[test]
    fn point_in_poly() {
        let q = square(0.0, 0.0, 1.0);
        assert!(super::point_in_poly(Vec2::new(0.5, 0.5), &q, 0.0));
        assert!(!super::point_in_poly(Vec2::new(1.5, 0.5), &q, 0.0));
        assert!(super::point_in_poly(Vec2::new(0.5, 0.0), &q, 1e-9)); // boundary
    }

    #[test]
    fn overlap_modes() {
        use crate::params::OverlapDetectionMode as M;
        let a = square(0.0, 0.0, 1.0);
        let b = square(0.5, 0.5, 1.0);
        assert!(overlap(&a, &b, M::AnyPart, 0.0));
        assert!(overlap(&a, &b, M::Exact, 0.0));
        // Bboxes overlap but polygons don't: two diagonal "diamonds" whose
        // bbox ranges intersect on both axes but whose interiors are clear
        // (L1 distance between centers 0.24 > 0.2 = sum of the radii).
        let c = vec![
            Vec2::new(1.0, 0.9),
            Vec2::new(1.1, 1.0),
            Vec2::new(1.0, 1.1),
            Vec2::new(0.9, 1.0),
        ];
        let d = vec![
            Vec2::new(1.12, 1.02),
            Vec2::new(1.22, 1.12),
            Vec2::new(1.12, 1.22),
            Vec2::new(1.02, 1.12),
        ];
        assert!(!overlap(&c, &d, M::AnyPart, 0.0));
        assert!(overlap(&c, &d, M::Exact, 0.0));
    }

    #[test]
    fn self_intersection() {
        let bowtie = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 1.0),
        ];
        assert!(self_intersects(&bowtie));
        assert!(!self_intersects(&square(0.0, 0.0, 1.0)));
    }

    #[test]
    fn snap_grid() {
        let q = square(0.123, 0.456, 0.111);
        let s = snap_to_grid(&q, 0.1);
        assert!((s[0].u - 0.1).abs() < 1e-12);
        assert!((s[0].v - 0.5).abs() < 1e-12);
    }
}
