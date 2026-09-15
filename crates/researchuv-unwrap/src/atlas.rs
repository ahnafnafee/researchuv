//! The island-packer bridge — runs the configurable `researchuv-pack` engine
//! on the pipeline's rectified charts and converts its per-island transforms
//! back into final UV coordinates.
//!
//! Each chart's rectified `[0,1]²` UVs are re-scaled by their true 2-D extent
//! (`u·ext.u, v·ext.v`) so relative island sizes survive packing (a texel on
//! a big chart covers the same UV distance as on a small one). The pack
//! engine consumes *outline polygons*: a chart's ring set is its border loops
//! mapped to world units — the largest loop is the outer outline, loops
//! strictly inside it become holes (annulus-like charts), and loops that lie
//! beside the outer loop (the top/bottom edges of a cylinder strip) fall back
//! to the convex hull of their union. Closed charts (no loops at all) use the
//! convex hull of all their points.

use crate::pack::Placed;
use crate::segment::Chart;
use researchuv_math::Vec2;

/// A chart's packable ring set: the outer outline plus interior holes, in
/// world-scaled UV units.
#[derive(Clone, Debug, Default)]
pub struct ChartRings {
    pub outer: Vec<Vec2>,
    pub holes: Vec<Vec<Vec2>>,
}

/// Run the island packer over per-chart ring sets.
///
/// `rings[i]` is chart `i`'s outline + holes in world-scaled UV units;
/// `params` is the full pack option surface (rotation, margins, scale mode,
/// target box, …). Returns the per-chart placement transforms (in input
/// order; `None` where the island did not fit), the packer's own result
/// contract, and the largest applied per-island scale.
pub fn pack_outlines(
    rings: Vec<ChartRings>,
    params: &researchuv_pack::PackParams,
) -> (Vec<Option<researchuv_pack::PlacedTransform>>, researchuv_pack::PackResult, f64) {
    let mut islands: Vec<researchuv_pack::Island> = rings
        .into_iter()
        .map(|r| researchuv_pack::Island::from_polygon_with_holes(r.outer, r.holes))
        .collect();
    let result = researchuv_pack::pipeline::pack(&mut islands, params);
    let scale = result
        .placed
        .iter()
        .flatten()
        .map(|t| t.scale)
        .fold(f64::NEG_INFINITY, f64::max);
    let scale = if scale.is_finite() { scale } else { 0.0 };
    (result.placed.clone(), result, scale)
}

/// The ring set of a chart for packing: border loops mapped to world-scaled
/// rectified UVs. The largest-|area| loop is the outer outline; loops
/// strictly inside it become holes; loops beside it join the (hull) outline.
/// Closed charts use the convex hull of all points.
pub fn chart_rings(chart: &Chart, rectified: &[Vec2], ext: Vec2) -> ChartRings {
    let world = |p: Vec2| Vec2::new(p.u * ext.u.max(1e-30), p.v * ext.v.max(1e-30));
    if chart.border_loops.is_empty() {
        let all: Vec<Vec2> = rectified.iter().map(|&p| world(p)).collect();
        return ChartRings { outer: convex_hull(&all), holes: Vec::new() };
    }
    // Map every loop into world UV space (drop the closing repeat).
    let mut loops: Vec<Vec<Vec2>> = Vec::with_capacity(chart.border_loops.len());
    for l in &chart.border_loops {
        let closed = l.first() == l.last() && l.len() > 1;
        let verts: &[u32] = if closed { &l[..l.len() - 1] } else { l.as_slice() };
        let pts: Vec<Vec2> = verts
            .iter()
            .filter_map(|&v| chart.local_of(v))
            .map(|li| world(rectified[li]))
            .collect();
        if pts.len() >= 3 {
            loops.push(pts);
        }
    }
    if loops.is_empty() {
        let all: Vec<Vec2> = rectified.iter().map(|&p| world(p)).collect();
        return ChartRings { outer: convex_hull(&all), holes: Vec::new() };
    }
    // Outer = the largest |signed area| loop.
    let mut outer_idx = 0usize;
    let mut outer_area = f64::NEG_INFINITY;
    for (i, l) in loops.iter().enumerate() {
        let a = shoelace(l).abs();
        if a > outer_area {
            outer_area = a;
            outer_idx = i;
        }
    }
    let outer = loops.swap_remove(outer_idx);
    // Remaining loops: holes when strictly inside the outer loop, else they
    // border the outline from outside (cylinder strips) → hull fallback.
    let mut holes: Vec<Vec<Vec2>> = Vec::new();
    let mut outside: Vec<Vec2> = Vec::new();
    for l in loops {
        let inside = l.iter().all(|&p| point_strictly_inside(p, &outer));
        if inside {
            holes.push(l);
        } else {
            outside.extend(l);
        }
    }
    if outside.is_empty() {
        return ChartRings { outer, holes };
    }
    // Hull of the outer + outside loops (conservative union outline); holes
    // that survive inside the hull are kept.
    let mut pts = outer.clone();
    pts.extend(outside);
    let hull = convex_hull(&pts);
    let holes = holes
        .into_iter()
        .filter(|h| h.iter().all(|&p| point_strictly_inside(p, &hull)))
        .collect();
    ChartRings { outer: hull, holes }
}

fn shoelace(poly: &[Vec2]) -> f64 {
    let n = poly.len();
    let mut a = 0.0;
    for i in 0..n {
        let p = poly[i];
        let q = poly[(i + 1) % n];
        a += p.u * q.v - q.u * p.v;
    }
    0.5 * a
}

/// Strict interior test (excludes the boundary by `eps`).
fn point_strictly_inside(p: Vec2, poly: &[Vec2]) -> bool {
    if poly.len() < 3 {
        return false;
    }
    let eps = 1e-9;
    let mut inside = false;
    let n = poly.len();
    let mut j = n - 1;
    for i in 0..n {
        let a = poly[i];
        let b = poly[j];
        if (a.v > p.v) != (b.v > p.v) {
            let t = (p.v - a.v) / (b.v - a.v);
            let xu = a.u + t * (b.u - a.u);
            if p.u < xu {
                inside = !inside;
            }
        }
        // On the boundary → not strictly inside.
        if on_segment(a, b, p, eps) {
            return false;
        }
        j = i;
    }
    inside
}

fn on_segment(a: Vec2, b: Vec2, p: Vec2, eps: f64) -> bool {
    let cross = (b.u - a.u) * (p.v - a.v) - (b.v - a.v) * (p.u - a.u);
    if cross.abs() > eps * ((b - a).len() + 1e-30) {
        return false;
    }
    p.u >= a.u.min(b.u) - eps
        && p.u <= a.u.max(b.u) + eps
        && p.v >= a.v.min(b.v) - eps
        && p.v <= a.v.max(b.v) + eps
}

/// Andrew's monotone-chain convex hull (CCW, no collinear points).
pub fn convex_hull(points: &[Vec2]) -> Vec<Vec2> {
    let mut pts = points.to_vec();
    pts.sort_by(|a, b| {
        a.u.partial_cmp(&b.u)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.v.partial_cmp(&b.v).unwrap_or(std::cmp::Ordering::Equal))
    });
    pts.dedup_by(|a, b| (*a - *b).len() < 1e-12);
    if pts.len() < 3 {
        return pts;
    }
    let cross = |o: Vec2, a: Vec2, b: Vec2| (a.u - o.u) * (b.v - o.v) - (a.v - o.v) * (b.u - o.u);
    let mut lower: Vec<Vec2> = Vec::new();
    for &p in &pts {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 1e-18
        {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper: Vec<Vec2> = Vec::new();
    for &p in pts.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 1e-18
        {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/// Convert a pack placement transform into the pipeline's [`Placed`] box.
pub fn placed_of(t: &researchuv_pack::PlacedTransform) -> Placed {
    Placed {
        x: t.box_.min.u,
        y: t.box_.min.v,
        w: t.box_.width(),
        h: t.box_.height(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hull_of_a_square_is_the_square() {
        let pts = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(0.0, 1.0),
            Vec2::new(0.5, 0.5),
            Vec2::new(0.25, 0.75),
        ];
        let h = convex_hull(&pts);
        assert_eq!(h.len(), 4);
        let area: f64 = shoelace(&h);
        assert!((area.abs() - 1.0).abs() < 1e-12, "hull area {area}");
    }

    #[test]
    fn hull_is_ccw() {
        let pts = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(2.0, 1.0),
            Vec2::new(0.0, 1.0),
        ];
        let h = convex_hull(&pts);
        assert!(shoelace(&h) > 0.0, "CCW hull has positive signed area");
    }

    #[test]
    fn two_square_ring_sets_pack_into_the_unit_target() {
        let rings = vec![
            ChartRings {
                outer: vec![
                    Vec2::new(0.0, 0.0),
                    Vec2::new(1.0, 0.0),
                    Vec2::new(1.0, 1.0),
                    Vec2::new(0.0, 1.0),
                ],
                holes: Vec::new(),
            },
            ChartRings {
                outer: vec![
                    Vec2::new(0.0, 0.0),
                    Vec2::new(1.0, 0.0),
                    Vec2::new(1.0, 1.0),
                    Vec2::new(0.0, 1.0),
                ],
                holes: Vec::new(),
            },
        ];
        let params = researchuv_pack::PackParams::default();
        let (placed, result, _scale) = pack_outlines(rings, &params);
        assert_eq!(placed.len(), 2);
        assert!(placed.iter().all(|p| p.is_some()), "both squares placed");
        assert_eq!(result.retcode, researchuv_pack::params::UvpmRetcode::Success);
        // Transforms map the world outlines into the unit target.
        for t in placed.iter().flatten() {
            assert!(t.box_.min.u >= -1e-9 && t.box_.min.v >= -1e-9);
            assert!(t.box_.max.u <= 1.0 + 1e-9 && t.box_.max.v <= 1.0 + 1e-9);
        }
    }

    #[test]
    fn hole_packs_with_another_island_in_the_gap() {
        // An annulus-like chart (outer 1×1 with a central 0.5×0.5 hole) and a
        // small island: the small island may legally occupy the hole area —
        // ring-set validation must not flag it.
        let rings = vec![
            ChartRings {
                outer: vec![
                    Vec2::new(0.0, 0.0),
                    Vec2::new(1.0, 0.0),
                    Vec2::new(1.0, 1.0),
                    Vec2::new(0.0, 1.0),
                ],
                holes: vec![vec![
                    Vec2::new(0.3, 0.3),
                    Vec2::new(0.7, 0.3),
                    Vec2::new(0.7, 0.7),
                    Vec2::new(0.3, 0.7),
                ]],
            },
            ChartRings {
                outer: vec![
                    Vec2::new(0.0, 0.0),
                    Vec2::new(0.2, 0.0),
                    Vec2::new(0.2, 0.2),
                    Vec2::new(0.0, 0.2),
                ],
                holes: Vec::new(),
            },
        ];
        let mut params = researchuv_pack::PackParams::default();
        params.rotation_enable = false;
        params.scale_mode = researchuv_pack::params::ScaleMode::FixedScale;
        params.scale = 0.4;
        let (placed, result, _s) = pack_outlines(rings, &params);
        assert!(placed.iter().all(|p| p.is_some()));
        // The placement engine works on boxes, so the small island may sit in
        // the hole; the ring-set (filled-region) validation stays quiet.
        assert!(result.validation.overlapping.is_empty(), "{:?}", result.validation.overlapping);
        assert_eq!(result.retcode, researchuv_pack::params::UvpmRetcode::Success);
    }

    #[test]
    fn placed_of_rounds_the_transform_box() {
        let t = researchuv_pack::PlacedTransform::from_parts(
            0.0,
            false,
            0.5,
            0.1,
            0.2,
            researchuv_pack::box2::Box2::new(Vec2::new(0.1, 0.2), Vec2::new(0.6, 0.7)),
        );
        let p = placed_of(&t);
        assert!((p.x - 0.1).abs() < 1e-12);
        assert!((p.y - 0.2).abs() < 1e-12);
        assert!((p.w - 0.5).abs() < 1e-12);
        assert!((p.h - 0.5).abs() < 1e-12);
    }
}
