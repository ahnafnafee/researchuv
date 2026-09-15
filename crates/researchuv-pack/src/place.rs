//! Placement engine — the CPU analog of the engine's Vulkan free-space search
//! (`CSMainVulkanFindBestRow`, `CSVulkanFindLocationGroups`, `walkingSearch`).
//!
//! The GPU engine maintains a free-space representation (CDT triangles +
//! scanline) and searches for the best (row, location) for each island. This
//! CPU analog is exact for axis-aligned placements: candidate anchor
//! coordinates are the target-box origin and the (gap-offset) extents of every
//! already-placed island; candidate scales are the largest scale that keeps the
//! placed bbox disjoint from every neighbor and inside the target. The result
//! — the same (transform, flags, retcode) contract — is independent of the
//! representation.
//!
//! Placement operates on island **bounding boxes** (the engine's `SBox2` /
//! `max_island_dimension` vocabulary); the rotated/flipped island outline is
//! emitted afterward. Overlap of the *outlines* is detected by validation, per
//! `UvpmOverlapDetectionMode`.

use crate::box2::{Box2, BoxCorner};
use crate::island::{Island, PlacedTransform};
use crate::params::{PackParams, PackStrategy, ScaleMode};
use crate::rng::SplitMix64;
use researchuv_math::Vec2;

/// A placed island: transform + placed bbox in target space.
#[derive(Clone, Debug)]
pub struct Placed {
    pub island_index: u32,
    pub transform: PlacedTransform,
    /// Placed bbox in target space (what neighbors are measured against).
    pub box_: Box2,
}

/// Bounding size of the island at a rotation (unflipped), for scale decisions.
pub fn rotated_size(island: &Island, rotation: f64) -> (f64, f64) {
    let w = island.bbox.width();
    let h = island.bbox.height();
    let (sr, cr) = (rotation.sin().abs(), rotation.cos().abs());
    // Rotating a w×h box by θ gives a bounding box of
    // (w·|cr| + h·|sr|) × (w·|sr| + h·|cr|).
    (w * cr + h * sr, w * sr + h * cr)
}

/// The min corner of the island's bbox after rotation/flip/scale (no
/// translation). Used to map the island's bbox min corner onto the anchor so
/// the placed `box` exactly matches the transformed geometry.
pub fn local_min_corner(island: &Island, rotation: f64, flipped: bool, scale: f64) -> (f64, f64) {
    let (b0, b1) = (island.bbox.min, island.bbox.max);
    let (sr, cr) = (rotation.sin(), rotation.cos());
    let fx = if flipped { -1.0 } else { 1.0 };
    let mut mnx = f64::INFINITY;
    let mut mny = f64::INFINITY;
    for &(pu, pv) in [(b0.u, b0.v), (b1.u, b0.v), (b0.u, b1.v), (b1.u, b1.v)].iter() {
        let q = (
            fx * (cr * scale * pu - sr * scale * pv),
            fx * (sr * scale * pu + cr * scale * pv),
        );
        mnx = mnx.min(q.0);
        mny = mny.min(q.1);
    }
    (mnx, mny)
}

/// Max scale that keeps a box of size (w,h) at anchor inside `target` with
/// `border` gaps on all sides (the box extends +u/+v from the anchor).
pub fn max_scale_in_box(
    w: f64,
    h: f64,
    anchor: Vec2,
    target: &Box2,
    border: f64,
) -> Option<f64> {
    if anchor.u < target.min.u + border - 1e-12 || anchor.v < target.min.v + border - 1e-12 {
        return None;
    }
    if w <= 0.0 || h <= 0.0 {
        return Some(1.0);
    }
    let sx = (target.max.u - border - anchor.u) / w;
    let sy = (target.max.v - border - anchor.v) / h;
    Some(sx.min(sy).max(0.0))
}

/// The border gap as a function of scale: constant in the pixel-margin modes
/// (px / tex size), proportional to the island extent in the relative mode.
pub fn border_at_scale(params: &PackParams, w0: f64, s: f64) -> f64 {
    if params.pixel_border_margin_enable || params.pixel_margin_enable {
        params.border_gap(1.0)
    } else {
        params.border_gap(w0 * s)
    }
}

/// Max scale that keeps the box of size (w,h) at `anchor` inside `target`
/// with the scale-dependent border gap (relative margins: `border(s) =
/// margin·extent·s` — the feasibility predicate is monotone in s, so the
/// largest feasible scale is found by binary search between 0 and the
/// border-free upper bound). The box may be nudged inward from `anchor` to
/// clear the border; the (possibly nudged) box min corner is returned in
/// `out_anchor`.
pub fn max_scale_in_box_rel(
    w: f64,
    h: f64,
    anchor: Vec2,
    target: &Box2,
    params: &PackParams,
    w0: f64,
    out_anchor: &mut Vec2,
) -> Option<f64> {
    if w <= 0.0 || h <= 0.0 {
        *out_anchor = Vec2::new(
            anchor.u.max(target.min.u),
            anchor.v.max(target.min.v),
        );
        return Some(1.0);
    }
    // Feasibility at scale s: the box [a(s), a(s) + (w,h)·s] inside
    // [min + b(s), max − b(s)], where a(s) = max(anchor, min + b(s)) (the
    // box is nudged inward to clear the border). Monotone in s: in relative
    // mode the border grows with s, so the feasible set is [0, s*].
    let fits = |s: f64| -> bool {
        let b = border_at_scale(params, w0, s);
        let au = anchor.u.max(target.min.u + b);
        let av = anchor.v.max(target.min.v + b);
        au + w * s <= target.max.u - b + 1e-12 && av + h * s <= target.max.v - b + 1e-12
    };
    if !fits(0.0) {
        return None;
    }
    // Border-free upper bound (≥ s*): the box extends from the anchor
    // clamped into the target, with no border.
    let au0 = anchor.u.max(target.min.u);
    let av0 = anchor.v.max(target.min.v);
    let mut hi = (((target.max.u - au0) / w).min((target.max.v - av0) / h)).max(0.0);
    if hi <= 0.0 {
        return None;
    }
    let mut lo = 0.0f64;
    for _ in 0..64 {
        let mid = (lo + hi) / 2.0;
        if fits(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let s = lo;
    if s <= 1e-15 {
        return None;
    }
    let b = border_at_scale(params, w0, s);
    *out_anchor = Vec2::new(anchor.u.max(target.min.u + b), anchor.v.max(target.min.v + b));
    Some(s)
}

/// Max scale such that the box at anchor with size (w,h)·s is disjoint (gap on
/// the placed side) from every already-placed neighbor.
///
/// Per neighbor the disjoint condition is
/// `(s ≤ (n.min.u − gap − x)/w) OR (s ≤ (n.min.v − gap − y)/h)` (or the
/// neighbor is already clear at s = 0). Intersecting over neighbors gives
/// `s ≤ max(sx_k, sy_k)` per neighbor, i.e. `s ≤ min_k max(sx_k, sy_k)`.
pub fn max_scale_clear_of(
    w: f64,
    h: f64,
    anchor: Vec2,
    placed: &[Placed],
    gap: f64,
) -> f64 {
    let mut s = f64::INFINITY;
    for p in placed {
        if p.box_.max.u + gap <= anchor.u || p.box_.max.v + gap <= anchor.v {
            continue; // clear at any scale
        }
        let sx = if w > 0.0 { (p.box_.min.u - gap - anchor.u) / w } else { f64::INFINITY };
        let sy = if h > 0.0 { (p.box_.min.v - gap - anchor.v) / h } else { f64::INFINITY };
        let allowed = sx.max(sy);
        if allowed < s {
            s = allowed;
        }
    }
    s
}

/// Anchor candidate coordinates: the target origin, the target max (the
/// bottom-right corner), the (gap-offset) edges of every placed island, and
/// the *free strips* the border gap leaves around the target. The gap-offset
/// edges matter because the max-clear-scale can only shrink the island, so a
/// strip that only fits at small scale (e.g. to the right of an island that
/// nearly fills the target) must be entered through its gap-offset edge.
pub fn anchor_axis(target: &Box2, placed: &[Placed], params: &PackParams) -> Vec<f64> {
    let mut xs: Vec<f64> = vec![target.min.u, target.max.u];
    for p in placed {
        // Max scale at which an island could still clear the target border.
        let s = (target.width().max(target.height()) / (p.box_.max_extent().max(1e-12) + 1e-12))
            .min(1.0)
            .max(0.0);
        let gap = params.island_gap(p.box_.max_extent(), p.box_.max_extent().max(s));
        xs.push(p.box_.max.u + gap);
        xs.push(p.box_.min.u - gap);
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    xs.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
    xs
}

/// Score a placement per the pack strategy (lower is better).
///
/// - `SideToSideVert`: side-to-side along u (columns); a new row (larger v)
///   is the dominant cost, so rows advance only once the current row is full.
/// - `SideToSideHori`: side-to-side along v (rows); a new column (larger u)
///   is the dominant cost.
/// - `Square` / `Automatic`: Manhattan distance from the start corner.
pub fn placement_score(
    strategy: PackStrategy,
    anchor: Vec2,
    target: &Box2,
    start: BoxCorner,
) -> f64 {
    let u_at_start = match start {
        BoxCorner::Bl | BoxCorner::Tl => target.min.u,
        BoxCorner::Br | BoxCorner::Tr => target.max.u,
    };
    let v_at_start = match start {
        BoxCorner::Bl | BoxCorner::Br => target.min.v,
        BoxCorner::Tl | BoxCorner::Tr => target.max.v,
    };
    let du = (anchor.u - u_at_start).abs();
    let dv = (anchor.v - v_at_start).abs();
    match strategy {
        PackStrategy::SideToSideVert => dv * 1e6 + du,
        PackStrategy::SideToSideHori => du * 1e6 + dv,
        PackStrategy::Square | PackStrategy::Automatic => du + dv,
    }
}

/// Find the best placement for `island` given the `placed` set.
///
/// Returns `None` when no (orientation, scale, anchor) fits — the caller marks
/// it non-packed (and optionally arranges it outside the target).
pub fn find_best_placement(
    island: &Island,
    placed: &[Placed],
    target: &Box2,
    params: &PackParams,
    rng: &mut SplitMix64,
) -> Option<Placed> {
    let w0 = island.bbox.width();
    let h0 = island.bbox.height();
    if w0 <= 0.0 || h0 <= 0.0 {
        return None; // degenerate — nothing to place
    }

    let strategy = effective_strategy(params);
    let rotations = params.rotation_candidates(0);
    let mut flips = vec![false];
    if params.flipping_enable {
        flips.push(true);
    }
    let ax = anchor_axis(target, placed, params);
    let ay = anchor_axis(target, placed, params);

    let mut best: Option<(f64, f64, Placed)> = None; // (scale, score, Placed)
    let mut tried = 0u64;

    'outer: for &rot in rotations.iter() {
        let (rw, rh) = rotated_size(island, rot);
        // Rotation-specific global scale bound. MaxScale mode means "the
        // largest scale at which ALL islands fit" — the engine picks one
        // uniform scale for the batch. Per island we approximate that by
        // requiring room for a *second copy* of the same size: two copies fit
        // if they sit side-by-side along u (width `2·rw·s + 3·m·ext·s`,
        // height `rh·s + 2·m·ext·s`) OR stacked along v (the transpose). In
        // relative mode border = gap = m·(placed extent) = m·ext·s per island,
        // so the per-unit-scale budgets are as below; the bound is the max of
        // the two arrangements (a thin island may fill its long axis when the
        // second copy fits beside it along the short axis).
        let s_max_global_rot = match params.scale_mode {
            ScaleMode::MaxScale => {
                let m = params.margin;
                let ext = rw.max(rh);
                if params.pixel_border_margin_enable || params.pixel_margin_enable {
                    // Pixel margins are fixed UV offsets (not scale-dependent).
                    let px = if params.pixel_border_margin_enable {
                        params.pixel_border_margin
                    } else {
                        params.pixel_margin
                    };
                    let m_uv = px as f64 / params.pixel_margin_tex_size.max(1) as f64;
                    let side = (target.width() / (2.0 * rw + 3.0 * m_uv))
                        .min(target.height() / (rh + 2.0 * m_uv));
                    let stack = (target.width() / (rw + 2.0 * m_uv))
                        .min(target.height() / (2.0 * rh + 3.0 * m_uv));
                    side.max(stack)
                } else {
                    let side = (target.width() / (2.0 * rw + 3.0 * m * ext))
                        .min(target.height() / (rh + 2.0 * m * ext));
                    let stack = (target.width() / (rw + 2.0 * m * ext))
                        .min(target.height() / (2.0 * rh + 3.0 * m * ext));
                    side.max(stack)
                }
                .max(1e-9)
            }
            ScaleMode::FixedScale | ScaleMode::FixedScaleMaxMargin => params.scale.max(1e-9),
        };
        if s_max_global_rot <= 0.0 {
            return None;
        }
        // The border at the rotation's max scale (the anchor is nudged inward
        // by this amount when the island is placed at its max scale).
        let b_max = border_at_scale(params, w0, s_max_global_rot);
        // Reference scale for gap-offset anchors: the largest scale at which
        // the island could clear the target border, i.e. its own placement
        // scale when nothing else is in the way (capped at the global bound).
        let s_ref = s_max_global_rot;
        let gap_ref = params.island_gap(rw * s_ref, rh * s_ref);
        // Border-aware + gap-aware anchor set: the raw edges, each edge
        // offset by the reference gap (so a new island can sit flush against
        // a neighbor's edge + its gap at full scale), and each edge nudged
        // by the scale-dependent border (so an island placed at its max
        // scale can sit flush against the target edge with its border
        // clearing).
        let mut ax_b: Vec<f64> = Vec::with_capacity(ax.len() * 5);
        for &a in ax.iter() {
            ax_b.push(a);
            ax_b.push(a + gap_ref);
            ax_b.push(a - gap_ref);
            ax_b.push(a + gap_ref + b_max);
            ax_b.push(a - gap_ref - b_max);
        }
        ax_b.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        ax_b.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
        let mut ay_b: Vec<f64> = Vec::with_capacity(ay.len() * 5);
        for &a in ay.iter() {
            ay_b.push(a);
            ay_b.push(a + gap_ref);
            ay_b.push(a - gap_ref);
            ay_b.push(a + gap_ref + b_max);
            ay_b.push(a - gap_ref - b_max);
        }
        ay_b.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        ay_b.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
        for &fl in flips.iter() {
            for &axu in ax_b.iter() {
                for &ayv in ay_b.iter() {
                    let anchor = Vec2::new(axu, ayv);
                    // Border-aware max scale (nudges the box inward to clear
                    // the scale-dependent border gap).
                    let mut anchor_eff = anchor;
                    let Some(mut s) =
                        max_scale_in_box_rel(rw, rh, anchor, target, params, w0, &mut anchor_eff)
                    else {
                        continue;
                    };
                    s = s.min(s_max_global_rot);
                    if s <= 0.0 {
                        continue;
                    }
                    // Clearance at the candidate scale.
                    let gap = params.island_gap(w0 * s, h0 * s);
                    let s_clear = max_scale_clear_of(rw, rh, anchor_eff, placed, gap);
                    s = s.min(s_clear);
                    if s <= 0.0 {
                        continue;
                    }
                    // Refine once: the clearance gap depends on the final scale.
                    let gap_f = params.island_gap(w0 * s, h0 * s);
                    let s_clear_f = max_scale_clear_of(rw, rh, anchor_eff, placed, gap_f);
                    s = s.min(s_clear_f);
                    if s <= 0.0 {
                        continue;
                    }
                    let s_final = match params.scale_mode {
                        ScaleMode::MaxScale => s,
                        ScaleMode::FixedScale | ScaleMode::FixedScaleMaxMargin => {
                            let fixed = params.scale;
                            if fixed > s + 1e-9 {
                                continue; // doesn't fit at the fixed scale
                            }
                            fixed
                        }
                    };

                    // Row alignment (side-to-side): if this island sits
                    // side-by-side with an already-placed island, snap its v
                    // to that island's row so same-row islands share a common
                    // v baseline (independent of each island's own scale and
                    // scale-dependent border nudge).
                    if let Some(rv) = row_snap_v(anchor_eff, rw * s_final, rh * s_final, placed) {
                        anchor_eff.v = rv;
                    }

                    let b = Box2::new(
                        anchor_eff,
                        Vec2::new(
                            anchor_eff.u + rw * s_final,
                            anchor_eff.v + rh * s_final,
                        ),
                    );
                    // Final verification (covers the FixedScale path).
                    let gap_f = params.island_gap(w0 * s_final, h0 * s_final);
                    if !island_box_clear(&b, placed, gap_f)
                        || !target.contains_box_eps(&b, 1e-9)
                    {
                        continue;
                    }

                    tried += 1;
                    let jitter = if tried % 7 == 0 { rng.next_f64() * 1e-12 } else { 0.0 };
                    let score =
                        placement_score(strategy, anchor_eff, target, params.start_corner)
                            + jitter;

                    let (lmin_u, lmin_v) = local_min_corner(island, rot, fl, s_final);
                    let transform = PlacedTransform::from_parts(
                        rot,
                        fl,
                        s_final,
                        anchor_eff.u - lmin_u,
                        anchor_eff.v - lmin_v,
                        b,
                    );
                    let cand = Placed {
                        island_index: 0, // set by the caller
                        transform,
                        box_: b,
                    };
                    let is_better = match best {
                        Some((bs, bsc, _)) => {
                            // MaxScale: the scale is the objective — a larger
                            // scale wins outright; at (near-)equal scale the
                            // strategy score decides. Fixed-scale modes all
                            // share the scale, so the score decides.
                            s_final > bs + 1e-9 || (s_final >= bs - 1e-9 && score < bsc)
                        }
                        None => true,
                    };
                    if is_better {
                        best = Some((s_final, score, cand));
                    }
                }
            }
        }
        if best.is_some() && !params.heuristic_enable {
            break 'outer;
        }
    }

    best.map(|(_, _, p)| p)
}

/// Is `box` disjoint (gap applied) from every placed island?
pub fn island_box_clear(b: &Box2, placed: &[Placed], gap: f64) -> bool {
    for p in placed {
        let xok = b.max.u + gap <= p.box_.min.u || p.box_.max.u + gap <= b.min.u;
        let yok = b.max.v + gap <= p.box_.min.v || p.box_.max.v + gap <= b.min.v;
        if !(xok || yok) {
            return false;
        }
    }
    true
}

/// Row alignment (side-to-side packing): if the box of size (w, h) at
/// `anchor` sits side-by-side with an already-placed island (horizontally
/// separated, vertically overlapping), return that island's `min.v` so the
/// new island aligns to the same row. Same-row islands share a common v
/// baseline regardless of their individual scales / border nudges. Returns
/// `None` when there is no horizontally-separated neighbor whose v-range
/// overlaps.
fn row_snap_v(anchor: Vec2, w: f64, h: f64, placed: &[Placed]) -> Option<f64> {
    let eps = 1e-9;
    for p in placed {
        // Side-by-side = horizontally separated (no u-overlap). The exact
        // separation can differ slightly from the island gap (anchors use a
        // reference gap), so only require horizontal separation.
        let to_right = anchor.u >= p.box_.max.u - eps;
        let to_left = anchor.u + w <= p.box_.min.u + eps;
        if !to_right && !to_left {
            continue;
        }
        if anchor.v < p.box_.max.v + eps && anchor.v + h > p.box_.min.v - eps {
            return Some(p.box_.min.v);
        }
    }
    None
}

/// Resolve `Automatic` to a concrete strategy (the engine's default heuristic:
/// side-to-side vertical for wide targets, horizontal for tall ones).
pub fn effective_strategy(params: &PackParams) -> PackStrategy {
    match params.pack_strategy {
        PackStrategy::Automatic => {
            let b = params.effective_box();
            if b.width() >= b.height() {
                PackStrategy::SideToSideVert
            } else {
                PackStrategy::SideToSideHori
            }
        }
        s => s,
    }
}

/// Arrange islands that did not fit (the `arrange_non_packed` option): place
/// them outside the target box, stacked along +v at the target's left edge.
pub fn arrange_non_packed(
    island: &Island,
    placed: &[Placed],
    target: &Box2,
    params: &PackParams,
) -> Option<Placed> {
    if !params.arrange_non_packed {
        return None;
    }
    let w0 = island.bbox.width();
    let h0 = island.bbox.height();
    if w0 <= 0.0 || h0 <= 0.0 {
        return None;
    }
    let gap = params.island_gap(w0, h0);
    let x = target.min.u;
    let mut y = target.max.v + gap;
    for p in placed {
        if p.box_.min.u <= x + 1e-9 && p.box_.max.u >= x + w0 - 1e-9 {
            y = y.max(p.box_.max.v + gap);
        }
    }
    let b = Box2::new(Vec2::new(x, y), Vec2::new(x + w0, y + h0));
    let (lmin_u, lmin_v) = local_min_corner(island, 0.0, false, 1.0);
    Some(Placed {
        island_index: 0,
        transform: PlacedTransform::from_parts(0.0, false, 1.0, x - lmin_u, y - lmin_v, b),
        box_: b,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::island::Island;

    fn sq(s: f64) -> Island {
        Island::from_polygon(vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(s, 0.0),
            Vec2::new(s, s),
            Vec2::new(0.0, s),
        ])
    }

    #[test]
    fn rotated_size_quarter() {
        let i = Island::from_polygon(vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(2.0, 1.0),
            Vec2::new(0.0, 1.0),
        ]);
        // 90° rotation of a 2×1 box → 1×2.
        let (w, h) = rotated_size(&i, std::f64::consts::FRAC_PI_2);
        assert!((w - 1.0).abs() < 1e-9);
        assert!((h - 2.0).abs() < 1e-9);
    }

    #[test]
    fn two_squares_stack_side_by_side() {
        let isls = vec![sq(0.4), sq(0.4)];
        let p = PackParams::default();
        let target = p.effective_box();
        let mut rng = SplitMix64::new(0);
        let mut placed: Vec<Placed> = Vec::new();
        for (idx, isl) in isls.iter().enumerate() {
            let pl = find_best_placement(isl, &placed, &target, &p, &mut rng).unwrap();
            placed.push(Placed {
                island_index: idx as u32,
                ..pl
            });
        }
        assert!(target.contains_box_eps(&placed[0].box_, 1e-9));
        assert!(target.contains_box_eps(&placed[1].box_, 1e-9));
        // Side-to-side (automatic → square target → vertical strategy):
        // second island is in the same row (same v), further along u.
        assert!(placed[1].box_.min.u > placed[0].box_.min.u);
        assert!((placed[1].box_.min.v - placed[0].box_.min.v).abs() < 1e-9);
    }

    #[test]
    fn scale_down_when_too_big() {
        let isl = sq(0.9); // 0.9² won't fit two side-by-side in the unit box
        let p = PackParams::default();
        let target = p.effective_box();
        let mut rng = SplitMix64::new(0);
        let mut placed: Vec<Placed> = Vec::new();
        for _ in 0..2 {
            let pl = find_best_placement(&isl, &placed, &target, &p, &mut rng).unwrap();
            placed.push(pl);
        }
        assert!(target.contains_box_eps(&placed[0].box_, 1e-9));
        assert!(target.contains_box_eps(&placed[1].box_, 1e-9));
        assert!(placed[0].transform.scale < 0.9);
        assert!(placed[1].transform.scale < 0.9);
    }

    #[test]
    fn fixed_scale_honored() {
        let isl = sq(0.5);
        let mut p = PackParams::default();
        p.scale_mode = ScaleMode::FixedScale;
        p.scale = 0.25;
        let target = p.effective_box();
        let mut rng = SplitMix64::new(0);
        let pl = find_best_placement(&isl, &[], &target, &p, &mut rng).unwrap();
        assert!((pl.transform.scale - 0.25).abs() < 1e-9);
        assert!((pl.box_.width() - 0.125).abs() < 1e-9); // 0.5 * 0.25
    }

    #[test]
    fn degenerate_island_not_placed() {
        let isl = Island::from_polygon(vec![Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0)]);
        let p = PackParams::default();
        let mut rng = SplitMix64::new(0);
        assert!(find_best_placement(&isl, &[], &p.effective_box(), &p, &mut rng).is_none());
    }

    #[test]
    fn rotation_used_to_fit_tall_island() {
        let isl = Island::from_polygon(vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(0.1, 0.0),
            Vec2::new(0.1, 0.9),
            Vec2::new(0.0, 0.9),
        ]);
        let mut p = PackParams::default();
        p.rotation_step = 90;
        let target = p.effective_box();
        let mut rng = SplitMix64::new(0);
        let pl = find_best_placement(&isl, &[], &target, &p, &mut rng).unwrap();
        // Unrotated it fills the box at scale 1.0 (0.9 tall ≤ 1.0). Rotated
        // 90° it is 0.9 wide × 0.1 tall and can scale to ~1.11 (border-aware).
        // Either way the placement is inside the box; the rotated candidate
        // must not be worse than the unrotated corner fill.
        assert!(target.contains_box_eps(&pl.box_, 1e-9));
        assert!(pl.transform.scale >= 0.95);
    }
}
