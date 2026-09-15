//! Heuristic search — the engine's time-limited stochastic placement
//! improvement (`heuristic_enable`, `heuristic_search_time`,
//! `advanced_heuristic` Auto/Disable/Enable, `heuristic_allow_mixed_scales`,
//! the "DoMaxima" improvement pass).
//!
//! Evidence: engine strings `heuristic_enable`, `heuristic_search_time`,
//! `advanced_heuristic`, `heuristic_allow_mixed_scales`, `DoMaxima`;
//! addon `spipeline/engine/props.py` (`HEURISTIC_ENABLE` default False,
//! `HEURISTIC_SEARCH_TIME` 0..3600 default 30.0, `ADVANCED_HEURISTIC`
//! enum Auto/Disable/Enable, `HEURISTIC_ALLOW_MIXED_SCALES` default False).
//!
//! Model: the base placement from [`crate::place::find_best_placement`] is
//! treated as the initial solution; then, for the duration of the time
//! budget, candidate re-placements of random islands are sampled
//! (rotation/flip re-rolls, anchors jittered around the candidate axes) and
//! accepted when the island stays inside the target (border gap applied),
//! stays clear of every other island (island gap applied), and the global
//! packing score (sum of per-island strategy scores from the start corner)
//! strictly decreases. With `heuristic_allow_mixed_scales` the island's
//! scale may grow up to the maximum feasible at the new location; otherwise
//! the island's current scale is kept.

use std::time::Instant;

use crate::box2::Box2;
use crate::island::{Island, PlacedTransform};
use crate::params::{AdvancedHeuristicMode, PackParams};
use crate::place::{
    anchor_axis, effective_strategy, island_box_clear, local_min_corner, max_scale_clear_of,
    max_scale_in_box, placement_score, rotated_size, Placed,
};
use crate::rng::SplitMix64;
use researchuv_math::Vec2;

/// The island count at which the `Auto` advanced heuristic activates.
pub const AUTO_THRESHOLD: usize = 8;

/// Statistics of a heuristic run.
#[derive(Clone, Debug, Default)]
pub struct HeuristicStats {
    /// Candidate placements evaluated.
    pub candidates: u64,
    /// Relocations accepted.
    pub relocations: u64,
    /// Wall time actually consumed (seconds).
    pub elapsed_secs: f64,
}

/// Is the advanced heuristic active for this configuration and island count?
pub fn advanced_heuristic_active(params: &PackParams, island_count: usize) -> bool {
    if !params.heuristic_enable {
        return false;
    }
    match params.advanced_heuristic {
        AdvancedHeuristicMode::Enable => true,
        AdvancedHeuristicMode::Disable => false,
        AdvancedHeuristicMode::Auto => island_count >= AUTO_THRESHOLD,
    }
}

/// Packing-quality score: the sum of the per-island strategy scores
/// (lower = more compact near the start corner).
pub fn packing_score(placed: &[Placed], target: &Box2, params: &PackParams) -> f64 {
    let strategy = effective_strategy(params);
    placed
        .iter()
        .map(|p| placement_score(strategy, p.box_.min, target, params.start_corner))
        .sum()
}

/// Improve `placed` (an initial solution from the base placement pass) over a
/// time budget (the engine's `heuristic_search_time` seconds; a budget of 0
/// runs exactly one improvement pass).
///
/// Only islands that are *not* static are moved; static islands stay put.
pub fn heuristic_refine(
    islands: &[Island],
    placed: &mut [Placed],
    target: &Box2,
    params: &PackParams,
    rng: &mut SplitMix64,
) -> HeuristicStats {
    let start = Instant::now();
    let budget = params.heuristic_search_time.max(0.0);
    let n = placed.len().min(islands.len());
    if n < 2 {
        let mut s = HeuristicStats::default();
        s.elapsed_secs = start.elapsed().as_secs_f64();
        return s;
    }
    let strategy = effective_strategy(params);
    let mut stats = HeuristicStats::default();
    let mut score = packing_score(placed, target, params);
    let rotations = params.rotation_candidates(0);
    let mut pass = 0u32;
    let mut stagnant = 0u32;

    loop {
        if budget > 0.0 && start.elapsed().as_secs_f64() >= budget {
            break;
        }
        if budget == 0.0 && pass > 0 {
            break; // zero budget: exactly one pass
        }
        pass += 1;
        let mut moved = 0usize;

        // Seeded random scan order.
        let mut order: Vec<usize> = (0..n).collect();
        rng.shuffle(&mut order);

        for &i in order.iter() {
            if budget > 0.0 && start.elapsed().as_secs_f64() >= budget {
                break;
            }
            let cur = placed[i].clone();
            let island_idx = cur.island_index as usize;
            if island_idx >= islands.len() {
                continue;
            }
            let isl = &islands[island_idx];
            if isl.is_static || isl.verts.is_empty() {
                continue;
            }
            let extent = isl.bbox.max_extent();
            if extent <= 0.0 {
                continue;
            }

            let others: Vec<Placed> = placed
                .iter()
                .enumerate()
                .filter(|(k, _)| *k != i)
                .map(|(_, p)| p.clone())
                .collect();
            let ax = anchor_axis(target, &others, params);
            let ay = anchor_axis(target, &others, params);
            let cur_scale = cur.transform.scale;
            let gap = params.island_gap(cur_scale * extent, extent);
            let cur_island_score =
                placement_score(strategy, cur.box_.min, target, params.start_corner);

            for _try in 0..8 {
                stats.candidates += 1;
                let rot = if rotations.len() > 1 && rng.next_f64() < 0.5 {
                    rotations[(rng.next_u64() as usize) % rotations.len()]
                } else {
                    cur.transform.rotation
                };
                let flipped = params.flipping_enable && rng.next_f64() < 0.5;
                let anchor = Vec2::new(
                    ax[(rng.next_u64() as usize) % ax.len()] + rng.range_f64(0.0, 1e-6) * extent,
                    ay[(rng.next_u64() as usize) % ay.len()] + rng.range_f64(0.0, 1e-6) * extent,
                );

                let (rw, rh) = rotated_size(isl, rot);
                let border = params.border_gap(cur_scale * extent);
                let s_box = max_scale_in_box(rw, rh, anchor, target, border).unwrap_or(0.0);
                let s_clear = max_scale_clear_of(rw, rh, anchor, &others, gap);
                let s_fit = s_box.min(s_clear);
                let s = if params.heuristic_allow_mixed_scales {
                    s_fit
                } else {
                    cur_scale.min(s_fit)
                };
                if s <= 1e-12 {
                    continue;
                }
                let b = Box2::new(
                    Vec2::new(anchor.u, anchor.v),
                    Vec2::new(anchor.u + rw * s, anchor.v + rh * s),
                );
                if !island_box_clear(&b, &others, gap) {
                    continue;
                }
                // Re-verify the border clearance at the *chosen* scale (the
                // border is proportional to the island's extent at scale s).
                let border_eff = params.border_gap(s * extent);
                if b.max.u > target.max.u - border_eff + 1e-9
                    || b.max.v > target.max.v - border_eff + 1e-9
                    || b.min.u < target.min.u + border_eff - 1e-9
                    || b.min.v < target.min.v + border_eff - 1e-9
                {
                    continue;
                }
                let new_island_score =
                    placement_score(strategy, b.min, target, params.start_corner);
                if new_island_score + (score - cur_island_score) < score - 1e-12 {
                    let (lmin_u, lmin_v) = local_min_corner(isl, rot, flipped, s);
                    let transform = PlacedTransform::from_parts(
                        rot,
                        flipped,
                        s,
                        b.min.u - lmin_u,
                        b.min.v - lmin_v,
                        b,
                    );
                    placed[i] = Placed {
                        island_index: cur.island_index,
                        transform,
                        box_: b,
                    };
                    score += new_island_score - cur_island_score;
                    stats.relocations += 1;
                    moved += 1;
                    break; // island moved; sample its new neighborhood next pass
                }
            }
        }

        // Stagnation exit: the budget is a wall-time ceiling, not a target.
        // Two consecutive passes with no accepting move means the layout is
        // a local optimum under this sampler — stop rather than spin.
        if moved == 0 {
            stagnant += 1;
            if stagnant >= 2 {
                break;
            }
        } else {
            stagnant = 0;
        }
    }

    stats.elapsed_secs = start.elapsed().as_secs_f64();
    stats
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sq(s: f64) -> Island {
        Island::from_polygon(vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(s, 0.0),
            Vec2::new(s, s),
            Vec2::new(0.0, s),
        ])
    }

    fn place_all(isls: &[Island], p: &PackParams) -> Vec<Placed> {
        let target = p.effective_box();
        let mut rng = SplitMix64::new(p.seed);
        let mut placed = Vec::new();
        for (i, isl) in isls.iter().enumerate() {
            if let Some(mut pl) = crate::place::find_best_placement(isl, &placed, &target, p, &mut rng) {
                pl.island_index = i as u32;
                placed.push(pl);
            }
        }
        placed
    }

    #[test]
    fn active_flags() {
        let mut p = PackParams::default();
        p.heuristic_enable = true;
        assert!(!advanced_heuristic_active(&p, 4)); // Auto, small count
        p.advanced_heuristic = AdvancedHeuristicMode::Enable;
        assert!(advanced_heuristic_active(&p, 1));
        p.advanced_heuristic = AdvancedHeuristicMode::Disable;
        assert!(!advanced_heuristic_active(&p, 1000));
        p.heuristic_enable = false;
        p.advanced_heuristic = AdvancedHeuristicMode::Enable;
        assert!(!advanced_heuristic_active(&p, 1000));
    }

    #[test]
    fn packing_score_ordering() {
        let p = PackParams::default();
        let t = p.effective_box();
        let a = Placed {
            island_index: 0,
            transform: PlacedTransform::from_parts(0.0, false, 1.0, 0.0, 0.0, Box2::new(Vec2::new(0.0, 0.0), Vec2::new(0.2, 0.2))),
            box_: Box2::new(Vec2::new(0.0, 0.0), Vec2::new(0.2, 0.2)),
        };
        let b = Placed {
            island_index: 1,
            transform: PlacedTransform::from_parts(0.0, false, 1.0, 0.9, 0.9, Box2::new(Vec2::new(0.9, 0.9), Vec2::new(1.0, 1.0))),
            box_: Box2::new(Vec2::new(0.9, 0.9), Vec2::new(1.0, 1.0)),
        };
        assert!(packing_score(&[a.clone()], &t, &p) < packing_score(&[b.clone()], &t, &p));
    }

    #[test]
    fn refine_keeps_valid_placements() {
        // 16 unit squares in a unit target with margin — they must scale
        // down; the heuristic must not break containment or introduce
        // overlaps.
        let isls: Vec<Island> = (0..16).map(|_| sq(0.3)).collect();
        let p = PackParams::default();
        let target = p.effective_box();
        let mut placed = place_all(&isls, &p);
        let before = packing_score(&placed, &target, &p);
        let mut rng = SplitMix64::new(123);
        let stats = heuristic_refine(&isls, &mut placed, &target, &p, &mut rng);
        assert!(stats.candidates > 0);
        // No overlap, everything inside.
        for (i, a) in placed.iter().enumerate() {
            assert!(target.contains_box_eps(&a.box_, 1e-6));
            for b in placed.iter().skip(i + 1) {
                assert!(!crate::poly::boxes_overlap(&a.box_, &b.box_, -1e-9));
            }
        }
        let after = packing_score(&placed, &target, &p);
        assert!(after <= before + 1e-9);
    }

    #[test]
    fn refine_improves_a_bad_initial_layout() {
        // Two islands initially stacked in the tall column; the side-by-side
        // layout (the automatic strategy's preference) has the lower score.
        let isls = vec![sq(0.4), sq(0.4)];
        let mut p = PackParams::default();
        p.pack_strategy = crate::params::PackStrategy::SideToSideVert;
        let target = p.effective_box();
        // Force a sub-optimal start: island 0 bottom-left, island 1 top.
        let b0 = Box2::new(Vec2::new(0.0, 0.0), Vec2::new(0.4, 0.4));
        let b1 = Box2::new(Vec2::new(0.0, 0.5), Vec2::new(0.4, 0.9));
        let mut placed = vec![
            Placed {
                island_index: 0,
                transform: PlacedTransform::from_parts(0.0, false, 1.0, 0.0, 0.0, b0),
                box_: b0,
            },
            Placed {
                island_index: 1,
                transform: PlacedTransform::from_parts(0.0, false, 1.0, 0.0, 0.5, b1),
                box_: b1,
            },
        ];
        let before = packing_score(&placed, &target, &p);
        let mut rng = SplitMix64::new(7);
        let stats = heuristic_refine(&isls, &mut placed, &target, &p, &mut rng);
        let after = packing_score(&placed, &target, &p);
        assert!(stats.relocations > 0);
        assert!(after < before);
    }
}
