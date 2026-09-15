//! Split-overlap — the engine's `align_split_overlapping` scenario: after
//! packing, islands that still overlap get pushed apart by *integer tile
//! offsets* (each island moves whole tiles in a row-major layout along x)
//! until no overlapping pair remains.
//!
//! Evidence: addon `scenario/align_split_overlapping.py` — iterative rounds;
//! each round computes the overlap map over `to_process + processed`; islands
//! with no overlaps are marked processed; the remaining overlapping islands
//! are sorted by (−align_priority when enabled, −overlap count) and each gets
//! the smallest non-negative integer offset `k` not already used by an
//! assigned overlapping neighbour; offsets accumulate into the
//! `split_offset_x` / `split_offset_y` iparams; with `max_tile_x > 0` the
//! x-offset wraps at `max_tile_x` carrying a +1 y-row
//! (`free_to_split_offset_delta`); islands whose final X coord falls outside
//! `[0, max_tile_x)` abort with InputError "Some UVs have the X coord lower
//! than 0 or greater than the 'Max Tile (X)' value"; `dont_split_priorities`
//! ignores overlaps between equal-priority pairs.
//!
//! The offsets are expressed in *tile units*: the caller adds
//! `(dx, dy)` to each island's UVs (one tile per UV unit in the packed tile
//! space) and writes the accumulated result back to the `split_offset_x/y`
//! iparam channels.

use crate::island::Island;
use crate::params::{iparam, OverlapDetectionMode, SplitOverlapParams};
use crate::poly;
use researchuv_math::Vec2;

/// The InputError text from the addon scenario.
pub const INPUT_ERROR_TILE_RANGE: &str = "Some UVs have the X coord lower than 0 or the \
'Max Tile (X)' value is exceeded";

/// An island that could not be moved inside the tile range.
#[derive(Clone, Debug, PartialEq)]
pub struct TileRangeError {
    /// Island index (into the input slice).
    pub island: u32,
    /// The offending final X coord (UV units).
    pub x: f64,
}

impl core::fmt::Display for TileRangeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{INPUT_ERROR_TILE_RANGE} (island {}, x = {:.6})", self.island, self.x)
    }
}

/// The align-priority of an island (the `align_priority` iparam channel).
fn island_priority(island: &Island) -> u32 {
    island
        .iparam_channel(iparam::ALIGN_PRIORITY)
        .unwrap_or(0.0) as u32
}

/// Accumulated split offset of an island from its iparams
/// (`(0, 0)` when the channels are unset).
fn base_offset(island: &Island) -> (i32, i32) {
    let x = island.iparam_channel(iparam::SPLIT_OFFSET_X).unwrap_or(iparam::SPLIT_OFFSET_UNSET);
    let y = island.iparam_channel(iparam::SPLIT_OFFSET_Y).unwrap_or(iparam::SPLIT_OFFSET_UNSET);
    let x = if x <= iparam::SPLIT_OFFSET_UNSET { 0.0 } else { x };
    let y = if y <= iparam::SPLIT_OFFSET_UNSET { 0.0 } else { y };
    (x.round() as i32, y.round() as i32)
}

/// Translate a polygon by a tile offset.
fn translate(poly_pts: &[Vec2], dx: i32, dy: i32) -> Vec<Vec2> {
    poly_pts
        .iter()
        .map(|&p| Vec2::new(p.u + dx as f64, p.v + dy as f64))
        .collect()
}

/// Two islands overlap (per the detection mode) after their accumulated
/// offsets are applied: bbox overlap for `Exact`, full polygon intersection
/// on the translated outlines for `AnyPart`.
fn overlap_at(
    a: &Island,
    b: &Island,
    oa: (i32, i32),
    ob: (i32, i32),
    mode: OverlapDetectionMode,
) -> bool {
    let pa = translate(&a.verts, oa.0, oa.1);
    let pb = translate(&b.verts, ob.0, ob.1);
    poly::overlap(&pa, &pb, mode, 1e-9)
}

/// Pack an integer offset `k` into `(dx, dy)` tile units: with
/// `max_tile_x > 0`, `dx` wraps at `max_tile_x` carrying a +1 y-row
/// (`free_to_split_offset_delta`); otherwise all of `k` goes to x.
pub fn offset_to_delta(k: i32, max_tile_x: u32) -> (i32, i32) {
    if max_tile_x > 0 {
        let mt = max_tile_x as i32;
        let dy = k.div_euclid(mt);
        let dx = k.rem_euclid(mt);
        (dx, dy)
    } else {
        (k, 0)
    }
}

/// Run the split-overlap pass over `islands` (their `bbox`/`verts` are the
/// packed positions, 1 tile = 1 UV unit along x).
///
/// Returns the final `(dx, dy)` tile offset per island (including any
/// accumulated `split_offset_x/y` iparam values). Fails with
/// [`TileRangeError`] when an island ends outside the X range with
/// `max_tile_x > 0`.
pub fn split_overlapping(
    islands: &[Island],
    params: &SplitOverlapParams,
    align_priority_enable: bool,
) -> Result<Vec<(i32, i32)>, TileRangeError> {
    let n = islands.len();
    if n == 0 || !params.enable {
        return Ok(vec![(0, 0); n]);
    }
    let mt = params.max_tile_x;
    let mode = params.detection_mode;
    let threshold = params.dont_split_priorities as f64;

    // Accumulated offsets (tile units).
    let mut off: Vec<(i32, i32)> = islands.iter().map(base_offset).collect();

    // Priority of each island.
    let prio: Vec<f64> = islands.iter().map(|i| island_priority(i) as f64).collect();

    // The overlap map is rebuilt every round over the current positions.
    let mut processed = vec![false; n];

    for _round in 0..n * n + 1 {
        // Current overlap map: i → indices overlapping i (position-aware).
        let mut overlap_of: Vec<Vec<u32>> = vec![Vec::new(); n];
        for i in 0..n {
            for j in (i + 1)..n {
                // dont_split_priorities: ignore overlaps between equal-
                // priority pairs at or above the threshold.
                if params.dont_split_priorities > 0
                    && prio[i] >= threshold
                    && (prio[i] - prio[j]).abs() < 1e-9
                {
                    continue;
                }
                if overlap_at(&islands[i], &islands[j], off[i], off[j], mode) {
                    overlap_of[i].push(j as u32);
                    overlap_of[j].push(i as u32);
                }
            }
        }

        // Islands with no overlap partners are settled.
        for i in 0..n {
            if !processed[i] && overlap_of[i].is_empty() {
                processed[i] = true;
            }
        }
        if processed.iter().all(|&p| p) {
            break;
        }

        // The unsettled overlapping islands, ordered by (−priority when
        // enabled, −overlap count, index).
        let mut pending: Vec<u32> = (0..n)
            .filter(|&i| !processed[i] && !overlap_of[i as usize].is_empty())
            .map(|i| i as u32)
            .collect();
        pending.sort_by(|&a, &b| {
            let a = a as usize;
            let b = b as usize;
            let mut c = overlap_of[b].len().cmp(&overlap_of[a].len());
            if c.is_eq() && align_priority_enable {
                c = prio[b].partial_cmp(&prio[a]).unwrap_or(std::cmp::Ordering::Equal);
            }
            c.then(a.cmp(&b))
        });
        if pending.is_empty() {
            // Nothing left to place; recheck whether everything is settled.
            if (0..n).all(|i| processed[i] || overlap_of[i].is_empty()) {
                break;
            }
        }

        // Track the slots (final total offsets) assigned this round so a
        // pending island never collides with an already-assigned neighbour.
        let mut assigned: Vec<Option<(i32, i32)>> = vec![None; n];
        let mut moved = 0usize;
        for &i in pending.iter() {
            let i = i as usize;
            // Slots already taken: this round's assignments plus the current
            // offsets of settled (processed) overlapping neighbours.
            let mut used: Vec<(i32, i32)> = Vec::new();
            for &j in overlap_of[i].iter() {
                let j = j as usize;
                if let Some(s) = assigned[j] {
                    used.push(s);
                } else if processed[j] {
                    used.push(off[j]);
                }
            }
            // Smallest k such that the island's new total offset is free.
            let mut k = 0i32;
            let collides = |k: i32| {
                let d = offset_to_delta(k, mt);
                let t = (off[i].0 + d.0, off[i].1 + d.1);
                used.iter().any(|&(ux, uy)| ux == t.0 && uy == t.1)
            };
            while collides(k) {
                k += 1;
            }
            let (dx, dy) = offset_to_delta(k, mt);
            off[i] = (off[i].0 + dx, off[i].1 + dy);
            assigned[i] = Some(off[i]);
            processed[i] = true;
            moved += 1;
        }
        if moved == 0 {
            // No progress possible — break to avoid an infinite loop (this
            // only happens with degenerate overlap cycles).
            break;
        }
    }

    // Final X-range validation (max_tile_x > 0): every island's working box
    // must lie in [0, max_tile_x) along x.
    if mt > 0 {
        let limit = mt as f64;
        for i in 0..n {
            let b = islands[i].bbox;
            let minx = b.min.u + off[i].0 as f64;
            let maxx = b.max.u + off[i].0 as f64;
            if minx < -1e-9 || maxx >= limit - 1e-9 {
                return Err(TileRangeError {
                    island: i as u32,
                    x: if minx < -1e-9 { minx } else { maxx },
                });
            }
        }
    }

    Ok(off)
}

/// Write the accumulated offsets back to the `split_offset_x/y` iparam
/// channels of each island's first face (the addon serializes per-face
/// iparams; the engine applies the same value to all faces of an island).
pub fn write_split_offsets(islands: &mut [Island], offsets: &[(i32, i32)]) {
    for (i, isl) in islands.iter_mut().enumerate() {
        let (x, y) = offsets[i];
        if let Some(f) = isl.faces.first_mut() {
            f.iparams[iparam::SPLIT_OFFSET_X] = x as f64;
            f.iparams[iparam::SPLIT_OFFSET_Y] = y as f64;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sq(x: f64, y: f64, s: f64) -> Island {
        Island::from_polygon(vec![
            Vec2::new(x, y),
            Vec2::new(x + s, y),
            Vec2::new(x + s, y + s),
            Vec2::new(x, y + s),
        ])
    }

    #[test]
    fn no_overlaps_means_no_offsets() {
        let isls = vec![sq(0.0, 0.0, 0.4), sq(1.5, 0.0, 0.4), sq(0.0, 1.5, 0.4)];
        let p = SplitOverlapParams {
            enable: true,
            ..Default::default()
        };
        let off = split_overlapping(&isls, &p, false).unwrap();
        assert_eq!(off, vec![(0, 0); 3]);
    }

    #[test]
    fn overlapping_pair_gets_separate_offsets() {
        // Two identical overlapping squares → the second moves 1 tile.
        let a = sq(0.1, 0.1, 0.8);
        let b = sq(0.2, 0.1, 0.8);
        let p = SplitOverlapParams {
            enable: true,
            ..Default::default()
        };
        let off = split_overlapping(&[a.clone(), b.clone()], &p, false).unwrap();
        assert_eq!(off[0], (0, 0)); // first in order keeps its slot
        assert_eq!(off[1], (1, 0));
        // After applying, the boxes are disjoint.
        let ba = translate(&a.verts, off[0].0, off[0].1);
        let bb = translate(&b.verts, off[1].0, off[1].1);
        assert!(!poly::overlap(&ba, &bb, OverlapDetectionMode::AnyPart, 1e-9));
    }

    #[test]
    fn wrap_carry_with_max_tile_x() {
        let k = 5;
        assert_eq!(offset_to_delta(k, 2), (1, 2)); // 5 = 2*2 + 1
        assert_eq!(offset_to_delta(0, 3), (0, 0));
        assert_eq!(offset_to_delta(7, 0), (7, 0)); // no wrap when unlimited
    }

    #[test]
    fn priority_order_moves_lower_priority_first() {
        // Both overlap; island 1 has higher priority → it keeps its slot.
        let a = sq(0.1, 0.1, 0.8);
        let mut b = sq(0.2, 0.1, 0.8);
        b.faces = vec![crate::island::Face::default()];
        b.faces[0].iparams[iparam::ALIGN_PRIORITY] = 50.0;
        let p = SplitOverlapParams {
            enable: true,
            ..Default::default()
        };
        let off = split_overlapping(&[a, b], &p, true).unwrap();
        assert_eq!(off[1], (0, 0));
        assert_eq!(off[0], (1, 0));
    }

    #[test]
    fn out_of_tile_range_errors() {
        // Island already at x = 0.9 in a 1-tile-wide field: moving it +1
        // overflows the range → InputError.
        let a = sq(0.1, 0.1, 0.5);
        let b = sq(0.6, 0.1, 0.5);
        let p = SplitOverlapParams {
            enable: true,
            max_tile_x: 1,
            ..Default::default()
        };
        let r = split_overlapping(&[a, b], &p, false);
        assert!(r.is_err());
    }

    #[test]
    fn disabled_returns_zeros() {
        let a = sq(0.0, 0.0, 1.0);
        let b = sq(0.0, 0.0, 1.0);
        let p = SplitOverlapParams::default(); // enable = false
        let off = split_overlapping(&[a, b], &p, false).unwrap();
        assert_eq!(off, vec![(0, 0); 2]);
    }

    #[test]
    fn base_offsets_accumulate() {
        let mut a = sq(0.0, 0.0, 1.0);
        a.faces = vec![crate::island::Face::default()];
        a.faces[0].iparams[iparam::SPLIT_OFFSET_X] = 3.0;
        a.faces[0].iparams[iparam::SPLIT_OFFSET_Y] = 1.0;
        let p = SplitOverlapParams {
            enable: true,
            ..Default::default()
        };
        let off = split_overlapping(&[a], &p, false).unwrap();
        assert_eq!(off[0], (3, 1));
    }
}
