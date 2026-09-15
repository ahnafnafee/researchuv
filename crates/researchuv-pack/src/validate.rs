//! Island validation — the engine's `ProcessIntersections` +
//! `UvpmValidation` feature: detects overlapping islands, islands outside the
//! target box, and self-intersecting outlines; stamps the `UvpmIslandFlags`
//! bits and produces the `UvpmRetcode` result.
//!
//! Evidence: engine strings `ProcessIntersections`, `has_self_intersections`,
//! `check_holes`, `invalid_islands`, `overlapping_islands`, `non_packed_islands`;
//! addon `UvpmRetcode` (`InvalidIslands = 4`, `NoSpace = 2`, `Warning = 7`).

use crate::box2::Box2;
use crate::island::{Island, OVERLAPS, OUTSIDE_TARGET_BOX};
use crate::params::{PackParams, UvpmRetcode};
use crate::poly;
use researchuv_math::Vec2;

/// Per-island validation verdict.
#[derive(Clone, Debug, Default)]
pub struct ValidationReport {
    /// Which islands overlap each other (pair flags on both).
    pub overlapping: Vec<u32>,
    /// Which islands are outside the target box.
    pub outside: Vec<u32>,
    /// Which islands have self-intersecting outlines.
    pub self_intersecting: Vec<u32>,
    /// Which islands have holes (non-simple outline).
    pub with_holes: Vec<u32>,
    /// Overall retcode.
    pub retcode: UvpmRetcode,
}

impl ValidationReport {
    fn new() -> Self {
        Self {
            retcode: UvpmRetcode::Success,
            ..Default::default()
        }
    }
    /// Any problem found?
    pub fn has_errors(&self) -> bool {
        !matches!(self.retcode, UvpmRetcode::Success)
    }
}

/// Validate a placement of `islands` (with their placed outlines in `placed`
/// and placed hole loops in `placed_holes`) against `target`. Returns the
/// report and stamps `island.flags`.
pub fn validate_islands(
    islands: &mut [Island],
    placed: &[Vec<Vec2>],
    placed_holes: &[Vec<Vec<Vec2>>],
    target: &Box2,
    params: &PackParams,
) -> ValidationReport {
    let mut rep = ValidationReport::new();
    let n = islands.len();
    debug_assert_eq!(placed.len(), n);

    // Clear stale result bits so re-validation is idempotent.
    for isl in islands.iter_mut() {
        isl.flags &= !(OVERLAPS | OUTSIDE_TARGET_BOX);
    }

    // Self-intersection + holes.
    for i in 0..n {
        let p = &placed[i];
        if poly::self_intersects(p) {
            rep.self_intersecting.push(i as u32);
        }
        if params.similarity.check_holes && placed_holes[i].is_empty() && poly::has_holes(p) {
            rep.with_holes.push(i as u32);
        }
    }

    // Outside target box (with `fully_inside` semantics).
    if params.fully_inside {
        let eps = 1e-9;
        for i in 0..n {
            let b = poly::bbox_of(&placed[i]);
            if !target.contains_box_eps(&b, eps) {
                rep.outside.push(i as u32);
                islands[i].flags |= OUTSIDE_TARGET_BOX;
            }
        }
    }

    // Pairwise overlap (filled regions: outer minus holes).
    for i in 0..n {
        for j in (i + 1)..n {
            let mode = params.overlap_detection_mode;
            if mode == crate::params::OverlapDetectionMode::Disabled {
                continue;
            }
            if poly::ring_sets_intersect(&placed[i], &placed_holes[i], &placed[j], &placed_holes[j])
                || poly::overlap(&placed[i], &placed[j], mode, 1e-9)
            {
                rep.overlapping.push(i as u32);
                rep.overlapping.push(j as u32);
                islands[i].flags |= OVERLAPS;
                islands[j].flags |= OVERLAPS;
            }
        }
    }

    // Retcode.
    if rep.self_intersecting.is_empty() && rep.outside.is_empty() && rep.overlapping.is_empty() {
        rep.retcode = UvpmRetcode::Success;
    } else if rep.self_intersecting.is_empty() && rep.outside.is_empty() {
        // Overlaps only: warning (the engine's `lock_overlapping` path).
        rep.retcode = if params.lock_overlapping {
            UvpmRetcode::Warning
        } else {
            UvpmRetcode::Success
        };
    } else if !rep.self_intersecting.is_empty() {
        rep.retcode = UvpmRetcode::InvalidIslands;
    } else {
        rep.retcode = UvpmRetcode::NoSpace;
    }

    rep
}

/// Convenience: clear the `OVERLAPS` / `OUTSIDE_TARGET_BOX` bits before a fresh
/// validation pass.
pub fn clear_result_flags(islands: &mut [Island]) {
    for i in islands.iter_mut() {
        i.flags &= !(OVERLAPS | OUTSIDE_TARGET_BOX);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use researchuv_math::Vec2;

    fn sq(x: f64, y: f64, s: f64) -> Island {
        Island::from_polygon(vec![
            Vec2::new(x, y),
            Vec2::new(x + s, y),
            Vec2::new(x + s, y + s),
            Vec2::new(x, y + s),
        ])
    }

    #[test]
    fn no_problems() {
        let mut isls = vec![sq(0.0, 0.0, 0.4), sq(0.6, 0.0, 0.4)];
        let placed: Vec<Vec<Vec2>> = isls.iter().map(|i| i.verts.clone()).collect();
        let holes: Vec<Vec<Vec<Vec2>>> = isls.iter().map(|i| i.holes.clone()).collect();
        let p = PackParams::default();
        let r = validate_islands(&mut isls, &placed, &holes, &Box2::unit(), &p);
        assert_eq!(r.retcode, UvpmRetcode::Success);
        assert!(r.overlapping.is_empty());
        assert!(r.outside.is_empty());
    }

    #[test]
    fn overlap_detected() {
        let mut isls = vec![sq(0.0, 0.0, 0.6), sq(0.4, 0.0, 0.6)];
        let placed: Vec<Vec<Vec2>> = isls.iter().map(|i| i.verts.clone()).collect();
        let holes: Vec<Vec<Vec<Vec2>>> = isls.iter().map(|i| i.holes.clone()).collect();
        let p = PackParams::default();
        let r = validate_islands(&mut isls, &placed, &holes, &Box2::unit(), &p);
        assert_eq!(r.retcode, UvpmRetcode::Success);
        assert_eq!(r.overlapping, vec![0u32, 1]);
        assert!(isls[0].overlaps_flag());
        assert!(isls[1].overlaps_flag());
    }

    #[test]
    fn outside_detected() {
        let mut isls = vec![sq(-0.3, 0.0, 0.4), sq(0.0, 0.5, 0.4)];
        let placed: Vec<Vec<Vec2>> = isls.iter().map(|i| i.verts.clone()).collect();
        let holes: Vec<Vec<Vec<Vec2>>> = isls.iter().map(|i| i.holes.clone()).collect();
        let p = PackParams::default();
        let r = validate_islands(&mut isls, &placed, &holes, &Box2::unit(), &p);
        assert_eq!(r.retcode, UvpmRetcode::NoSpace);
        assert_eq!(r.outside, vec![0u32]);
        assert!(isls[0].outside_flag());
    }

    #[test]
    fn self_intersecting_island() {
        let bowtie = Island::from_polygon(vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 1.0),
        ]);
        let mut isls = vec![bowtie.clone()];
        let placed: Vec<Vec<Vec2>> = isls.iter().map(|i| i.verts.clone()).collect();
        let holes: Vec<Vec<Vec<Vec2>>> = isls.iter().map(|i| i.holes.clone()).collect();
        let p = PackParams::default();
        let r = validate_islands(&mut isls, &placed, &holes, &Box2::unit(), &p);
        assert_eq!(r.retcode, UvpmRetcode::InvalidIslands);
        assert_eq!(r.self_intersecting, vec![0u32]);
    }
}
