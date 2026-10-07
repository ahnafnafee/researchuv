//! Packing-quality reporting — utilization plus, for the equal-square
//! subtype, the certified optimality-gap reference from [`crate::sqbounds`].
//!
//! Two utilization numbers are reported because they answer different
//! questions:
//!
//! - **Polygon utilization** — the sum of the placed islands' filled areas
//!   (outline minus holes, at the final scale) over the target area. This is
//!   the honest "how much of the atlas is texture" figure.
//! - **Bounding-box utilization** — the same over placed bounding boxes; the
//!   customary UV-packing figure, comparable across packers.
//!
//! When every placed island is a (near-)square of the same size, the result
//! also carries an [`EqualSquaresReference`]: the certified lower and
//! best-known upper bounds on the minimal container side `s(n)` for `n` unit
//! squares, and the certified utilization *ceiling* `n / s(n)²`. Comparing
//! the achieved utilization against that ceiling turns "the packer seems
//! good" into a bounded claim: the gap to a *proven* floor.

use crate::island::Island;
use crate::params::PackParams;
use crate::sqbounds;

/// Relative tolerance for calling two island extents "equal" and a bbox
/// "square" when detecting the equal-square subtype.
const EQUAL_SQUARES_TOL: f64 = 1e-6;

/// The equal-square optimality-gap reference for a packing whose islands are
/// all squares of one size.
#[derive(Clone, Copy, Debug)]
pub struct EqualSquaresReference {
    /// How many equal square islands were placed.
    pub n: u32,
    /// Certified lower bound on the minimal container side `s(n)`.
    pub certified_lower_side: f64,
    /// Best-known upper bound on `s(n)` (an explicit packing achieves it).
    pub best_known_side: f64,
    /// `s(n)` is proven optimal (lower == upper).
    pub proven_optimal: bool,
    /// Certified ceiling on utilization: `n / s(n)²` from the lower bound.
    /// An equal-square packing can never exceed it.
    pub certified_max_utilization: f64,
    /// Where the bounds come from (see [`crate::sqbounds`]).
    pub source: &'static str,
}

/// Packing-quality summary attached to every [`crate::PackResult`].
#[derive(Clone, Copy, Debug, Default)]
pub struct QualityReport {
    /// Islands handed to the packer.
    pub island_count: u32,
    /// Islands with a placement.
    pub placed_count: u32,
    /// Effective target-box area.
    pub target_area: f64,
    /// Sum of placed islands' filled polygon areas (outline − holes) at the
    /// final scale.
    pub placed_area: f64,
    /// Sum of placed islands' bounding-box areas at the final scale.
    pub bbox_area: f64,
    /// `placed_area / target_area` — how much of the atlas is texture.
    pub utilization: f64,
    /// `bbox_area / target_area` — the customary UV-packing figure.
    pub bbox_utilization: f64,
    /// The certified reference when all placed islands are equal squares.
    pub equal_squares: Option<EqualSquaresReference>,
}

impl QualityReport {
    /// The certified utilization ceiling, when the equal-square reference is
    /// present (otherwise `None`: no certified ceiling is known for general
    /// island sets).
    pub fn certified_utilization_ceiling(&self) -> Option<f64> {
        self.equal_squares.map(|e| e.certified_max_utilization)
    }

    /// `utilization / certified ceiling` when a ceiling exists: 1.0 means the
    /// packing saturates a proven bound.
    pub fn certified_gap(&self) -> Option<f64> {
        let ceil = self.certified_utilization_ceiling()?;
        if ceil > 0.0 {
            Some(self.utilization / ceil)
        } else {
            None
        }
    }
}

/// Compute the quality summary for a finished pack: `islands` are the raw
/// inputs (post flag stamping) and `placed` their final transforms.
pub fn quality_report(islands: &[Island], placed: &[Option<crate::island::PlacedTransform>], params: &PackParams) -> QualityReport {
    let target = params.effective_box();
    let target_area = target.width() * target.height();
    let mut rep = QualityReport {
        island_count: islands.len() as u32,
        target_area,
        ..QualityReport::default()
    };
    let mut square_extents: Vec<f64> = Vec::new();
    let mut all_square = true;
    for (i, t) in placed.iter().enumerate() {
        let Some(t) = t else { continue };
        rep.placed_count += 1;
        let s2 = t.scale * t.scale;
        rep.placed_area += islands[i].area().max(0.0) * s2;
        rep.bbox_area += t.box_.width() * t.box_.height();
        if all_square {
            let b = &islands[i].bbox;
            let (w, h) = (b.width(), b.height());
            let ext = w.max(h);
            if ext <= 0.0 || (w - h).abs() > EQUAL_SQUARES_TOL * ext {
                all_square = false;
            } else {
                square_extents.push(ext);
            }
        }
    }
    if target_area > 0.0 {
        rep.utilization = rep.placed_area / target_area;
        rep.bbox_utilization = rep.bbox_area / target_area;
    }
    if all_square && square_extents.len() >= 2 {
        // All extents equal within tolerance → the equal-square subtype.
        let e0 = square_extents[0];
        if square_extents.iter().all(|&e| (e - e0).abs() <= EQUAL_SQUARES_TOL * e0) {
            let n = square_extents.len() as u32;
            let b = sqbounds::bound(n);
            rep.equal_squares = Some(EqualSquaresReference {
                n,
                certified_lower_side: b.lower,
                best_known_side: b.upper,
                proven_optimal: b.exact,
                certified_max_utilization: sqbounds::certified_max_utilization(n),
                source: b.source,
            });
        }
    }
    rep
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::PackParams;
    use crate::pipeline::pack;
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
    fn utilization_of_two_half_squares() {
        let mut isls = vec![sq(0.0, 0.0, 0.5), sq(0.0, 0.0, 0.5)];
        let mut p = PackParams::default();
        p.margin = 0.0; // no gaps: two half-squares tile exactly
        let r = pack(&mut isls, &p);
        let q = &r.quality;
        assert_eq!(q.placed_count, 2);
        assert_eq!(q.island_count, 2);
        // Two 0.5² squares tile half of the unit atlas exactly (polygon
        // area = bbox area here; margins are zero).
        assert!((q.utilization - 0.5).abs() < 1e-9, "{}", q.utilization);
        assert!((q.bbox_utilization - 0.5).abs() < 1e-9, "{}", q.bbox_utilization);
        // Equal-square reference: n = 2 → s(2) = 2 proven, ceiling 2/4 = 0.5.
        let e = q.equal_squares.expect("equal squares detected");
        assert_eq!(e.n, 2);
        assert!(e.proven_optimal);
        assert!((e.certified_max_utilization - 0.5).abs() < 1e-9);
        assert!((q.certified_gap().unwrap() - 1.0).abs() < 1e-9, "saturates s(2)=2");
    }

    #[test]
    fn mixed_sizes_have_no_equal_square_reference() {
        let mut isls = vec![sq(0.0, 0.0, 0.5), sq(0.0, 0.0, 0.25)];
        let p = PackParams::default();
        let r = pack(&mut isls, &p);
        assert!(r.quality.equal_squares.is_none());
        assert!(r.quality.certified_gap().is_none());
    }
}
