//! Target boxes — the `unit_box` / `target_box` / `flipped_box` / `BoxCorner`
//! vocabulary of the engine string cluster, plus the tile targets
//! (`TILE_GRID` / `TILE_RANGE` / `DYNAMIC_TILES`, `tiles_in_row`, start tile).
//!
//! A *target box* is the region islands are packed into. The default is the
//! **unit box** `[0,1]²` (UVPM packs into the UV editor's working space). A
//! *flipped box* mirrors the box across its center (used by the
//! `flipping_enable` option and by corner-based strategies).

use researchuv_math::Vec2;

/// Axis-aligned box with min/max corners (the engine's `Box2`).
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Box2 {
    pub min: Vec2,
    pub max: Vec2,
}

impl Box2 {
    pub fn new(min: Vec2, max: Vec2) -> Self {
        Self { min, max }
    }

    /// The unit box `[0,1]²` (`unit_box`).
    pub fn unit() -> Self {
        Self::new(Vec2::new(0.0, 0.0), Vec2::new(1.0, 1.0))
    }

    /// Box from two opposite corners (any corner order).
    pub fn from_corners(a: Vec2, b: Vec2) -> Self {
        Self::new(
            Vec2::new(a.u.min(b.u), a.v.min(b.v)),
            Vec2::new(a.u.max(b.u), a.v.max(b.v)),
        )
    }

    pub fn width(&self) -> f64 {
        self.max.u - self.min.u
    }
    pub fn height(&self) -> f64 {
        self.max.v - self.min.v
    }
    pub fn area(&self) -> f64 {
        self.width() * self.height()
    }
    /// Max extent (used to normalize the relative margin and pixel sizes).
    pub fn max_extent(&self) -> f64 {
        self.width().max(self.height())
    }
    pub fn center(&self) -> Vec2 {
        Vec2::new(0.5 * (self.min.u + self.max.u), 0.5 * (self.min.v + self.max.v))
    }

    /// Does `p` lie inside (inclusive, with `eps` slack)?
    pub fn contains_eps(&self, p: Vec2, eps: f64) -> bool {
        p.u >= self.min.u - eps
            && p.u <= self.max.u + eps
            && p.v >= self.min.v - eps
            && p.v <= self.max.v + eps
    }
    /// Is the whole box `b` inside (inclusive, with `eps` slack)?
    pub fn contains_box_eps(&self, b: &Box2, eps: f64) -> bool {
        b.min.u >= self.min.u - eps
            && b.min.v >= self.min.v - eps
            && b.max.u <= self.max.u + eps
            && b.max.v <= self.max.v + eps
    }
    /// Axis-aligned intersection (returns `None` when disjoint).
    pub fn intersect(&self, o: &Box2) -> Option<Box2> {
        let mn = Vec2::new(self.min.u.max(o.min.u), self.min.v.max(o.min.v));
        let mx = Vec2::new(self.max.u.min(o.max.u), self.max.v.min(o.max.v));
        if mx.u < mn.u || mx.v < mn.v {
            None
        } else {
            Some(Box2::new(mn, mx))
        }
    }
    /// Box enlarged by `m` on all sides.
    pub fn grow(&self, m: f64) -> Box2 {
        Box2::new(self.min - Vec2::new(m, m), self.max + Vec2::new(m, m))
    }
    /// Mirror the box across its center (`flipped_box`); returns the box of a
    /// point/bbox mirrored in this box.
    pub fn flipped(&self) -> Box2 {
        let c = self.center();
        Box2::new(
            Vec2::new(2.0 * c.u - self.max.u, 2.0 * c.v - self.max.v),
            Vec2::new(2.0 * c.u - self.min.u, 2.0 * c.v - self.min.v),
        )
    }
}

/// Corners of a box (`BoxCorner`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BoxCorner {
    /// Bottom-left (min corner).
    Bl,
    /// Bottom-right.
    Br,
    /// Top-right.
    Tr,
    /// Top-left.
    Tl,
}

impl BoxCorner {
    /// Corner position of `b` for this enum value.
    pub fn point(&self, b: &Box2) -> Vec2 {
        match self {
            BoxCorner::Bl => b.min,
            BoxCorner::Br => Vec2::new(b.max.u, b.min.v),
            BoxCorner::Tr => b.max,
            BoxCorner::Tl => Vec2::new(b.min.u, b.max.v),
        }
    }

    /// All four corners in BL→BR→TR→TL order.
    pub const ALL: [BoxCorner; 4] = [BoxCorner::Bl, BoxCorner::Br, BoxCorner::Tr, BoxCorner::Tl];
}

/// A pack target: the box to pack into.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TargetBox {
    /// The unit box `[0,1]²` (default).
    Unit,
    /// An explicit box (`custom_target_box`).
    Explicit(Box2),
    /// Flip the effective target across its center.
    Flipped(Box2),
}

impl TargetBox {
    /// The effective (unflipped) box to pack into.
    pub fn box_(&self) -> Box2 {
        match self {
            TargetBox::Unit => Box2::unit(),
            TargetBox::Explicit(b) => *b,
            TargetBox::Flipped(b) => *b,
        }
    }
    /// Is the box flipped (pack order mirrored)?
    pub fn is_flipped(&self) -> bool {
        matches!(self, TargetBox::Flipped(_))
    }
}

/// Tile geometry for `TILE_GRID` / `TILE_RANGE` / `DYNAMIC_TILES` targets
/// (`tile_from_number`, `max_island_dimension`, tile grid start/size).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TileTarget {
    /// Number of tiles along x (grid width).
    pub count_x: u32,
    /// Number of tiles along y (grid height).
    pub count_y: u32,
    /// Tile size (in UV units) along x.
    pub size_x: f64,
    /// Tile size along y.
    pub size_y: f64,
    /// Starting tile index (col, row) of the grid origin.
    pub start: (u32, u32),
    /// Optional bounding rectangle (UV units) the grid is fitted into.
    pub within: Option<Box2>,
}

impl TileTarget {
    /// The box of tile `(ix, iy)` (local grid indices).
    pub fn tile_box(&self, ix: u32, iy: u32) -> Box2 {
        let (sx, sy) = self.start;
        let ox = (ix as f64 + sx as f64) * self.size_x;
        let oy = (iy as f64 + sy as f64) * self.size_y;
        Box2::new(Vec2::new(ox, oy), Vec2::new(ox + self.size_x, oy + self.size_y))
    }
    /// Total grid box (from start through the last tile).
    pub fn grid_box(&self) -> Box2 {
        let b = self.tile_box(0, 0);
        let b2 = self.tile_box(self.count_x.saturating_sub(1), self.count_y.saturating_sub(1));
        Box2::new(
            b.min,
            Vec2::new(b2.max.u, b2.max.v),
        )
    }
    /// Fit the grid so it covers `b` exactly: derive tile size + count.
    pub fn fit(b: &Box2, count_x: u32, count_y: u32) -> Self {
        let cx = count_x.max(1);
        let cy = count_y.max(1);
        Self {
            count_x: cx,
            count_y: cy,
            size_x: b.width() / cx as f64,
            size_y: b.height() / cy as f64,
            start: (0, 0),
            within: Some(*b),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_box_metrics() {
        let b = Box2::unit();
        assert_eq!(b.width(), 1.0);
        assert_eq!(b.height(), 1.0);
        assert_eq!(b.center(), Vec2::new(0.5, 0.5));
        assert!(b.contains_box_eps(&b, 0.0));
    }

    #[test]
    fn corners_and_flip() {
        let b = Box2::unit();
        assert_eq!(BoxCorner::Bl.point(&b), Vec2::new(0.0, 0.0));
        assert_eq!(BoxCorner::Br.point(&b), Vec2::new(1.0, 0.0));
        assert_eq!(BoxCorner::Tr.point(&b), Vec2::new(1.0, 1.0));
        assert_eq!(BoxCorner::Tl.point(&b), Vec2::new(0.0, 1.0));
        // Flipping the unit box is itself.
        assert_eq!(b.flipped(), b);
        let b2 = Box2::new(Vec2::new(0.0, 0.0), Vec2::new(2.0, 1.0));
        let f = b2.flipped();
        assert_eq!(f.min, Vec2::new(0.0, 0.0));
        assert_eq!(f.max, Vec2::new(2.0, 1.0));
    }

    #[test]
    fn tile_grid() {
        let t = TileTarget::fit(&Box2::unit(), 4, 2);
        let b00 = t.tile_box(0, 0);
        let b31 = t.tile_box(3, 1);
        assert!((b00.min.u - 0.0).abs() < 1e-12);
        assert!((b00.width() - 0.25).abs() < 1e-12);
        assert!((b00.height() - 0.5).abs() < 1e-12);
        assert!((b31.max.u - 1.0).abs() < 1e-12);
        assert!((b31.max.v - 1.0).abs() < 1e-12);
    }
}
