//! Tile targets for the rasterizer placement path.
//!
//! The addon's tile vocabulary: a pack target of `cols × rows` unit tiles,
//! filled either simultaneously (all tiles are one target; the scoring
//! naturally spreads islands) or one by one (tile 0 fills before tile 1
//! opens). The rasterizer handles tiles by *scoring*: a placement's score
//! gets `tile_index · 1e6` added, so the anchor scan fills the earliest
//! tile first under either filling method; `OneByOne` additionally marks
//! later tiles' cells occupied until the current tile is full (an island
//! only moves to the next tile when nothing fits).
//!
//! All tiles share the extent-class margin grids — the dilation state
//! carries over the whole tile grid (the raster grid spans
//! `cols × rows` unit tiles).

use crate::box2::Box2;
use researchuv_math::Vec2;

/// The resolved tile geometry for one pack run.
#[derive(Clone, Copy, Debug)]
pub struct TileGrid {
    /// Tile columns.
    pub cols: u32,
    /// Tile rows.
    pub rows: u32,
    /// The filling method (affects scoring/occupancy only).
    pub one_by_one: bool,
}

impl TileGrid {
    /// Resolve the grid from the pack parameters and the island count:
    /// `DynamicTiles` sizes the grid from the total island area (rounded up
    /// to whole tile rows of `tiles_in_row`); `TileGrid`/`TileRange` take
    /// their explicit counts.
    pub fn from_params(
        params: &crate::params::PackParams,
        total_island_area: f64,
        tile_extent: f64,
    ) -> TileGrid {
        use crate::params::TileTargetMode;
        let cols = params.tiles_in_row.max(1);
        let (cols, rows) = match params.tile_target {
            TileTargetMode::TileGrid => (
                params.tile_count_x.max(1),
                params.tile_count_y.max(1),
            ),
            TileTargetMode::TileRange => {
                let total = params.tile_count_x.max(1).saturating_mul(params.tile_count_y.max(1)).max(1);
                (cols, total.div_ceil(cols))
            }
            TileTargetMode::DynamicTiles => {
                let tile_area = tile_extent * tile_extent;
                let tiles_needed = (total_island_area / tile_area.max(1e-30))
                    .ceil()
                    .max(1.0) as u32;
                let rows = tiles_needed.div_ceil(cols).max(1);
                (cols, rows)
            }
        };
        TileGrid {
            cols,
            rows,
            one_by_one: params.tile_filling_method
                == crate::params::TileFillingMethod::OneByOne,
        }
    }

    /// The full grid box `[0, cols] × [0, rows]` (unit tiles).
    pub fn box_(self) -> Box2 {
        Box2::new(Vec2::new(0.0, 0.0), Vec2::new(self.cols as f64, self.rows as f64))
    }

    /// The box of tile `(ix, iy)` in grid coordinates.
    pub fn tile_box(self, ix: u32, iy: u32) -> Box2 {
        Box2::new(
            Vec2::new(ix as f64, iy as f64),
            Vec2::new((ix + 1) as f64, (iy + 1) as f64),
        )
    }

    /// The tile containing `p` (grid coordinates), or the nearest tile on
    /// overflow.
    pub fn tile_of(self, p: Vec2) -> (u32, u32) {
        let ix = (p.u.floor().max(0.0) as u32).min(self.cols.saturating_sub(1));
        let iy = (p.v.floor().max(0.0) as u32).min(self.rows.saturating_sub(1));
        (ix, iy)
    }

    /// Row-major tile index of `(ix, iy)`.
    pub fn tile_index(self, ix: u32, iy: u32) -> u32 {
        iy * self.cols + ix
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::{PackParams, TileFillingMethod, TileTargetMode};

    #[test]
    fn dynamic_tiles_size_from_area() {
        let mut p = PackParams::default();
        p.tile_target = TileTargetMode::DynamicTiles;
        p.tiles_in_row = 4;
        // 10 unit-islands of area 0.25 in unit tiles → ⌈2.5⌉ = 3 tiles →
        // ⌈3/4⌉ = 1 row of 4.
        let g = TileGrid::from_params(&p, 10.0 * 0.25, 1.0);
        assert_eq!((g.cols, g.rows), (4, 1));
        // 30 islands → 8 tiles → 2 rows.
        let g2 = TileGrid::from_params(&p, 30.0 * 0.25, 1.0);
        assert_eq!((g2.cols, g2.rows), (4, 2));
    }

    #[test]
    fn explicit_grid_and_range() {
        let mut p = PackParams::default();
        p.tile_target = TileTargetMode::TileGrid;
        p.tile_count_x = 3;
        p.tile_count_y = 2;
        let g = TileGrid::from_params(&p, 99.0, 1.0);
        assert_eq!((g.cols, g.rows), (3, 2));
        p.tile_target = TileTargetMode::TileRange;
        p.tiles_in_row = 2; // TileRange: tile_count_x is the TOTAL count
        p.tile_count_x = 10;
        p.tile_count_y = 1;
        let g2 = TileGrid::from_params(&p, 99.0, 1.0);
        assert_eq!((g2.cols, g2.rows), (2, 5));
    }

    #[test]
    fn tile_boxes_and_indices() {
        let g = TileGrid { cols: 3, rows: 2, one_by_one: false };
        assert_eq!(g.box_().max, Vec2::new(3.0, 2.0));
        let b = g.tile_box(1, 1);
        assert_eq!(b.min, Vec2::new(1.0, 1.0));
        assert_eq!(b.max, Vec2::new(2.0, 2.0));
        assert_eq!(g.tile_index(2, 1), 5);
        assert_eq!(g.tile_of(Vec2::new(0.5, 1.5)), (0, 1));
        assert_eq!(g.tile_of(Vec2::new(9.0, 9.0)), (2, 1), "clamped");
        assert!(!g.one_by_one);
        let g2 = TileGrid { cols: 1, rows: 1, one_by_one: true };
        assert!(g2.one_by_one);
        let _ = TileFillingMethod::OneByOne;
    }
}
