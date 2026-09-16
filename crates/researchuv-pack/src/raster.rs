//! Rasterizer placement — the free-space occupancy-grid placement path.
//!
//! When [`PackParams::raster_resolution`] is set (a power-of-two multiple of
//! 32; 0 disables), the movable-island placement stage switches from the
//! exact anchor-enumeration planner to the GPU rasterizer: every placed
//! island is rasterized into its extent class's occupancy grid, the grids
//! are dilated by the per-class margin radii, and each island's footprint
//! (outer ring + holes, transformed by its shadow pre-transform) is searched
//! against the free space with the strategy scoring — **per rotation
//! candidate** from the packer's rotation step, keeping the best-scoring
//! placement. The result is a [`Placed`] exactly like the exact planner's,
//! so the rest of the pipeline (heuristic, alignment, validation) is
//! unchanged.
//!
//! The grid quantizes the target box: `resolution` texels span its larger
//! side, and the margin becomes `ceil(margin · E · resolution)` texels for
//! extent-class bounds `E` (per-island-extent margins — see
//! [`researchuv_gpu::raster`]).

use crate::box2::Box2;
use crate::island::{Island, PlacedTransform};
use crate::params::{PackParams, PackStrategy};
use crate::place::Placed;
use researchuv_math::Vec2;

/// The per-class dilation radii in texels: `ceil(margin · E_j · resolution)`
/// with a one-texel floor.
pub fn margin_radii(margin: f64, resolution: u32) -> [u32; researchuv_gpu::N_CLASSES] {
    let mut out = [1u32; researchuv_gpu::N_CLASSES];
    for (j, &e) in researchuv_gpu::EXTENT_CLASSES.iter().enumerate() {
        out[j] = ((margin * e * resolution as f64).ceil() as u32).max(1);
    }
    out
}

/// The extent class of an island placed at `scale` within `region`.
fn class_of(island: &Island, scale: f64, region: &Box2) -> usize {
    let frac = island.bbox.max_extent() * scale / region.max_extent().max(1e-30);
    researchuv_gpu::extent_class(frac)
}

/// One rotation candidate's footprint in cell space (min-shifted) plus its
/// cell bounding box.
struct Footprint {
    rings: Vec<Vec<(f64, f64)>>,
    min_c: (f64, f64),
    w: u32,
    h: u32,
}

fn footprint(island: &Island, theta: f64, fit_scale: f64, origin: Vec2, k: f64) -> Footprint {
    let (cos, sin) = (theta.cos(), theta.sin());
    let xf = |p: Vec2| -> Vec2 {
        // Uniform scale commutes with rotation; rotate-then-scale matches
        // PlacedTransform::from_parts' T·R·S composition.
        let r = Vec2::new(cos * p.u - sin * p.v, sin * p.u + cos * p.v);
        Vec2::new(r.u * fit_scale, r.v * fit_scale)
    };
    let cell = |q: Vec2| -> (f64, f64) { ((q.u - origin.u) * k, (q.v - origin.v) * k) };
    let mut rings: Vec<Vec<(f64, f64)>> = Vec::with_capacity(1 + island.holes.len());
    let mut min_c = (f64::MAX, f64::MAX);
    let mut max_c = (f64::MIN, f64::MIN);
    for pts in std::iter::once(&island.verts).chain(island.holes.iter()) {
        let ring: Vec<(f64, f64)> = pts.iter().map(|p| cell(xf(*p))).collect();
        for c in &ring {
            min_c = (min_c.0.min(c.0), min_c.1.min(c.1));
            max_c = (max_c.0.max(c.0), max_c.1.max(c.1));
        }
        rings.push(ring);
    }
    // Shift by the footprint min: masks live in their own bounding box.
    for ring in rings.iter_mut() {
        for c in ring.iter_mut() {
            c.0 -= min_c.0;
            c.1 -= min_c.1;
        }
    }
    // Sub-texel shrink before ceil: an axis-aligned edge that is exactly
    // n texels wide must not round up to n+1 on a 1-ulp coordinate error.
    Footprint {
        rings,
        min_c,
        w: ((max_c.0 - min_c.0) - 1e-9).ceil().max(1.0) as u32,
        h: ((max_c.1 - min_c.1) - 1e-9).ceil().max(1.0) as u32,
    }
}

/// The placement search for one island against the raster state: every
/// rotation candidate is rasterized and searched; the best-scoring anchor
/// wins (ties: the earlier rotation in the ladder wins). Returns the placed
/// box in target space, or `None` when no rotation fits.
pub fn find_placement_raster(
    st: &mut researchuv_gpu::RasterState,
    island: &Island,
    region: &Box2,
    params: &PackParams,
    fit_scale: f64,
    tiles: Option<crate::tiles::TileGrid>,
) -> Option<Placed> {
    let res = st.resolution as f64;
    // With tiles, `region` is the whole tile-grid box and one resolution
    // spans ONE tile side; without tiles it spans the region's larger side.
    let ext = region.max_extent().max(1e-30);
    // With tiles the square state spans max(cols, rows) tiles per side at
    // `resolution / max(cols, rows)` cells per tile; one UV unit is one
    // tile side, so cells-per-UV-unit = per-tile cells.
    let k = match tiles {
        Some(tg) => (st.resolution / tg.cols.max(tg.rows)).max(1) as f64,
        None => res / ext,
    };
    let origin = region.min;
    let fit_scale = fit_scale.max(1e-9);
    let mode = match tiles {
        Some(tg) => researchuv_gpu::RasterMode::Tiles {
            // The raster grid spans per_tile × cols cells; one tile's side
            // in cells is therefore resolution / cols.
            tile_cells: (st.resolution / tg.cols.max(1)).max(1),
            tile_cols: tg.cols,
        },
        None => match crate::place::effective_strategy(params) {
            PackStrategy::SideToSideVert => researchuv_gpu::RasterMode::SideToSideVert,
            PackStrategy::SideToSideHori => researchuv_gpu::RasterMode::SideToSideHori,
            PackStrategy::Square | PackStrategy::Automatic => researchuv_gpu::RasterMode::Corner,
        },
    };
    // The extent class is relative to ONE TILE (the margin grids' E_j
    // bounds are fractions of a tile side), not the whole tile grid.
    let class = match tiles {
        Some(tg) => {
            let per_tile_extent = region.max_extent() / tg.cols.max(tg.rows) as f64;
            researchuv_gpu::extent_class(
                island.bbox.max_extent() * fit_scale / per_tile_extent.max(1e-30),
            )
        }
        None => class_of(island, fit_scale, region),
    };
    let radii = margin_radii(params.margin, st.resolution);
    if !st.dilate_all(&radii) {
        return None;
    }
    let mut best: Option<(f64, f64, u32, u32, (f64, f64), u32, u32)> = None;
    for &theta in params.rotation_candidates(-1).iter() {
        let fp = footprint(island, theta, fit_scale, origin, k);
        if fp.w > st.resolution || fp.h > st.resolution {
            continue;
        }
        let ring_refs: Vec<&[(f64, f64)]> = fp.rings.iter().map(|r| r.as_slice()).collect();
        let mask = researchuv_gpu::DeviceMask::rasterize(&ring_refs, fp.w, fp.h)?;
        if let Some((cx, cy, score)) = st.find_best(&mask, mode, class) {
            let better = match best {
                Some((bs, ..)) => score < bs,
                None => true,
            };
            if better {
                best = Some((score, theta, cx, cy, fp.min_c, fp.w, fp.h));
            }
        }
    }
    let (_, theta, cx, cy, min_c, w, h) = best?;
    // The mask is the min-shifted footprint, so placing it at cell (cx, cy)
    // puts the footprint's min corner at anchor = origin + (cx, cy)/k. The
    // translation achieving that under T·R·S is
    //   t = anchor − min(R·S·p) = origin + (cx, cy)/k − (origin + min_c/k)
    //     = ((cx, cy) − min_c) / k
    // (negative translations are normal for raw geometry far from origin).
    let anchor = Vec2::new(origin.u + cx as f64 / k, origin.v + cy as f64 / k);
    let box_ = Box2::new(
        anchor,
        Vec2::new(anchor.u + w as f64 / k, anchor.v + h as f64 / k),
    );
    let transform = PlacedTransform::from_parts(
        theta,
        false,
        fit_scale,
        (cx as f64 - min_c.0) / k,
        (cy as f64 - min_c.1) / k,
        box_,
    );
    Some(Placed { island_index: 0, transform, box_ })
}

/// Rasterize an already-placed island (transform applied) into its extent
/// class's occupancy grid. Returns the class used.
pub fn rasterize_placed(
    st: &mut researchuv_gpu::RasterState,
    island: &Island,
    transform: &PlacedTransform,
    region: &Box2,
    tiles: Option<crate::tiles::TileGrid>,
) -> Option<usize> {
    let res = st.resolution as f64;
    let ext = region.max_extent().max(1e-30);
    // Tiled: one UV unit = one tile side = resolution / side tiles cells.
    let k = match tiles {
        Some(tg) => (st.resolution / tg.cols.max(tg.rows)).max(1) as f64,
        None => res / ext,
    };
    let origin = region.min;
    let class = match tiles {
        Some(tg) => {
            let per_tile = region.max_extent() / tg.cols.max(tg.rows) as f64;
            researchuv_gpu::extent_class(
                island.bbox.max_extent() * transform.scale / per_tile.max(1e-30),
            )
        }
        None => class_of(island, transform.scale, region),
    };
    let cell_pts: Vec<(f64, f64)> = island
        .verts
        .iter()
        .map(|p| {
            let q = transform.apply(*p);
            ((q.u - origin.u) * k, (q.v - origin.v) * k)
        })
        .collect();
    if !st.rasterize_ring(&cell_pts, class) {
        return None;
    }
    for hole in &island.holes {
        let pts: Vec<(f64, f64)> = hole
            .iter()
            .map(|p| {
                let q = transform.apply(*p);
                ((q.u - origin.u) * k, (q.v - origin.v) * k)
            })
            .collect();
        if !st.rasterize_ring(&pts, class) {
            return None;
        }
    }
    Some(class)
}

/// The uniform placement scale for the raster path: the largest scale at
/// which the movable islands' bounding-box areas fit ~85% of the target and
/// every island's extent clears the border band (the MaxScale intent,
/// grid-quantized).
pub fn fit_scale(shadows: &[Island], movable: &[usize], region: &Box2, margin: f64) -> f64 {
    let total_bbox: f64 = movable
        .iter()
        .map(|&i| shadows[i].bbox.width() * shadows[i].bbox.height())
        .sum();
    if total_bbox <= 0.0 {
        return 1.0;
    }
    let band = (margin * 2.0).min(0.5);
    let avail = (region.width() * (1.0 - band)) * (region.height() * (1.0 - band));
    let by_area = (0.85 * avail / total_bbox).sqrt();
    let max_ext = movable
        .iter()
        .map(|&i| shadows[i].bbox.max_extent())
        .fold(1e-30, f64::max);
    let by_extent = region.max_extent() * (1.0 - band) / max_ext;
    by_area.min(by_extent).min(1.0).max(1e-6)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn margin_radii_scale_with_class_bounds() {
        let r = margin_radii(0.01, 256);
        assert_eq!(r[0], (0.01f64 * 0.125 * 256.0).ceil() as u32); // 1
        assert_eq!(r[3], (0.01f64 * 1.0 * 256.0).ceil() as u32); // 3
        assert!(r[0] <= r[1] && r[1] <= r[2] && r[2] <= r[3]);
    }

    #[test]
    fn footprint_rotates_and_shifts_to_its_own_origin() {
        // A 4×1 strip at a nonzero raw position: rotated 90° its mask is
        // 1 wide × 4 tall, and the ring coordinates start at (0, 0).
        let island = Island::from_polygon(vec![
            Vec2::new(5.0, 7.0),
            Vec2::new(9.0, 7.0),
            Vec2::new(9.0, 8.0),
            Vec2::new(5.0, 8.0),
        ]);
        let fp = footprint(&island, std::f64::consts::FRAC_PI_2, 1.0, Vec2::new(0.0, 0.0), 1.0);
        assert_eq!((fp.w, fp.h), (1, 4), "the strip rotates to 1×4");
        // The shifted rings live inside the mask's bounding box at (0, 0).
        for c in fp.rings.iter().flatten() {
            assert!(c.0 >= -1e-9 && c.0 <= fp.w as f64 + 1e-9);
            assert!(c.1 >= -1e-9 && c.1 <= fp.h as f64 + 1e-9);
        }
        let unrot = footprint(&island, 0.0, 1.0, Vec2::new(0.0, 0.0), 1.0);
        assert_eq!((unrot.w, unrot.h), (4, 1));
        for c in unrot.rings.iter().flatten() {
            assert!(c.0 >= -1e-9 && c.0 <= unrot.w as f64 + 1e-9);
            assert!(c.1 >= -1e-9 && c.1 <= unrot.h as f64 + 1e-9);
        }
    }
}
