//! Rasterizer placement — the free-space occupancy-grid placement path.
//!
//! When [`PackParams::raster_resolution`] is set (a power-of-two multiple of
//! 32; 0 disables), the movable-island placement stage switches from the
//! exact anchor-enumeration planner to the GPU rasterizer: every placed
//! island is rasterized into the occupancy grid, the grid is dilated by the
//! margin radius, and each island's footprint (outer ring + holes,
//! transformed by its shadow pre-transform) is searched against the free
//! space with the strategy scoring. The result is a [`Placed`] exactly like
//! the exact planner's, so the rest of the pipeline (heuristic, alignment,
//! validation) is unchanged. Translation-only: rotations stay with the
//! exact planner and the pre-rotation logic.
//!
//! The grid quantizes the target box: `resolution` texels span its larger
//! side, the margin becomes `ceil(margin · resolution)` texels of separation
//! (plus the guard word), and the anchor maps back to UV by the same scale.

use crate::box2::Box2;
use crate::island::{Island, PlacedTransform};
use crate::params::{PackParams, PackStrategy};
use crate::place::Placed;
use researchuv_math::Vec2;

/// Map a UV point to cell coordinates in the grid.
fn to_cell(p: Vec2, origin: Vec2, scale: f64) -> (f64, f64) {
    ((p.u - origin.u) * scale, (p.v - origin.v) * scale)
}

/// The placement search for one island against the raster state.
/// Returns the placed box in target space, or `None` when the footprint
/// fits nowhere.
pub fn find_placement_raster(
    st: &mut researchuv_gpu::RasterState,
    island: &Island,
    region: &Box2,
    params: &PackParams,
    fit_scale: f64,
) -> Option<Placed> {
    let res = st.resolution as f64;
    let ext = region.max_extent().max(1e-30);
    let scale = res / ext;
    let origin = region.min;
    let fit_scale = fit_scale.max(1e-9);
    // The footprint in cell space: outline + holes under the identity
    // placement (the raster path searches translations only).
    let (mut min_c, mut max_c) = ((f64::MAX, f64::MAX), (f64::MIN, f64::MIN));
    for p in island.verts.iter().chain(island.holes.iter().flatten()) {
        let q = Vec2::new(p.u * fit_scale, p.v * fit_scale);
        let c = to_cell(q, origin, scale);
        min_c = (min_c.0.min(c.0), min_c.1.min(c.1));
        max_c = (max_c.0.max(c.0), max_c.1.max(c.1));
    }
    let w = (max_c.0 - min_c.0).ceil().max(1.0) as u32;
    let h = (max_c.1 - min_c.1).ceil().max(1.0) as u32;
    if w > st.resolution || h > st.resolution {
        return None;
    }
    let mut rings: Vec<Vec<(f64, f64)>> = Vec::with_capacity(1 + island.holes.len());
    for pts in std::iter::once(&island.verts).chain(island.holes.iter()) {
        rings.push(
            pts.iter()
                .map(|p| {
                    let q = Vec2::new(p.u * fit_scale, p.v * fit_scale);
                    to_cell(q, origin, scale)
                })
                .collect(),
        );
    }
    let ring_refs: Vec<&[(f64, f64)]> = rings.iter().map(|r| r.as_slice()).collect();
    let mask = researchuv_gpu::DeviceMask::rasterize(&ring_refs, w, h)?;
    let mode = match crate::place::effective_strategy(params) {
        PackStrategy::SideToSideVert => researchuv_gpu::RasterMode::SideToSideVert,
        PackStrategy::SideToSideHori => researchuv_gpu::RasterMode::SideToSideHori,
        PackStrategy::Square | PackStrategy::Automatic => researchuv_gpu::RasterMode::Corner,
    };
    let margin_radius = (params.margin * res).ceil().max(1.0) as u32;
    if !st.dilate(margin_radius) {
        return None;
    }
    let (cx, cy, _score) = st.find_best(&mask, mode)?;
    // Cell anchor → UV anchor, then shift the shadow geometry there.
    let anchor = Vec2::new(
        origin.u + cx as f64 / scale + min_c.0 / scale,
        origin.v + cy as f64 / scale + min_c.1 / scale,
    );
    // The placed box in UV: the anchor spans the footprint's cell bbox.
    let box_ = Box2::new(anchor, Vec2::new(anchor.u + w as f64 / scale, anchor.v + h as f64 / scale));
    // Border: the search must keep the box inside the region (the candidate
    // range already bounds cells to the grid, which IS the region).
    let mut pl = Placed {
        island_index: 0,
        transform: PlacedTransform::identity(),
        box_,
    };
    pl.transform = PlacedTransform::from_parts(
        0.0,
        false,
        fit_scale,
        anchor.u - island.bbox.min.u * fit_scale,
        anchor.v - island.bbox.min.v * fit_scale,
        pl.box_,
    );
    pl.box_ = Box2::new(
        Vec2::new(anchor.u, anchor.v),
        Vec2::new(
            anchor.u + island.bbox.width() * fit_scale,
            anchor.v + island.bbox.height() * fit_scale,
        ),
    );
    Some(pl)
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

/// Rasterize an already-placed island (transform applied) into the
/// occupancy grid.
pub fn rasterize_placed(
    st: &mut researchuv_gpu::RasterState,
    island: &Island,
    transform: &PlacedTransform,
    region: &Box2,
) -> bool {
    let res = st.resolution as f64;
    let ext = region.max_extent().max(1e-30);
    let scale = res / ext;
    let origin = region.min;
    let cell_pts: Vec<(f64, f64)> = island
        .verts
        .iter()
        .map(|p| to_cell(transform.apply(*p), origin, scale))
        .collect();
    if !st.rasterize_ring(&cell_pts) {
        return false;
    }
    for hole in &island.holes {
        let pts: Vec<(f64, f64)> = hole
            .iter()
            .map(|p| to_cell(transform.apply(*p), origin, scale))
            .collect();
        if !st.rasterize_ring(&pts) {
            return false;
        }
    }
    true
}
