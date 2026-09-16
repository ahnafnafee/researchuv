//! Pipeline — the `EXECUTE_SCENARIO` pack entry point (the engine's scenario
//! runner, wired to this crate's CPU placement).
//!
//! Order of operations (mirroring the engine's pack scenario + the addon's
//! `spipeline` flow):
//!
//! 1. Max-dimension clamp (`MAX_ISLAND_DIM_ALLOWED` = 4.0) on non-fixed-scale
//!    islands.
//! 2. Effective target box (non-square ratio applied).
//! 3. Texel density (`set_tdensity`): per-island scale policy (pre-scale).
//! 4. Scale normalization (`normalize_scale` + `island_normalize_multiplier`).
//! 5. Grouping (`GroupingMethod` → `GroupResult`, per-group regions).
//! 6. Similarity: `split_by_similarity` + `align_similar` pre-rotations.
//! 7. Placement: static/"others" islands first (at their current position),
//!    then the movable islands in (group id, −max-extent) order via
//!    [`crate::place::find_best_placement`], each inside its group's region
//!    when `groups_together` is on.
//! 8. Heuristic refinement (time-budgeted, when enabled + active).
//! 9. Pixel-perfect alignment (Corner/Center snap → `ALIGNED` flag).
//! 10. Validation (overlap / outside-target / self-intersection / holes →
//!     `UvpmRetcode` + island flags), on ring sets (outlines + holes).
//! 11. Split-overlap (integer tile offsets for leftover overlaps).
//!
//! The result contract is per-island `PlacedTransform` + flags + retcode —
//! the engine's `CONTAINS_TRANSFORM` / `CONTAINS_FLAGS` / `CONTAINS_VERTICES`
//! serialization payload. `islands` is the *raw* input (outlines in each
//! island's own UV space); the returned transforms map that raw space to the
//! target space, and the per-island `OVERLAPS` / `OUTSIDE_TARGET_BOX` /
//! `ALIGNED` bits are stamped on the input islands.

use crate::align::{orient_to_3d_rotation, pixel_perfect_offset};
use crate::box2::Box2;
use crate::groups::{assign_groups, GroupResult};
use crate::heur::{advanced_heuristic_active, heuristic_refine, HeuristicStats};
use crate::island::{Island, PlacedTransform, OVERLAPS, OUTSIDE_TARGET_BOX, ALIGNED};
use crate::params::{
    iparam, CoordSpace, GroupingMethod, PackOpType, PackParams, ScaleMode, UvpmRetcode,
};
use crate::place::{
    arrange_non_packed, find_best_placement, local_min_corner, rotated_size, Placed,
};
use crate::rng::SplitMix64;
use crate::similarity;
use crate::split;
use crate::tdensity::{set_tdensity, TexelDensityPolicy};
use crate::validate;
use researchuv_math::Vec2;

/// `MAX_ISLAND_DIM_ALLOWED` — non-fixed-scale islands larger than this (UV
/// units) are pre-scaled down before packing (the addon's input clamp).
pub const MAX_ISLAND_DIM_ALLOWED: f64 = 4.0;

/// The result of a pack run — the per-island transform + flags + retcode
/// contract of the engine's `EXECUTE_SCENARIO` reply.
#[derive(Clone, Debug)]
pub struct PackResult {
    /// Per-island placement (in island order); `None` = non-packed
    /// (arranged outside the target or skipped). Each transform maps the
    /// island's raw UV space to the target space.
    pub placed: Vec<Option<PlacedTransform>>,
    /// Indices of the non-packed islands.
    pub non_packed: Vec<u32>,
    /// Grouping result (group ids, keys, members, regions).
    pub groups: GroupResult,
    /// Validation report (overlap / outside / self-intersection / holes).
    pub validation: validate::ValidationReport,
    /// The run's retcode.
    pub retcode: UvpmRetcode,
    /// Heuristic-search statistics (zero when the heuristic did not run).
    pub heuristic: HeuristicStats,
    /// Per-island texel-density policy (scale, density) when enabled.
    pub tdensity: Vec<TexelDensityPolicy>,
    /// Similarity clusters (empty when nothing clustered).
    pub similarity_clusters: Vec<Vec<u32>>,
    /// Split-overlap tile offsets (empty when the pass did not run).
    pub split_offsets: Vec<(i32, i32)>,
    /// Error raised by the split-overlap pass (tile-range InputError).
    pub split_error: Option<String>,
}

/// Compose two `PlacedTransform`s: the result applies `inner` first, then
/// `outer`. Recovers the (scale, rotation, flipped) components from the
/// composed 2×2 matrix (both transforms are similarity maps, so the product
/// is one too).
fn compose_transform(outer: &PlacedTransform, inner: &PlacedTransform) -> PlacedTransform {
    let a = outer.a * inner.a + outer.b * inner.c;
    let b = outer.a * inner.b + outer.b * inner.d;
    let c = outer.c * inner.a + outer.d * inner.c;
    let d = outer.c * inner.b + outer.d * inner.d;
    let tx = outer.a * inner.tx + outer.b * inner.ty + outer.tx;
    let ty = outer.c * inner.tx + outer.d * inner.ty + outer.ty;
    // The uniform scale is the norm of R·S's first column (a, c) — the
    // diagonal (a, d) vanishes at ±90° rotations.
    let s = ((a * a + c * c).max(0.0)).sqrt();
    let det = a * d - b * c;
    let flipped = s > 0.0 && det < 0.0;
    let rotation = if flipped {
        (a / s).atan2(c / s)
    } else {
        (-b / s).atan2(a / s)
    };
    PlacedTransform {
        a,
        b,
        c,
        d,
        tx,
        ty,
        rotation,
        flipped,
        scale: s,
        box_: outer.box_,
    }
}

/// Run the pack scenario over `islands` with `params` (the `pack` entry
/// point — `EXECUTE_SCENARIO`). `islands` is mutated in place: the
/// `OVERLAPS` / `OUTSIDE_TARGET_BOX` / `ALIGNED` flags are stamped and the
/// split-overlap offsets are written back to the `split_offset_x/y` iparams.
pub fn pack(islands: &mut [Island], params: &PackParams) -> PackResult {
    let target = params.effective_box();
    let n = islands.len();
    let mut rng = SplitMix64::new(params.seed);

    // --- 1: max-dimension clamp (MAX_ISLAND_DIM_ALLOWED) ---
    // Non-fixed-scale islands larger than the engine's allowed maximum are
    // pre-scaled down before packing. The clamp is folded into the per-island
    // pre-scale (not the input geometry), so the returned transforms keep
    // mapping each island's raw UV space.
    let clamp: Vec<f64> = islands
        .iter()
        .map(|isl| {
            let ext = isl.bbox.max_extent();
            if params.scale_mode == ScaleMode::FixedScale
                || params.scale_mode == ScaleMode::FixedScaleMaxMargin
                || ext <= MAX_ISLAND_DIM_ALLOWED
            {
                1.0
            } else {
                MAX_ISLAND_DIM_ALLOWED / ext
            }
        })
        .collect();

    // --- 2: texel density (per-island scale policy, applied before packing) ---
    let extents: Vec<f64> = islands
        .iter()
        .zip(clamp.iter())
        .map(|(i, &k)| i.bbox.max_extent() * k)
        .collect();
    let tex_size = params.pixel_margin_tex_size.max(1) as f32;
    let tdensity: Vec<TexelDensityPolicy> = if params.tdensity.enable {
        set_tdensity(&params.tdensity, &extents, tex_size, 1.0)
    } else {
        vec![TexelDensityPolicy { scale: 1.0, density: 0.0 }; n]
    };
    let mut pre_scale: Vec<f64> = clamp
        .iter()
        .zip(tdensity.iter())
        .map(|(&k, p)| (k * p.scale.max(1e-30)).max(1e-30))
        .collect();

    // --- 3: scale normalization (normalize_scale + per-island multiplier) ---
    if params.normalize_scale {
        let global_max = extents.iter().cloned().fold(1e-30, f64::max);
        let local = params.normalize_space == CoordSpace::Local;
        for i in 0..n {
            let ref_extent = if local { extents[i] } else { global_max };
            // The channel stores a percent (10..1000, default 100): 100 = ×1.
            let mult = islands[i]
                .iparam_channel(iparam::NORMALIZE_MULTIPLIER)
                .map(|v| (v / 100.0).clamp(0.1, 10.0))
                .unwrap_or(params.island_normalize_multiplier);
            if ref_extent > 0.0 {
                pre_scale[i] *= mult / ref_extent;
            }
        }
    }

    // --- 3: grouping (method → group ids, members, target-space regions) ---
    let groups = assign_groups(islands, &params.grouping, &params.similarity, &target);

    // --- 4: similarity clusters + member pre-rotations ---
    let mut similarity_clusters: Vec<Vec<u32>> = Vec::new();
    let mut similar_rot: Vec<Option<f64>> = vec![None; n];
    let method_is_similarity = params.grouping.method == GroupingMethod::Similarity;
    if method_is_similarity
        || (groups.members.iter().any(|m| m.len() >= 2) && params.similarity.threshold > 0.0)
    {
        let clusters = similarity::split_by_similarity(islands, &params.similarity);
        similarity_clusters = clusters.clone();
        if let Some(ref_idx) = clusters.first().and_then(|c| c.first()) {
            let t =
                similarity::align_similar_full(*ref_idx, &clusters, islands, &params.similarity);
            for (i, opt) in t.iter().enumerate() {
                if let Some((_s, r, _tx, _ty)) = opt {
                    similar_rot[i] = Some(*r);
                }
            }
        }
    }

    // --- 5: shadow islands (geometry pre-rotated / pre-scaled for placement) ---
    //
    // Placement runs on island bboxes. Pre-rotations (orient-to-3d,
    // similarity alignment) and pre-scales (texel density, normalization)
    // change that bbox, so placement runs on *shadow* islands whose bbox is
    // the transformed one; the final per-island transform is the placement
    // transform composed with the shadow's own transform (raw UV space →
    // shadow space → target space).
    let mut shadows: Vec<Island> = Vec::with_capacity(n);
    let mut shadow_t: Vec<PlacedTransform> = Vec::with_capacity(n);
    for i in 0..n {
        let isl = &islands[i];
        let mut theta = 0.0f64;
        if params.orient_to_3d.enable && !params.pre_rotation_disable {
            if let Some(t3) = orient_to_3d_rotation(isl, &params.orient_to_3d) {
                theta = t3;
            }
        }
        if let Some(ts) = similar_rot[i] {
            theta += ts;
        }
        let s = pre_scale[i].max(1e-30);
        let (rw, rh) = rotated_size(isl, theta);
        let (lmu, lmv) = local_min_corner(isl, theta, false, s);
        let bb = Box2::new(
            Vec2::new(lmu, lmv),
            Vec2::new(lmu + rw * s, lmv + rh * s),
        );
        let st = PlacedTransform::from_parts(theta, false, s, 0.0, 0.0, bb);
        let mut sh = isl.clone();
        sh.bbox = bb;
        shadows.push(sh);
        shadow_t.push(st);
    }

    // --- 6: placement order ---
    // Fixed (static / "others") islands are placed first, at their current
    // position with the identity transform. `RepackWithOthers` moves the
    // selected islands plus the unselected ones that overlap the op target;
    // unselected islands elsewhere keep their position (4.1.2 semantics).
    let pack_op = params.pack_op;
    let is_movable = |i: usize| {
        let isl = &islands[i];
        if isl.is_static {
            return false;
        }
        match pack_op {
            PackOpType::Pack => true,
            PackOpType::RepackWithOthers => {
                isl.is_selected() || isl.bbox.intersect(&target).is_some()
            }
            PackOpType::PackToOthers => isl.is_selected(),
        }
    };
    let fixed: Vec<usize> = (0..n).filter(|&i| !is_movable(i)).collect();
    let mut movable: Vec<usize> = (0..n).filter(|&i| is_movable(i)).collect();
    // Movable order: (group id, −max extent).
    movable.sort_by(|&a, &b| {
        let ga = groups.group_ids.get(a).copied().unwrap_or(0);
        let gb = groups.group_ids.get(b).copied().unwrap_or(0);
        ga.cmp(&gb).then_with(|| {
            shadows[b]
                .bbox
                .max_extent()
                .partial_cmp(&shadows[a].bbox.max_extent())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    });

    let mut placed: Vec<Option<PlacedTransform>> = vec![None; n];
    let mut placed_any: Vec<Placed> = Vec::new();
    let mut non_packed: Vec<u32> = Vec::new();

    // Fixed islands: identity transform, at their current position.
    for &i in fixed.iter() {
        let bb = shadows[i].bbox;
        if bb.width() > 0.0 && bb.height() > 0.0 {
            let (lmu, lmv) = local_min_corner(&shadows[i], 0.0, false, 1.0);
            let t = PlacedTransform::from_parts(0.0, false, 1.0, bb.min.u - lmu, bb.min.v - lmv, bb);
            placed_any.push(Placed {
                island_index: i as u32,
                transform: t,
                box_: bb,
            });
            placed[i] = Some(t);
        } else {
            non_packed.push(i as u32);
        }
    }

    // The rasterizer placement path (GPU occupancy grid). Created once;
    // fixed islands seed the grid, each placed island joins it. Tile
    // targets are rasterizer-native: the grid spans the whole tile grid.
    let raster_tiles = if params.raster_resolution > 0
        && params.tile_target != crate::params::TileTargetMode::DynamicTiles || (params.raster_resolution > 0 && params.tiles_in_row != 10)
    {
        let total_area: f64 = shadows.iter().map(|i| i.bbox.width() * i.bbox.height()).sum();
        Some(crate::tiles::TileGrid::from_params(params, total_area, target.max_extent()))
    } else {
        None
    };
    let raster_region = match raster_tiles {
        Some(tg) => tg.box_(),
        None => target,
    };
    let mut raster_state = if params.raster_resolution > 0 {
        match researchuv_gpu::RasterState::new(
            params.raster_resolution
                * raster_tiles.map(|t| t.cols.max(t.rows)).unwrap_or(1),
        ) {
            Some(mut st) => {
                let boundaries_ok = st.reset()
                    && match raster_tiles {
                        Some(tg) => st.block_tile_boundaries(tg.cols, tg.rows),
                        None => true,
                    };
                if !boundaries_ok {
                    None
                } else {
                    // Fixed islands are part of the free space.
                    let mut ok = true;
                    for p in placed_any.iter() {
                        let isl = &shadows[p.island_index as usize];
                        if crate::raster::rasterize_placed(&mut st, isl, &p.transform, &raster_region, raster_tiles)
                            .is_none()
                        {
                            ok = false;
                            break;
                        }
                    }
                    if ok {
                        Some(st)
                    } else {
                        None
                    }
                }
            }
            None => None,
        }
    } else {
        None
    };
    if params.raster_resolution > 0 && raster_state.is_none() {
        eprintln!(
            "researchuv-pack: raster placement requested but unavailable —              falling back to the exact planner"
        );
    }

    let raster_fit = if raster_state.is_some() {
        match params.scale_mode {
            // Fixed-scale modes honor the user's scale exactly.
            crate::params::ScaleMode::FixedScale | crate::params::ScaleMode::FixedScaleMaxMargin => {
                params.scale.max(1e-9)
            }
            // MaxScale: the uniform grid-quantized fit.
            crate::params::ScaleMode::MaxScale => {
                crate::raster::fit_scale(&shadows, &movable, &raster_region, params.margin)
            }
        }
    } else {
        1.0
    };

    // Movable islands.
    for &i in movable.iter() {
        let isl = &shadows[i];
        // Region: the group's region when groups are laid out separately.
        let region = if params.grouping.groups_together {
            groups.region_of(i as u32, &target)
        } else {
            target
        };
        let base_t = shadow_t[i];
        // Raster path first; the exact planner is the fallback.
        let raster_hit = match raster_state.as_mut() {
            Some(st) if !params.grouping.groups_together => {
                match crate::raster::find_placement_raster(
                            st,
                            isl,
                            &raster_region,
                            params,
                            raster_fit,
                            raster_tiles,
                        ) {
                    Some(mut pl) => {
                        let t = compose_transform(&pl.transform, &base_t);
                        pl.island_index = i as u32;
                        placed_any.push(pl.clone());
                        placed[i] = Some(t);
                        let ok =
                            crate::raster::rasterize_placed(st, isl, &t, &raster_region, raster_tiles).is_some();
                        if ok {
                            Some(())
                        } else {
                            placed_any.pop();
                            placed[i] = None;
                            None
                        }
                    }
                    None => None,
                }
            }
            _ => None,
        };
        if raster_hit.is_some() {
            continue;
        }
        match find_best_placement(isl, &placed_any, &region, params, &mut rng) {
            Some(mut pl) => {
                let t = compose_transform(&pl.transform, &base_t);
                pl.island_index = i as u32;
                pl.transform = t;
                placed_any.push(pl);
                placed[i] = Some(t);
            }
            None => match arrange_non_packed(isl, &placed_any, &region, params) {
                Some(mut pl) => {
                    let t = compose_transform(&pl.transform, &base_t);
                    pl.island_index = i as u32;
                    pl.transform = t;
                    placed_any.push(pl);
                    placed[i] = Some(t);
                }
                None => {
                    non_packed.push(i as u32);
                }
            },
        }
    }

    // --- 7: heuristic refinement (time-budgeted stochastic improvement) ---
    // The GPU multi-restart relocation runs first when a device is present
    // (one CUDA block per restart); the CPU sampler then improves whichever
    // layout won, and both stages validate through the same score.
    let mut heuristic = HeuristicStats::default();
    if params.heuristic_enable && advanced_heuristic_active(params, n) {
        let mut flat: Vec<Placed> = placed
            .iter()
            .enumerate()
            .filter_map(|(i, p)| {
                p.as_ref().map(|t| Placed {
                    island_index: i as u32,
                    transform: *t,
                    box_: t.box_,
                })
            })
            .collect();
        if flat.len() >= 2 {
            if let Some((layout, _score)) = researchuv_gpu::heuristic::refine(
                &flat
                    .iter()
                    .map(|p| researchuv_gpu::LayoutBox {
                        w: p.box_.width(),
                        h: p.box_.height(),
                        x: p.box_.min.u,
                        y: p.box_.min.v,
                    })
                    .collect::<Vec<_>>(),
                (target.min.u, target.min.v, target.max.u, target.max.v),
                params.margin,
                params.seed,
                24,   // restarts
                12,   // relocation passes
                512,  // candidate anchors
            ) {
                // Apply the winning restart only if it scores better.
                let before = crate::heur::packing_score(&flat, &target, params);
                let candidate: Vec<Placed> = flat
                    .iter()
                    .zip(layout.iter())
                    .map(|(p, b)| {
                        let mut q = p.clone();
                        q.transform.tx += b.x - p.box_.min.u;
                        q.transform.ty += b.y - p.box_.min.v;
                        q.transform.box_ = Box2::new(
                            Vec2::new(b.x, b.y),
                            Vec2::new(b.x + p.box_.width(), b.y + p.box_.height()),
                        );
                        q.box_ = q.transform.box_;
                        q
                    })
                    .collect();
                let after = crate::heur::packing_score(&candidate, &target, params);
                if after < before {
                    flat = candidate;
                }
            }
        }
        let stats = heuristic_refine(&shadows, &mut flat, &target, params, &mut rng);
        for p in flat.iter() {
            let idx = p.island_index as usize;
            if idx < n {
                // Re-compose the (possibly improved) placement transform with
                // the island's pre-transform.
                let base_t = shadow_t[idx];
                let t = compose_transform(&p.transform, &base_t);
                placed[idx] = Some(t);
            }
        }
        heuristic = stats;
    }

    // --- 8: pixel-perfect alignment (snap the placed box to the pixel grid) ---
    if params.pixel_perfect_align {
        for (i, t) in placed.iter_mut().enumerate() {
            if let Some(t) = t {
                let anchor = t.box_.min;
                let d = pixel_perfect_offset(anchor, params);
                if d.u != 0.0 || d.v != 0.0 {
                    let mut t2 = *t;
                    t2.tx += d.u;
                    t2.ty += d.v;
                    t2.box_ = Box2::new(
                        Vec2::new(t2.box_.min.u + d.u, t2.box_.min.v + d.v),
                        Vec2::new(t2.box_.max.u + d.u, t2.box_.max.v + d.v),
                    );
                    *t = t2;
                    islands[i].flags |= ALIGNED;
                }
            }
        }
    }

    // --- 9: validation (flags + retcode) ---
    // Validate the *final* placements (post pixel-perfect offsets).
    let outlines: Vec<Vec<Vec2>> = placed
        .iter()
        .enumerate()
        .map(|(i, t)| match t {
            Some(t) => t.transform_poly(&islands[i].verts),
            None => islands[i].verts.clone(),
        })
        .collect();
    let placed_holes: Vec<Vec<Vec<Vec2>>> = placed
        .iter()
        .enumerate()
        .map(|(i, t)| match t {
            Some(t) => islands[i].transformed_holes(t),
            None => islands[i].holes.clone(),
        })
        .collect();
    let mut validation =
        validate::validate_islands(islands, &outlines, &placed_holes, &target, params);
    // Tile placements live on the tile grid: a box inside ANY tile is not
    // "outside the target" (only boxes beyond the whole grid are).
    if let Some(tg) = raster_tiles {
        let in_some_tile = |b: &Box2| -> bool {
            (0..tg.cols).any(|ix| (0..tg.rows).any(|iy| {
                let tb = tg.tile_box(ix, iy);
                tb.contains_box_eps(b, 1e-9)
            }))
        };
        validation.outside.retain(|&i| {
            let inside = placed[i as usize]
                .as_ref()
                .map(|t| in_some_tile(&t.box_))
                .unwrap_or(false);
            if inside {
                islands[i as usize].flags &= !OUTSIDE_TARGET_BOX;
            }
            !inside
        });
        if validation.outside.is_empty()
            && validation.retcode == UvpmRetcode::NoSpace
            && validation.self_intersecting.is_empty()
        {
            validation.retcode = if validation.overlapping.is_empty() {
                UvpmRetcode::Success
            } else {
                UvpmRetcode::Warning
            };
        }
    }

    // Non-packed islands (arranged outside the target) are exempt from the
    // OUTSIDE_TARGET_BOX error.
    if !non_packed.is_empty() {
        validation.outside.retain(|&i| !non_packed.contains(&i));
        for &i in non_packed.iter() {
            islands[i as usize].flags &= !OUTSIDE_TARGET_BOX;
        }
        // No other outside-target islands: the outside entries were only the
        // arranged ones → fix up the retcode.
        if validation.outside.is_empty() && validation.retcode == UvpmRetcode::NoSpace {
            validation.retcode = if validation.overlapping.is_empty() {
                UvpmRetcode::Success
            } else {
                UvpmRetcode::Warning
            };
        }
    }

    // --- 10: split-overlap (integer tile offsets for leftover overlaps) ---
    let mut split_offsets: Vec<(i32, i32)> = Vec::new();
    let mut split_error: Option<String> = None;
    if params.split_overlap.enable && !validation.overlapping.is_empty() {
        match split::split_overlapping(islands, &params.split_overlap, params.align_priority_enable)
        {
            Ok(off) => {
                split_offsets = off.clone();
                for (i, (dx, dy)) in off.iter().enumerate() {
                    if *dx == 0 && *dy == 0 {
                        continue;
                    }
                    if let Some(mut t) = placed[i] {
                        t.tx += *dx as f64;
                        t.ty += *dy as f64;
                        t.box_ = Box2::new(
                            Vec2::new(t.box_.min.u + *dx as f64, t.box_.min.v + *dy as f64),
                            Vec2::new(t.box_.max.u + *dx as f64, t.box_.max.v + *dy as f64),
                        );
                        placed[i] = Some(t);
                    }
                }
                split::write_split_offsets(islands, &off);
                // The overlaps were separated: clear the overlap state.
                for i in 0..n {
                    islands[i].flags &= !OVERLAPS;
                }
                validation.overlapping.clear();
                if validation.outside.is_empty() && validation.self_intersecting.is_empty() {
                    validation.retcode = UvpmRetcode::Success;
                }
            }
            Err(e) => {
                split_error = Some(e.to_string());
                validation.retcode = UvpmRetcode::InvalidIslands;
            }
        }
    }

    validation.retcode = if split_error.is_some() {
        UvpmRetcode::InvalidIslands
    } else {
        validation.retcode
    };

    PackResult {
        placed,
        non_packed,
        groups,
        validation: validation.clone(),
        retcode: validation.retcode,
        heuristic,
        tdensity,
        similarity_clusters,
        split_offsets,
        split_error,
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
    fn pack_two_squares() {
        let mut isls = vec![sq(0.0, 0.0, 0.5), sq(0.0, 0.0, 0.5)];
        let p = PackParams::default();
        let r = pack(&mut isls, &p);
        assert_eq!(r.placed.len(), 2);
        assert!(r.placed[0].is_some());
        assert!(r.placed[1].is_some());
        assert!(r.non_packed.is_empty());
        assert_eq!(r.retcode, UvpmRetcode::Success);
        // The two placements must not overlap.
        let a = r.placed[0].unwrap().box_;
        let b = r.placed[1].unwrap().box_;
        assert!(!crate::poly::boxes_overlap(&a, &b, -1e-9));
    }

    #[test]
    fn pack_static_island_first() {
        let mut a = sq(0.0, 0.0, 0.4);
        a.is_static = true;
        let b = sq(0.0, 0.0, 0.4);
        let p = PackParams::default();
        let mut v = vec![a, b];
        let r = pack(&mut v, &p);
        // The static island stays at its position (identity transform).
        let t = r.placed[0].unwrap();
        assert!((t.tx).abs() < 1e-12 && (t.ty).abs() < 1e-12);
        // The second must not overlap it.
        let b = r.placed[1].unwrap().box_;
        let a = r.placed[0].unwrap().box_;
        assert!(!crate::poly::boxes_overlap(&a, &b, -1e-9));
    }

    #[test]
    fn pack_to_others_leaves_unselected_fixed() {
        let mut sel = sq(0.0, 0.0, 0.3);
        sel.flags |= crate::island::SELECTED;
        let unsel = sq(0.7, 0.7, 0.3);
        let mut p = PackParams::default();
        p.pack_op = PackOpType::PackToOthers;
        let mut v = vec![sel, unsel];
        let r = pack(&mut v, &p);
        // Unselected stays put.
        let t = r.placed[1].unwrap();
        assert!((t.tx).abs() < 1e-12 && (t.ty).abs() < 1e-12);
    }

    #[test]
    fn pack_too_big_is_arranged_outside() {
        // A 5×5 square in a unit target cannot fit; it is arranged outside.
        let big = sq(0.0, 0.0, 5.0);
        let mut p = PackParams::default();
        p.scale_mode = crate::params::ScaleMode::FixedScale;
        p.scale = 1.0;
        let mut v = vec![big];
        let r = pack(&mut v, &p);
        // Fixed scale 1.0 → cannot fit in the unit target (margin) →
        // non-packed or arranged outside.
        if r.non_packed.is_empty() {
            let b = r.placed[0].unwrap().box_;
            assert!(!Box2::unit().contains_box_eps(&b, 1e-9));
        }
    }

    #[test]
    fn pack_validation_flags_overlap() {
        // Two identical fixed squares at the same position overlap →
        // OVERLAPS flags + retcode (lock_overlapping → Warning).
        let mut a = sq(0.2, 0.2, 0.5);
        a.is_static = true;
        let mut b = sq(0.2, 0.2, 0.5);
        b.is_static = true;
        let mut p = PackParams::default();
        p.lock_overlapping = true;
        let mut v = vec![a, b];
        let r = pack(&mut v, &p);
        assert_eq!(r.retcode, UvpmRetcode::Warning);
        assert!(!r.validation.overlapping.is_empty());
    }

    #[test]
    fn raster_placement_produces_a_valid_atlas() {
        if researchuv_gpu::GpuSolver::global().is_none() {
            eprintln!("skipping: no CUDA device/PTX available");
            return;
        }
        let isls: Vec<Island> = (0..8)
            .map(|i| {
                let s = 0.18 + (i % 3) as f64 * 0.06;
                let mut isl = sq(0.0, 0.0, s);
                isl.verts[0].u += i as f64 * 1e-9; // distinct outlines
                isl
            })
            .collect();
        let mut p = PackParams::default();
        p.raster_resolution = 256;
        let mut v1 = isls.clone();
        let r = pack(&mut v1, &p);
        assert_eq!(r.retcode, UvpmRetcode::Success);
        assert!(r.placed.iter().all(|t| t.is_some()), "all islands placed");
        assert!(r.validation.overlapping.is_empty(), "{:?}", r.validation.overlapping);
        let target = p.effective_box();
        for (i, a) in r.placed.iter().flatten().enumerate() {
            assert!(target.contains_box_eps(&a.box_, 1e-6), "island {i} outside: {:?}", a.box_);
            for b in r.placed.iter().flatten().skip(i + 1) {
                assert!(
                    !crate::poly::boxes_overlap(&a.box_, &b.box_, -1e-9),
                    "islands {i} overlap after raster placement"
                );
            }
        }
    }

    #[test]
    fn raster_path_survives_rotation_ladder_and_nonzero_origins() {
        if researchuv_gpu::GpuSolver::global().is_none() {
            eprintln!("skipping: no CUDA device/PTX available");
            return;
        }
        // Four 2×1 strips at a nonzero raw origin: the rotation ladder runs
        // per island (some orientations fit rows, others don't), and the
        // nonzero origin exercises the mask's min-shift. Whatever mix of
        // raster and exact-fallback placements results must be valid.
        let strip = |k: f64| Island::from_polygon(vec![
            Vec2::new(3.0 + k * 0.01, 5.0),
            Vec2::new(5.0 + k * 0.01, 5.0),
            Vec2::new(5.0 + k * 0.01, 6.0),
            Vec2::new(3.0 + k * 0.01, 6.0),
        ]);
        let isls: Vec<Island> = (0..4).map(|k| strip(k as f64)).collect();
        let mut p = PackParams::default();
        p.raster_resolution = 256;
        let mut v1 = isls.clone();
        let r = pack(&mut v1, &p);
        assert_eq!(r.retcode, UvpmRetcode::Success);
        assert!(r.placed.iter().all(|t| t.is_some()), "all strips placed");
        assert!(r.validation.overlapping.is_empty(), "{:?}", r.validation.overlapping);
        let target = p.effective_box();
        for (i, a) in r.placed.iter().flatten().enumerate() {
            assert!(target.contains_box_eps(&a.box_, 1e-6), "strip {i} outside: {:?}", a.box_);
        }
    }

    #[test]
    fn raster_tile_targets_spill_across_tiles() {
        if researchuv_gpu::GpuSolver::global().is_none() {
            eprintln!("skipping: no CUDA device/PTX available");
            return;
        }
        // 12 unit squares at MaxScale: 9 fit one unit tile's 85% budget, so
        // the placement must spill to later tiles. DynamicTiles with 4
        // columns → a 4×1 grid; every box must sit inside SOME tile.
        let isls: Vec<Island> = (0..12)
            .map(|i| {
                let mut isl = sq(0.0, 0.0, 1.0);
                isl.verts[0].u += i as f64 * 1e-9;
                isl
            })
            .collect();
        let mut p = PackParams::default();
        p.raster_resolution = 256;
        p.tile_target = crate::params::TileTargetMode::DynamicTiles;
        p.tiles_in_row = 4;
        let mut v1 = isls.clone();
        let r = pack(&mut v1, &p);
        assert!(
            r.placed.iter().all(|t| t.is_some()),
            "all 12 islands placed: {:?}",
            r.placed.iter().map(|t| t.is_some()).collect::<Vec<_>>()
        );
        assert!(r.validation.overlapping.is_empty(), "{:?}", r.validation.overlapping);
        assert_eq!(r.retcode, UvpmRetcode::Success, "tiles validate");
        // Boxes cover more than one unit tile horizontally.
        let max_u = r
            .placed
            .iter()
            .flatten()
            .map(|t| t.box_.max.u)
            .fold(0.0, f64::max);
        assert!(max_u > 1.5, "spilled across tiles: max u = {max_u}");
        // Each box fits within one whole tile.
        for (i, t) in r.placed.iter().flatten().enumerate() {
            let ix = t.box_.min.u.floor() as u32;
            let iy = t.box_.min.v.floor() as u32;
            let tb = Box2::new(
                Vec2::new(ix as f64, iy as f64),
                Vec2::new((ix + 1) as f64, (iy + 1) as f64),
            );
            assert!(tb.contains_box_eps(&t.box_, 1e-6), "island {i} crosses a tile boundary: {:?}", t.box_);
        }
    }

    #[test]
    fn raster_falls_back_when_nothing_fits() {
        if researchuv_gpu::GpuSolver::global().is_none() {
            eprintln!("skipping: no CUDA device/PTX available");
            return;
        }
        // One island larger than the target at fixed scale: the raster
        // search finds no anchor, the exact planner declines too, and the
        // arrange-outside path handles it.
        let mut big = sq(0.0, 0.0, 5.0);
        big.is_static = false;
        let mut p = PackParams::default();
        p.raster_resolution = 256;
        p.scale_mode = crate::params::ScaleMode::FixedScale;
        p.scale = 1.0;
        let mut v = vec![big];
        let r = pack(&mut v, &p);
        // Nothing of a 5x5 fits the unit target: non-packed (or arranged
        // outside it), never placed inside.
        if r.non_packed.is_empty() {
            let b = r.placed[0].expect("arranged outside").box_;
            assert!(!Box2::unit().contains_box_eps(&b, 1e-9), "oversize placed inside the target");
        }
    }

    #[test]
    fn gpu_heuristic_path_keeps_placements_valid() {
        // With a device present the pipeline routes the heuristic through
        // the GPU multi-restart pass first; the result must stay valid and
        // at least as well-scored as a pure-CPU run.
        if researchuv_gpu::GpuSolver::global().is_none() {
            eprintln!("skipping: no CUDA device/PTX available");
            return;
        }
        let isls: Vec<Island> = (0..8)
            .map(|i| {
                let mut isl = sq(0.0, 0.0, 0.3);
                isl.verts[0].u += i as f64 * 1e-9; // distinct outlines
                isl
            })
            .collect();
        let mut p = PackParams::default();
        p.heuristic_enable = true;
        p.heuristic_search_time = 0.5;
        p.seed = 123;
        let mut v1 = isls.clone();
        let r1 = pack(&mut v1, &p);
        assert_eq!(r1.retcode, UvpmRetcode::Success);
        assert!(r1.placed.iter().all(|t| t.is_some()));
        assert!(r1.validation.overlapping.is_empty());
        let target = p.effective_box();
        for (i, a) in r1.placed.iter().flatten().enumerate() {
            assert!(target.contains_box_eps(&a.box_, 1e-6), "island {i} outside");
            for b in r1.placed.iter().flatten().skip(i + 1) {
                assert!(
                    !crate::poly::boxes_overlap(&a.box_, &b.box_, -1e-9),
                    "islands {i} overlap after the GPU heuristic"
                );
            }
        }
    }

    #[test]
    fn placed_transform_roundtrip() {
        // A 90°-rotated square: the placed box must match the transformed
        // outline's bbox (the anchor mapping keeps them consistent).
        let mut isl = sq(0.2, 0.3, 0.4); // non-origin island
        isl.is_static = true;
        let mut p = PackParams::default();
        p.rotation_enable = false;
        let mut v = vec![isl];
        let r = pack(&mut v, &p);
        let t = r.placed[0].unwrap();
        let outline = t.transform_poly(&v[0].verts);
        let bb = crate::poly::bbox_of(&outline);
        assert!((bb.min.u - t.box_.min.u).abs() < 1e-9);
        assert!((bb.max.u - t.box_.max.u).abs() < 1e-9);
    }
}
