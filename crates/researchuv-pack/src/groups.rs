//! Island grouping — the engine's `UVPM4_GroupingScheme` / `groups_together` /
//! `grouping_compactness` / `find_trackers` / numbered-groups feature: assign
//! each island a group id by one of the documented `GroupingMethod`s, lay the
//! groups out in the target box per `GroupLayoutMode`, and expose the
//! numbered-group (lock/stack/track/norm) iparam channels.
//!
//! Evidence: engine strings `groups_together`, `grouping_compactness`,
//! `find_trackers`, `numbered_groups`; addon `spipeline/engine/props.py`
//! (`GROUPING_METHOD`, `GROUP_LAYOUT`, `GROUPS_TOGETHER` default False,
//! `GROUPING_COMPACTNESS` 0.0..1.0 default 0.0), `types.py`
//! (`UvpmGroupingMethod`, `UvpmGroupLayout`), `island_params.py`
//! (`NumberedGroupIParamInfo` lock/stack/track/norm channels,
//! `MIN_VALUE + 1` = "unset"), `pack_utils/__init__.py` (track-group
//! semantics: an island whose `track_group` iparam differs from the default
//! is a *tracker island*), `scenario/align_split_by_similarity.py`
//! (numbered group values written to islands).
//!
//! Grouping keys are read from the island's *faces*: material name, mesh-part
//! id, object id, tile, vertex color, collection name. An island's key is the
//! first face value it carries; islands without the value stay ungrouped.

use crate::box2::Box2;
use crate::island::Island;
use crate::params::{iparam, GroupLayoutMode, GroupingMethod, SimilarityParams};
use crate::similarity;
use researchuv_math::Vec2;

pub use crate::params::{GroupParams, NumberedGroups};

/// Per-island group key (what the islands are grouped *by*).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum GroupKey {
    /// `GroupingMethod::Material`.
    Material(String),
    /// `GroupingMethod::Mesh` (mesh part id).
    Mesh(u64),
    /// `GroupingMethod::Object`.
    Object(u64),
    /// `GroupingMethod::Tile` (grid column, row).
    Tile(u32, u32),
    /// `GroupingMethod::VertexColor` (quantized to 1/255 per channel).
    VertexColor(u8, u8, u8),
    /// `GroupingMethod::Collection` (Blender collection name).
    Collection(String),
    /// `GroupingMethod::Manual` (explicit `Island::group`).
    Manual(u32),
    /// `GroupingMethod::Similarity` (similarity cluster id).
    Similarity(u32),
}

/// The result of the grouping pass over `n` islands.
#[derive(Clone, Debug, Default)]
pub struct GroupResult {
    /// Per-island group id (`0` = no group / not grouped).
    pub group_ids: Vec<u32>,
    /// Per-island group key (`None` when `group_ids[i] == 0`).
    pub group_keys: Vec<Option<GroupKey>>,
    /// Per-group island indices (group `g` = `members[g - 1]`).
    pub members: Vec<Vec<u32>>,
    /// Target-space region per group (index `g - 1`). For
    /// `GroupLayoutMode::Automatic` / `Manual`, or when `groups_together` is
    /// off, each region is the whole target box.
    pub group_regions: Vec<Box2>,
    /// The layout that was applied.
    pub layout: GroupLayoutMode,
}

impl GroupResult {
    /// The group region for island `i` (the whole target when ungrouped).
    pub fn region_of(&self, island: u32, target: &Box2) -> Box2 {
        let g = self.group_ids.get(island as usize).copied().unwrap_or(0);
        if g == 0 || g as usize > self.group_regions.len() {
            return *target;
        }
        self.group_regions[(g - 1) as usize]
    }
}

/// Quantize a 3-D color to 8 bits per channel (the `VertexColor` grouping
/// key — the engine compares vertex colors with an 8-bit tolerance).
pub fn quantize_color(c: &researchuv_math::Vec3) -> (u8, u8, u8) {
    let q = |x: f64| (x.clamp(0.0, 1.0) * 255.0).round() as u8;
    (q(c.x), q(c.y), q(c.z))
}

/// The dominant key of an island's faces for the given grouping method
/// (`None` when no face carries the value).
fn island_key(island: &Island, method: GroupingMethod) -> Option<GroupKey> {
    match method {
        GroupingMethod::Material => island
            .faces
            .iter()
            .find_map(|f| f.material.clone())
            .map(GroupKey::Material),
        GroupingMethod::Mesh => island
            .faces
            .iter()
            .find_map(|f| f.mesh_part)
            .map(GroupKey::Mesh),
        GroupingMethod::Object => island
            .faces
            .iter()
            .find_map(|f| f.object)
            .map(GroupKey::Object),
        GroupingMethod::Tile => island
            .faces
            .iter()
            .find_map(|f| f.tile)
            .map(|(x, y)| GroupKey::Tile(x, y)),
        GroupingMethod::VertexColor => island
            .faces
            .iter()
            .find_map(|f| f.vertex_color.as_ref())
            .map(|c| {
                let (r, g, b) = quantize_color(c);
                GroupKey::VertexColor(r, g, b)
            }),
        GroupingMethod::Collection => island
            .faces
            .iter()
            .find_map(|f| f.collection.clone())
            .map(GroupKey::Collection),
        GroupingMethod::Manual => {
            if island.group != 0 {
                Some(GroupKey::Manual(island.group))
            } else {
                None
            }
        }
        GroupingMethod::Similarity => None, // handled separately
    }
}

/// Assign group ids to `islands` per `grouping` (the grouping pass).
///
/// `GroupingMethod::Similarity` clusters the islands with
/// [`similarity::split_by_similarity`] using `similarity`; single-element
/// clusters stay ungrouped.
///
/// `target` is the effective target box — group regions are computed in
/// target space when the groups are laid out (see [`layout_group_regions`]).
pub fn assign_groups(
    islands: &[Island],
    grouping: &GroupParams,
    similarity: &SimilarityParams,
    target: &Box2,
) -> GroupResult {
    let method = grouping.method;
    let n = islands.len();
    let mut keys: Vec<Option<GroupKey>> = vec![None; n];

    if method == GroupingMethod::Similarity {
        let clusters = similarity::split_by_similarity(islands, similarity);
        for (c, members) in clusters.iter().enumerate() {
            if members.len() < 2 {
                continue; // singletons stay ungrouped
            }
            let key = GroupKey::Similarity(c as u32);
            for &m in members.iter() {
                keys[m as usize] = Some(key.clone());
            }
        }
    } else {
        for i in 0..n {
            keys[i] = island_key(&islands[i], method);
        }
    }

    // Stable ids in first-seen order.
    let mut id_of: Vec<(GroupKey, u32)> = Vec::new();
    let mut ids = vec![0u32; n];
    let mut members: Vec<Vec<u32>> = Vec::new();
    for i in 0..n {
        if let Some(k) = keys[i].clone() {
            let id = match id_of.iter().position(|(kk, _)| *kk == k) {
                Some(pos) => id_of[pos].1,
                None => {
                    let id = id_of.len() as u32 + 1;
                    id_of.push((k, id));
                    members.push(Vec::new());
                    id
                }
            };
            ids[i] = id;
            members[(id - 1) as usize].push(i as u32);
        }
    }

    // Layout regions in target space.
    let regions = layout_group_regions(target, &ids, &members, islands, grouping);

    GroupResult {
        group_ids: ids,
        group_keys: keys,
        members,
        group_regions: regions,
        layout: grouping.layout,
    }
}

/// Compute the per-group target-space regions for a layout.
///
/// Region sizing is area-proportional: group `g` gets width
/// `target.width * sqrt(A_g) / Σ sqrt(A)` (Horizontal, and the Automatic
/// default), the mirror along v (Vertical), an equal grid cell
/// (TileGrid / TextureAtlas, with `cols = ceil(sqrt(groups))`), or the whole
/// target per group (Manual). `grouping_compactness` (0.0..1.0) shrinks the
/// allocated space: factor `1.0 − 0.5·compactness` (1.0 at the default 0.0,
/// 0.5 at 1.0), re-normalized so the regions tile the target.
///
/// Callers pass the *effective target box* to get regions in target space.
pub fn layout_group_regions(
    target: &Box2,
    _ids: &[u32],
    members: &[Vec<u32>],
    islands: &[Island],
    grouping: &GroupParams,
) -> Vec<Box2> {
    let n_groups = members.len();
    if n_groups == 0 {
        return Vec::new();
    }
    let layout = grouping.layout;
    let compact = grouping.grouping_compactness.clamp(0.0, 1.0);
    let slack = 1.0 - 0.5 * compact;

    // Group areas (outline areas; scale-1 footprints).
    let mut areas: Vec<f64> = Vec::with_capacity(n_groups);
    for m in members.iter() {
        let a: f64 = m.iter().map(|&i| islands[i as usize].area()).sum();
        areas.push(a.max(1e-12));
    }

    let regions = match layout {
        GroupLayoutMode::Horizontal | GroupLayoutMode::Automatic => {
            // Area-proportional columns.
            let sum_sqrt: f64 = areas.iter().map(|a| a.sqrt()).sum();
            let mut widths: Vec<f64> = areas
                .iter()
                .map(|&a| target.width() * (a.sqrt() / sum_sqrt.max(1e-12)) * slack)
                .collect();
            // Renormalize to tile the target exactly.
            let sum: f64 = widths.iter().sum();
            if sum > 0.0 {
                for w in widths.iter_mut() {
                    *w *= target.width() / sum;
                }
            }
            let mut x = target.min.u;
            let mut out = Vec::with_capacity(n_groups);
            for w in widths.iter() {
                let region = Box2::new(
                    Vec2::new(x, target.min.v),
                    Vec2::new(x + w, target.max.v),
                );
                x += w;
                out.push(region);
            }
            out
        }
        GroupLayoutMode::Vertical => {
            // Area-proportional rows.
            let sum_sqrt: f64 = areas.iter().map(|a| a.sqrt()).sum();
            let mut heights: Vec<f64> = areas
                .iter()
                .map(|&a| target.height() * (a.sqrt() / sum_sqrt.max(1e-12)) * slack)
                .collect();
            let sum: f64 = heights.iter().sum();
            if sum > 0.0 {
                for h in heights.iter_mut() {
                    *h *= target.height() / sum;
                }
            }
            let mut y = target.min.v;
            let mut out = Vec::with_capacity(n_groups);
            for h in heights.iter() {
                let region = Box2::new(
                    Vec2::new(target.min.u, y),
                    Vec2::new(target.max.u, y + h),
                );
                y += h;
                out.push(region);
            }
            out
        }
        GroupLayoutMode::TileGrid | GroupLayoutMode::TextureAtlas => {
            // Equal grid cells.
            let cols = ((n_groups as f64).sqrt().ceil() as u32).max(1);
            let rows = (n_groups as u32).div_ceil(cols);
            let cw = target.width() / cols as f64;
            let ch = target.height() / rows as f64;
            let mut out = Vec::with_capacity(n_groups);
            for g in 0..n_groups {
                let cx = (g as u32) % cols;
                let cy = (g as u32) / cols;
                out.push(Box2::new(
                    Vec2::new(target.min.u + cx as f64 * cw, target.min.v + cy as f64 * ch),
                    Vec2::new(target.min.u + (cx + 1) as f64 * cw, target.min.v + (cy + 1) as f64 * ch),
                ));
            }
            out
        }
        GroupLayoutMode::Manual => {
            // Manual: each group keeps the whole target (no subdivision).
            vec![*target; n_groups]
        }
    };

    regions
}

/// The numbered-group channels of an island (lock/stack/track/norm group ids,
/// `None` = channel unset). The channel default is `GROUP_UNSET` (−10000,
/// the addon's `MIN_VALUE + 1`).
pub fn numbered_group_ids(island: &Island) -> [Option<u32>; 4] {
    let get = |i: usize| {
        island
            .iparam_channel(i)
            .filter(|&v| v > iparam::GROUP_UNSET)
            .map(|v| v as u32)
    };
    [
        get(iparam::LOCK_GROUP),
        get(iparam::STACK_GROUP),
        get(iparam::TRACK_GROUP),
        get(iparam::NORM_GROUP),
    ]
}

/// Are two islands in the same numbered group on channel `i`
/// (`lock` = 0, `stack` = 1, `track` = 2, `norm` = 3)?
pub fn same_numbered_group(a: &Island, b: &Island, channel: usize) -> bool {
    let va = a.iparam_channel(channel).unwrap_or(iparam::GROUP_UNSET);
    let vb = b.iparam_channel(channel).unwrap_or(iparam::GROUP_UNSET);
    va != iparam::GROUP_UNSET && va == vb
}

/// Find the tracker islands (`find_trackers`): islands whose `track_group`
/// channel carries the active track-group id (the addon's `pack_utils`
/// treats an island as a *tracker island* when its `track_group` iparam
/// differs from the default).
pub fn find_trackers(islands: &[Island], ng: &NumberedGroups) -> Vec<u32> {
    if !ng.track_enable {
        return Vec::new();
    }
    let target = ng.track_group as f64;
    islands
        .iter()
        .enumerate()
        .filter(|(_, i)| i.iparam_channel(iparam::TRACK_GROUP) == Some(target))
        .map(|(i, _)| i as u32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use researchuv_math::Vec2;

    fn sq(x: f64, s: f64) -> Island {
        Island::from_polygon(vec![
            Vec2::new(x, 0.0),
            Vec2::new(x + s, 0.0),
            Vec2::new(x + s, s),
            Vec2::new(x, s),
        ])
    }

    fn face_with_material(name: &str) -> crate::island::Face {
        let mut f = crate::island::Face::default();
        f.uv = [
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 1.0),
        ];
        f.material = Some(name.to_string());
        f
    }

    #[test]
    fn material_grouping() {
        let mut a = sq(0.0, 0.5);
        a.faces = vec![face_with_material("M1")];
        let mut b = sq(0.6, 0.5);
        b.faces = vec![face_with_material("M2")];
        let mut c = sq(0.0, 0.5);
        c.verts[0] = Vec2::new(0.0, 0.6);
        c.faces = vec![face_with_material("M1")];
        c.rebuild_bbox();
        let p = GroupParams::default(); // method = Material
        let r = assign_groups(&[a, b, c], &p, &SimilarityParams::default(), &Box2::unit());
        assert_eq!(r.group_ids, vec![1, 2, 1]);
        assert_eq!(r.members.len(), 2);
        assert_eq!(r.members[0], vec![0u32, 2]);
        assert_eq!(r.members[1], vec![1u32]);
    }

    #[test]
    fn manual_grouping() {
        let mut a = sq(0.0, 0.4);
        a.group = 5;
        let mut b = sq(0.5, 0.4);
        b.group = 5;
        let c = sq(0.0, 0.4);
        let mut p = GroupParams::default();
        p.method = GroupingMethod::Manual;
        let r = assign_groups(&[a, b, c], &p, &SimilarityParams::default(), &Box2::unit());
        assert_eq!(r.group_ids, vec![1, 1, 0]);
    }

    #[test]
    fn similarity_grouping() {
        // Two identical squares + one wide rect.
        let a = sq(0.0, 1.0);
        let b = sq(2.0, 1.0);
        let mut c = sq(0.0, 1.0);
        c.verts[1] = Vec2::new(4.0, 0.0);
        c.verts[2] = Vec2::new(4.0, 0.4);
        c.verts[3] = Vec2::new(0.0, 0.4);
        c.rebuild_bbox();
        let mut p = GroupParams::default();
        p.method = GroupingMethod::Similarity;
        let r = assign_groups(&[a, b, c], &p, &SimilarityParams::default(), &Box2::unit());
        assert_eq!(r.group_ids, vec![1, 1, 0]);
    }

    #[test]
    fn horizontal_regions_tile_target() {
        let a = sq(0.0, 0.5);
        let b = sq(0.0, 1.0); // 4× the area
        let mut p = GroupParams::default();
        p.method = GroupingMethod::Manual;
        p.groups_together = true;
        let mut a2 = a.clone();
        a2.group = 1;
        let mut b2 = b.clone();
        b2.group = 2;
        let r = assign_groups(&[a2, b2], &p, &SimilarityParams::default(), &Box2::unit());
        // Regions are computed for the unit box by assign_groups.
        let t = Box2::unit();
        let r0 = r.region_of(0, &t);
        let r1 = r.region_of(1, &t);
        // sqrt(0.25) : sqrt(1.0) = 0.5 : 1 → widths 1/3 and 2/3.
        assert!((r0.width() - 1.0 / 3.0).abs() < 1e-9);
        assert!((r1.width() - 2.0 / 3.0).abs() < 1e-9);
        assert!((r0.min.u + r0.width() - r1.min.u).abs() < 1e-9);
    }

    #[test]
    fn find_trackers_finds_channel_match() {
        let mut a = sq(0.0, 0.5);
        a.faces = vec![crate::island::Face::default()];
        a.faces[0].iparams[iparam::TRACK_GROUP] = 3.0;
        let b = sq(0.5, 0.5);
        let mut ng = NumberedGroups::default();
        ng.track_enable = true;
        ng.track_group = 3;
        let t = find_trackers(&[a, b], &ng);
        assert_eq!(t, vec![0u32]);
    }

    #[test]
    fn numbered_group_ids_reads_channels() {
        let mut a = sq(0.0, 0.5);
        a.faces = vec![crate::island::Face::default()];
        a.faces[0].iparams[iparam::LOCK_GROUP] = 7.0;
        let ids = numbered_group_ids(&a);
        assert_eq!(ids, [Some(7), None, None, None]);
    }
}
