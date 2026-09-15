//! # researchuv-pack — UVPackmaster-grade island packing engine
//!
//! A Rust clone of **UVPackmaster 4's packing engine** (`uvpm.exe`): the island
//! data model, the full documented option surface (from the GPL addon's
//! `UVPM4_MainProps` / `Labels` / `types` / `island_params` specs), and the
//! packing algorithms (placement search, scale modes, margins, rotation/flip,
//! overlap detection, similarity, texel density, pixel-perfect alignment,
//! grouping, numbered groups, split-overlap, heuristic search).
//!
//! Evidence base (all read from the installed `uvpm.exe` + addon on this machine):
//!
//! - **Engine string cluster** @ `~2.89 MB` of `uvpm.exe`: `PackStrategy`,
//!   `SimilarityParams`, `PackStrategyParams`, `PackParams`, `PackResult`,
//!   `PixelPerfectAlignTarget`, `pack_to_single_box`, `unit_box`, `target_box`,
//!   `flipped_box`, `CSMainVulkanFindBestRow` / `CSVulkanFindLocationGroups`
//!   (Vulkan GPU free-space search), `walkingSearchTrianglesAt` +
//!   `src\uvpm_core\algorithms\CDT\include\Triangulation.hpp` (constrained
//!   Delaunay free-space triangulation), `ProcessIntersections` /
//!   `has_self_intersections` / `check_holes` (Boost.Geometry validation),
//!   `find_similar` / `align_similar` / `is_similar` / `split_by_similarity`,
//!   `texel_density`, `find_trackers`, `groups_together`,
//!   `grouping_compactness`, `heuristic_allow_mixed_scales`,
//!   `overlapping_islands` / `invalid_islands` / `non_packed_islands` /
//!   `static_islands`, `send_iparams` / `IntIParamsManager` /
//!   `StrIParamsManager`, `pixel_perfect_align_target`,
//!   `pixel_perfect_vert_align`, `DoMaxima error`, `match_3d_axis`.
//! - **Addon spec** (`uvpackmaster4` addon, GPL): `spipeline/engine/props.py`
//!   (option defaults + bounds), `spipeline/engine/labels.py` (authoritative
//!   per-option semantics), `spipeline/engine/types.py` (enums + retcodes +
//!   island flags + serialization flags + feature codes),
//!   `spipeline/engine/island_params.py` (per-face island parameter channels,
//!   `__uvpm4_v1_` prefix, MAX_COUNT = 16), `spipeline/engine/props_base.py`
//!   (tile/group/orient constants).
//!
//! The original engine is GPU-accelerated (Vulkan) with a CDT free-space
//! representation; this crate is the deterministic CPU analog: the same option
//! semantics, the same result contract (per-island transform + flags +
//! retcode), with the "find best row" GPU kernel replaced by an exact
//! wall/edge candidate search (equivalent placement results, no GPU required).
//!
//! | Module            | UVPM counterpart                                                        |
//! |-------------------|-------------------------------------------------------------------------|
//! | [`island`]        | `Island` / `send_vertices` / island flags (`UvpmIslandFlags`)           |
//! | [`box2`]          | `unit_box` / `target_box` / `flipped_box` / `BoxCorner` / tiles         |
//! | [`params`]        | `UVPM4_MainProps` + `PackParams` / `SimilarityParams` / `PackStrategyParams` |
//! | [`poly`]          | Boost.Geometry overlay tests (`has_self_intersections`, `check_holes`)  |
//! | [`validate`]      | `ProcessIntersections`, retcodes (`UvpmRetcode`), island flags          |
//! | [`place`]         | `CSMainVulkanFindBestRow` / `CSVulkanFindLocation` analog (CPU)         |
//! | [`tdensity`]      | `texel_density` + `UVPM4_TDensity*` (units, tiers, set-before-pack)     |
//! | [`align`]         | `pixel_perfect_align_target`, `orient_to_3d` (`match_3d_axis`)          |
//! | [`similarity`]    | `find_similar` / `align_similar` / `split_by_similarity`                |
//! | [`groups`]        | `UVPM4_GroupingScheme`, `groups_together`, `find_trackers`, numbered groups |
//! | [`split`]         | `split_overlap` (detection mode, `max_tile_x`, `dont_split_priorities`) |
//! | [`heur`]          | `heuristic_enable` / `heuristic_search_time`, `DoMaxima`                |
//! | [`pipeline`]      | `EXECUTE_SCENARIO` pack entry point                                     |

#![forbid(unsafe_code)]

pub mod align;
pub mod box2;
pub mod groups;
pub mod heur;
pub mod island;
pub mod params;
pub mod pipeline;
pub mod place;
pub mod poly;
pub mod rng;
pub mod similarity;
pub mod split;
pub mod tdensity;
pub mod validate;

pub use box2::{Box2, BoxCorner, TileTarget, TargetBox};
pub use groups::{GroupParams, GroupResult, NumberedGroups};
pub use island::{Island, IslandFlags, PlacedTransform};
pub use params::{
    AdvancedHeuristicMode, AlignPriority, CoordSpace, GroupingMethod, GroupLayoutMode,
    OrientTo3dParams, PackOpType, PackParams, PackStrategy, PixelPerfectAlignTarget,
    PixelPerfectVertAlignMode, ScaleMode, SimilarityMode, SimilarityParams, SplitOverlapParams,
    TexelDensityUnit, TileFillingMethod, TileTargetMode, UvpmAxis, UvpmFeatureCode, UvpmRetcode,
};
pub use pipeline::{pack, PackResult};
pub use rng::SplitMix64;
pub use similarity::{align_similar, find_similar, is_similar, split_by_similarity};
pub use tdensity::{set_tdensity, TexelDensityParams, TexelDensityPolicy};
pub use validate::{validate_islands, ValidationReport};
