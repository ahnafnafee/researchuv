//! Pack parameters — the full `UVPM4_MainProps` option surface, with the
//! defaults and bounds documented in the addon's `spipeline/engine/props.py`
//! and `props_base.py`, and the enum values from `spipeline/engine/types.py`.
//!
//! Every field carries its addon evidence (property id + default) in the doc
//! comment. `PackParams::default()` reproduces the addon's factory state.

use crate::box2::{Box2, BoxCorner, TargetBox, TileTarget};
use researchuv_math::Vec2;

/// Canonical per-face island-parameter channel indices (`IParamInfo`,
/// `MAX_COUNT = 16`). The engine registers channels in a fixed order; the
/// addon's `island_params.py` names them (script name = channel name under
/// the `__uvpm4_v1_` prefix):
///
/// | Index | Channel                  | Range        | Default      |
/// |-------|--------------------------|--------------|--------------|
/// | 0     | `align_priority`         | 0..100       | 0            |
/// | 1     | `normalize_multiplier`   | 0.0001..1000 | 1.0          |
/// | 2     | `rotation_step`          | 0..180       | 0            |
/// | 3     | `island_rot_step`        | 0..180       | 0            |
/// | 4     | `split_offset_x`         | −10000..10000| −10000       |
/// | 5     | `split_offset_y`         | −10000..10000| −10000       |
/// | 6     | `lock_group` (numbered)  | —            | MIN_VALUE+1  |
/// | 7     | `stack_group` (numbered) | —            | MIN_VALUE+1  |
/// | 8     | `track_group` (numbered) | —            | MIN_VALUE+1  |
/// | 9     | `norm_group` (numbered)  | —            | MIN_VALUE+1  |
pub mod iparam {
    /// `align_priority` channel index.
    pub const ALIGN_PRIORITY: usize = 0;
    /// `normalize_multiplier` channel index.
    pub const NORMALIZE_MULTIPLIER: usize = 1;
    /// `rotation_step` channel index.
    pub const ROTATION_STEP: usize = 2;
    /// `island_rot_step` channel index.
    pub const ISLAND_ROT_STEP: usize = 3;
    /// `split_offset_x` channel index.
    pub const SPLIT_OFFSET_X: usize = 4;
    /// `split_offset_y` channel index.
    pub const SPLIT_OFFSET_Y: usize = 5;
    /// `lock_group` (numbered groups) channel index.
    pub const LOCK_GROUP: usize = 6;
    /// `stack_group` (numbered groups) channel index.
    pub const STACK_GROUP: usize = 7;
    /// `track_group` (numbered groups) channel index.
    pub const TRACK_GROUP: usize = 8;
    /// `norm_group` (numbered groups) channel index.
    pub const NORM_GROUP: usize = 9;

    /// "Not set" sentinel for the split offsets (default −10000).
    pub const SPLIT_OFFSET_UNSET: f64 = -10_000.0;
    /// "Not set" sentinel for numbered-group channels (`MIN_VALUE + 1`).
    pub const GROUP_UNSET: f64 = -10_000.0;
}

/// Align priority — the `align_priority` iparam channel (0..100, default 0).
///
/// Higher-priority islands are placed first and, in the split-overlap pass,
/// keep their position while lower-priority islands are moved.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AlignPriority(pub u32);

impl AlignPriority {
    /// Max representable priority.
    pub const MAX: u32 = 100;

    /// Clamp `v` to the valid range.
    pub const fn new(v: u32) -> Self {
        if v > Self::MAX {
            Self(Self::MAX)
        } else {
            Self(v)
        }
    }

    /// The raw value.
    pub const fn value(&self) -> u32 {
        self.0
    }

    /// Read an island's align priority from iparam channel 0 (default 0).
    pub fn of(island: &crate::island::Island) -> u32 {
        island
            .iparam_channel(iparam::ALIGN_PRIORITY)
            .unwrap_or(0.0)
            .clamp(0.0, Self::MAX as f64) as u32
    }
}

/// Engine return codes — `UvpmRetcode` (`types.py`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum UvpmRetcode {
    /// No problem.
    #[default]
    Success = 0,
    FatalError = 1,
    NoSpace = 2,
    Cancelled = 3,
    InvalidIslands = 4,
    NoUvs = 5,
    InvalidInput = 6,
    Warning = 7,
}

impl UvpmRetcode {
    pub const ALL: [UvpmRetcode; 8] = [
        UvpmRetcode::Success,
        UvpmRetcode::FatalError,
        UvpmRetcode::NoSpace,
        UvpmRetcode::Cancelled,
        UvpmRetcode::InvalidIslands,
        UvpmRetcode::NoUvs,
        UvpmRetcode::InvalidInput,
        UvpmRetcode::Warning,
    ];
}

/// Island rotation step (degrees) — `rotation_step` (1..180, def 90);
/// `island_rot_step` (0..180, def 0) is the per-island override.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RotationStep {
    pub degrees: u32,
}

impl RotationStep {
    pub const GLOBAL: RotationStep = RotationStep { degrees: 90 };
    pub const ISLAND: RotationStep = RotationStep { degrees: 0 };

    /// The rotation candidate angles (radians, CCW) for this step.
    /// A step of `s` yields `{0, s, 2s, …}` up to 360° (deduped at 360).
    pub fn angles(&self) -> Vec<f64> {
        if self.degrees == 0 {
            return vec![0.0];
        }
        let s = self.degrees as f64 * std::f64::consts::PI / 180.0;
        let mut out = vec![0.0];
        let mut k = 1.0;
        while k * s < 360.0 * (std::f64::consts::PI / 180.0) {
            out.push(k * s);
            k += 1.0;
        }
        out
    }
}

/// Scale mode — `DEF__SCALE_MODE` (`UvpmScaleMode`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum ScaleMode {
    /// Pack as large as possible (default).
    #[default]
    MaxScale,
    /// Use the fixed scale factor (`params.scale`).
    FixedScale,
    /// Fixed scale factor; margins maximized (islands placed at the fixed
    /// scale and pushed to the least-wasteful positions).
    FixedScaleMaxMargin,
}

/// Pack strategy — `UvpmPackStrategy` (start corner defaults to bottom-left).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum PackStrategy {
    /// Auto (default).
    #[default]
    Automatic,
    /// Side-to-side, vertical (fill left→right, stack bottom→top).
    SideToSideVert,
    /// Side-to-side, horizontal (fill bottom→top, advance left→right).
    SideToSideHori,
    /// Square (balanced square-ish packing).
    Square,
}

/// Similarity mode — `UvpmSimilarityMode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum SimilarityMode {
    /// Compare outline shape (default).
    #[default]
    BorderShape,
    /// Compare vertex positions.
    VertexPosition,
    /// Compare face topology.
    Topology,
}

/// Overlap detection mode — `UvpmOverlapDetectionMode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum OverlapDetectionMode {
    /// Any part: polygon-based overlap of the islands' areas (default).
    #[default]
    AnyPart,
    /// Exact: bounding-box overlap test.
    Exact,
}

/// Pixel-perfect alignment target — `UvpmPixelPerfectAlignTarget`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum PixelPerfectAlignTarget {
    /// Align each island's min corner to the pixel grid (default).
    #[default]
    Corner,
    /// Align each island's center to the pixel grid.
    Center,
}

/// Vertex alignment mode — `UvpmPixelPerfectVertAlignMode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum PixelPerfectVertAlignMode {
    /// No vertex alignment (default).
    #[default]
    None,
    /// Snap the bounding-box corner vertices to the grid.
    BboxCorners,
    /// Snap all vertices on the bounding-box border to the grid.
    Bbox,
    /// Snap the outline (border) vertices to the grid.
    BorderEdges,
    /// Snap every vertex to the grid.
    All,
}

/// Coordinate space — `UvpmCoordSpace`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum CoordSpace {
    /// Local (per-island) space (default).
    #[default]
    Local,
    /// Global (shared) space.
    Global,
}

/// 3-D axis — `UvpmAxis`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum UvpmAxis {
    /// X axis (default).
    #[default]
    X,
    Y,
    Z,
}

impl UvpmAxis {
    /// Unit vector of this axis.
    pub fn vector(self) -> (f64, f64, f64) {
        match self {
            UvpmAxis::X => (1.0, 0.0, 0.0),
            UvpmAxis::Y => (0.0, 1.0, 0.0),
            UvpmAxis::Z => (0.0, 0.0, 1.0),
        }
    }
    pub fn from_index(i: usize) -> UvpmAxis {
        match i {
            0 => UvpmAxis::X,
            1 => UvpmAxis::Y,
            _ => UvpmAxis::Z,
        }
    }
}

/// Tile filling method — `UvpmTileFillingMethod`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum TileFillingMethod {
    /// Fill all tiles at once (default).
    #[default]
    Simultaneously,
    /// Fill tiles one by one.
    OneByOne,
}

/// Tile target mode — `TileTargetMode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum TileTargetMode {
    /// Dynamic tiles (default): grid derived from `tiles_in_row`.
    #[default]
    DynamicTiles,
    /// Explicit tile grid (`count_x × count_y`).
    TileGrid,
    /// Tile range: pack into a fixed tile rectangle.
    TileRange,
}

/// Grouping method — `GroupingMethod`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum GroupingMethod {
    /// By material name (default).
    #[default]
    Material,
    /// By mesh part (connected components).
    Mesh,
    /// By object.
    Object,
    /// By tile.
    Tile,
    /// By vertex color.
    VertexColor,
    /// By collection.
    Collection,
    /// Manual (explicit `island.group`).
    Manual,
    /// By similarity (see `SimilarityMode`).
    Similarity,
}

/// Group layout mode — `GroupLayoutMode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum GroupLayoutMode {
    /// Automatic (default).
    #[default]
    Automatic,
    /// Groups laid out side-by-side horizontally.
    Horizontal,
    /// Groups stacked vertically.
    Vertical,
    /// Groups arranged in a tile grid.
    TileGrid,
    /// Groups arranged as a texture atlas.
    TextureAtlas,
    /// Manual placement (group order only).
    Manual,
}

/// Pack operation type — `PackOpType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum PackOpType {
    /// Plain pack (default).
    #[default]
    Pack,
    /// Pack selected islands into the area without touching the unselected
    /// (static) islands.
    PackToOthers,
    /// Repack selected islands together with the unselected islands that
    /// overlap the target area (all movable).
    RepackWithOthers,
}

/// Advanced heuristic mode — `UvpmAdvancedHeuristicMode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum AdvancedHeuristicMode {
    /// Automatic (default).
    #[default]
    Auto,
    Disable,
    Enable,
}

/// Texel density unit — `TexelDensityUnit` (px per unit length).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum TexelDensityUnit {
    /// px per meter (default; multiplier 1,000,000).
    #[default]
    PxPerMeter,
    /// px per centimeter (multiplier 10,000).
    PxPerCentimeter,
    /// px per inch (multiplier 2,540).
    PxPerInch,
    /// px per foot (multiplier 304.8).
    PxPerFoot,
}

impl TexelDensityUnit {
    /// px-per-unit multiplier (UV units assumed to be meters for PxPerMeter).
    pub fn multiplier(self) -> f64 {
        match self {
            TexelDensityUnit::PxPerMeter => 1_000_000.0,
            TexelDensityUnit::PxPerCentimeter => 10_000.0,
            TexelDensityUnit::PxPerInch => 2_540.0,
            TexelDensityUnit::PxPerFoot => 304.8,
        }
    }
}

/// Feature codes — `UvpmFeatureCode` (edition gating).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum UvpmFeatureCode {
    Demo = 0,
    IslandRotation = 1,
    OverlapCheck = 2,
    PackingDepth = 3,
    HeuristicSearch = 4,
    PackRatio = 5,
    PackToOthers = 6,
    Grouping = 7,
    LockOverlapping = 8,
    AdvancedHeuristic = 9,
    SelfIntersectProcessing = 10,
    Validation = 11,
    MultiDevicePack = 12,
    TargetBox = 13,
    IslandRotationStep = 14,
    PackToTiles = 15,
}

/// Similarity parameters — `SimilarityParams` (the engine's C++ struct,
/// populated by `UVPM4_MainProps.get_similar_params`).
#[derive(Clone, Debug)]
pub struct SimilarityParams {
    pub mode: SimilarityMode,
    /// Comparison precision (default `params.precision` = 500).
    pub precision: u32,
    /// Similarity threshold, 0.0..0.75 (default 0.1).
    pub threshold: f64,
    /// Treat holes as dissimilar (default false).
    pub check_holes: bool,
    /// Rescale before comparing (default false).
    pub adjust_scale: bool,
    /// Tolerance for non-uniform scaling, 0.0..1.0 (default 1.0).
    pub non_uniform_scaling_tolerance: f64,
    /// 3-D axis to match (default X).
    pub match_3d_axis: UvpmAxis,
    /// Space for the 3-D axis match (default Local).
    pub match_3d_axis_space: CoordSpace,
    /// Snap vertices before comparing (default false).
    pub correct_vertices: bool,
    /// Vertex correction threshold, 0.001..0.1 (default 0.01).
    pub vertex_threshold: f64,
}

impl Default for SimilarityParams {
    fn default() -> Self {
        Self {
            mode: SimilarityMode::default(),
            precision: 500,
            threshold: 0.1,
            check_holes: false,
            adjust_scale: false,
            non_uniform_scaling_tolerance: 1.0,
            match_3d_axis: UvpmAxis::X,
            match_3d_axis_space: CoordSpace::Local,
            correct_vertices: false,
            vertex_threshold: 0.01,
        }
    }
}

/// Numbered groups — lock / stack / track / norm (per-group channel values).
#[derive(Clone, Debug, Default)]
pub struct NumberedGroups {
    /// Lock group enable + id + precision.
    pub lock_enable: bool,
    pub lock_group: u32,
    pub lock_precision: u32,
    /// Stack group enable + id + max tiles + precision.
    pub stack_enable: bool,
    pub stack_group: u32,
    pub stack_max_tile_x: u32,
    pub stack_precision: u32,
    /// Track group enable + id + precision.
    pub track_enable: bool,
    pub track_group: u32,
    pub track_precision: u32,
    /// Norm group enable + id + max tiles.
    pub norm_enable: bool,
    pub norm_group: u32,
    pub norm_max_tile_x: u32,
}

/// Split-overlap parameters — `UVPM4_SplitOverlapProps`.
#[derive(Clone, Debug, Default)]
pub struct SplitOverlapParams {
    /// Enable split-overlap.
    pub enable: bool,
    /// Detection mode (default AnyPart).
    pub detection_mode: OverlapDetectionMode,
    /// Max tiles along x (0 = unlimited).
    pub max_tile_x: u32,
    /// Don't split islands with `align_priority` >= this value.
    pub dont_split_priorities: u32,
}

/// Orient-to-3-D parameters — `UVPM4_OrientTo3dProps`.
#[derive(Clone, Debug)]
pub struct OrientTo3dParams {
    pub enable: bool,
    /// Primary 3-D axis to match (default Z).
    pub prim_3d_axis: UvpmAxis,
    /// Primary UV axis it maps to (default Y).
    pub prim_uv_axis: UvpmAxis,
    /// Secondary 3-D axis (default X).
    pub sec_3d_axis: UvpmAxis,
    /// Secondary UV axis (default X).
    pub sec_uv_axis: UvpmAxis,
    /// Space (default Local).
    pub axes_space: CoordSpace,
    /// Primary/secondary bias, 0..100 (default 80).
    pub prim_sec_bias: f64,
}

impl Default for OrientTo3dParams {
    fn default() -> Self {
        Self {
            enable: false,
            prim_3d_axis: UvpmAxis::Z,
            prim_uv_axis: UvpmAxis::Y,
            sec_3d_axis: UvpmAxis::X,
            sec_uv_axis: UvpmAxis::X,
            axes_space: CoordSpace::Local,
            prim_sec_bias: 80.0,
        }
    }
}

/// Group parameters — `UVPM4_GroupingScheme` / `UVPM4_GroupParams`.
#[derive(Clone, Debug)]
pub struct GroupParams {
    pub method: GroupingMethod,
    pub layout: GroupLayoutMode,
    /// Keep each group's islands adjacent (default false).
    pub groups_together: bool,
    /// Group compactness, 0.0..1.0 (default 0.0).
    pub grouping_compactness: f64,
}

impl Default for GroupParams {
    fn default() -> Self {
        Self {
            method: GroupingMethod::default(),
            layout: GroupLayoutMode::default(),
            groups_together: false,
            grouping_compactness: 0.0,
        }
    }
}

/// The full main option set — `UVPM4_MainProps` (defaults from `props.py`).
#[derive(Clone, Debug)]
pub struct PackParams {
    // --- determinism / resources ---
    /// Stochastic-search seed, 0..10000 (default 0).
    pub seed: u64,
    /// Max CPU threads (default: host core count; 0 = auto).
    pub thread_count: u32,
    /// Topology parse timeout, seconds; 0 disables (default 30.0).
    pub topology_parse_timeout: f64,

    // --- placement precision ---
    /// Placement search precision, 10..10000 (default 500).
    pub precision: u32,
    /// Relative margin (UV fraction), 0.0..0.2 (default 0.003).
    pub margin: f64,

    // --- pixel margins ---
    /// Enable pixel margin (overrides `margin`).
    pub pixel_margin_enable: bool,
    /// Margin between islands, px, 1..256 (default 5).
    pub pixel_margin: u32,
    /// Enable border margin.
    pub pixel_border_margin_enable: bool,
    /// Island-to-border margin, px, 0..256 (default 1).
    pub pixel_border_margin: u32,
    /// Extra margin to "others" (unselected/static), px, 0..128 (default 0).
    pub extra_pixel_margin_to_others: u32,
    /// Texture size the pixel margins refer to, 16..32768 (default 1024).
    pub pixel_margin_tex_size: u32,

    // --- alignment ---
    /// Pixel-perfect alignment enable.
    pub pixel_perfect_align: bool,
    /// Alignment target (Corner/Center).
    pub pixel_perfect_align_target: PixelPerfectAlignTarget,
    /// Vertex alignment mode.
    pub pixel_perfect_vert_align: PixelPerfectVertAlignMode,
    /// Per-island align-priority channel enable (iparam 0..100, def 0).
    pub align_priority_enable: bool,

    // --- rotation / flipping ---
    /// Allow island rotation (default true).
    pub rotation_enable: bool,
    /// Pre-rotation disable (packer does not pre-rotate islands).
    pub pre_rotation_disable: bool,
    /// Allow island flipping (default true).
    pub flipping_enable: bool,
    /// Rotation step, degrees, 1..180 (default 90).
    pub rotation_step: u32,
    /// Per-island rotation step override (0 = use global).
    pub island_rot_step: u32,

    // --- scale ---
    /// Scale mode (default MaxScale).
    pub scale_mode: ScaleMode,
    /// Fixed scale factor (for FixedScale modes; default 1.0).
    pub scale: f64,
    /// Normalize island scale before packing (default false).
    pub normalize_scale: bool,
    /// Normalization space (Local/Global).
    pub normalize_space: CoordSpace,
    /// Per-island normalize multiplier (iparam; default 1.0).
    pub island_normalize_multiplier: f64,

    // --- containment ---
    /// Islands must be fully inside the target box (default true).
    pub fully_inside: bool,
    /// Target box (default unit box).
    pub target_box: TargetBox,
    /// Non-square packing ratio (w/h of the effective box; 1.0 = square).
    pub non_square_packing: f64,

    // --- strategy / tiling ---
    /// Pack strategy (default Automatic).
    pub pack_strategy: PackStrategy,
    /// Strategy start corner (default BL).
    pub start_corner: BoxCorner,
    /// Pack everything into a single box (default false).
    pub pack_to_single_box: bool,
    /// Tile target mode (default DynamicTiles).
    pub tile_target: TileTargetMode,
    /// Tiles per row (dynamic tiles), 1..100 (default 10).
    pub tiles_in_row: u32,
    /// Explicit tile count x/y, 1..100 (default 1).
    pub tile_count_x: u32,
    pub tile_count_y: u32,
    /// Grid origin tile (col, row) (default (0,0)).
    pub start_tile: (u32, u32),
    /// Tile filling method (default Simultaneously).
    pub tile_filling_method: TileFillingMethod,

    // --- overlaps / lock ---
    /// Lock overlapping islands (default false).
    pub lock_overlapping: bool,
    /// Overlap detection mode (default AnyPart).
    pub overlap_detection_mode: OverlapDetectionMode,

    // --- heuristic ---
    /// Enable heuristic (time-limited) search (default false).
    pub heuristic_enable: bool,
    /// Heuristic search time budget, seconds, 0..3600 (default 30.0).
    pub heuristic_search_time: f64,
    /// Advanced heuristic mode (default Auto).
    pub advanced_heuristic: AdvancedHeuristicMode,
    /// Allow mixed scales in heuristic search.
    pub heuristic_allow_mixed_scales: bool,

    // --- misc ---
    /// Arrange non-packed islands outside the target (default true).
    pub arrange_non_packed: bool,
    /// High-precision topology analysis.
    pub high_precision_topology_analysis: bool,
    /// Suppress immediate UV updates.
    pub disabled_immediate_uv_updates: bool,

    // --- sub-params ---
    /// Similarity settings (default: border shape, threshold 0.1).
    pub similarity: SimilarityParams,
    /// Orient-to-3-D settings.
    pub orient_to_3d: OrientTo3dParams,
    /// Texel density settings.
    pub tdensity: crate::tdensity::TexelDensityParams,
    /// Grouping settings.
    pub grouping: GroupParams,
    /// Numbered groups.
    pub numbered_groups: NumberedGroups,
    /// Split-overlap settings.
    pub split_overlap: SplitOverlapParams,
    /// Pack operation type (default Pack).
    pub pack_op: PackOpType,
}

impl Default for PackParams {
    fn default() -> Self {
        Self {
            seed: 0,
            thread_count: 0,
            topology_parse_timeout: 30.0,
            precision: 500,
            margin: 0.003,
            pixel_margin_enable: false,
            pixel_margin: 5,
            pixel_border_margin_enable: false,
            pixel_border_margin: 1,
            extra_pixel_margin_to_others: 0,
            pixel_margin_tex_size: 1024,
            pixel_perfect_align: false,
            pixel_perfect_align_target: PixelPerfectAlignTarget::default(),
            pixel_perfect_vert_align: PixelPerfectVertAlignMode::default(),
            align_priority_enable: false,
            rotation_enable: true,
            pre_rotation_disable: false,
            flipping_enable: true,
            rotation_step: 90,
            island_rot_step: 0,
            scale_mode: ScaleMode::default(),
            scale: 1.0,
            normalize_scale: false,
            normalize_space: CoordSpace::Local,
            island_normalize_multiplier: 1.0,
            fully_inside: true,
            target_box: TargetBox::Unit,
            non_square_packing: 1.0,
            pack_strategy: PackStrategy::default(),
            start_corner: BoxCorner::Bl,
            pack_to_single_box: false,
            tile_target: TileTargetMode::default(),
            tiles_in_row: 10,
            tile_count_x: 1,
            tile_count_y: 1,
            start_tile: (0, 0),
            tile_filling_method: TileFillingMethod::default(),
            lock_overlapping: false,
            overlap_detection_mode: OverlapDetectionMode::default(),
            heuristic_enable: false,
            heuristic_search_time: 30.0,
            advanced_heuristic: AdvancedHeuristicMode::default(),
            heuristic_allow_mixed_scales: false,
            arrange_non_packed: true,
            high_precision_topology_analysis: false,
            disabled_immediate_uv_updates: false,
            similarity: SimilarityParams::default(),
            orient_to_3d: OrientTo3dParams::default(),
            tdensity: crate::tdensity::TexelDensityParams::default(),
            grouping: GroupParams::default(),
            numbered_groups: NumberedGroups::default(),
            split_overlap: SplitOverlapParams::default(),
            pack_op: PackOpType::default(),
        }
    }
}

impl PackParams {
    /// The effective target box, with the non-square ratio applied.
    pub fn effective_box(&self) -> Box2 {
        let mut b = self.target_box.box_();
        if self.non_square_packing != 1.0 {
            let cx = b.center();
            let w = b.max_extent() * self.non_square_packing;
            let h = b.max_extent();
            b = Box2::new(
                Vec2::new(cx.u - w / 2.0, cx.v - h / 2.0),
                Vec2::new(cx.u + w / 2.0, cx.v + h / 2.0),
            );
        }
        b
    }

    /// UV gap between two islands (relative mode): the margin fraction of the
    /// larger island's max extent, clamped to at least the border gap.
    pub fn island_gap(&self, a_extent: f64, b_extent: f64) -> f64 {
        if self.pixel_margin_enable {
            let tex = self.pixel_margin_tex_size.max(1) as f64;
            (self.pixel_margin as f64) / tex
        } else {
            self.margin * a_extent.max(b_extent)
        }
    }

    /// UV gap between an island and the target-box border.
    pub fn border_gap(&self, extent: f64) -> f64 {
        if self.pixel_border_margin_enable {
            let tex = self.pixel_margin_tex_size.max(1) as f64;
            (self.pixel_border_margin as f64) / tex
        } else if self.pixel_margin_enable {
            let tex = self.pixel_margin_tex_size.max(1) as f64;
            (self.pixel_margin as f64) / tex
        } else {
            self.margin * extent
        }
    }

    /// UV gap from an island to a static/"other" island (pixel margins add the
    /// `extra_pixel_margin_to_others` px).
    pub fn other_gap(&self, a_extent: f64, b_extent: f64) -> f64 {
        let base = self.island_gap(a_extent, b_extent);
        if self.pixel_margin_enable {
            let tex = self.pixel_margin_tex_size.max(1) as f64;
            base + (self.extra_pixel_margin_to_others as f64) / tex
        } else {
            base
        }
    }

    /// The rotation candidates (radians) for an island with optional
    /// per-island rotation step override.
    pub fn rotation_candidates(&self, island_step_override: u32) -> Vec<f64> {
        if !self.rotation_enable {
            return vec![0.0];
        }
        let step = if island_step_override > 0 {
            island_step_override
        } else {
            self.island_rot_step.max(self.rotation_step)
        };
        RotationStep { degrees: step }.angles()
    }

    /// The tile target geometry for the effective box.
    pub fn tile_target(&self) -> Option<TileTarget> {
        match self.tile_target {
            TileTargetMode::TileGrid => Some(TileTarget {
                count_x: self.tile_count_x.max(1),
                count_y: self.tile_count_y.max(1),
                size_x: self.effective_box().width() / self.tile_count_x.max(1) as f64,
                size_y: self.effective_box().height() / self.tile_count_y.max(1) as f64,
                start: self.start_tile,
                within: Some(self.effective_box()),
            }),
            TileTargetMode::TileRange => Some(TileTarget {
                count_x: self.tile_count_x.max(1),
                count_y: self.tile_count_y.max(1),
                size_x: self.effective_box().width() / self.tile_count_x.max(1) as f64,
                size_y: self.effective_box().height() / self.tile_count_y.max(1) as f64,
                start: self.start_tile,
                within: Some(self.effective_box()),
            }),
            TileTargetMode::DynamicTiles => {
                let b = self.effective_box();
                let per_row = self.tiles_in_row.max(1);
                Some(TileTarget {
                    count_x: per_row,
                    count_y: 1,
                    size_x: b.width() / per_row as f64,
                    size_y: b.height(),
                    start: self.start_tile,
                    within: Some(b),
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_addon() {
        let p = PackParams::default();
        assert_eq!(p.seed, 0);
        assert_eq!(p.topology_parse_timeout, 30.0);
        assert_eq!(p.precision, 500);
        assert!((p.margin - 0.003).abs() < 1e-12);
        assert!(!p.pixel_margin_enable);
        assert_eq!(p.pixel_margin, 5);
        assert_eq!(p.pixel_border_margin, 1);
        assert_eq!(p.extra_pixel_margin_to_others, 0);
        assert_eq!(p.pixel_margin_tex_size, 1024);
        assert!(p.rotation_enable);
        assert!(p.flipping_enable);
        assert_eq!(p.rotation_step, 90);
        assert_eq!(p.scale_mode, ScaleMode::MaxScale);
        assert!(p.fully_inside);
        assert_eq!(p.pack_strategy, PackStrategy::Automatic);
        assert_eq!(p.start_corner, BoxCorner::Bl);
        assert_eq!(p.tiles_in_row, 10);
        assert!(!p.lock_overlapping);
        assert!(!p.heuristic_enable);
        assert!((p.heuristic_search_time - 30.0).abs() < 1e-12);
        assert!(p.arrange_non_packed);
        assert_eq!(p.similarity.threshold, 0.1);
        assert_eq!(p.orient_to_3d.prim_3d_axis, UvpmAxis::Z);
        assert!((p.orient_to_3d.prim_sec_bias - 80.0).abs() < 1e-12);
        assert_eq!(p.grouping.layout, GroupLayoutMode::Automatic);
        assert!(!p.grouping.groups_together);
    }

    #[test]
    fn rotation_candidates() {
        let p = PackParams {
            rotation_step: 90,
            ..PackParams::default()
        };
        let a = p.rotation_candidates(0);
        assert_eq!(a.len(), 4); // 0, 90, 180, 270
        let p2 = PackParams {
            rotation_enable: false,
            ..PackParams::default()
        };
        assert_eq!(p2.rotation_candidates(0), vec![0.0]);
        let p3 = PackParams {
            rotation_step: 180,
            ..PackParams::default()
        };
        assert_eq!(p3.rotation_candidates(0).len(), 2);
    }

    #[test]
    fn pixel_margins() {
        let p = PackParams {
            pixel_margin_enable: true,
            pixel_margin: 10,
            pixel_margin_tex_size: 1024,
            ..PackParams::default()
        };
        let gap = p.island_gap(0.5, 0.25);
        assert!((gap - 10.0 / 1024.0).abs() < 1e-12);
        let p2 = PackParams {
            pixel_border_margin_enable: true,
            pixel_border_margin: 8,
            pixel_margin_enable: true,
            pixel_margin: 10,
            pixel_margin_tex_size: 1024,
            ..PackParams::default()
        };
        assert!((p2.border_gap(0.5) - 8.0 / 1024.0).abs() < 1e-12);
    }

    #[test]
    fn tdensity_units() {
        assert!((TexelDensityUnit::PxPerMeter.multiplier() - 1_000_000.0).abs() < 1e-9);
        assert!((TexelDensityUnit::PxPerCentimeter.multiplier() - 10_000.0).abs() < 1e-9);
        assert!((TexelDensityUnit::PxPerInch.multiplier() - 2_540.0).abs() < 1e-9);
        assert!((TexelDensityUnit::PxPerFoot.multiplier() - 304.8).abs() < 1e-9);
    }

    #[test]
    fn retcode_values() {
        assert_eq!(UvpmRetcode::Success as u8, 0);
        assert_eq!(UvpmRetcode::NoSpace as u8, 2);
        assert_eq!(UvpmRetcode::InvalidIslands as u8, 4);
        assert_eq!(UvpmRetcode::Warning as u8, 7);
    }
}
