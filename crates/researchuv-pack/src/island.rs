//! Island data model — the engine's `Island` (faces → outline polygon), the
//! `UvpmIslandFlags` bits, and the per-face island parameters
//! (`__uvpm4_v1_*` channels, `MAX_COUNT = 16`).
//!
//! Evidence: `UvpmIslandFlags` (`OVERLAPS = 1`, `OUTSIDE_TARGET_BOX = 2`,
//! `ALIGNED = 4`, `SELECTED = 8`), `UvpmOutIslandsSerializationFlags`
//! (`CONTAINS_TRANSFORM = 1`, `CONTAINS_IPARAMS = 2`, `CONTAINS_FLAGS = 4`,
//! `CONTAINS_VERTICES = 8`), face input flags (`SELECTED = 1`,
//! `UV_SET_IDX_OFFSET = 1 << 16`), `UvpmIParamInfo` / `IParamInfo.MAX_COUNT = 16`,
//! `send_vertices` / `send_iparams` / `IntIParamsManager` / `StrIParamsManager`.

use crate::box2::Box2;
use researchuv_math::{SBox2, Vec2, Vec3};

/// Island flags — `UvpmIslandFlags` bit set (raw `u32`).
pub type IslandFlags = u32;

pub const OVERLAPS: IslandFlags = 1;
pub const OUTSIDE_TARGET_BOX: IslandFlags = 2;
pub const ALIGNED: IslandFlags = 4;
pub const SELECTED: IslandFlags = 8;

/// Per-face island parameter channels — `IParamInfo` (`MAX_COUNT = 16`).
///
/// The addon serializes per-face iparams into UV layers named
/// `__uvpm4_v1_<script_name>`; known channels: `align_priority` (0..100, def 0),
/// `normalize_multiplier` (10..1000 %, def 100), `rotation_step` (−1..180,
/// def −1 = global), `island_rot_step` (−1..180, def −1 = global),
/// `split_offset_x` / `split_offset_y` (−10000..10000, def −10000), and the
/// numbered groups (`lock_group`, `stack_group`, `track_group`, `norm_group`,
/// 0..1000, def 0 = 'N' unset, set values ≥ 1).
pub const IPARAM_MAX: usize = 16;

/// A mesh face contributing to an island (triangle in UV + optional 3D).
#[derive(Clone, Debug)]
pub struct Face {
    /// UV coordinates of the 3 face vertices (in UV space, the island's space).
    pub uv: [Vec2; 3],
    /// 3-D positions of the same 3 vertices (for orient-to-3D / 3D-axis matching).
    pub pos3d: Option<[Vec3; 3]>,
    /// Material name (`GroupingMethod::Material`).
    pub material: Option<String>,
    /// Object id (`GroupingMethod::Object`).
    pub object: Option<u64>,
    /// Tile `(ix, iy)` (`GroupingMethod::Tile`).
    pub tile: Option<(u32, u32)>,
    /// Vertex color (`GroupingMethod::VertexColor`).
    pub vertex_color: Option<Vec3>,
    /// Mesh part id (`GroupingMethod::Mesh`).
    pub mesh_part: Option<u64>,
    /// Collection name (`GroupingMethod::Collection`).
    pub collection: Option<String>,
    /// Per-face island parameter values for the 16 iparam channels
    /// (the documented addon defaults when the channel is unset).
    pub iparams: [f64; IPARAM_MAX],
    /// Face input flags: `SELECTED = 1`, `UV_SET_IDX_OFFSET = 1 << 16`.
    pub flags: u32,
}

impl Default for Face {
    /// A face with the addon's documented iparam channel defaults
    /// (`align_priority` 0, `normalize_multiplier` 100 %, `rotation_step` /
    /// `island_rot_step` −1 (use the global), `split_offset_x/y` at the unset
    /// sentinel −10000, and the four numbered-group channels at the unset
    /// sentinel 0 = 'N').
    fn default() -> Self {
        use crate::params::iparam as c;
        let mut iparams = [0.0f64; IPARAM_MAX];
        iparams[c::NORMALIZE_MULTIPLIER] = 100.0;
        iparams[c::ROTATION_STEP] = c::ROT_STEP_UNSET;
        iparams[c::ISLAND_ROT_STEP] = c::ROT_STEP_UNSET;
        iparams[c::SPLIT_OFFSET_X] = c::SPLIT_OFFSET_UNSET;
        iparams[c::SPLIT_OFFSET_Y] = c::SPLIT_OFFSET_UNSET;
        iparams[c::LOCK_GROUP] = c::GROUP_UNSET;
        iparams[c::STACK_GROUP] = c::GROUP_UNSET;
        iparams[c::TRACK_GROUP] = c::GROUP_UNSET;
        iparams[c::NORM_GROUP] = c::GROUP_UNSET;
        Self {
            uv: [Vec2::new(0.0, 0.0); 3],
            pos3d: None,
            material: None,
            object: None,
            tile: None,
            vertex_color: None,
            mesh_part: None,
            collection: None,
            iparams,
            flags: 0,
        }
    }
}

impl Face {
    pub fn is_selected(&self) -> bool {
        self.flags & 1 != 0
    }
    pub fn uv_set_idx(&self) -> i32 {
        (self.flags >> 16) as i32
    }
}

/// A UV island — a connected set of faces plus its outline polygon.
#[derive(Clone, Debug)]
pub struct Island {
    /// The island's faces (for iparams, grouping, topology similarity).
    pub faces: Vec<Face>,
    /// Outline polygon vertices, in order (CCW or CW; sign handled by area()).
    pub verts: Vec<Vec2>,
    /// Hole polygons inside the outline (annulus-like islands; each hole is
    /// subtracted from the filled area for overlap detection and area).
    pub holes: Vec<Vec<Vec2>>,
    /// 3-D positions corresponding to `verts` (orient-to-3D).
    pub verts3d: Option<Vec<Vec3>>,
    /// Island flags (see `UvpmIslandFlags`); mutable, set by validation/placement.
    pub flags: IslandFlags,
    /// Static island: placed first and never moved
    /// (`static_islands`; "Pack To Others" / pinned UVs).
    pub is_static: bool,
    /// Group id (manual/material grouping; `0` = no group).
    pub group: u32,
    /// Bounding box (recomputed by [`Island::rebuild_bbox`]).
    pub bbox: Box2,
}

impl Island {
    /// Build an island from a bare outline polygon (no faces — e.g. the
    /// chart outlines produced by the unwrap pipeline).
    pub fn from_polygon(verts: Vec<Vec2>) -> Self {
        Self::from_polygon_with_holes(verts, Vec::new())
    }

    /// Build an island from an outline polygon plus interior hole loops.
    pub fn from_polygon_with_holes(verts: Vec<Vec2>, holes: Vec<Vec<Vec2>>) -> Self {
        let mut isl = Self {
            faces: Vec::new(),
            verts,
            holes,
            verts3d: None,
            flags: 0,
            is_static: false,
            group: 0,
            bbox: Box2::unit(),
        };
        isl.rebuild_bbox();
        isl
    }

    /// Build an island from faces (union outline computed by the caller or
    /// left empty); sets the bbox from all face + outline vertices.
    pub fn from_faces(faces: Vec<Face>) -> Self {
        let mut isl = Self {
            faces,
            verts: Vec::new(),
            holes: Vec::new(),
            verts3d: None,
            flags: 0,
            is_static: false,
            group: 0,
            bbox: Box2::unit(),
        };
        isl.rebuild_bbox();
        isl
    }

    /// Recompute `bbox` from the outline (or faces when no outline).
    pub fn rebuild_bbox(&mut self) {
        let mut b: Option<Box2> = None;
        let grow = |p: Vec2, b: &mut Option<Box2>| {
            match b {
                Some(bb) => *bb = Box2::new(
                    Vec2::new(bb.min.u.min(p.u), bb.min.v.min(p.v)),
                    Vec2::new(bb.max.u.max(p.u), bb.max.v.max(p.v)),
                ),
                None => *b = Some(Box2::new(p, p)),
            }
        };
        for p in &self.verts {
            grow(*p, &mut b);
        }
        if b.is_none() {
            for f in &self.faces {
                for p in f.uv.iter() {
                    grow(*p, &mut b);
                }
            }
        }
        self.bbox = b.unwrap_or_else(Box2::unit);
    }

    /// Signed area of the outline (shoelace); `0.0` when no outline.
    pub fn signed_area(&self) -> f64 {
        if self.verts.len() < 3 {
            return 0.0;
        }
        let n = self.verts.len();
        let mut a = 0.0f64;
        for i in 0..n {
            let p0 = self.verts[i];
            let p1 = self.verts[(i + 1) % n];
            a += p0.u * p1.v - p1.u * p0.v;
        }
        0.5 * a
    }
    /// Absolute area of the outline minus the holes.
    pub fn area(&self) -> f64 {
        let outer = self.signed_area().abs();
        let holes: f64 = self
            .holes
            .iter()
            .map(|h| crate::poly::signed_area(h).abs())
            .sum();
        (outer - holes).max(0.0)
    }

    /// Does this island carry `SELECTED`?
    pub fn is_selected(&self) -> bool {
        self.flags & SELECTED != 0
    }
    /// Does this island carry `OVERLAPS`?
    pub fn overlaps_flag(&self) -> bool {
        self.flags & OVERLAPS != 0
    }
    /// Does this island carry `OUTSIDE_TARGET_BOX`?
    pub fn outside_flag(&self) -> bool {
        self.flags & OUTSIDE_TARGET_BOX != 0
    }

    /// Per-face iparam channel values (the `i`-th channel across faces).
    /// Returns the max over faces (UVPM's "get value" semantics for grouping),
    /// or `None` if no face carries the channel.
    pub fn iparam_channel(&self, i: usize) -> Option<f64> {
        let mut best: Option<f64> = None;
        for f in &self.faces {
            let v = f.iparams[i];
            best = Some(match best {
                None => v,
                Some(b) => b.max(v),
            });
        }
        best
    }

    /// The GPUPack-style AABB (`SBox2`) of the island.
    pub fn sbox(&self) -> SBox2 {
        SBox2::new(self.bbox.min, self.bbox.max)
    }

    /// Transform the outline by a placement (used to emit final UVs).
    pub fn transformed_outline(&self, t: &PlacedTransform) -> Vec<Vec2> {
        self.verts.iter().map(|p| t.apply(*p)).collect()
    }

    /// Transform the hole loops by a placement.
    pub fn transformed_holes(&self, t: &PlacedTransform) -> Vec<Vec<Vec2>> {
        self.holes.iter().map(|h| t.transform_poly(h)).collect()
    }
}

/// Placement transform of one island (the `CONTAINS_TRANSFORM` payload).
///
/// The engine's `STransform2` is `[sx, sy, tx, ty]` (scale + translate); the
/// full rotation/flip is carried by the transformed vertices
/// (`CONTAINS_VERTICES`). This struct keeps both: the 2×2 affine for the
/// general case and the axis-aligned components for the `STransform2` path.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlacedTransform {
    /// `x' = a·x + b·y + tx`.
    pub a: f64,
    pub b: f64,
    /// `y' = c·x + d·y + ty`.
    pub c: f64,
    pub d: f64,
    pub tx: f64,
    pub ty: f64,
    /// Rotation (radians, CCW) applied to the island.
    pub rotation: f64,
    /// Mirror (flip) applied to the island.
    pub flipped: bool,
    /// Uniform scale factor (the pack scale for this island).
    pub scale: f64,
    /// The placed bounding box in target space.
    pub box_: Box2,
}

impl PlacedTransform {
    /// Identity transform.
    pub fn identity() -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 1.0,
            tx: 0.0,
            ty: 0.0,
            rotation: 0.0,
            flipped: false,
            scale: 1.0,
            box_: Box2::unit(),
        }
    }

    /// Build from rotation (CCW radians), flip, uniform scale, and translation
    /// (applied in order: scale → rotate → flip → translate).
    pub fn from_parts(
        rotation: f64,
        flipped: bool,
        scale: f64,
        tx: f64,
        ty: f64,
        box_: Box2,
    ) -> Self {
        let (sr, cr) = (rotation.sin(), rotation.cos());
        let fx = if flipped { -1.0 } else { 1.0 };
        // M = T · Flip · R(θ) · S, with R CCW: [[cr, −sr],[sr, cr]].
        let m00 = cr * scale;
        let m01 = -sr * scale;
        let m10 = sr * scale;
        let m11 = cr * scale;
        Self {
            a: fx * m00,
            b: fx * m01,
            c: fx * m10,
            d: fx * m11,
            tx,
            ty,
            rotation,
            flipped,
            scale,
            box_,
        }
    }

    /// Apply the affine to a point.
    pub fn apply(&self, p: Vec2) -> Vec2 {
        Vec2::new(self.a * p.u + self.b * p.v + self.tx, self.c * p.u + self.d * p.v + self.ty)
    }

    /// The `STransform2`-compatible 4-vector (valid for axis-aligned placements:
    /// `sx = a`, `sy = d`, `tx`, `ty`).
    pub fn stransform2(&self) -> (f32, f32, f32, f32) {
        (self.a as f32, self.d as f32, self.tx as f32, self.ty as f32)
    }

    /// Transform an island polygon.
    pub fn transform_poly(&self, poly: &[Vec2]) -> Vec<Vec2> {
        poly.iter().map(|p| self.apply(*p)).collect()
    }
}

/// The `SVec2` alias re-export for GPUPack ABI compatibility.
pub use researchuv_math::SVec2;

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_square() -> Island {
        Island::from_polygon(vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(0.0, 1.0),
        ])
    }

    #[test]
    fn polygon_area_and_bbox() {
        let i = unit_square();
        assert!((i.area() - 1.0).abs() < 1e-12);
        assert_eq!(i.bbox.min, Vec2::new(0.0, 0.0));
        assert_eq!(i.bbox.max, Vec2::new(1.0, 1.0));
    }

    #[test]
    fn transform_roundtrip() {
        let i = unit_square();
        let t = PlacedTransform::from_parts(
            std::f64::consts::FRAC_PI_2,
            false,
            0.5,
            0.1,
            0.2,
            Box2::unit(),
        );
        let p = t.apply(Vec2::new(1.0, 1.0));
        // rotate 90° CCW: (1,1) → (−1,1); scale 0.5 → (−0.5, 0.5); translate → (−0.4, 0.7)
        assert!((p.u + 0.4).abs() < 1e-12 && (p.v - 0.7).abs() < 1e-12);
        let _ = i.transformed_outline(&t);
    }

    #[test]
    fn flags_bits() {
        assert_eq!(OVERLAPS | OUTSIDE_TARGET_BOX, 3);
        assert_eq!(ALIGNED | SELECTED, 12);
        let mut i = unit_square();
        i.flags = OVERLAPS;
        assert!(i.overlaps_flag());
        assert!(!i.outside_flag());
    }

    #[test]
    fn iparam_channel_max() {
        let mut f1 = Face::default();
        f1.uv = [
            Vec2::new(0.0, 0.0),
            Vec2::new(0.5, 0.0),
            Vec2::new(0.0, 0.5),
        ];
        f1.iparams[3] = 42.0;
        let mut f2 = f1.clone();
        f2.iparams[3] = 7.0;
        let mut isl = Island::from_faces(vec![f1, f2]);
        isl.rebuild_bbox();
        assert_eq!(isl.iparam_channel(3), Some(42.0));
        // Channel 4 is `split_offset_x`: the default face carries the
        // documented unset sentinel, and a channel with no default is 0.0.
        assert_eq!(
            isl.iparam_channel(4),
            Some(crate::params::iparam::SPLIT_OFFSET_UNSET)
        );
        assert_eq!(isl.iparam_channel(10), Some(0.0));
    }
}
