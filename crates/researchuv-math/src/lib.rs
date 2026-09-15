//! Math and geometry primitives for ResearchUV.
//!
//! | Module | Responsibility |
//! | --- | --- |
//! | [`mod@vec`] | Two- and three-dimensional vectors |
//! | [`mat`] | Small matrices for geometry calculations |
//! | [`geo`] | Per-triangle factors and tangent-frame contributions |
//! | [`pack`] | Bounding boxes and affine island transforms |

#![forbid(unsafe_code)]

pub mod geo;
pub mod mat;
pub mod pack;
pub mod vec;

pub use geo::{compute_geo_tri2, rhs_pull, CGeoTri2};
pub use mat::{Mat2, Mat4};
pub use pack::{SBox2, STransform2, SVec2};
pub use vec::{Vec2, Vec3};

/// Documented `.rdata` constant VA (base `0x140000000`) for the area half-factor `1/2`.
/// File offset = VA − 0x140000000 − 0x1600 (`.rdata` delta). See ALGORITHMS.md §2.3.
pub const CONST_AREA_HALF_FACTOR_VA: u64 = 0x140F84DC8;
