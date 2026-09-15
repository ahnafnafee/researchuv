//! Mesh topology, task dispatch, configuration, and typed parameter values.
//!
//! | Module | Responsibility |
//! | --- | --- |
//! | [`model`] | Half-edge surfaces, UV islands, and multi-mesh documents |
//! | [`val`] | Named values, attributes, and binary serialization |
//! | [`app`] | Task registration, configuration, reports, and undo/redo |

pub mod app;
pub mod model;
pub mod val;

pub use app::{App, Config, DataReport, Edition, Task};
pub use model::{Island, MultiMesh, PrimType, SurfaceMesh, WorkingSet};
pub use val::{CRef, Val};

/// Base address used by the retained numerical evidence annotations.
pub const IMAGE_BASE: u64 = 0x140000000;
