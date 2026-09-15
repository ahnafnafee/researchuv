//! Mesh topology, task dispatch, configuration, and typed parameter values.
//!
//! | Module | Responsibility |
//! | --- | --- |
//! | [`model`] | Half-edge surfaces, UV islands, and multi-mesh documents |
//! | [`val`] | Named values, attributes, and binary serialization |
//! | [`app`] | Task registration, configuration, reports, and undo/redo |
//! | [`io`] | OBJ/STL import, OBJ/STL export, and atlas SVG rendering |

pub mod app;
pub mod io;
pub mod model;
pub mod val;

pub use app::{App, Config, DataReport, Edition, Task};
pub use io::{
    atlas_svg, parse_obj, parse_stl, read_obj, read_stl, write_atlas_svg_file, write_obj,
    write_obj_file, write_stl, write_stl_file, IoError,
};
pub use model::{Island, MultiMesh, PrimType, SurfaceMesh, WorkingSet};
pub use val::{CRef, Val};

/// Base address used by the retained numerical evidence annotations.
pub const IMAGE_BASE: u64 = 0x140000000;
