//! # researchuv-api — the structured API catalog
//!
//! The call surface a ResearchUV host can drive: every entry names its
//! request parameters and reply fields (with the wire type tags of
//! [`researchuv_core::val`]), so a host can validate and marshal calls
//! before dispatch. [`researchuv_link`] implements the entries against the
//! engine; this crate stays data-only so embedders can build their own
//! dispatch (FFI, sockets, scripts) from the same catalog.

#![forbid(unsafe_code)]

use researchuv_core::val::CRef;

/// The catalog format version (bumped when entries change).
pub const API_VERSION: u32 = 1;

/// One request/response field: name, wire tag, and meaning.
#[derive(Clone, Copy, Debug)]
pub struct FieldSpec {
    /// The field name (as it appears in the `CRef` values).
    pub name: &'static str,
    /// The wire type tag (`Val::type_tag` vocabulary: `Double`, `Int64`,
    /// `String`, `Bool`, …).
    pub tag: &'static str,
    /// Whether the call tolerates the field being absent.
    pub optional: bool,
    /// Human-readable meaning.
    pub doc: &'static str,
}

/// One catalog entry: a callable by name with its parameter and reply fields.
#[derive(Clone, Copy, Debug)]
pub struct ApiEntry {
    /// The dispatch name (e.g. `"Unwrap.Run"`).
    pub name: &'static str,
    /// One-line summary.
    pub summary: &'static str,
    /// Request fields (all required unless marked optional).
    pub params: &'static [FieldSpec],
    /// Reply fields (present on success).
    pub returns: &'static [FieldSpec],
}

const fn req(name: &'static str, tag: &'static str, doc: &'static str) -> FieldSpec {
    FieldSpec { name, tag, optional: false, doc }
}

const fn opt(name: &'static str, tag: &'static str, doc: &'static str) -> FieldSpec {
    FieldSpec { name, tag, optional: true, doc }
}

const fn entry(
    name: &'static str,
    summary: &'static str,
    params: &'static [FieldSpec],
    returns: &'static [FieldSpec],
) -> ApiEntry {
    ApiEntry { name, summary, params, returns }
}

/// The full catalog, in dispatch order.
pub const CATALOG: &[ApiEntry] = &[
    entry(
        "Api.Info",
        "Catalog metadata: version, entry count, edition.",
        &[],
        &[
            req("Version", "Int64", "The API catalog version."),
            req("Entries", "Int64", "Number of catalog entries."),
            req("Edition", "String", "The host edition label (RS/VS)."),
        ],
    ),
    entry(
        "App.New",
        "Reset the engine state (document, config defaults, undo stacks).",
        &[opt("Edition", "String", "\"RS\" or \"VS\" (default RS).")],
        &[req("Ok", "Bool", "Always true.")],
    ),
    entry(
        "Config.Get",
        "Read one configuration value by dotted key.",
        &[req("Key", "String", "The config key (e.g. \"Vars.AutoSelect.SharpEdges.Angle\").")],
        &[req("Value", "Double", "The value (0 when the key is unset)."), req("Found", "Bool", "Whether the key exists.")],
    ),
    entry(
        "Config.Set",
        "Write one configuration value by dotted key.",
        &[req("Key", "String", "The config key."), req("Value", "Double", "The value to store.")],
        &[req("Ok", "Bool", "Always true.")],
    ),
    entry(
        "Doc.Fixture",
        "Load a procedural fixture mesh into the document.",
        &[
            req("Fixture", "String", "One of: cube, sphere, torus, annulus, grid, cylinder."),
            opt("N", "Int64", "The subdivision parameter (fixture-specific, default 6)."),
        ],
        &[
            req("Vertices", "Int64", "Welded vertex count."),
            req("Faces", "Int64", "Welded face count."),
        ],
    ),
    entry(
        "Doc.Import",
        "Import a mesh file (OBJ or STL) into the document.",
        &[
            req("Path", "String", "The file path (.obj/.stl, case-insensitive)."),
            opt("WeldTol", "Double", "Position weld tolerance (default 1e-12)."),
        ],
        &[
            req("Vertices", "Int64", "Welded vertex count."),
            req("Faces", "Int64", "Welded face count."),
        ],
    ),
    entry(
        "Doc.Stats",
        "Document statistics.",
        &[],
        &[
            req("HasDocument", "Bool", "Whether a document is open."),
            req("Vertices", "Int64", "Source vertex count (0 when closed)."),
            req("Faces", "Int64", "Source face count (0 when closed)."),
            req("Islands", "Int64", "Island count after an unwrap."),
        ],
    ),
    entry(
        "Unwrap.Run",
        "Run the full unwrap pipeline on the current document.",
        &[
            opt("Angle", "Double", "Sharp-edge cut angle in degrees (default 30)."),
            opt("WeldTol", "Double", "Position weld tolerance (default 1e-12)."),
            opt("MaxIter", "Int64", "Unfold driver iterations (default 100)."),
            opt("SeamCut", "Bool", "Cut closed charts open (default false)."),
            opt("Packer", "String", "\"shelf\" or \"islands\" (default shelf)."),
            opt("Padding", "Double", "Shelf packer gutter (default 0.01)."),
        ],
        &[
            req("Charts", "Int64", "Chart count."),
            req("Placed", "Int64", "Placed island count."),
            req("Scale", "Double", "The pack scale."),
            req("ConformalMean", "Double", "Mean conformal ratio over all charts."),
            req("AreaRatioMean", "Double", "Mean area ratio over all charts."),
            req("Flips", "Int64", "Total flipped/degenerate triangles."),
            req("Warnings", "Int64", "Validation findings (errors + warnings)."),
        ],
    ),
    entry(
        "Validate.Run",
        "Re-run validation on the current document and last unwrap.",
        &[],
        &[
            req("MeshErrors", "Int64", "Malformed-mesh errors."),
            req("MeshWarnings", "Int64", "Malformed-mesh warnings."),
            req("AtlasErrors", "Int64", "Atlas errors (unplaced/overlap/outside)."),
            req("AtlasWarnings", "Int64", "Atlas warnings (flips)."),
        ],
    ),
    entry(
        "Export.Obj",
        "Export the unwrapped atlas as an OBJ (v/vt/f) file.",
        &[req("Path", "String", "The output path.")],
        &[req("Ok", "Bool", "True once written.")],
    ),
    entry(
        "Export.Svg",
        "Render the packed islands to an SVG file.",
        &[
            req("Path", "String", "The output path."),
            opt("Size", "Int64", "The canvas size in px (default 512)."),
        ],
        &[req("Ok", "Bool", "True once written.")],
    ),
    entry(
        "Undo.Redo",
        "Redo the last undone task; returns the task name.",
        &[],
        &[req("Name", "String", "The redone task (empty when nothing to redo).")],
    ),
    entry(
        "Undo.Undo",
        "Undo the last task; returns the task name.",
        &[],
        &[req("Name", "String", "The undone task (empty when nothing to undo).")],
    ),
];

/// Look up an entry by dispatch name.
pub fn lookup(name: &str) -> Option<&'static ApiEntry> {
    CATALOG.iter().find(|e| e.name == name)
}

/// All dispatch names, in catalog order.
pub fn names() -> impl Iterator<Item = &'static str> {
    CATALOG.iter().map(|e| e.name)
}

/// A call failure.
#[derive(Clone, Debug, PartialEq)]
pub enum ApiError {
    /// No entry with that dispatch name.
    UnknownEntry(String),
    /// A required parameter is missing or has the wrong type.
    BadRequest(String),
    /// The request bytes did not decode.
    Decode(&'static str),
    /// The entry ran and failed.
    Task(String),
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ApiError::UnknownEntry(n) => write!(f, "unknown api entry {n:?}"),
            ApiError::BadRequest(m) => write!(f, "bad request: {m}"),
            ApiError::Decode(m) => write!(f, "decode error: {m}"),
            ApiError::Task(m) => write!(f, "task failed: {m}"),
        }
    }
}

impl std::error::Error for ApiError {}

/// The dispatch contract: invoke one catalog entry by name.
pub trait ApiHandler {
    fn invoke(&mut self, name: &str, params: &CRef) -> Result<CRef, ApiError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_names_are_unique_and_dotted() {
        let mut seen = std::collections::BTreeSet::new();
        for e in CATALOG {
            assert!(e.name.contains('.'), "{} is namespaced", e.name);
            assert!(seen.insert(e.name), "{} is unique", e.name);
            assert!(!e.summary.is_empty(), "{} has a summary", e.name);
        }
    }

    #[test]
    fn lookup_finds_entries() {
        assert!(lookup("Unwrap.Run").is_some());
        assert!(lookup("Nope.Run").is_none());
        assert_eq!(names().count(), CATALOG.len());
    }

    #[test]
    fn param_tags_are_wire_tags() {
        for e in CATALOG {
            for f in e.params.iter().chain(e.returns.iter()) {
                assert!(
                    matches!(f.tag, "Bool" | "Int64" | "Double" | "String" | "Array<>" | "Null"),
                    "{}.{} has tag {}",
                    e.name,
                    f.name,
                    f.tag
                );
            }
        }
    }

    #[test]
    fn errors_display() {
        assert_eq!(
            ApiError::UnknownEntry("X".into()).to_string(),
            "unknown api entry \"X\""
        );
        assert_eq!(ApiError::Decode("eof").to_string(), "decode error: eof");
    }
}
