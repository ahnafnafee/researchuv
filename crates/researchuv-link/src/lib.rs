//! # researchuv-link — the host integration layer
//!
//! Wires the [`researchuv_api`] catalog to the engine: [`HostLink`] holds an
//! engine [`App`] plus the last pipeline result and implements
//! [`ApiHandler`] for every catalog entry. Hosts that live out of process
//! use the wire framing ([`encode_request`], [`decode_request`],
//! [`encode_response`], [`decode_response`]) — length-prefixed `CRef`
//! documents over any byte transport, with errors carried in the response
//! envelope instead of the channel.
//!
//! ```no_run
//! use researchuv_api::ApiHandler;
//! use researchuv_link::HostLink;
//!
//! let mut host = HostLink::new();
//! let params = researchuv_core::val::CRef::new("Doc.Fixture");
//! host.invoke("Doc.Fixture", &params).expect("fixture loaded");
//! let out = host.invoke("Unwrap.Run", &researchuv_core::val::CRef::new("Unwrap.Run"))
//!     .expect("unwrap ran");
//! ```

#![forbid(unsafe_code)]

use researchuv_api::{ApiError, ApiHandler};
use researchuv_core::app::{App, Edition};
use researchuv_core::io as mesh_io;
use researchuv_core::val::{self, CRef, Val};
use researchuv_math::{Vec2, Vec3};
use researchuv_unwrap::pipeline::{Packer, PipelineOptions, PipelineResult};

/// The frame magic prefix (`RUV1`).
pub const FRAME_MAGIC: [u8; 4] = *b"RUV1";

/// One engine host: the app state plus the last unwrap result (for
/// validation and export entries).
pub struct HostLink {
    /// The engine application state (document, config, undo/redo).
    pub app: App,
    /// The source soup kept for re-runs (positions/faces as imported).
    source: Option<(Vec<Vec3>, Vec<[u32; 3]>)>,
    /// The last pipeline result (set by `Unwrap.Run`).
    pub last_result: Option<PipelineResult>,
}

impl Default for HostLink {
    fn default() -> Self {
        Self::new()
    }
}

impl HostLink {
    pub fn new() -> Self {
        Self::with_edition(Edition::RealSpace)
    }

    pub fn with_edition(edition: Edition) -> Self {
        Self {
            app: App::new(edition),
            source: None,
            last_result: None,
        }
    }

    fn require_source(&self) -> Result<(Vec<Vec3>, Vec<[u32; 3]>), ApiError> {
        self.source
            .clone()
            .ok_or_else(|| ApiError::Task("no document open (Doc.Import or Doc.Fixture first)".into()))
    }

    fn get_f64(&self, params: &CRef, key: &str) -> Option<f64> {
        match params.get(key) {
            Some(Val::Double(d)) => Some(*d),
            Some(Val::Int(i)) => Some(*i as f64),
            _ => None,
        }
    }

    fn get_i64(&self, params: &CRef, key: &str) -> Option<i64> {
        match params.get(key) {
            Some(Val::Int(i)) => Some(*i),
            Some(Val::Double(d)) => Some(*d as i64),
            _ => None,
        }
    }

    fn get_bool(&self, params: &CRef, key: &str) -> Option<bool> {
        match params.get(key) {
            Some(Val::Bool(b)) => Some(*b),
            Some(Val::Int(i)) => Some(*i != 0),
            _ => None,
        }
    }

    fn get_str<'a>(&self, params: &'a CRef, key: &str) -> Option<&'a str> {
        match params.get(key) {
            Some(Val::Str(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    fn build_fixture(name: &str, n: usize) -> Result<(Vec<Vec3>, Vec<[u32; 3]>), ApiError> {
        researchuv_unwrap::meshgen::fixture(name, n).ok_or_else(|| {
            ApiError::BadRequest(format!(
                "unknown fixture {name:?} (expected cube, sphere, torus, annulus, grid, cylinder)"
            ))
        })
    }

    fn load_mesh(&mut self, positions: Vec<Vec3>, faces: Vec<[u32; 3]>, weld_tol: f64) -> Result<CRef, ApiError> {
        let mesh = researchuv_unwrap::weld::weld(positions.clone(), faces.clone(), weld_tol);
        let (v, f) = (mesh.positions.len() as i64, mesh.faces.len() as i64);
        self.app.document = Some(researchuv_core::model::MultiMesh::new(mesh));
        self.source = Some((positions, faces));
        self.last_result = None;
        Ok(reply(&[
            ("Vertices", Val::Int(v)),
            ("Faces", Val::Int(f)),
        ]))
    }

    fn face_uvs(&self) -> Option<Vec<[Vec2; 3]>> {
        let res = self.last_result.as_ref()?;
        // Face → chart index (charts partition the faces).
        let mut owner: Vec<usize> = vec![usize::MAX; res.mesh.faces.len()];
        for (ci, cr) in res.charts.iter().enumerate() {
            for &fi in &cr.chart.face_ids {
                owner[fi] = ci;
            }
        }
        // Assemble in SOURCE face order so the triplets align 1:1 with the
        // OBJ face list; the face's source vertices → chart-local → island UV.
        let mut out = Vec::with_capacity(res.mesh.faces.len());
        for (fi, face) in res.mesh.faces.iter().enumerate() {
            let ci = owner[fi];
            if ci == usize::MAX {
                return None;
            }
            let ch = &res.charts[ci].chart;
            let island = &res.multi.islands[ci];
            let mut uv_tri = [Vec2::new(0.0, 0.0); 3];
            for (k, &v) in face.iter().enumerate() {
                match ch.local_of(v) {
                    Some(li) => uv_tri[k] = island.uv[li],
                    None => return None,
                }
            }
            out.push(uv_tri);
        }
        Some(out)
    }
}

fn reply(fields: &[(&str, Val)]) -> CRef {
    let mut c = CRef::new("Response");
    for (k, v) in fields {
        c.values.push((k.to_string(), v.clone()));
    }
    c
}

impl ApiHandler for HostLink {
    fn invoke(&mut self, name: &str, params: &CRef) -> Result<CRef, ApiError> {
        let entry = researchuv_api::lookup(name)
            .ok_or_else(|| ApiError::UnknownEntry(name.to_string()))?;
        // Required-parameter presence check.
        for spec in entry.params {
            if !spec.optional && params.get(spec.name).is_none() {
                return Err(ApiError::BadRequest(format!(
                    "missing required parameter {}",
                    spec.name
                )));
            }
        }
        match name {
            "Api.Info" => Ok(reply(&[
                ("Version", Val::Int(researchuv_api::API_VERSION as i64)),
                ("Entries", Val::Int(researchuv_api::CATALOG.len() as i64)),
                ("Edition", Val::Str(self.app.edition.as_str().to_string())),
            ])),
            "App.New" => {
                let edition = match self.get_str(params, "Edition") {
                    Some("VS") => Edition::VirtualSpace,
                    _ => Edition::RealSpace,
                };
                self.app = App::new(edition);
                self.source = None;
                self.last_result = None;
                Ok(reply(&[("Ok", Val::Bool(true))]))
            }
            "Config.Get" => {
                let key = self.get_str(params, "Key").unwrap_or_default();
                match self.app.config.get(key) {
                    Some(Val::Double(d)) => Ok(reply(&[("Value", Val::Double(*d)), ("Found", Val::Bool(true))])),
                    Some(Val::Bool(b)) => Ok(reply(&[("Value", Val::Double(if *b { 1.0 } else { 0.0 })), ("Found", Val::Bool(true))])),
                    Some(Val::Int(i)) => Ok(reply(&[("Value", Val::Double(*i as f64)), ("Found", Val::Bool(true))])),
                    _ => Ok(reply(&[("Value", Val::Double(0.0)), ("Found", Val::Bool(false))])),
                }
            }
            "Config.Set" => {
                let key = self.get_str(params, "Key").unwrap_or_default().to_string();
                let value = self.get_f64(params, "Value").unwrap_or(0.0);
                self.app.config.set(&key, Val::Double(value));
                Ok(reply(&[("Ok", Val::Bool(true))]))
            }
            "Doc.Fixture" => {
                let fixture = self.get_str(params, "Fixture").unwrap_or_default();
                let n = self.get_i64(params, "N").unwrap_or(6).clamp(2, 128) as usize;
                let (p, f) = Self::build_fixture(fixture, n)?;
                let weld = self.get_f64(params, "WeldTol").unwrap_or(1e-12);
                self.load_mesh(p, f, weld)
            }
            "Doc.Import" => {
                let path = self.get_str(params, "Path").unwrap_or_default();
                let path = std::path::Path::new(path);
                let lower = path
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(|e| e.to_ascii_lowercase())
                    .unwrap_or_default();
                let (p, f) = match lower.as_str() {
                    "obj" => mesh_io::read_obj(path).map_err(|e| ApiError::Task(e.to_string()))?,
                    "stl" => mesh_io::read_stl(path).map_err(|e| ApiError::Task(e.to_string()))?,
                    other => {
                        return Err(ApiError::BadRequest(format!(
                            "unsupported import format {other:?} (expected .obj or .stl)"
                        )))
                    }
                };
                let weld = self.get_f64(params, "WeldTol").unwrap_or(1e-12);
                self.load_mesh(p, f, weld)
            }
            "Doc.Stats" => {
                let (v, f, i) = match (&self.app.document, &self.last_result) {
                    (Some(doc), Some(res)) => (
                        doc.source.positions.len() as i64,
                        doc.source.faces.len() as i64,
                        res.multi.islands.len() as i64,
                    ),
                    (Some(doc), None) => (
                        doc.source.positions.len() as i64,
                        doc.source.faces.len() as i64,
                        0,
                    ),
                    (None, _) => (0, 0, 0),
                };
                Ok(reply(&[
                    ("HasDocument", Val::Bool(self.app.document.is_some())),
                    ("Vertices", Val::Int(v)),
                    ("Faces", Val::Int(f)),
                    ("Islands", Val::Int(i)),
                ]))
            }
            "Unwrap.Run" => {
                let (positions, faces) = self.require_source()?;
                let mut opts = PipelineOptions::default();
                opts.angle_min_deg = self.get_f64(params, "Angle").unwrap_or(opts.angle_min_deg);
                opts.weld_tol = self.get_f64(params, "WeldTol").unwrap_or(opts.weld_tol);
                opts.unfold.max_iter =
                    self.get_i64(params, "MaxIter").map(|i| i.max(1) as usize).unwrap_or(opts.unfold.max_iter);
                opts.seam_cut.enable = self.get_bool(params, "SeamCut").unwrap_or(opts.seam_cut.enable);
                opts.recut.enable = self.get_bool(params, "SplitCut").unwrap_or(opts.recut.enable);
                opts.threads = self.get_i64(params, "Threads").map(|t| t.clamp(0, 1024) as u32).unwrap_or(opts.threads);
                if self.get_bool(params, "Gpu").unwrap_or(false) {
                    opts.unfold.solver = researchuv_unwrap::SolverBackend::Gpu;
                }
                opts.padding = self.get_f64(params, "Padding").unwrap_or(opts.padding);
                if let Some(packer) = self.get_str(params, "Packer") {
                    opts.packer = match packer {
                        "islands" => Packer::Islands,
                        "shelf" => Packer::Shelf,
                        other => {
                            return Err(ApiError::BadRequest(format!(
                                "unknown packer {other:?} (expected \"shelf\" or \"islands\")"
                            )))
                        }
                    };
                }
                let res = researchuv_unwrap::pipeline::run(positions, faces, &opts)
                    .map_err(|e| ApiError::Task(e.to_string()))?;
                let charts = res.charts.len() as i64;
                let placed = res.placed.iter().filter(|p| p.is_some()).count() as i64;
                let n_charts = res.charts.len().max(1) as f64;
                let conformal = res.charts.iter().map(|c| c.metrics.conformal_mean).sum::<f64>() / n_charts;
                let area_ratio = res.charts.iter().map(|c| c.metrics.area_ratio_mean).sum::<f64>() / n_charts;
                let flips = res.charts.iter().map(|c| c.metrics.flips).sum::<usize>() as i64;
                let findings = res.mesh_report.findings.len() + res.atlas_report.findings.len();
                // Publish the islands into the app document (undo point).
                if let Some(doc) = self.app.document.as_mut() {
                    doc.islands = res.multi.islands.clone();
                }
                let out = reply(&[
                    ("Charts", Val::Int(charts)),
                    ("Placed", Val::Int(placed)),
                    ("Scale", Val::Double(res.scale)),
                    ("ConformalMean", Val::Double(conformal)),
                    ("AreaRatioMean", Val::Double(area_ratio)),
                    ("Flips", Val::Int(flips)),
                    ("Warnings", Val::Int(findings as i64)),
                ]);
                self.last_result = Some(res);
                Ok(out)
            }
            "Validate.Run" => {
                let res = self
                    .last_result
                    .as_ref()
                    .ok_or_else(|| ApiError::Task("nothing to validate (Unwrap.Run first)".into()))?;
                Ok(reply(&[
                    ("MeshErrors", Val::Int(res.mesh_report.errors().count() as i64)),
                    ("MeshWarnings", Val::Int(res.mesh_report.warnings().count() as i64)),
                    ("AtlasErrors", Val::Int(res.atlas_report.errors().count() as i64)),
                    ("AtlasWarnings", Val::Int(res.atlas_report.warnings().count() as i64)),
                ]))
            }
            "Atlas.Get" => {
                let res = self
                    .last_result
                    .as_ref()
                    .ok_or_else(|| ApiError::Task("no atlas yet (Unwrap.Run first)".into()))?;
                let mut islands_out: Vec<Val> = Vec::with_capacity(res.multi.islands.len());
                for (i, isl) in res.multi.islands.iter().enumerate() {
                    let mut o = CRef::new("Island");
                    // Interleaved (u, v) doubles.
                    o.values.push((
                        "Uv".into(),
                        Val::Array(isl.uv.iter().flat_map(|p| [Val::Double(p.u), Val::Double(p.v)]).collect()),
                    ));
                    o.values.push((
                        "Tris".into(),
                        Val::Array(isl.tris.iter().flat_map(|t| [Val::Int(t[0] as i64), Val::Int(t[1] as i64), Val::Int(t[2] as i64)]).collect()),
                    ));
                    if let Some(cr) = res.charts.get(i) {
                        o.values.push(("ConformalMean".into(), Val::Double(cr.metrics.conformal_mean)));
                        o.values.push(("ConformalMax".into(), Val::Double(cr.metrics.conformal_max)));
                        o.values.push(("AreaRatio".into(), Val::Double(cr.metrics.area_ratio_mean)));
                        o.values.push(("Folds".into(), Val::Int(cr.metrics.folds as i64)));
                        o.values.push(("Flips".into(), Val::Int(cr.metrics.flips as i64)));
                    }
                    islands_out.push(Val::Object(o));
                }
                Ok(reply(&[
                    ("Islands", Val::Array(islands_out)),
                    ("Charts", Val::Int(res.charts.len() as i64)),
                ]))
            }
            "Export.Obj" => {
                let res = self
                    .last_result
                    .as_ref()
                    .ok_or_else(|| ApiError::Task("nothing to export (Unwrap.Run first)".into()))?;
                let path = self.get_str(params, "Path").unwrap_or_default().to_string();
                let face_uvs = self.face_uvs();
                let face_uvs = face_uvs.as_deref();
                mesh_io::write_obj_file(std::path::Path::new(&path), &res.mesh, face_uvs)
                    .map_err(|e| ApiError::Task(e.to_string()))?;
                Ok(reply(&[("Ok", Val::Bool(true))]))
            }
            "Export.Svg" => {
                let res = self
                    .last_result
                    .as_ref()
                    .ok_or_else(|| ApiError::Task("nothing to export (Unwrap.Run first)".into()))?;
                let path = self.get_str(params, "Path").unwrap_or_default().to_string();
                let size = self.get_i64(params, "Size").unwrap_or(512).clamp(64, 4096) as f64;
                let island_tris = mesh_io::atlas_tris(&res.multi.islands);
                mesh_io::write_atlas_svg_file(std::path::Path::new(&path), &island_tris, size)
                    .map_err(|e| ApiError::Task(e.to_string()))?;
                Ok(reply(&[("Ok", Val::Bool(true))]))
            }
            "Undo.Undo" => {
                let name = self.app.undo().unwrap_or_default();
                Ok(reply(&[("Name", Val::Str(name))]))
            }
            "Undo.Redo" => {
                let name = self.app.redo().unwrap_or_default();
                Ok(reply(&[("Name", Val::Str(name))]))
            }
            other => Err(ApiError::UnknownEntry(other.to_string())),
        }
    }
}

// ---- Wire framing -----------------------------------------------------------------------------

/// Encode a request `CRef` into a framed message (`RUV1` magic + the core
/// binary codec, length-prefixed).
pub fn encode_request(cref: &CRef) -> Vec<u8> {
    let body = val::encode(cref);
    let mut out = Vec::with_capacity(FRAME_MAGIC.len() + body.len());
    out.extend_from_slice(&FRAME_MAGIC);
    out.extend_from_slice(&body);
    out
}

/// Decode a framed request produced by [`encode_request`].
pub fn decode_request(bytes: &[u8]) -> Result<CRef, ApiError> {
    if bytes.len() < FRAME_MAGIC.len() || bytes[..FRAME_MAGIC.len()] != FRAME_MAGIC {
        return Err(ApiError::Decode("missing RUV1 magic"));
    }
    val::decode(&bytes[FRAME_MAGIC.len()..]).map_err(ApiError::Decode)
}

/// The response envelope: `{ Ok: Bool, Error: String, Payload: CRef }`.
#[derive(Clone, Debug, PartialEq)]
pub struct Response {
    pub ok: bool,
    pub error: String,
    pub payload: CRef,
}

impl Response {
    /// A successful response carrying `payload`.
    pub fn ok(payload: CRef) -> Self {
        Self { ok: true, error: String::new(), payload }
    }

    /// A failed response carrying the error text.
    pub fn err(error: impl Into<String>) -> Self {
        Self { ok: false, error: error.into(), payload: CRef::new("Response") }
    }

    /// Convert into a `CRef` document.
    pub fn to_cref(&self) -> CRef {
        let mut c = CRef::new("Response");
        c.values.push(("Ok".into(), Val::Bool(self.ok)));
        c.values.push(("Error".into(), Val::Str(self.error.clone())));
        c.values.push(("Payload".into(), Val::Object(self.payload.clone())));
        c
    }

    /// Parse from a `CRef` document.
    pub fn from_cref(c: &CRef) -> Result<Self, ApiError> {
        let ok = match c.get("Ok") {
            Some(Val::Bool(b)) => *b,
            _ => return Err(ApiError::Decode("response missing Ok field")),
        };
        let error = match c.get("Error") {
            Some(Val::Str(s)) => s.clone(),
            _ => return Err(ApiError::Decode("response missing Error field")),
        };
        let payload = match c.get("Payload") {
            Some(Val::Object(o)) => o.clone(),
            _ => CRef::new("Response"),
        };
        Ok(Self { ok, error, payload })
    }
}

/// Encode a response envelope as a framed message.
pub fn encode_response(r: &Response) -> Vec<u8> {
    encode_request(&r.to_cref())
}

/// Decode a framed response produced by [`encode_response`].
pub fn decode_response(bytes: &[u8]) -> Result<Response, ApiError> {
    let cref = decode_request(bytes)?;
    Response::from_cref(&cref)
}

#[cfg(test)]
mod tests {
    use super::*;
    use researchuv_api::ApiHandler;

    #[test]
    fn info_reports_catalog_metadata() {
        let mut host = HostLink::new();
        let r = host.invoke("Api.Info", &CRef::new("Api.Info")).unwrap();
        assert_eq!(r.get("Version"), Some(&Val::Int(researchuv_api::API_VERSION as i64)));
        assert_eq!(r.get("Entries"), Some(&Val::Int(researchuv_api::CATALOG.len() as i64)));
    }

    #[test]
    fn unknown_and_missing_param_are_errors() {
        let mut host = HostLink::new();
        assert!(matches!(
            host.invoke("Nope", &CRef::new("Nope")),
            Err(ApiError::UnknownEntry(_))
        ));
        // Path is required for Doc.Import.
        assert!(matches!(
            host.invoke("Doc.Import", &CRef::new("Doc.Import")),
            Err(ApiError::BadRequest(_))
        ));
    }

    #[test]
    fn fixture_unwrap_and_export_roundtrip() {
        let mut host = HostLink::new();
        let mut fix = CRef::new("Doc.Fixture");
        fix.values.push(("Fixture".into(), Val::Str("cube".into())));
        fix.values.push(("N".into(), Val::Int(4)));
        let r = host.invoke("Doc.Fixture", &fix).unwrap();
        assert_eq!(r.get("Vertices"), Some(&Val::Int(98)));

        let mut run = CRef::new("Unwrap.Run");
        run.values.push(("Packer".into(), Val::Str("islands".into())));
        let r = host.invoke("Unwrap.Run", &run).unwrap();
        assert_eq!(r.get("Charts"), Some(&Val::Int(6)));
        assert_eq!(r.get("Placed"), Some(&Val::Int(6)));

        // Stats now see islands.
        let s = host.invoke("Doc.Stats", &CRef::new("Doc.Stats")).unwrap();
        assert_eq!(s.get("Islands"), Some(&Val::Int(6)));

        // Export OBJ with per-face UVs.
        let dir = std::env::temp_dir().join("researchuv-link-test");
        std::fs::create_dir_all(&dir).unwrap();
        let obj_path = dir.join("cube-uv.obj");
        let mut ex = CRef::new("Export.Obj");
        ex.values.push(("Path".into(), Val::Str(obj_path.to_string_lossy().into_owned())));
        host.invoke("Export.Obj", &ex).unwrap();
        let text = std::fs::read_to_string(&obj_path).unwrap();
        assert!(text.contains("vt "));
        let (p, f) = researchuv_core::io::parse_obj(&text).unwrap();
        assert_eq!(p.len(), 98);
        assert_eq!(f.len(), 192); // 12 · 4² faces for the n=4 cube

        // Export SVG.
        let svg_path = dir.join("cube-uv.svg");
        let mut ex = CRef::new("Export.Svg");
        ex.values.push(("Path".into(), Val::Str(svg_path.to_string_lossy().into_owned())));
        host.invoke("Export.Svg", &ex).unwrap();
        assert!(std::fs::read_to_string(&svg_path).unwrap().contains("<svg"));

        // Validation reflects a clean atlas.
        let v = host.invoke("Validate.Run", &CRef::new("Validate.Run")).unwrap();
        assert_eq!(v.get("AtlasErrors"), Some(&Val::Int(0)));
    }

    #[test]
    fn import_reads_obj_from_disk() {
        let dir = std::env::temp_dir().join("researchuv-link-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tri.obj");
        std::fs::write(&path, "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n").unwrap();
        let mut host = HostLink::new();
        let mut p = CRef::new("Doc.Import");
        p.values.push(("Path".into(), Val::Str(path.to_string_lossy().into_owned())));
        let r = host.invoke("Doc.Import", &p).unwrap();
        assert_eq!(r.get("Vertices"), Some(&Val::Int(3)));
        assert_eq!(r.get("Faces"), Some(&Val::Int(1)));
    }

    #[test]
    fn config_get_set_roundtrip() {
        let mut host = HostLink::new();
        let mut set = CRef::new("Config.Set");
        set.values.push(("Key".into(), Val::Str("Vars.AutoSelect.SharpEdges.Angle".into())));
        set.values.push(("Value".into(), Val::Double(45.0)));
        host.invoke("Config.Set", &set).unwrap();
        let mut get = CRef::new("Config.Get");
        get.values.push(("Key".into(), Val::Str("Vars.AutoSelect.SharpEdges.Angle".into())));
        let r = host.invoke("Config.Get", &get).unwrap();
        assert_eq!(r.get("Value"), Some(&Val::Double(45.0)));
        assert_eq!(r.get("Found"), Some(&Val::Bool(true)));
    }

    #[test]
    fn frames_roundtrip() {
        let mut req = CRef::new("Doc.Fixture");
        req.values.push(("Fixture".into(), Val::Str("cube".into())));
        let bytes = encode_request(&req);
        let back = decode_request(&bytes).unwrap();
        assert_eq!(back, req);
        let resp = Response::ok(req);
        let bytes = encode_response(&resp);
        let back = decode_response(&bytes).unwrap();
        assert!(back.ok);
        assert_eq!(back.payload, resp.payload);
        let err = Response::err("boom");
        let back = decode_response(&encode_response(&err)).unwrap();
        assert!(!back.ok);
        assert_eq!(back.error, "boom");
        // Bad magic rejected.
        assert!(decode_request(b"XXXX").is_err());
    }
}
