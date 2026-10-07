//! Replayable packing certificates — the soundness counterpart to the
//! optimality-gap benchmarks, modeled on the squares project's certificate
//! discipline (<https://jlevy.github.io/squares/>, verification rungs V0–V5).
//!
//! A [`PackingCertificate`] captures everything needed to *recheck* a pack
//! result independently of the packer run: one transform per island, the
//! target box, and a digest of the validation-affecting parameters.
//! [`PackingCertificate::replay`] rebuilds the placed outlines from the raw
//! islands and re-runs the exact polygon-level validation
//! ([`crate::validate::validate_islands`]); a certificate is *sound* only if
//! the replay finds no overlaps, nothing outside the target, and no
//! self-intersections.
//!
//! Two disciplines come along for free:
//!
//! - **Persistence**: `to_json`/`from_json` (a tiny std-only codec — the
//!   workspace has no external dependencies) let a test or regression
//!   baseline store a certificate and replay it later, against the same
//!   islands and parameters.
//! - **Mutant checks**: deliberately corrupting a valid certificate (nudge
//!   one transform into a neighbor, shove one outside the box, replay
//!   against changed margins) *must* fail. A validator that accepts every
//!   mutant is vacuous; the unit tests here pin that it does not.
//!
//! The `params_digest` is an FNV-1a hash (not cryptographic — a change
//! detector) over the parameters that affect whether a given layout is
//! *valid*: margins, pixel margins, containment, the target box, and the
//! overlap-detection mode.

use crate::box2::Box2;
use crate::island::{Island, PlacedTransform};
use crate::params::PackParams;
use crate::validate;
use researchuv_math::Vec2;

/// Certificate format version (bump on breaking layout changes).
pub const FORMAT_VERSION: u32 = 1;

/// One island's placement: the components of its [`PlacedTransform`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CertificateEntry {
    pub island_index: u32,
    /// Rotation (CCW radians).
    pub rotation: f64,
    pub flipped: bool,
    pub scale: f64,
    pub tx: f64,
    pub ty: f64,
}

/// Why a certificate failed to replay.
#[derive(Clone, Debug)]
pub enum CertificateError {
    /// The certificate's island count does not match the islands provided.
    CountMismatch { certificate: u32, provided: usize },
    /// The validation-affecting parameters changed since the certificate was
    /// issued (different margins, target, containment, overlap mode).
    DigestMismatch { certificate: u64, replayed: u64 },
    /// An entry refers to an island index that does not exist.
    EntryOutOfBounds { index: u32 },
    /// Malformed JSON.
    Json(String),
    /// The replayed layout is invalid (overlaps / outside / self-intersecting).
    ReplayFailed(validate::ValidationReport),
}

impl std::fmt::Display for CertificateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CertificateError::CountMismatch { certificate, provided } => write!(
                f,
                "certificate covers {certificate} islands but {provided} were provided"
            ),
            CertificateError::DigestMismatch { certificate, replayed } => write!(
                f,
                "params digest changed: certificate {certificate:#x}, replayed {replayed:#x}"
            ),
            CertificateError::EntryOutOfBounds { index } => {
                write!(f, "entry refers to nonexistent island {index}")
            }
            CertificateError::Json(m) => write!(f, "malformed certificate JSON: {m}"),
            CertificateError::ReplayFailed(rep) => write!(
                f,
                "replay found an invalid layout: {} overlapping, {} outside, {} self-intersecting",
                rep.overlapping.len(),
                rep.outside.len(),
                rep.self_intersecting.len()
            ),
        }
    }
}

impl std::error::Error for CertificateError {}

/// FNV-1a over the validation-affecting parameters (a change detector, not a
/// cryptographic digest).
fn mix_u64(h: &mut u64, v: u64) {
    for b in v.to_le_bytes() {
        *h ^= b as u64;
        *h = h.wrapping_mul(0x100_0000_01b3);
    }
}

fn mix_f64(h: &mut u64, v: f64) {
    mix_u64(h, v.to_bits());
}

/// Digest of the parameters that decide whether a given layout validates.
pub fn params_digest(params: &PackParams) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    mix_u64(&mut h, FORMAT_VERSION as u64);
    mix_f64(&mut h, params.margin);
    mix_u64(&mut h, params.pixel_margin_enable as u64);
    mix_u64(&mut h, params.pixel_margin as u64);
    mix_u64(&mut h, params.pixel_border_margin_enable as u64);
    mix_u64(&mut h, params.pixel_border_margin as u64);
    mix_u64(&mut h, params.extra_pixel_margin_to_others as u64);
    mix_u64(&mut h, params.pixel_margin_tex_size as u64);
    mix_u64(&mut h, params.fully_inside as u64);
    let b = params.effective_box();
    mix_f64(&mut h, b.min.u);
    mix_f64(&mut h, b.min.v);
    mix_f64(&mut h, b.max.u);
    mix_f64(&mut h, b.max.v);
    mix_f64(&mut h, params.non_square_packing);
    mix_u64(
        &mut h,
        match params.overlap_detection_mode {
            crate::params::OverlapDetectionMode::Disabled => 0u64,
            crate::params::OverlapDetectionMode::AnyPart => 1,
            crate::params::OverlapDetectionMode::Exact => 2,
        },
    );
    mix_u64(&mut h, params.lock_overlapping as u64);
    // Mask to the f64-significand range so the digest survives the JSON
    // round trip bit-for-bit (numbers travel as f64).
    h & ((1u64 << 53) - 1)
}

/// A replayable packing result: per-island transforms + target + parameter
/// digest, serializable to JSON.
#[derive(Clone, Debug, PartialEq)]
pub struct PackingCertificate {
    pub format: u32,
    pub island_count: u32,
    /// Target box `[min.u, min.v, max.u, max.v]`.
    pub target: [f64; 4],
    pub params_digest: u64,
    /// One entry per placed island, in island order (all islands placed).
    pub entries: Vec<CertificateEntry>,
}

impl PackingCertificate {
    /// Capture a finished pack. Returns `None` unless every island was
    /// placed inside the target (partial packs and arranged-outside islands
    /// have no full-layout certificate).
    pub fn from_pack(
        islands: &[Island],
        params: &PackParams,
        result: &crate::pipeline::PackResult,
    ) -> Option<Self> {
        if islands.len() != result.placed.len() || result.placed.iter().any(|p| p.is_none()) {
            return None;
        }
        if !result.non_packed.is_empty() {
            return None;
        }
        let b = params.effective_box();
        let inside = result.placed.iter().all(|p| {
            p.as_ref()
                .map(|t| b.contains_box_eps(&t.box_, 1e-9))
                .unwrap_or(false)
        });
        if !inside {
            return None;
        }
        Some(Self {
            format: FORMAT_VERSION,
            island_count: islands.len() as u32,
            target: [b.min.u, b.min.v, b.max.u, b.max.v],
            params_digest: params_digest(params),
            entries: result
                .placed
                .iter()
                .enumerate()
                .map(|(i, t)| {
                    let t = t.as_ref().expect("all placed");
                    CertificateEntry {
                        island_index: i as u32,
                        rotation: t.rotation,
                        flipped: t.flipped,
                        scale: t.scale,
                        tx: t.tx,
                        ty: t.ty,
                    }
                })
                .collect(),
        })
    }

    /// Rebuild the placed outlines from the raw islands and re-run the exact
    /// validation. `Ok(report)` means the layout is still sound: no overlaps,
    /// nothing outside, no self-intersections.
    pub fn replay(
        &self,
        islands: &[Island],
        params: &PackParams,
    ) -> Result<validate::ValidationReport, CertificateError> {
        if self.island_count as usize != islands.len() {
            return Err(CertificateError::CountMismatch {
                certificate: self.island_count,
                provided: islands.len(),
            });
        }
        if self.params_digest != params_digest(params) {
            return Err(CertificateError::DigestMismatch {
                certificate: self.params_digest,
                replayed: params_digest(params),
            });
        }
        if self.entries.len() != islands.len() {
            return Err(CertificateError::CountMismatch {
                certificate: self.entries.len() as u32,
                provided: islands.len(),
            });
        }
        let target = Box2::new(
            Vec2::new(self.target[0], self.target[1]),
            Vec2::new(self.target[2], self.target[3]),
        );
        // Rebuild transforms from the certificate's components; the placed
        // box is recomputed from the transformed outline (the replay trusts
        // no stored bounding boxes).
        let mut scratch: Vec<Island> = islands.to_vec();
        let mut outlines: Vec<Vec<Vec2>> = Vec::with_capacity(islands.len());
        let mut holes: Vec<Vec<Vec<Vec2>>> = Vec::with_capacity(islands.len());
        for (i, e) in self.entries.iter().enumerate() {
            if e.island_index as usize >= islands.len() {
                return Err(CertificateError::EntryOutOfBounds { index: e.island_index });
            }
            if e.island_index as usize != i {
                return Err(CertificateError::EntryOutOfBounds { index: e.island_index });
            }
            let poly = e.apply(&islands[i].verts);
            let bb = crate::poly::bbox_of(&poly);
            let t = PlacedTransform::from_parts(e.rotation, e.flipped, e.scale, e.tx, e.ty, bb);
            outlines.push(t.transform_poly(&islands[i].verts));
            holes.push(islands[i].transformed_holes(&t));
        }
        let rep = validate::validate_islands(&mut scratch, &outlines, &holes, &target, params);
        if rep.overlapping.is_empty()
            && rep.outside.is_empty()
            && rep.self_intersecting.is_empty()
        {
            Ok(rep)
        } else {
            Err(CertificateError::ReplayFailed(rep))
        }
    }

    /// Serialize to a compact JSON object (std-only codec).
    pub fn to_json(&self) -> String {
        let mut s = String::with_capacity(128 + 96 * self.entries.len());
        s.push_str("{\"format\":");
        s.push_str(&self.format.to_string());
        s.push_str(",\"island_count\":");
        s.push_str(&self.island_count.to_string());
        s.push_str(",\"target\":[");
        for (k, v) in self.target.iter().enumerate() {
            if k > 0 {
                s.push(',');
            }
            s.push_str(&format_f64(*v));
        }
        s.push_str("],\"params_digest\":");
        s.push_str(&self.params_digest.to_string());
        s.push_str(",\"entries\":[");
        for (k, e) in self.entries.iter().enumerate() {
            if k > 0 {
                s.push(',');
            }
            s.push_str(&format!(
                "{{\"i\":{},\"rot\":{},\"flip\":{},\"scale\":{},\"tx\":{},\"ty\":{}}}",
                e.island_index,
                format_f64(e.rotation),
                e.flipped,
                format_f64(e.scale),
                format_f64(e.tx),
                format_f64(e.ty)
            ));
        }
        s.push_str("]}");
        s
    }

    /// Parse the JSON produced by [`to_json`].
    pub fn from_json(text: &str) -> Result<Self, CertificateError> {
        let mut p = Parser { b: text.as_bytes(), i: 0 };
        p.ws();
        let obj = p.object()?;
        p.ws();
        if p.i != p.b.len() {
            return Err(CertificateError::Json("trailing bytes".into()));
        }
        let mut cert = PackingCertificate {
            format: 0,
            island_count: 0,
            target: [0.0; 4],
            params_digest: 0,
            entries: Vec::new(),
        };
        let mut seen_target = false;
        let mut seen_digest = false;
        for (k, v) in obj {
            match k.as_str() {
                "format" => cert.format = v.as_num("format")? as u32,
                "island_count" => cert.island_count = v.as_num("island_count")? as u32,
                "target" => {
                    let arr = v.as_arr("target")?;
                    if arr.len() != 4 {
                        return Err(CertificateError::Json("target must have 4 numbers".into()));
                    }
                    for (j, x) in arr.iter().enumerate() {
                        cert.target[j] = x.as_num("target")?;
                    }
                    seen_target = true;
                }
                "params_digest" => {
                    cert.params_digest = v.as_num("params_digest")? as u64;
                    seen_digest = true;
                }
                "entries" => {
                    for item in v.as_arr("entries")? {
                        let fields = item.as_obj("entries[]")?;
                        let mut e = CertificateEntry {
                            island_index: 0,
                            rotation: 0.0,
                            flipped: false,
                            scale: 1.0,
                            tx: 0.0,
                            ty: 0.0,
                        };
                        for (fk, fv) in fields {
                            match fk.as_str() {
                                "i" => e.island_index = fv.as_num("i")? as u32,
                                "rot" => e.rotation = fv.as_num("rot")?,
                                "flip" => e.flipped = fv.as_bool("flip")?,
                                "scale" => e.scale = fv.as_num("scale")?,
                                "tx" => e.tx = fv.as_num("tx")?,
                                "ty" => e.ty = fv.as_num("ty")?,
                                other => {
                                    return Err(CertificateError::Json(format!(
                                        "unknown entry field {other}"
                                    )))
                                }
                            }
                        }
                        cert.entries.push(e);
                    }
                }
                other => {
                    return Err(CertificateError::Json(format!("unknown field {other}")));
                }
            }
        }
        if cert.format != FORMAT_VERSION {
            return Err(CertificateError::Json(format!(
                "unsupported format {}",
                cert.format
            )));
        }
        if !seen_target || !seen_digest {
            return Err(CertificateError::Json("missing target or params_digest".into()));
        }
        Ok(cert)
    }
}

impl CertificateEntry {
    /// Apply this entry's transform components to a polygon (the same
    /// scale → rotate → flip → translate composition as
    /// [`PlacedTransform::from_parts`]).
    fn apply(&self, poly: &[Vec2]) -> Vec<Vec2> {
        let (sr, cr) = (self.rotation.sin(), self.rotation.cos());
        let fx = if self.flipped { -1.0 } else { 1.0 };
        poly.iter()
            .map(|p| {
                let (x, y) = (p.u * self.scale, p.v * self.scale);
                Vec2::new(
                    fx * (cr * x - sr * y) + self.tx,
                    fx * (sr * x + cr * y) + self.ty,
                )
            })
            .collect()
    }
}

/// Shortest round-trippable f64 formatting: try increasing precision until
/// the parse reproduces the bits exactly.
fn format_f64(v: f64) -> String {
    for prec in 1..=17 {
        let s = format!("{v:.prec$e}");
        if s.parse::<f64>().map(|p| p.to_bits() == v.to_bits()).unwrap_or(false) {
            return s;
        }
    }
    format!("{v:e}")
}

// --- minimal JSON parser (objects, arrays, numbers, bools; keys only) ---

#[derive(Clone, Debug)]
enum Json {
    Num(f64),
    Bool(bool),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    fn as_num(&self, what: &str) -> Result<f64, CertificateError> {
        match self {
            Json::Num(v) => Ok(*v),
            _ => Err(CertificateError::Json(format!("{what} must be a number"))),
        }
    }
    fn as_bool(&self, what: &str) -> Result<bool, CertificateError> {
        match self {
            Json::Bool(v) => Ok(*v),
            _ => Err(CertificateError::Json(format!("{what} must be a bool"))),
        }
    }
    fn as_arr(&self, what: &str) -> Result<&[Json], CertificateError> {
        match self {
            Json::Arr(v) => Ok(v),
            _ => Err(CertificateError::Json(format!("{what} must be an array"))),
        }
    }
    fn as_obj(&self, what: &str) -> Result<&[(String, Json)], CertificateError> {
        match self {
            Json::Obj(v) => Ok(v),
            _ => Err(CertificateError::Json(format!("{what} must be an object"))),
        }
    }
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Parser<'a> {
    fn ws(&mut self) {
        while self.i < self.b.len() && self.b[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }
    fn peek(&mut self) -> Result<u8, CertificateError> {
        self.ws();
        self.b
            .get(self.i)
            .copied()
            .ok_or_else(|| CertificateError::Json("unexpected end of input".into()))
    }
    fn eat(&mut self, c: u8) -> Result<(), CertificateError> {
        if self.peek()? == c {
            self.i += 1;
            Ok(())
        } else {
            Err(CertificateError::Json(format!("expected '{}'", c as char)))
        }
    }
    fn object(&mut self) -> Result<Vec<(String, Json)>, CertificateError> {
        self.eat(b'{')?;
        let mut out = Vec::new();
        if self.peek()? == b'}' {
            self.i += 1;
            return Ok(out);
        }
        loop {
            let key = self.string()?;
            self.eat(b':')?;
            let val = self.value()?;
            out.push((key, val));
            match self.peek()? {
                b',' => {
                    self.i += 1;
                }
                b'}' => {
                    self.i += 1;
                    return Ok(out);
                }
                c => {
                    return Err(CertificateError::Json(format!(
                        "expected ',' or '}}', got '{}'",
                        c as char
                    )))
                }
            }
        }
    }
    fn value(&mut self) -> Result<Json, CertificateError> {
        match self.peek()? {
            b'{' => Ok(Json::Obj(self.object()?)),
            b'[' => {
                self.i += 1;
                let mut out = Vec::new();
                if self.peek()? == b']' {
                    self.i += 1;
                    return Ok(Json::Arr(out));
                }
                loop {
                    out.push(self.value()?);
                    match self.peek()? {
                        b',' => {
                            self.i += 1;
                        }
                        b']' => {
                            self.i += 1;
                            return Ok(Json::Arr(out));
                        }
                        c => {
                            return Err(CertificateError::Json(format!(
                                "expected ',' or ']', got '{}'",
                                c as char
                            )))
                        }
                    }
                }
            }
            b'"' => {
                // Keys only; a string in value position is an error we accept
                // as-is (the writer never produces them).
                let _ = self.string()?;
                Err(CertificateError::Json("unexpected string value".into()))
            }
            b't' => {
                self.lit("true")?;
                Ok(Json::Bool(true))
            }
            b'f' => {
                self.lit("false")?;
                Ok(Json::Bool(false))
            }
            _ => self.number().map(Json::Num),
        }
    }
    fn lit(&mut self, s: &str) -> Result<(), CertificateError> {
        if self.b[self.i..].starts_with(s.as_bytes()) {
            self.i += s.len();
            Ok(())
        } else {
            Err(CertificateError::Json(format!("expected {s}")))
        }
    }
    fn string(&mut self) -> Result<String, CertificateError> {
        self.eat(b'"')?;
        let start = self.i;
        while self.i < self.b.len() && self.b[self.i] != b'"' {
            self.i += 1;
        }
        if self.i >= self.b.len() {
            return Err(CertificateError::Json("unterminated string".into()));
        }
        let s = String::from_utf8_lossy(&self.b[start..self.i]).into_owned();
        self.i += 1;
        Ok(s)
    }
    fn number(&mut self) -> Result<f64, CertificateError> {
        self.ws();
        let start = self.i;
        while self.i < self.b.len()
            && matches!(self.b[self.i], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E')
        {
            self.i += 1;
        }
        if start == self.i {
            return Err(CertificateError::Json("expected a number".into()));
        }
        std::str::from_utf8(&self.b[start..self.i])
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .ok_or_else(|| CertificateError::Json("bad number".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::ScaleMode;
    use crate::pipeline::pack;

    fn sq(s: f64) -> Island {
        Island::from_polygon(vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(s, 0.0),
            Vec2::new(s, s),
            Vec2::new(0.0, s),
        ])
    }

    fn packed(n: usize, size: f64) -> (Vec<Island>, PackParams, crate::pipeline::PackResult) {
        let isls: Vec<Island> = (0..n)
            .map(|i| {
                let mut isl = sq(size);
                isl.verts[0].u += i as f64 * 1e-9; // distinct outlines
                isl
            })
            .collect();
        // Fixed uniform scale with a side that trivially fits (a 4×4 grid
        // of slots). A small margin keeps replay strict: the packer's
        // validation flags touching islands, so a marginless layout would
        // not replay cleanly by design.
        let mut params = PackParams::default();
        params.scale_mode = ScaleMode::FixedScale;
        params.scale = 0.25;
        params.margin = 0.01;
        let mut run = isls.clone();
        let result = pack(&mut run, &params);
        assert_eq!(result.retcode, crate::params::UvpmRetcode::Success);
        assert!(result.validation.overlapping.is_empty());
        (isls, params, result)
    }

    #[test]
    fn certificate_roundtrip_and_replay() {
        let (isls, params, result) = packed(8, 0.3);
        let cert = PackingCertificate::from_pack(&isls, &params, &result).expect("all placed");
        assert_eq!(cert.island_count, 8);
        assert_eq!(cert.entries.len(), 8);
        // Replay against the pristine islands.
        cert.replay(&isls, &params).expect("valid layout replays");
        // JSON round trip preserves the certificate bit-for-bit.
        let text = cert.to_json();
        let back = PackingCertificate::from_json(&text).expect("parses");
        assert_eq!(cert, back);
        back.replay(&isls, &params).expect("parsed cert replays");
    }

    #[test]
    fn mutant_overlap_is_rejected() {
        let (isls, params, result) = packed(8, 0.3);
        let mut cert = PackingCertificate::from_pack(&isls, &params, &result).unwrap();
        // Mutant: island 1 teleports exactly onto island 0.
        cert.entries[1].tx = cert.entries[0].tx;
        cert.entries[1].ty = cert.entries[0].ty;
        cert.entries[1].rotation = cert.entries[0].rotation;
        let err = cert.replay(&isls, &params).unwrap_err();
        match err {
            CertificateError::ReplayFailed(rep) => assert!(!rep.overlapping.is_empty()),
            other => panic!("expected ReplayFailed, got {other}"),
        }
    }

    #[test]
    fn mutant_outside_target_is_rejected() {
        let (isls, params, result) = packed(4, 0.4);
        let mut cert = PackingCertificate::from_pack(&isls, &params, &result).unwrap();
        // Mutant: island 2 shoved past the target's right edge.
        cert.entries[2].tx += 10.0;
        let err = cert.replay(&isls, &params).unwrap_err();
        match err {
            CertificateError::ReplayFailed(rep) => assert!(!rep.outside.is_empty()),
            other => panic!("expected ReplayFailed, got {other}"),
        }
    }

    #[test]
    fn mutant_params_change_is_rejected() {
        let (isls, params, result) = packed(4, 0.4);
        let cert = PackingCertificate::from_pack(&isls, &params, &result).unwrap();
        // Replay against different margins: the digest must trip first.
        // (packed() uses margin 0.01, so mutate to a different value.)
        let mut other = params.clone();
        other.margin = 0.02;
        match cert.replay(&isls, &other) {
            Err(CertificateError::DigestMismatch { .. }) => {}
            Err(other) => panic!("expected DigestMismatch, got {other}"),
            Ok(_) => panic!("digest failed to detect the margin change"),
        }
    }

    #[test]
    fn mutant_island_set_is_rejected() {
        let (isls, params, result) = packed(4, 0.4);
        let cert = PackingCertificate::from_pack(&isls, &params, &result).unwrap();
        // Replay against three islands instead of four.
        let short = &isls[..3];
        match cert.replay(short, &params) {
            Err(CertificateError::CountMismatch { .. }) => {}
            Err(other) => panic!("expected CountMismatch, got {other}"),
            Ok(_) => panic!("count check failed"),
        }
    }

    #[test]
    fn partial_pack_has_no_certificate() {
        // Fixed scale too large to place: not every island placed.
        let isls: Vec<Island> = (0..4).map(|_| sq(0.9)).collect();
        let mut params = PackParams::default();
        params.scale_mode = ScaleMode::FixedScale;
        params.scale = 1.0;
        params.margin = 0.0;
        let mut run = isls.clone();
        let result = pack(&mut run, &params);
        // Either arranged outside or non-packed; never a full placement.
        assert!(PackingCertificate::from_pack(&isls, &params, &result).is_none());
    }

    #[test]
    fn json_rejects_garbage() {
        assert!(PackingCertificate::from_json("not json").is_err());
        assert!(PackingCertificate::from_json("{\"format\":1}").is_err());
        assert!(PackingCertificate::from_json("{\"format\":2,\"island_count\":0,\"target\":[0,0,1,1],\"params_digest\":0,\"entries\":[]}").is_err());
    }
}
