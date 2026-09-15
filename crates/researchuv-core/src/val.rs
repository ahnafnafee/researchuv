//! Typed task parameters and their binary representation.
//!
//! [`CRef`] carries a type name, ordered values, and attributes. [`encode`] and
//! [`decode`] serialize nested values using length-prefixed fields, type tags,
//! and UTF-8 strings.

use std::collections::BTreeMap;

/// A typed value — one of the kinds documented in API.md §6.
#[derive(Clone, Debug, PartialEq)]
pub enum Val {
    Null,
    Bool(bool),
    /// Integer-valued kind (`Int`/`UInt`/`Int64`/`Short`/`Char`/`UChar` collapse here).
    Int(i64),
    /// Floating kind (`Float`/`Double`).
    Double(f64),
    /// `String` — stored UTF-16LE on the wire in the original; UTF-8 here.
    Str(String),
    /// `Array<>`.
    Array(Vec<Val>),
    /// A nested `CRef` (object) with its own type name / values / attributes.
    Object(CRef),
}

impl Val {
    /// The wire type tag (`fieldType` in the `fieldType#fieldName` descriptor).
    pub fn type_tag(&self) -> &str {
        match self {
            Val::Null => "Null",
            Val::Bool(_) => "Bool",
            Val::Int(_) => "Int64",
            Val::Double(_) => "Double",
            Val::Str(_) => "String",
            Val::Array(_) => "Array<>",
            Val::Object(o) => &o.type_name_tag,
        }
    }
}

/// A named, attributed reference — `CRef`. Ordered parameter map + attribute map.
#[derive(Clone, Debug, PartialEq)]
pub struct CRef {
    /// The registered task/params class name (e.g. `"CTaskUnfoldParams"`); the tag written as
    /// `TypeName\0` (task-name registry VA `0x14040C0F0`).
    pub type_name_tag: String,
    /// Parameter values (`values[]`), insertion-ordered.
    pub values: Vec<(String, Val)>,
    /// Attributes (`attributes[]`) — e.g. the `"Island"` selector attribute.
    pub attributes: BTreeMap<String, Val>,
}

impl CRef {
    pub fn new(type_name: impl Into<String>) -> Self {
        Self { type_name_tag: type_name.into(), values: Vec::new(), attributes: BTreeMap::new() }
    }
    /// `CRef::AddAttribute("Island", …)` — the documented "Island" selector attribute form.
    pub fn add_attribute(&mut self, key: &str, v: Val) {
        self.attributes.insert(key.to_string(), v);
    }
    pub fn get(&self, key: &str) -> Option<&Val> {
        self.values.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
}

// ---- Binary codec (structural match to API.md §6 framing) ------------------------------------

/// Write a `[u32 len][bytes]\0` string (type names, field descriptors and `String` values).
fn write_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u32).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
    out.push(0); // NUL terminator, per the documented `TypeName\0` framing
}

/// Write only a value's *payload* (its type tag/name is written by the caller as a descriptor).
fn write_payload(out: &mut Vec<u8>, v: &Val) {
    match v {
        Val::Null => {}
        Val::Bool(b) => out.push(*b as u8),
        Val::Int(i) => out.extend_from_slice(&i.to_le_bytes()),
        Val::Double(d) => out.extend_from_slice(&d.to_bits().to_le_bytes()),
        Val::Str(s) => write_str(out, s),
        Val::Array(a) => {
            out.extend_from_slice(&(a.len() as u32).to_le_bytes());
            for e in a {
                write_str(out, e.type_tag()); // array elements are self-describing
                write_payload(out, e);
            }
        }
        Val::Object(o) => write_object_body(out, o),
    }
}

/// Write an object body: `[u32 field_count]` then `field_count × [type#name\0][payload]`.
fn write_object_body(out: &mut Vec<u8>, cref: &CRef) {
    out.extend_from_slice(&(cref.values.len() as u32).to_le_bytes());
    for (name, v) in &cref.values {
        let mut desc = String::from(v.type_tag()); // [fieldType#fieldName\0]
        desc.push('#');
        desc.push_str(name);
        write_str(out, &desc);
        write_payload(out, v);
    }
}

/// Encode a `CRef` to the length-prefixed binary framing (`[u32 len][TypeName\0][fields…]`).
pub fn encode(cref: &CRef) -> Vec<u8> {
    let mut body = Vec::new();
    write_str(&mut body, &cref.type_name_tag);
    write_object_body(&mut body, cref);
    let mut out = Vec::with_capacity(4 + body.len());
    out.extend_from_slice(&(body.len() as u32).to_le_bytes()); // [u32 len]
    out.extend_from_slice(&body);
    out
}

/// Decode a `CRef` produced by [`encode`] (round-trips scalars, strings, arrays and objects).
pub fn decode(bytes: &[u8]) -> Result<CRef, &'static str> {
    let mut cur = Cursor { b: bytes, i: 0 };
    let len = cur.u32()? as usize;
    if cur.i + len > bytes.len() {
        return Err("length prefix exceeds buffer");
    }
    let name = cur.str()?; // [TypeName\0]
    decode_object_body(&mut cur, &name)
}

struct Cursor<'a> {
    b: &'a [u8],
    i: usize,
}
impl<'a> Cursor<'a> {
    fn u32(&mut self) -> Result<u32, &'static str> {
        let s = self.b.get(self.i..self.i + 4).ok_or("eof")?;
        self.i += 4;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }
    fn u64(&mut self) -> Result<u64, &'static str> {
        let s = self.b.get(self.i..self.i + 8).ok_or("eof")?;
        self.i += 8;
        Ok(u64::from_le_bytes(s.try_into().unwrap()))
    }
    fn byte(&mut self) -> Result<u8, &'static str> {
        let b = *self.b.get(self.i).ok_or("eof")?;
        self.i += 1;
        Ok(b)
    }
    /// Read a `[u32 len][bytes]\0` string.
    fn str(&mut self) -> Result<String, &'static str> {
        let n = self.u32()? as usize;
        let s = self.b.get(self.i..self.i + n).ok_or("eof")?;
        self.i += n;
        if self.byte()? != 0 {
            return Err("missing NUL terminator");
        }
        String::from_utf8(s.to_vec()).map_err(|_| "invalid utf-8")
    }
}

/// Read a value's payload given its type tag.
fn decode_payload(cur: &mut Cursor, tag: &str) -> Result<Val, &'static str> {
    Ok(match tag {
        "Null" => Val::Null,
        "Bool" => Val::Bool(cur.byte()? != 0),
        "Int64" | "Int" | "UInt" | "Short" | "Char" | "UChar" => Val::Int(cur.u64()? as i64),
        "Double" | "Float" => Val::Double(f64::from_bits(cur.u64()?)),
        "String" => Val::Str(cur.str()?),
        "Array<>" => {
            let n = cur.u32()? as usize;
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                let t = cur.str()?; // per-element tag
                v.push(decode_payload(cur, &t)?);
            }
            Val::Array(v)
        }
        other => Val::Object(decode_object_body(cur, other)?),
    })
}

fn decode_object_body(cur: &mut Cursor, name: &str) -> Result<CRef, &'static str> {
    let count = cur.u32()? as usize;
    let mut cref = CRef::new(name);
    for _ in 0..count {
        let desc = cur.str()?; // [fieldType#fieldName\0]
        let (tag, field) = desc.split_once('#').ok_or("malformed field descriptor")?;
        let v = decode_payload(cur, tag)?;
        cref.values.push((field.to_string(), v));
    }
    Ok(cref)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cref_roundtrip_scalar_and_string() {
        let mut c = CRef::new("CTaskUnfoldParams");
        c.values.push(("Iterations".into(), Val::Int(100)));
        c.values.push(("AngleDistanceMix".into(), Val::Double(1.0)));
        c.values.push(("PinMapName".into(), Val::Str("density".into())));
        let bytes = encode(&c);
        let back = decode(&bytes).unwrap();
        assert_eq!(back, c);
    }

    #[test]
    fn nested_object_and_array_roundtrip() {
        let mut inner = CRef::new("CTaskPackParams");
        inner.values.push(("Scale".into(), Val::Double(0.5)));
        let mut c = CRef::new("ResearchUV");
        c.values.push(("Task".into(), Val::Object(inner.clone())));
        c.values.push(("Ids".into(), Val::Array(vec![Val::Int(0), Val::Int(7)])));
        let back = decode(&encode(&c)).unwrap();
        assert_eq!(back, c);
    }
}
