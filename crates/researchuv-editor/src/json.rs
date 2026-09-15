//! A minimal JSON codec for [`Val`] — the editor's wire format.
//!
//! The engine's typed values map directly onto JSON: `Null`/`Bool`/`Int`/
//! `Double`/`Str`/`Array` map to their JSON kinds and an object maps to a
//! [`CRef`] with the type tag `"Json"` (objects keep insertion order, which
//! the catalog's reply schemas rely on). Non-finite doubles serialize as
//! `null` (JSON has no NaN/Infinity); integers stay integers when they carry
//! no fraction or exponent.

use researchuv_core::val::{CRef, Val};

/// Serialize a [`Val`] to JSON text.
pub fn to_json(v: &Val) -> String {
    let mut out = String::new();
    write_val(&mut out, v);
    out
}

fn write_val(out: &mut String, v: &Val) {
    match v {
        Val::Null => out.push_str("null"),
        Val::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Val::Int(i) => out.push_str(&i.to_string()),
        Val::Double(d) => {
            if d.is_finite() {
                out.push_str(&format!("{d}"));
            } else {
                out.push_str("null");
            }
        }
        Val::Str(s) => write_string(out, s),
        Val::Array(a) => {
            out.push('[');
            for (i, e) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_val(out, e);
            }
            out.push(']');
        }
        Val::Object(o) => {
            out.push('{');
            for (i, (k, e)) in o.values.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_string(out, k);
                out.push(':');
                write_val(out, e);
            }
            out.push('}');
        }
    }
}

fn write_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Parse JSON text into a [`Val`] (objects become `Val::Object` CRefs).
pub fn from_json(text: &str) -> Result<Val, String> {
    let mut p = Parser { b: text.as_bytes(), i: 0 };
    p.ws();
    let v = p.value()?;
    p.ws();
    if p.i != p.b.len() {
        return Err(format!("trailing data at byte {}", p.i));
    }
    Ok(v)
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Parser<'a> {
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }

    fn expect(&mut self, c: u8) -> Result<(), String> {
        if self.peek() == Some(c) {
            self.i += 1;
            Ok(())
        } else {
            Err(format!("expected {:?} at byte {}", c as char, self.i))
        }
    }

    fn lit(&mut self, word: &str, v: Val) -> Result<Val, String> {
        if self.b[self.i..].starts_with(word.as_bytes()) {
            self.i += word.len();
            Ok(v)
        } else {
            Err(format!("invalid literal at byte {}", self.i))
        }
    }

    fn value(&mut self) -> Result<Val, String> {
        match self.peek() {
            Some(b'n') => self.lit("null", Val::Null),
            Some(b't') => self.lit("true", Val::Bool(true)),
            Some(b'f') => self.lit("false", Val::Bool(false)),
            Some(b'"') => Ok(Val::Str(self.string()?)),
            Some(b'[') => {
                self.i += 1;
                let mut out = Vec::new();
                self.ws();
                if self.peek() == Some(b']') {
                    self.i += 1;
                    return Ok(Val::Array(out));
                }
                loop {
                    self.ws();
                    out.push(self.value()?);
                    self.ws();
                    match self.peek() {
                        Some(b',') => self.i += 1,
                        Some(b']') => {
                            self.i += 1;
                            return Ok(Val::Array(out));
                        }
                        _ => return Err(format!("expected , or ] at byte {}", self.i)),
                    }
                }
            }
            Some(b'{') => {
                self.i += 1;
                let mut c = CRef::new("Json");
                self.ws();
                if self.peek() == Some(b'}') {
                    self.i += 1;
                    return Ok(Val::Object(c));
                }
                loop {
                    self.ws();
                    let key = self.string()?;
                    self.ws();
                    self.expect(b':')?;
                    self.ws();
                    let val = self.value()?;
                    c.values.push((key, val));
                    self.ws();
                    match self.peek() {
                        Some(b',') => self.i += 1,
                        Some(b'}') => {
                            self.i += 1;
                            return Ok(Val::Object(c));
                        }
                        _ => return Err(format!("expected , or }} at byte {}", self.i)),
                    }
                }
            }
            Some(c) if c == b'-' || c.is_ascii_digit() => self.number(),
            _ => Err(format!("unexpected byte {} at {}", self.peek().unwrap_or(b'?') as char, self.i)),
        }
    }

    fn string(&mut self) -> Result<String, String> {
        self.expect(b'"')?;
        let mut out = String::new();
        loop {
            match self.peek() {
                None => return Err("unterminated string".into()),
                Some(b'"') => {
                    self.i += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.i += 1;
                    match self.peek() {
                        Some(b'"') => out.push('"'),
                        Some(b'\\') => out.push('\\'),
                        Some(b'/') => out.push('/'),
                        Some(b'b') => out.push('\u{8}'),
                        Some(b'f') => out.push('\u{c}'),
                        Some(b'n') => out.push('\n'),
                        Some(b'r') => out.push('\r'),
                        Some(b't') => out.push('\t'),
                        Some(b'u') => {
                            self.i += 1;
                            let hex = self
                                .b
                                .get(self.i..self.i + 4)
                                .ok_or("truncated \\u escape")?;
                            let s = std::str::from_utf8(hex).map_err(|_| "bad \\u escape")?;
                            let cp =
                                u32::from_str_radix(s, 16).map_err(|_| "bad \\u escape")?;
                            self.i += 3; // +1 below finishes the 4 hex digits
                            out.push(char::from_u32(cp).unwrap_or('\u{fffd}'));
                        }
                        _ => return Err(format!("bad escape at byte {}", self.i)),
                    }
                    self.i += 1;
                }
                Some(_) => {
                    // Consume one UTF-8 code point.
                    let start = self.i;
                    let len = utf8_len(self.b[start]);
                    let end = (start + len).min(self.b.len());
                    let s = std::str::from_utf8(&self.b[start..end])
                        .map_err(|_| "invalid utf-8 in string")?;
                    out.push_str(s);
                    self.i = end;
                }
            }
        }
    }

    fn number(&mut self) -> Result<Val, String> {
        let start = self.i;
        if self.peek() == Some(b'-') {
            self.i += 1;
        }
        let mut is_double = false;
        while let Some(c) = self.peek() {
            match c {
                b'0'..=b'9' => self.i += 1,
                b'.' | b'e' | b'E' | b'+' | b'-' => {
                    is_double = true;
                    self.i += 1;
                }
                _ => break,
            }
        }
        let text = std::str::from_utf8(&self.b[start..self.i]).map_err(|_| "bad number")?;
        if text.is_empty() || text == "-" {
            return Err(format!("bad number at byte {start}"));
        }
        if is_double {
            text.parse::<f64>()
                .map(Val::Double)
                .map_err(|e| format!("bad number {text:?}: {e}"))
        } else {
            text.parse::<i64>()
                .map(Val::Int)
                .map_err(|e| format!("bad number {text:?}: {e}"))
        }
    }
}

fn utf8_len(b: u8) -> usize {
    match b {
        0x00..=0x7F => 1,
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        _ => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalars_roundtrip() {
        for (v, text) in [
            (Val::Null, "null"),
            (Val::Bool(true), "true"),
            (Val::Bool(false), "false"),
            (Val::Int(-42), "-42"),
            (Val::Double(1.5), "1.5"),
            (Val::Str("hi".into()), "\"hi\""),
        ] {
            assert_eq!(to_json(&v), text);
            assert_eq!(from_json(text).unwrap(), v);
        }
    }

    #[test]
    fn nested_roundtrip() {
        let mut inner = CRef::new("Json");
        inner.values.push(("b".into(), Val::Bool(true)));
        let mut o = CRef::new("Json");
        o.values.push(("i".into(), Val::Int(7)));
        o.values.push(("s".into(), Val::Str("a\"b\\c\n".into())));
        o.values.push(("arr".into(), Val::Array(vec![Val::Int(1), Val::Double(0.5), Val::Null])));
        o.values.push(("obj".into(), Val::Object(inner)));
        let v = Val::Object(o);
        let text = to_json(&v);
        assert_eq!(from_json(&text).unwrap(), v);
    }

    #[test]
    fn escapes_and_unicode() {
        let v = Val::Str("日本語 — émoji 🎈".into());
        let back = from_json(&to_json(&v)).unwrap();
        assert_eq!(back, v);
        // \u escapes decode.
        assert_eq!(from_json("\"\\u0041\"").unwrap(), Val::Str("A".into()));
    }

    #[test]
    fn non_finite_doubles_become_null() {
        assert_eq!(to_json(&Val::Double(f64::NAN)), "null");
        assert_eq!(to_json(&Val::Double(f64::INFINITY)), "null");
    }

    #[test]
    fn malformed_json_is_rejected() {
        for bad in ["", "{", "[1,", "\"unterminated", "{\"a\":}", "1 2", "nul"] {
            assert!(from_json(bad).is_err(), "{bad:?} must not parse");
        }
    }

    #[test]
    fn integers_stay_integers() {
        assert_eq!(from_json("3").unwrap(), Val::Int(3));
        assert_eq!(from_json("-3").unwrap(), Val::Int(-3));
        assert_eq!(from_json("3.0").unwrap(), Val::Double(3.0));
        assert_eq!(from_json("1e2").unwrap(), Val::Double(100.0));
    }
}
