//! Mesh import/export — OBJ and STL readers/writers plus an atlas SVG
//! renderer, operating on the raw `(positions, faces)` soup the pipeline
//! consumes and the per-face UV triplets it produces.
//!
//! The OBJ reader accepts `v`/`f` records (fan triangulation for polygons,
//! 1-based and negative indices, `v/x/y/z`-style face qualifiers dropped) and
//! skips everything else (`vt`, `vn`, groups, materials, comments). The STL
//! reader auto-detects ASCII vs binary. Writers emit the common denominator
//! (positions, optional per-face UVs for OBJ; binary STL for the triangle
//! soup).

use crate::model::SurfaceMesh;
use researchuv_math::{Vec2, Vec3};

/// An import/export failure.
#[derive(Clone, Debug)]
pub enum IoError {
    /// A syntax problem in the source text.
    Parse { line: usize, msg: String },
    /// Binary framing/size mismatch.
    Format(String),
    /// The file could not be read or written.
    Io(String),
}

impl std::fmt::Display for IoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IoError::Parse { line, msg } => write!(f, "parse error at line {line}: {msg}"),
            IoError::Format(msg) => write!(f, "format error: {msg}"),
            IoError::Io(msg) => write!(f, "io error: {msg}"),
        }
    }
}

impl std::error::Error for IoError {}

impl From<std::io::Error> for IoError {
    fn from(e: std::io::Error) -> Self {
        IoError::Io(e.to_string())
    }
}

/// Parse an OBJ document into a triangle soup (fan triangulation).
pub fn parse_obj(text: &str) -> Result<(Vec<Vec3>, Vec<[u32; 3]>), IoError> {
    let mut positions: Vec<Vec3> = Vec::new();
    let mut faces: Vec<[u32; 3]> = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let line_no = i + 1;
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        let kind = parts.next().unwrap_or("");
        match kind {
            "v" => {
                let coords: Vec<f64> = parts
                    .take(3)
                    .map(|t| {
                        t.parse::<f64>().map_err(|_| IoError::Parse {
                            line: line_no,
                            msg: format!("bad vertex coordinate {t:?}"),
                        })
                    })
                    .collect::<Result<_, _>>()?;
                positions.push(Vec3::new(
                    coords.first().copied().unwrap_or(0.0),
                    coords.get(1).copied().unwrap_or(0.0),
                    coords.get(2).copied().unwrap_or(0.0),
                ));
            }
            "f" => {
                let idx: Vec<i64> = parts
                    .map(|t| {
                        // Face records may carry v/vt/vn qualifiers; the
                        // vertex index is the first slash-separated field.
                        let first = t.split('/').next().unwrap_or(t);
                        first.parse::<i64>().map_err(|_| IoError::Parse {
                            line: line_no,
                            msg: format!("bad face index {t:?}"),
                        })
                    })
                    .collect::<Result<_, _>>()?;
                if idx.len() < 3 {
                    return Err(IoError::Parse {
                        line: line_no,
                        msg: format!("face with {} vertices", idx.len()),
                    });
                }
                let resolve = |v: i64| -> Result<u32, IoError> {
                    let n = positions.len() as i64;
                    let r = if v < 0 { n + v } else { v - 1 };
                    if r < 0 || r >= n {
                        return Err(IoError::Parse {
                            line: line_no,
                            msg: format!("face index {v} out of range ({n} vertices)"),
                        });
                    }
                    Ok(r as u32)
                };
                let a = resolve(idx[0])?;
                // Fan triangulation: (v0, vk, vk+1) for k = 1..n−1.
                for k in 1..idx.len() - 1 {
                    faces.push([a, resolve(idx[k])?, resolve(idx[k + 1])?]);
                }
            }
            _ => {
                // vt/vn/o/g/usemtl/mtllib/s/l/… — ignored.
            }
        }
    }
    Ok((positions, faces))
}

/// Read and parse an OBJ file from disk.
pub fn read_obj(path: &std::path::Path) -> Result<(Vec<Vec3>, Vec<[u32; 3]>), IoError> {
    let text = std::fs::read_to_string(path)?;
    parse_obj(&text)
}

/// Build an OBJ document. `face_uvs` (when present) carries one UV triplet
/// per face; each distinct UV is emitted once as `vt` and faces reference
/// `v/vt` pairs. UVs are quantized to 9 decimals for deduplication.
pub fn write_obj(
    mesh: &SurfaceMesh,
    face_uvs: Option<&[[Vec2; 3]]>,
) -> Result<String, IoError> {
    if let Some(uvs) = face_uvs {
        if uvs.len() != mesh.faces.len() {
            return Err(IoError::Format(format!(
                "face_uvs has {} entries but the mesh has {} faces",
                uvs.len(),
                mesh.faces.len()
            )));
        }
    }
    let mut out = String::with_capacity(64 * mesh.positions.len());
    out.push_str("# ResearchUV export\n");
    for p in &mesh.positions {
        out.push_str(&format!("v {:.9} {:.9} {:.9}\n", p.x, p.y, p.z));
    }
    let mut vt_ids: Vec<[u32; 3]> = Vec::new();
    if let Some(uvs) = face_uvs {
        let mut lookup = std::collections::BTreeMap::new();
        let mut verts: Vec<Vec2> = Vec::new();
        let key = |p: Vec2| format!("{:.9},{:.9}", p.u, p.v);
        for tri in uvs {
            let mut ids = [0u32; 3];
            for (k, &p) in tri.iter().enumerate() {
                let kx = key(p);
                let next = match lookup.get(&kx) {
                    Some(&id) => id,
                    None => {
                        let id = verts.len() as u32 + 1; // 1-based
                        verts.push(p);
                        lookup.insert(kx, id);
                        id
                    }
                };
                ids[k] = next;
            }
            vt_ids.push(ids);
        }
        for t in &verts {
            out.push_str(&format!("vt {:.9} {:.9}\n", t.u, t.v));
        }
        for (f, ids) in mesh.faces.iter().zip(vt_ids.iter()) {
            out.push_str(&format!(
                "f {}/{} {}/{} {}/{}\n",
                f[0] + 1, ids[0], f[1] + 1, ids[1], f[2] + 1, ids[2]
            ));
        }
    } else {
        for f in &mesh.faces {
            out.push_str(&format!("f {} {} {}\n", f[0] + 1, f[1] + 1, f[2] + 1));
        }
    }
    Ok(out)
}

/// Write an OBJ file to disk.
pub fn write_obj_file(
    path: &std::path::Path,
    mesh: &SurfaceMesh,
    face_uvs: Option<&[[Vec2; 3]]>,
) -> Result<(), IoError> {
    std::fs::write(path, write_obj(mesh, face_uvs)?)?;
    Ok(())
}

/// Parse an STL file (auto-detects ASCII vs binary).
pub fn parse_stl(bytes: &[u8]) -> Result<(Vec<Vec3>, Vec<[u32; 3]>), IoError> {
    if looks_like_ascii_stl(bytes) {
        parse_stl_ascii(std::str::from_utf8(bytes).map_err(|_| IoError::Format(
            "ascii STL is not valid UTF-8".into(),
        ))?)
    } else {
        parse_stl_binary(bytes)
    }
}

/// The 80-byte binary header is free-form text; a file whose first token is
/// `solid` *and* that contains a `facet` keyword soon after is ASCII (a
/// binary header may also start with "solid").
fn looks_like_ascii_stl(bytes: &[u8]) -> bool {
    let head: Vec<u8> = bytes.iter().copied().take(512).collect();
    let head = String::from_utf8_lossy(&head).to_ascii_lowercase();
    if !head.trim_start().starts_with("solid") {
        return false;
    }
    head.contains("facet")
}

fn parse_stl_ascii(text: &str) -> Result<(Vec<Vec3>, Vec<[u32; 3]>), IoError> {
    let mut positions: Vec<Vec3> = Vec::new();
    let mut faces: Vec<[u32; 3]> = Vec::new();
    let mut current: Vec<Vec3> = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let line_no = i + 1;
        let mut parts = raw.trim().split_whitespace();
        let kind = parts.next().unwrap_or("");
        match kind {
            "vertex" => {
                let coords: Vec<f64> = parts
                    .take(3)
                    .map(|t| {
                        t.parse::<f64>().map_err(|_| IoError::Parse {
                            line: line_no,
                            msg: format!("bad vertex coordinate {t:?}"),
                        })
                    })
                    .collect::<Result<_, _>>()?;
                current.push(Vec3::new(
                    coords.first().copied().unwrap_or(0.0),
                    coords.get(1).copied().unwrap_or(0.0),
                    coords.get(2).copied().unwrap_or(0.0),
                ));
            }
            "endfacet" => {
                if current.len() != 3 {
                    return Err(IoError::Parse {
                        line: line_no,
                        msg: format!("facet with {} vertices", current.len()),
                    });
                }
                let base = positions.len() as u32;
                positions.extend(current.drain(..));
                faces.push([base, base + 1, base + 2]);
            }
            _ => {}
        }
    }
    Ok((positions, faces))
}

fn parse_stl_binary(bytes: &[u8]) -> Result<(Vec<Vec3>, Vec<[u32; 3]>), IoError> {
    // [80-byte header][u32 count][50 bytes per facet]
    if bytes.len() < 84 {
        return Err(IoError::Format("binary STL shorter than its header".into()));
    }
    let count = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;
    if bytes.len() < 84 + 50 * count {
        return Err(IoError::Format(format!(
            "binary STL truncated: {count} facets need {} bytes, file has {}",
            84 + 50 * count,
            bytes.len()
        )));
    }
    let mut positions: Vec<Vec3> = Vec::with_capacity(3 * count);
    let mut faces: Vec<[u32; 3]> = Vec::with_capacity(count);
    let read_f32 = |b: &[u8]| f64::from(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
    for f in 0..count {
        let base = 84 + 50 * f + 12; // skip the facet normal
        let mut tri = [0u32; 3];
        for k in 0..3 {
            let o = base + 12 * k;
            positions.push(Vec3::new(
                read_f32(&bytes[o..o + 4]),
                read_f32(&bytes[o + 4..o + 8]),
                read_f32(&bytes[o + 8..o + 12]),
            ));
            tri[k] = positions.len() as u32 - 1;
        }
        faces.push(tri);
    }
    Ok((positions, faces))
}

/// Read and parse an STL file from disk.
pub fn read_stl(path: &std::path::Path) -> Result<(Vec<Vec3>, Vec<[u32; 3]>), IoError> {
    parse_stl(&std::fs::read(path)?)
}

/// Encode the triangle soup as a binary STL (normals computed per facet).
pub fn write_stl(mesh: &SurfaceMesh) -> Vec<u8> {
    let mut out = Vec::with_capacity(84 + 50 * mesh.faces.len());
    let mut header = [0u8; 80];
    let title = b"ResearchUV binary STL";
    header[..title.len()].copy_from_slice(title);
    out.extend_from_slice(&header);
    out.extend_from_slice(&(mesh.faces.len() as u32).to_le_bytes());
    for f in &mesh.faces {
        let [a, b, c] = *f;
        let n = Vec3::cross(mesh.positions[a as usize], mesh.positions[b as usize], mesh.positions[c as usize]);
        let l = n.len();
        let unit = if l > 1e-30 { n / l } else { Vec3::new(0.0, 0.0, 0.0) };
        for v in [unit.x, unit.y, unit.z] {
            out.extend_from_slice(&(v as f32).to_le_bytes());
        }
        for v in [a, b, c] {
            let p = mesh.positions[v as usize];
            for c in [p.x, p.y, p.z] {
                out.extend_from_slice(&(c as f32).to_le_bytes());
            }
        }
        out.extend_from_slice(&[0, 0]); // attribute byte count
    }
    out
}

/// Write a binary STL file to disk.
pub fn write_stl_file(path: &std::path::Path, mesh: &SurfaceMesh) -> Result<(), IoError> {
    std::fs::write(path, write_stl(mesh))?;
    Ok(())
}

/// Per-island UV triangles of a finished atlas — the input shape
/// [`atlas_svg`] renders.
pub fn atlas_tris(islands: &[crate::model::Island]) -> Vec<Vec<[Vec2; 3]>> {
    islands
        .iter()
        .map(|isl| {
            isl.tris
                .iter()
                .map(|[a, b, c]| [isl.uv[*a as usize], isl.uv[*b as usize], isl.uv[*c as usize]])
                .collect()
        })
        .collect()
}

/// Render an atlas as an SVG document: one colored triangle group per
/// island. `island_tris[i]` is island `i`'s triangles as UV triplets.
pub fn atlas_svg(island_tris: &[Vec<[Vec2; 3]>], size_px: f64) -> String {
    let colors = [
        "#a7c5b0", "#e8ac87", "#ebcb83", "#98b7c8", "#c5b1a0", "#bfc795",
        "#d8b4c4", "#a8c5a0", "#c9b88a", "#9fb8c8",
    ];
    let pad = 24.0;
    let view = size_px + 2.0 * pad;
    let mut svg = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{view:.0}\" height=\"{view:.0}\" viewBox=\"0 0 {view:.0} {view:.0}\">\n\
         <rect width=\"{view:.0}\" height=\"{view:.0}\" fill=\"#f8f3e9\"/>\n\
         <rect x=\"{pad:.0}\" y=\"{pad:.0}\" width=\"{size_px:.0}\" height=\"{size_px:.0}\" fill=\"#fffdf8\" stroke=\"#bbc4b8\"/>\n"
    );
    for (i, tris) in island_tris.iter().enumerate() {
        let color = colors[i % colors.len()];
        svg.push_str(&format!(
            "<g fill=\"{color}\" stroke=\"#304d43\" stroke-width=\"0.6\" stroke-linejoin=\"round\">\n"
        ));
        for tri in tris {
            let pts: Vec<String> = tri
                .iter()
                .map(|p| {
                    format!(
                        "{:.2},{:.2}",
                        pad + size_px * p.u.clamp(0.0, 1.0),
                        pad + size_px * (1.0 - p.v.clamp(0.0, 1.0))
                    )
                })
                .collect();
            svg.push_str(&format!("<polygon points=\"{}\"/>\n", pts.join(" ")));
        }
        svg.push_str("</g>\n");
    }
    svg.push_str("</svg>\n");
    svg
}

/// Write the atlas SVG to disk.
pub fn write_atlas_svg_file(
    path: &std::path::Path,
    island_tris: &[Vec<[Vec2; 3]>],
    size_px: f64,
) -> Result<(), IoError> {
    std::fs::write(path, atlas_svg(island_tris, size_px))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn obj_roundtrip_with_uvs() {
        let mesh = SurfaceMesh::from_triangles(
            vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            vec![[0, 1, 2]],
        );
        let uvs = [[Vec2::new(0.1, 0.1), Vec2::new(0.9, 0.1), Vec2::new(0.1, 0.9)]];
        let text = write_obj(&mesh, Some(&uvs)).unwrap();
        assert!(text.contains("vt 0.1"));
        // Re-parse: vertices + faces come back (vt is skipped by the reader).
        let (p, f) = parse_obj(&text).unwrap();
        assert_eq!(p.len(), 3);
        assert_eq!(f, vec![[0, 1, 2]]);
    }

    #[test]
    fn obj_parses_polygons_negative_indices_and_comments() {
        let text = "# comment\n\
                    v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n\
                    f 1/1/1 2/2/2 3/3/3 4/4/4\n\
                    f -4 -3 -2\n";
        let (p, f) = parse_obj(text).unwrap();
        assert_eq!(p.len(), 4);
        // Quad → 2 triangles (fan), then one more triangle from negatives.
        assert_eq!(f, vec![[0, 1, 2], [0, 2, 3], [0, 1, 2]]);
    }

    #[test]
    fn obj_bad_index_is_a_parse_error() {
        let text = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 9\n";
        assert!(matches!(parse_obj(text), Err(IoError::Parse { .. })));
    }

    #[test]
    fn binary_stl_roundtrip() {
        let mesh = SurfaceMesh::from_triangles(
            vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            vec![[0, 1, 2]],
        );
        let bytes = write_stl(&mesh);
        assert_eq!(bytes.len(), 84 + 50);
        let (p, f) = parse_stl(&bytes).unwrap();
        assert_eq!(p.len(), 3);
        assert_eq!(f, vec![[0, 1, 2]]);
        assert!((p[1].x - 1.0).abs() < 1e-6);
    }

    #[test]
    fn ascii_stl_detected_and_parsed() {
        let text = "solid test\n\
                    facet normal 0 0 1\n\
                    outer loop\n\
                    vertex 0 0 0\n\
                    vertex 1 0 0\n\
                    vertex 0 1 0\n\
                    endloop\n\
                    endfacet\n\
                    endsolid test\n";
        let (p, f) = parse_stl(text.as_bytes()).unwrap();
        assert_eq!(p.len(), 3);
        assert_eq!(f, vec![[0, 1, 2]]);
    }

    #[test]
    fn truncated_binary_stl_is_an_error() {
        let mesh = SurfaceMesh::from_triangles(
            vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            vec![[0, 1, 2], [0, 1, 2]],
        );
        let mut bytes = write_stl(&mesh);
        bytes.truncate(bytes.len() - 10);
        assert!(matches!(parse_stl(&bytes), Err(IoError::Format(_))));
    }

    #[test]
    fn face_uv_mismatch_is_an_error() {
        let mesh = SurfaceMesh::from_triangles(
            vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            vec![[0, 1, 2]],
        );
        let uvs: [[Vec2; 3]; 0] = [];
        assert!(write_obj(&mesh, Some(&uvs)).is_err());
    }

    #[test]
    fn svg_contains_island_groups() {
        let tris = vec![vec![[
            Vec2::new(0.1, 0.1),
            Vec2::new(0.4, 0.1),
            Vec2::new(0.1, 0.4),
        ]]];
        let svg = atlas_svg(&tris, 400.0);
        assert!(svg.contains("<polygon"));
        assert!(svg.contains("</svg>"));
    }
}
