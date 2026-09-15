//! Procedural cube, sphere, and torus meshes for examples and tests.
//!
//! Cube faces have independent vertex grids that can be welded together.
//! The sphere duplicates its poles and longitude seam, while the torus uses
//! periodic indexing to form a connected manifold directly.

use researchuv_math::Vec3;

/// Cube as six `(n+1)×(n+1)` face grids: 6·(n+1)² raw vertices, 12·n² triangles.
/// Face definition: `(fixed axis, fixed value, axis-along-i, axis-along-j)`.
/// Returns `(positions, faces)`.
pub fn cube(n: usize) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    let mut p: Vec<Vec3> = Vec::with_capacity(6 * (n + 1) * (n + 1));
    let mut f: Vec<[u32; 3]> = Vec::with_capacity(12 * n * n);
    let face_defs: [(u32, f64, u32, u32); 6] = [
        (2, 0.0, 0, 1),
        (2, 1.0, 0, 1),
        (0, 0.0, 1, 2),
        (0, 1.0, 1, 2),
        (1, 0.0, 0, 2),
        (1, 1.0, 0, 2),
    ];
    for &(fx, fv, va, vb) in &face_defs {
        let base = p.len() as u32;
        for i in 0..=n {
            for j in 0..=n {
                let mut pt = [0.0f64; 3];
                for k in 0..3 {
                    pt[k] = if k == fx as usize {
                        fv
                    } else if k == va as usize {
                        (i as f64) / n as f64
                    } else if k == vb as usize {
                        (j as f64) / n as f64
                    } else {
                        0.0
                    };
                }
                p.push(Vec3::new(pt[0], pt[1], pt[2]));
            }
        }
        for i in 0..n {
            for j in 0..n {
                let a = base + (i as u32) * (n as u32 + 1) + j as u32;
                let b = a + 1;
                let c = a + (n as u32 + 1);
                let d = c + 1;
                f.push([a, b, d]);
                f.push([a, d, c]);
            }
        }
    }
    (p, f)
}

/// UV sphere: `nu × nv` grid over `u ∈ [0, 2π]` and `v ∈ [0, π]`, **both ends
/// included** (the reference `np.linspace(0, 2π, nu)` / `np.linspace(0, π, nv)`),
/// so the pole rings and the `u = 0/2π` seam are duplicated; weld them as in the
/// reference (`weld(make_uv_sphere(48, 24))` → 1036 verts / 2068 tris).
pub fn uv_sphere(nu: usize, nv: usize) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    let mut p: Vec<Vec3> = Vec::with_capacity(nu * nv);
    let mut f: Vec<[u32; 3]> = Vec::with_capacity(2 * nu * (nv - 1));
    let du = 2.0 * std::f64::consts::PI / (nu - 1) as f64;
    let dv = std::f64::consts::PI / (nv - 1) as f64;
    for j in 0..nv {
        let v = dv * j as f64;
        for i in 0..nu {
            let u = du * i as f64;
            p.push(Vec3::new(
                v.sin() * u.cos(),
                v.sin() * u.sin(),
                v.cos(),
            ));
        }
    }
    for i in 0..nu {
        for j in 0..nv - 1 {
            let a = (j * nu + i) as u32;
            let b = (j * nu + (i + 1) % nu) as u32;
            let c = a + nu as u32;
            let d = b + nu as u32;
            f.push([a, c, d]);
            f.push([a, d, b]);
        }
    }
    (p, f)
}

/// Torus: `nu × nvt` periodic grid (both directions `endpoint = false`, modulo
/// wrap) — already a welded manifold. `r_major` = major radius, `r_minor` =
/// minor radius.
pub fn torus(r_major: f64, r_minor: f64, nu: usize, nvt: usize) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    let mut p: Vec<Vec3> = Vec::with_capacity(nu * nvt);
    let mut f: Vec<[u32; 3]> = Vec::with_capacity(2 * nu * nvt);
    for j in 0..nvt {
        let v = 2.0 * std::f64::consts::PI * j as f64 / nvt as f64;
        for i in 0..nu {
            let u = 2.0 * std::f64::consts::PI * i as f64 / nu as f64;
            let cr = r_major + r_minor * v.cos();
            p.push(Vec3::new(cr * u.cos(), cr * u.sin(), r_minor * v.sin()));
        }
    }
    for i in 0..nu {
        for j in 0..nvt {
            let a = (j * nu + i) as u32;
            let b = (j * nu + (i + 1) % nu) as u32;
            let jn = (j + 1) % nvt;
            let c = (jn * nu + i) as u32;
            let d = (jn * nu + (i + 1) % nu) as u32;
            f.push([a, b, d]);
            f.push([a, d, c]);
        }
    }
    (p, f)
}

/// Torus annulus: [`torus`] with the full face ring band at minor index
/// `nvt / 2` (the inner equator) removed across all major angles, leaving a
/// cylinder with two border loops. Face index `k = 2·(i·nvt + j)` (major `i`
/// is the outer loop), so the band is every face with `(k / 2) % nvt == jc`.
pub fn torus_annulus(r_major: f64, r_minor: f64, nu: usize, nvt: usize) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    let (p, f) = torus(r_major, r_minor, nu, nvt);
    let jc = nvt / 2;
    let faces: Vec<[u32; 3]> = f
        .into_iter()
        .enumerate()
        .filter(|(k, _)| (*k / 2) % nvt != jc)
        .map(|(_, t)| t)
        .collect();
    (p, faces)
}

/// Flat grid plane: a `(w × h)`-quad sheet in the xy-plane (2·w·h triangles,
/// `(w+1)·(h+1)` vertices) — a large open single-chart fixture.
pub fn grid(w: usize, h: usize) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    let mut p: Vec<Vec3> = Vec::with_capacity((w + 1) * (h + 1));
    for j in 0..=h {
        for i in 0..=w {
            p.push(Vec3::new(i as f64, j as f64, 0.0));
        }
    }
    let mut f: Vec<[u32; 3]> = Vec::with_capacity(2 * w * h);
    let stride = (w + 1) as u32;
    for j in 0..h {
        for i in 0..w {
            let a = (j as u32) * stride + i as u32;
            let b = a + 1;
            let c = a + stride;
            let d = c + 1;
            f.push([a, b, d]);
            f.push([a, d, c]);
        }
    }
    (p, f)
}

/// Open cylinder: `nu` columns around, `nv` rows up (no caps) — a
/// single-chart fixture with two border loops. Seam vertices are duplicated
/// (welded by the pipeline).
pub fn cylinder(r: f64, height: f64, nu: usize, nv: usize) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    let mut p: Vec<Vec3> = Vec::with_capacity((nu + 1) * (nv + 1));
    for j in 0..=nv {
        let z = height * j as f64 / nv.max(1) as f64;
        for i in 0..=nu {
            let t = 2.0 * std::f64::consts::PI * i as f64 / nu.max(1) as f64;
            p.push(Vec3::new(r * t.cos(), r * t.sin(), z));
        }
    }
    let mut f: Vec<[u32; 3]> = Vec::with_capacity(2 * nu * nv);
    let stride = (nu + 1) as u32;
    for j in 0..nv {
        for i in 0..nu {
            let a = (j as u32) * stride + i as u32;
            let b = a + 1;
            let c = a + stride;
            let d = c + 1;
            f.push([a, b, d]);
            f.push([a, d, c]);
        }
    }
    (p, f)
}


/// A named fixture at subdivision `n` — the shared builder behind the CLI's
/// `--fixture` and the API's `Doc.Fixture`. `name` is one of `cube`,
/// `sphere`, `torus`, `annulus`, `grid`, `cylinder`.
pub fn fixture(name: &str, n: usize) -> Option<(Vec<Vec3>, Vec<[u32; 3]>)> {
    match name {
        "cube" => Some(cube(n)),
        "sphere" => Some(uv_sphere(n.max(8), (n / 2).max(4))),
        "torus" => Some(torus(2.0, 0.7, n * 8, n * 5)),
        "annulus" => Some(torus_annulus(2.0, 0.7, n * 8, n * 5)),
        "grid" => Some(grid(n, n)),
        "cylinder" => Some(cylinder(1.0, 2.0, n * 8, n * 4)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cube_counts_match_the_reference() {
        let (p, f) = cube(6);
        assert_eq!(p.len(), 294); // 6 × 7² raw
        assert_eq!(f.len(), 432); // 12 × 6²
    }

    #[test]
    fn sphere_counts_match_the_reference() {
        let (p, f) = uv_sphere(48, 24);
        // Raw (pre-weld) grid: 48 × 24 verts, 48 × (24-1) × 2 tris. Welding then
        // collapses the pole rings + longitude seam to 1036 verts / 2068 tris
        // (see the parity test).
        assert_eq!(p.len(), 1152);
        assert_eq!(f.len(), 2208);
    }

    #[test]
    fn torus_annulus_drops_one_ring_band() {
        let (p, f) = torus(2.0, 0.7, 64, 40);
        assert_eq!(p.len(), 2560);
        assert_eq!(f.len(), 4992 + 2 * 64); // full torus
        let (p2, f2) = torus_annulus(2.0, 0.7, 64, 40);
        assert_eq!(p2.len(), 2560);
        assert_eq!(f2.len(), 4992);
    }

    #[test]
    fn grid_counts_match() {
        let (p, f) = grid(10, 8);
        assert_eq!(p.len(), 11 * 9);
        assert_eq!(f.len(), 2 * 10 * 8);
        // Distinct positions (no weld needed).
        let mut s = std::collections::BTreeSet::new();
        for q in &p {
            s.insert((q.x.to_bits(), q.y.to_bits(), q.z.to_bits()));
        }
        assert_eq!(s.len(), p.len());
    }

    #[test]
    fn cylinder_welds_to_two_border_loops() {
        let (p, f) = cylinder(1.0, 2.0, 32, 16);
        assert_eq!(p.len(), 33 * 17); // raw, seam duplicated
        let m = crate::weld::weld(p, f, 1e-12);
        assert_eq!(m.positions.len(), 32 * 17); // seam column welded
        assert_eq!(m.faces.len(), 2 * 32 * 16);
        let (charts, _) = crate::segment::segment(&m, 30.0);
        assert_eq!(charts.len(), 1, "cylinder is one chart");
        assert_eq!(charts[0].border_loops.len(), 2, "top and bottom rims");
    }
}
