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
/// wrap) — already a welded manifold. `R` = major radius, `r` = minor radius.
pub fn torus(R: f64, r: f64, nu: usize, nvt: usize) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    let mut p: Vec<Vec3> = Vec::with_capacity(nu * nvt);
    let mut f: Vec<[u32; 3]> = Vec::with_capacity(2 * nu * nvt);
    for j in 0..nvt {
        let v = 2.0 * std::f64::consts::PI * j as f64 / nvt as f64;
        for i in 0..nu {
            let u = 2.0 * std::f64::consts::PI * i as f64 / nu as f64;
            let cr = R + r * v.cos();
            p.push(Vec3::new(cr * u.cos(), cr * u.sin(), r * v.sin()));
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
pub fn torus_annulus(R: f64, r: f64, nu: usize, nvt: usize) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    let (p, f) = torus(R, r, nu, nvt);
    let jc = nvt / 2;
    let faces: Vec<[u32; 3]> = f
        .into_iter()
        .enumerate()
        .filter(|(k, _)| (*k / 2) % nvt != jc)
        .map(|(_, t)| t)
        .collect();
    (p, faces)
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
}
