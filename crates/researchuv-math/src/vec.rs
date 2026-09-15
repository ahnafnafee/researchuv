//! Vectors: `Vec2` (2D UV / packing point) and `Vec3` (`CVector3`).
//!
//! The engine stores per-vertex UVs in an interleaved `CFVector` buffer of `(u, v)`
//! doubles ([ALGORITHMS.md §2.1], docstring `"CFastVector<double> &UV"`); [`Vec2`] is the
//! element type of that buffer. All math here is `f64`, matching the binary's
//! `double`/SSE2 scalar path (the AVX 3-double path is documented as "not yet analysed").

use std::ops::{Add, Div, Mul, Neg, Sub};

/// 2D double vector. Element of the interleaved UV buffer (`CFVector`) and of the
/// Packing `SVec2` value type ([`crate::pack`] re-exports this).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Vec2 {
    pub u: f64,
    pub v: f64,
}

impl Vec2 {
    #[inline]
    pub const fn new(u: f64, v: f64) -> Self {
        Self { u, v }
    }
    /// `CFVector` interleaved layout: read two consecutive doubles.
    #[inline]
    pub fn from_interleaved(buf: &[f64], i: usize) -> Self {
        Self::new(buf[2 * i], buf[2 * i + 1])
    }
    /// `CFVector` interleaved layout: write two consecutive doubles.
    #[inline]
    pub fn write_interleaved(self, buf: &mut [f64], i: usize) {
        buf[2 * i] = self.u;
        buf[2 * i + 1] = self.v;
    }
    /// `cross2d(a, b)` as a scalar (the z component of the 3D cross product).
    #[inline]
    pub fn cross2(self, o: Self) -> f64 {
        self.u * o.v - self.v * o.u
    }
    /// Euclidean length `sqrt(u² + v²)`.
    #[inline]
    pub fn len(self) -> f64 {
        (self.u * self.u + self.v * self.v).sqrt()
    }
}

impl Add for Vec2 {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        Self::new(self.u + o.u, self.v + o.v)
    }
}
impl Sub for Vec2 {
    type Output = Self;
    #[inline]
    fn sub(self, o: Self) -> Self {
        Self::new(self.u - o.u, self.v - o.v)
    }
}
impl Mul<f64> for Vec2 {
    type Output = Self;
    #[inline]
    fn mul(self, s: f64) -> Self {
        Self::new(self.u * s, self.v * s)
    }
}
impl Div<f64> for Vec2 {
    type Output = Self;
    #[inline]
    fn div(self, s: f64) -> Self {
        Self::new(self.u / s, self.v / s)
    }
}
impl Neg for Vec2 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.u, -self.v)
    }
}

/// 3D double vector — `CVector3` (positions and normals).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    #[inline]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
    /// `cross(a−b, c−b)` — the per-triangle geometric factor numerator
    /// ([ALGORITHMS.md §2.3], `andpd …0F135840` select mask + `mulsd …0F86478`).
    #[inline]
    pub fn cross(a: Self, b: Self, c: Self) -> Self {
        (a - b).cross_v(c - b)
    }
    #[inline]
    pub fn dot(self, o: Self) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    #[inline]
    pub fn cross_v(self, o: Self) -> Self {
        Self::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }
    #[inline]
    pub fn len(self) -> f64 {
        self.dot(self).sqrt()
    }
    /// `length² / 2` — the area weight of a chart (1/2 factor @ const `0x140F84DC8`).
    #[inline]
    pub fn half_len_sq(self) -> f64 {
        self.dot(self) * 0.5
    }
}

impl Add for Vec3 {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        Self::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}
impl Sub for Vec3 {
    type Output = Self;
    #[inline]
    fn sub(self, o: Self) -> Self {
        Self::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}
impl Mul<f64> for Vec3 {
    type Output = Self;
    #[inline]
    fn mul(self, s: f64) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }
}
impl Div<f64> for Vec3 {
    type Output = Self;
    #[inline]
    fn div(self, s: f64) -> Self {
        Self::new(self.x / s, self.y / s, self.z / s)
    }
}
impl Neg for Vec3 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec3_cross_matches_reference_formula() {
        // cross(a−b, c−b) for a right triangle at the origin.
        let (a, b, c) = (
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        let n = Vec3::cross(a, b, c);
        assert_eq!(n, Vec3::new(0.0, 0.0, 1.0));
        assert!((n.half_len_sq() - 0.5).abs() < 1e-12); // area of unit right triangle
    }

    #[test]
    fn interleaved_uv_roundtrip() {
        let mut buf = vec![0.0; 4];
        Vec2::new(0.25, 0.75).write_interleaved(&mut buf, 1);
        assert_eq!(Vec2::from_interleaved(&buf, 1), Vec2::new(0.25, 0.75));
    }
}
