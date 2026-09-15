//! Packing points, axis-aligned bounding boxes, and affine transforms.
//!
//! [`SVec2`] aliases the UV vector type; [`SBox2`] provides box growth and
//! measurements, while [`STransform2`] applies scale and translation.

use crate::vec::Vec2;

/// Packing point — `SVec2` (`x`/`y` doubles). Alias of [`Vec2`].
pub type SVec2 = Vec2;

/// Axis-aligned box with min/max corners — `SBox2`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct SBox2 {
    pub min: SVec2,
    pub max: SVec2,
}

impl SBox2 {
    #[inline]
    pub fn new(min: SVec2, max: SVec2) -> Self {
        Self { min, max }
    }
    /// `SBox2::Grow` — expand the box so it contains `p`.
    #[inline]
    pub fn grow(&mut self, p: SVec2) {
        self.min.u = self.min.u.min(p.u);
        self.min.v = self.min.v.min(p.v);
        self.max.u = self.max.u.max(p.u);
        self.max.v = self.max.v.max(p.v);
    }
    /// Grow by a symmetric margin (used for the `Margin` packing option).
    #[inline]
    pub fn grow_margin(&mut self, m: f64) {
        self.min = self.min - SVec2::new(m, m);
        self.max = self.max + SVec2::new(m, m);
    }
    #[inline]
    pub fn width(&self) -> f64 {
        self.max.u - self.min.u
    }
    #[inline]
    pub fn height(&self) -> f64 {
        self.max.v - self.min.v
    }
    /// `SBox2` area accessor.
    #[inline]
    pub fn area(&self) -> f64 {
        self.width() * self.height()
    }
    /// `SBox2` center accessor (used to normalize into `[0, 1]²`).
    #[inline]
    pub fn center(&self) -> SVec2 {
        SVec2::new(
            0.5 * (self.min.u + self.max.u),
            0.5 * (self.min.v + self.max.v),
        )
    }
}

/// Affine island transform — `STransform2` (`[f32;4]`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct STransform2 {
    pub m: [f32; 4],
}

impl STransform2 {
    /// Apply the affine to a packed point. Layout `[sx, sy, tx, ty]`.
    pub fn apply(&self, p: SVec2) -> SVec2 {
        SVec2::new(
            f64::from(self.m[0]) * p.u + f64::from(self.m[2]),
            f64::from(self.m[1]) * p.v + f64::from(self.m[3]),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sbox2_grow_and_metrics() {
        let mut b = SBox2::new(SVec2::new(0.0, 0.0), SVec2::new(1.0, 2.0));
        b.grow(SVec2::new(-1.0, 3.0));
        assert_eq!(b.min, SVec2::new(-1.0, 0.0));
        assert_eq!(b.max, SVec2::new(1.0, 3.0));
        assert!((b.area() - 2.0 * 3.0).abs() < 1e-12);
        assert_eq!(b.center(), SVec2::new(0.0, 1.5));
    }

    #[test]
    fn stransform2_apply() {
        let t = STransform2 { m: [2.0, 3.0, 1.0, -1.0] };
        assert_eq!(t.apply(SVec2::new(1.0, 1.0)), SVec2::new(3.0, 2.0));
    }
}
