//! Matrices: `Mat2` (`CMatrix2`, the per-triangle LSCM stiffness element) and `Mat4`.

use crate::vec::Vec2;

/// 2×2 double matrix — `CMatrix2`, stored row-major as `[m00, m01, m10, m11]`.
///
/// `CIsomap::ComputeLs` writes one `CMatrix2` per chart into a
/// `std::vector<CMatrix2>` (the conformal-Laplacian stiffness); see ALGORITHMS.md §2.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Mat2 {
    pub m: [f64; 4],
}

impl Mat2 {
    #[inline]
    pub const fn new(m00: f64, m01: f64, m10: f64, m11: f64) -> Self {
        Self { m: [m00, m01, m10, m11] }
    }
    #[inline]
    pub fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0)
    }
    /// Accumulate a symmetric stiffness contribution `s * [[a, b], [b, c]]`.
    #[inline]
    pub fn add_sym(&mut self, a: f64, b: f64, c: f64, s: f64) {
        self.m[0] += s * a;
        self.m[1] += s * b;
        self.m[2] += s * b;
        self.m[3] += s * c;
    }
    /// Solve `self x = rhs` for the 2×2 system (used by tests / tiny per-chart solves).
    pub fn solve(self, rhs: Vec2) -> Option<Vec2> {
        let [a, b, c, d] = self.m;
        let det = a * d - b * c;
        if det.abs() < f64::EPSILON {
            return None;
        }
        Some(Vec2::new((d * rhs.u - b * rhs.v) / det, (-c * rhs.u + a * rhs.v) / det))
    }
}

/// 4×4 double matrix (view/transform), row-major.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Mat4 {
    pub m: [f64; 16],
}

impl Mat4 {
    pub const IDENTITY: Self = Self {
        m: [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ],
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mat2_solve_identity() {
        let m = Mat2::new(1.0, 0.0, 0.0, 1.0);
        assert_eq!(m.solve(Vec2::new(3.0, -2.0)), Some(Vec2::new(3.0, -2.0)));
    }

    #[test]
    fn mat2_add_sym_accumulates() {
        let mut m = Mat2::zero();
        m.add_sym(1.0, 0.5, 2.0, 4.0);
        assert_eq!(m.m, [4.0, 2.0, 2.0, 8.0]); // symmetric: off-diagonal both += s*b
    }
}
