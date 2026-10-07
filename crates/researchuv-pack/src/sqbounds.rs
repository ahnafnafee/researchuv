//! Certified bounds for the equal-square packing problem — the reference
//! data behind the optimality-gap benchmarks and the packer's quality
//! report.
//!
//! The problem: `s(n)` is the side of the smallest square that contains `n`
//! non-overlapping unit squares (arbitrary rotation allowed). Exact values
//! are known only for small `n`; elsewhere the best that exists is a pair of
//! certified bounds — a *lower* bound (no packing of `n` unit squares fits in
//! a smaller square; proven) and a *best-known upper* bound (an explicit
//! packing achieves it).
//!
//! Sources, with gratitude:
//!
//! - **The squares project** (Joshua Levy, 2026, CC BY 4.0) —
//!   <https://jlevy.github.io/squares/>, data register
//!   `packing/frontier/results.yaml` in <https://github.com/jlevy/squares>.
//!   Cited below by register id (`T-xxx`). The register's verification rungs
//!   (V0–V5) grade how independently a claim has been machine-checked; every
//!   bound imported here is rung V3 (machine-checked with a replayable
//!   certificate) or better.
//! - Classical proven values for `n ≤ 13` (see *Square packing in a square*,
//!   Wikipedia, and the survey literature it summarizes). `s(5) = 2 + √2/2`
//!   and `s(10) = 3 + √2/2` are the two smallest counts whose optima need
//!   tilted squares; `s(11) = 3.87708359…` (Trump's 1979 packing) was proved
//!   optimal in 2026 by a Lean 4 formalization
//!   (<https://github.com/Queuingtheorydotcom/11SquaresFormalized>).
//!
//! The numbers are mathematical facts; the curated table reproduces them
//! with per-entry source strings so reports can cite where each bound came
//! from. Data from the squares project register is used under CC BY 4.0:
//! attribute "Joshua Levy, the squares project
//! (<https://github.com/jlevy/squares>)".

use std::f64::consts::SQRT_2;

/// One curated per-`n` record: certified lower side, best-known upper side,
/// and whether the two coincide (the value is proven optimal).
#[derive(Clone, Copy, Debug)]
pub struct SquarePackingBound {
    pub n: u32,
    /// No packing of `n` unit squares fits in a square smaller than this
    /// (certified).
    pub lower: f64,
    /// Some explicit packing achieves this side (or it is the trivial grid).
    pub upper: f64,
    /// `lower == upper`: the value is proven optimal.
    pub exact: bool,
    /// Where the numbers come from (for report citations).
    pub source: &'static str,
}

/// `2 + √2/2` — the proven optimum for five unit squares.
pub const S5: f64 = 2.0 + SQRT_2 / 2.0;
/// `3 + √2/2` — the proven optimum for ten unit squares.
pub const S10: f64 = 3.0 + SQRT_2 / 2.0;
/// Trump's 1979 packing of eleven unit squares, proved optimal in 2026
/// (Lean 4 formalization; the exact value is algebraic — a rational function
/// of a degree-8 polynomial root).
pub const S11: f64 = 3.877_083_590_022_814_2;

/// The curated register of certified bounds as raw tuples
/// `(n, lower, upper, exact, source)` (sorted by `n`). Values not listed
/// here still get bounds via the closed-form family bound and the trivial
/// constructions in [`bound`] — see that function.
static TABLE: &[(u32, f64, f64, bool, &str)] = &[
    (1u32, 1.0, 1.0, true, "trivial"),
    (2, 2.0, 2.0, true, "trivial"),
    (3, 2.0, 2.0, true, "classical (proven)"),
    (4, 2.0, 2.0, true, "trivial 2×2 grid"),
    (5, S5, S5, true, "classical (proven; first tilted optimum)"),
    (6, 3.0, 3.0, true, "classical (proven)"),
    (7, 3.0, 3.0, true, "classical (proven)"),
    (8, 3.0, 3.0, true, "classical (proven)"),
    (9, 3.0, 3.0, true, "trivial 3×3 grid"),
    (10, S10, S10, true, "classical (proven; tilted optimum)"),
    (
        11,
        S11,
        S11,
        true,
        "Trump 1979 (T-011); optimality proved 2026, Lean 4 (11SquaresFormalized); \
         independent certified lower T-061 = 3.875000003875",
    ),
    (
        12,
        15_680_000.0 / 3_949_423.0,
        4.0,
        false,
        "T-079 lower (15680000/3949423) / 3×4 grid upper",
    ),
    (13, 4.0, 4.0, true, "T-006 (Bentz 2010), s(k²−3) = k family"),
    (15, 4.0, 4.0, true, "T-083 family bound meets the 4×4 grid"),
    (16, 4.0, 4.0, true, "trivial 4×4 grid"),
    (
        17,
        461_300.0 / 99_853.0,
        4.675_530_093_604_551,
        false,
        "T-038 lower (461300/99853) / T-065 upper (Bidwell packing)",
    ),
    (
        18,
        22_529.0 / 5_000.0,
        4.822_876,
        false,
        "T-016 lower (22529/5000) / squares-project atlas upper",
    ),
    (
        19,
        22_529.0 / 5_000.0,
        4.885_618,
        false,
        "T-016 lower (by monotonicity) / squares-project atlas upper",
    ),
    (21, 5_000.0 / 1_001.0, 5.0, false, "T-050 lower (5000/1001) / 5×5 grid upper"),
    (22, 5.0, 5.0, true, "s(k²−3) = k family (T-008)"),
    (25, 5.0, 5.0, true, "trivial 5×5 grid"),
    (26, 1_377.0 / 250.0, 6.0, false, "T-047 lower (1377/250) / 6×6 grid upper"),
    (27, 1_377.0 / 250.0, 6.0, false, "T-047 lower (1377/250) / 6×6 grid upper"),
    (28, 1_377.0 / 250.0, 6.0, false, "T-047 lower (1377/250) / 6×6 grid upper"),
    (
        29,
        571.0 / 100.0,
        5.933_833_462_676_929,
        false,
        "T-047 lower (571/100) / T-009 upper (Krawczyk interval certificate)",
    ),
    (30, 571.0 / 100.0, 6.0, false, "T-047 lower (571/100) / 6×6 grid upper"),
    (31, 571.0 / 100.0, 6.0, false, "T-047 lower (571/100) / 6×6 grid upper"),
    (33, 6.0, 6.0, true, "s(k²−3) = k family (T-008)"),
    (36, 6.0, 6.0, true, "trivial 6×6 grid"),
    (46, 7.0, 7.0, true, "T-008, s(k²−3) = k family"),
    (49, 7.0, 7.0, true, "trivial 7×7 grid"),
    (50, 37.0 / 5.0, 8.0, false, "T-048 lower (37/5) / 8×8 grid upper"),
];

/// The curated record for `n`, when the register has one.
fn table_entry(n: u32) -> Option<SquarePackingBound> {
    TABLE
        .iter()
        .find(|t| t.0 == n)
        .map(|&(n, lower, upper, exact, source)| SquarePackingBound { n, lower, upper, exact, source })
}

/// The closed-form certified family lower bound (squares project `T-083`,
/// verification rung V3): for every nonsquare `n` with `8 ≤ n ≤ 324`,
/// `s(n) ≥ 1/2 + √(n − ⌊√n⌋ + 1/4)`.
pub fn family_lower_bound(n: u32) -> Option<f64> {
    let r = (n as f64).sqrt();
    if r.fract() == 0.0 || !(8..=324).contains(&n) {
        return None;
    }
    Some(0.5 + (n as f64 - r.floor() + 0.25).sqrt())
}

/// The strongest certified lower bound for `s(n)` from the curated table,
/// the closed-form family bound, and the area bound `√n`.
pub fn certified_lower_bound(n: u32) -> f64 {
    let mut best = (n as f64).sqrt(); // area bound: n unit squares need area n
    if let Some(f) = family_lower_bound(n) {
        best = best.max(f);
    }
    if let Some(t) = table_entry(n) {
        best = best.max(t.lower);
    }
    best
}

/// The best-known upper bound for `s(n)` from the curated table, never above
/// the trivial `⌈√n⌉` grid construction.
pub fn best_known_upper_bound(n: u32) -> f64 {
    let grid = (n as f64).sqrt().ceil();
    match table_entry(n) {
        Some(t) => t.upper.min(grid),
        None => grid,
    }
}

/// Is `s(n)` proven optimal (lower bound meets upper bound)?
pub fn proven_optimal(n: u32) -> bool {
    table_entry(n).map(|t| t.exact).unwrap_or(false)
}

/// The assembled bound record for `n` (table entry when present, otherwise
/// the family/area lower bound and the grid upper bound).
pub fn bound(n: u32) -> SquarePackingBound {
    if let Some(t) = table_entry(n) {
        return t;
    }
    let lower = certified_lower_bound(n);
    let upper = best_known_upper_bound(n);
    SquarePackingBound {
        n,
        lower,
        upper,
        exact: (upper - lower).abs() < 1e-12,
        source: if family_lower_bound(n).is_some() {
            "T-083 closed-form family bound / trivial grid"
        } else {
            "area bound √n / trivial grid"
        },
    }
}

/// The certified ceiling on atlas utilization for `n` equal square islands:
/// unit squares at uniform scale `σ` fit in the unit box only when
/// `σ ≤ 1/s(n)`, so the utilization `n·σ²` can never exceed `n/s(n)²`.
/// Certified means computed from the *lower* bound on `s(n)`.
pub fn certified_max_utilization(n: u32) -> f64 {
    let s = certified_lower_bound(n);
    if s <= 0.0 {
        return f64::INFINITY;
    }
    n as f64 / (s * s)
}

/// The default `n` counts exercised by the optimality-gap benchmark: the
/// proven tilted optima, the open-gap register entries, and a few exact
/// sanity anchors.
pub const BENCHMARK_N: &[u32] = &[2, 3, 5, 6, 8, 10, 11, 12, 13, 17, 18, 19, 21, 22, 29, 33, 50];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_is_sorted_and_consistent() {
        for w in TABLE.windows(2) {
            assert!(w[0].0 < w[1].0, "table must be sorted by n");
        }
        for &(n, lower, upper, exact, _) in TABLE.iter() {
            assert!(lower > 0.0 && upper >= lower - 1e-12, "n={n}");
            if exact {
                assert!((upper - lower).abs() < 1e-9, "n={n}");
            }
        }
    }

    #[test]
    fn classical_values() {
        assert!((certified_lower_bound(1) - 1.0).abs() < 1e-12);
        assert!((certified_lower_bound(4) - 2.0).abs() < 1e-12);
        assert!((certified_lower_bound(5) - S5).abs() < 1e-12);
        assert!((S5 - 2.707_106_781_186_547_6).abs() < 1e-12);
        assert!((S10 - 3.707_106_781_186_547_6).abs() < 1e-12);
        assert!((certified_lower_bound(9) - 3.0).abs() < 1e-12);
        assert!((certified_lower_bound(10) - S10).abs() < 1e-12);
        assert!(proven_optimal(11));
        assert!((certified_lower_bound(11) - S11).abs() < 1e-9);
        assert!((S11 - 3.877_083_590_022_814_2).abs() < 1e-12);
        assert!(!proven_optimal(12));
        assert!((certified_lower_bound(13) - 4.0).abs() < 1e-12);
    }

    #[test]
    fn family_bound_shape() {
        // Nonsquare 8..=324 only; squares are excluded.
        assert!(family_lower_bound(9).is_none());
        assert!(family_lower_bound(16).is_none());
        assert!(family_lower_bound(325).is_none());
        assert!(family_lower_bound(5).is_none());
        let f8 = family_lower_bound(8).unwrap();
        assert!((f8 - 3.0).abs() < 1e-12, "s(8) = 3 exactly");
        let f15 = family_lower_bound(15).unwrap();
        assert!((f15 - 4.0).abs() < 1e-12, "s(15) = 4 exactly");
        // Monotone-ish sanity at a few counts: never below the area bound.
        for n in [10u32, 17, 24, 35, 50, 99] {
            assert!(certified_lower_bound(n) >= (n as f64).sqrt() - 1e-12);
        }
    }

    #[test]
    fn upper_never_above_the_trivial_grid() {
        for n in 1u32..=60 {
            let b = bound(n);
            assert!(b.upper <= (n as f64).sqrt().ceil() + 1e-12);
            assert!(b.lower <= b.upper + 1e-9, "n={n}: {} > {}", b.lower, b.upper);
        }
    }

    #[test]
    fn utilization_ceilings() {
        // 4 unit squares tile the 2×2 square exactly: ceiling 1.0.
        assert!((certified_max_utilization(4) - 1.0).abs() < 1e-9);
        // 11 squares: n / s(n)² with the certified lower bound.
        let u11 = certified_max_utilization(11);
        assert!(u11 > 0.73 && u11 < 0.74, "{u11}");
        // Ceilings never exceed 1 (a lower bound ≥ √n gives n/s² ≤ 1).
        for n in 1u32..=324 {
            assert!(certified_max_utilization(n) <= 1.0 + 1e-9);
        }
    }
}
