//! Deterministic PRNG — the `seed` (0..10000) option of `UVPM4_MainProps`.
//!
//! UVPM takes a `seed` for its stochastic search (heuristic placement,
//! randomized restarts). The engine does not document the generator; this
//! crate uses SplitMix64 — deterministic, seed-stable, no dependencies.

/// SplitMix64 — deterministic 64-bit PRNG (seed-stable, fast).
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Create a generator from the packer `seed` (0..10000).
    pub fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    /// Next u64.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform f64 in `[0, 1)`.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform f64 in `[lo, hi)`.
    #[inline]
    pub fn range_f64(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }

    /// Fisher–Yates shuffle in place (deterministic for a given seed state).
    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            v.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_stable_sequence() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..16 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
        let mut c = SplitMix64::new(43);
        assert_ne!(a.next_u64(), c.next_u64());
    }

    #[test]
    fn range_and_shuffle_deterministic() {
        let mut a = SplitMix64::new(7);
        let mut b = SplitMix64::new(7);
        let va: Vec<f64> = (0..8).map(|_| a.range_f64(0.0, 10.0)).collect();
        let vb: Vec<f64> = (0..8).map(|_| b.range_f64(0.0, 10.0)).collect();
        assert_eq!(va, vb);
        assert!(va.iter().all(|x| (0.0..10.0).contains(x)));
        let mut x = vec![3, 1, 2, 5, 4];
        let mut y = x.clone();
        a.shuffle(&mut x);
        b.shuffle(&mut y);
        assert_eq!(x, y);
        assert_eq!(x.iter().sum::<i32>(), 15);
    }
}
