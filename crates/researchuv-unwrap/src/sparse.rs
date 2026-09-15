//! Sparse linear algebra — the direct sparse solve behind the unfold driver.
//!
//! Evidence: the shared LS solver driver (VA `0x14034CBB0`) solves each iteration's
//! system with **SuperLU 5.3.0_SEQ `dgssv`** (VA `0x1404E9080`) — a direct sparse LU.
//! This module mirrors that with a dependency-free sparse LU (Markowitz ordering +
//! partial pivoting) over the same `row/col/val` assembly the driver produces
//! (ALGORITHMS.md §3.2), so the Rust clone stays std-only by default.

use std::collections::BTreeMap;

/// Sparse matrix in row-indexed (COO-like) form with duplicate entries summed.
///
/// The LU factorization is memoized: `solve` reuses the cached factors while
/// the matrix is unchanged (the unfold driver re-solves an identical system
/// on its convergence-check iteration). The cache lives behind a `RefCell`
/// so the read-only `solve`/`lu` API stays `&self`; the crate is
/// single-threaded by design.
#[derive(Clone, Debug, Default)]
pub struct Sparse {
    pub n: usize,
    /// row -> (col -> value), sorted for deterministic elimination.
    rows: Vec<BTreeMap<usize, f64>>,
    /// Memoized LU factors (cleared by [`Sparse::add`]).
    lu_cache: std::cell::RefCell<Option<Box<LuFactors>>>,
}

/// The factorization payload: column permutation, partial-pivoting swaps,
/// combined L/U storage, and the factor nonzero count.
#[derive(Clone, Debug, Default)]
struct LuFactors {
    perm: Vec<usize>,
    swaps: Vec<(usize, usize)>,
    lu: BTreeMap<(usize, usize), f64>,
    nnz: usize,
}

impl Sparse {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            rows: vec![BTreeMap::new(); n],
            lu_cache: std::cell::RefCell::new(None),
        }
    }

    /// Add `val` at `(row, col)` (duplicates are summed, as in the driver's COO assembly).
    pub fn add(&mut self, row: usize, col: usize, val: f64) {
        if val != 0.0 {
            *self.rows[row].entry(col).or_insert(0.0) += val;
            *self.lu_cache.get_mut() = None;
        }
    }

    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.rows[row].get(&col).copied().unwrap_or(0.0)
    }

    pub fn nnz(&self) -> usize {
        self.rows.iter().map(|r| r.len()).sum()
    }

    /// Row access (column → value, sorted) for iterative solvers.
    pub fn row(&self, i: usize) -> &BTreeMap<usize, f64> {
        &self.rows[i]
    }

    /// Approximate minimum-degree column ordering: sort columns by their initial
    /// nonzero count (ties broken by column index). Cheap (`O(nnz + n log n)`) and
    /// effective for the structured systems the driver produces. (SuperLU's own
    /// Markowitz refinement is applied on top of such an ordering; the initial
    /// min-degree order is a faithful, dependency-free approximation.)
    pub fn markowitz_order(&self) -> Vec<usize> {
        let mut order: Vec<usize> = (0..self.n).collect();
        order.sort_by_key(|&c| (self.rows[c].len(), c));
        order
    }

    /// Sparse LU factorization with Markowitz ordering and partial pivoting.
    ///
    /// Returns `(perm, swaps, lu, nnz_lu)` where `perm` is the column/row
    /// permutation (variable `perm[i]` becomes the i-th pivot), `swaps` is the
    /// sequence of partial-pivoting row swaps `(k, piv)` applied during
    /// elimination (the RHS must be permuted with them too), and `lu` is the
    /// combined L (unit lower) / U (upper) storage. The result is memoized —
    /// repeated calls on an unchanged matrix return the cached factors.
    pub fn lu(&self) -> (Vec<usize>, Vec<(usize, usize)>, BTreeMap<(usize, usize), f64>, usize) {
        let f = self.cached_factors();
        (f.perm.clone(), f.swaps.clone(), f.lu.clone(), f.nnz)
    }

    /// The memoized factors: compute on first use, reuse until `add` clears.
    fn cached_factors(&self) -> std::cell::Ref<'_, Box<LuFactors>> {
        if self.lu_cache.borrow().is_none() {
            let f = self.factorize();
            *self.lu_cache.borrow_mut() = Some(f);
        }
        // The only other borrower above has ended; hand out a read guard.
        std::cell::Ref::map(self.lu_cache.borrow(), |c| c.as_ref().expect("just computed"))
    }

    fn factorize(&self) -> Box<LuFactors> {
        let perm = self.markowitz_order();
        let n = self.n;
        // Inverse permutation: inv[perm[i]] = i (O(1) lookups below).
        let mut inv = vec![0usize; n];
        for (i, &c) in perm.iter().enumerate() {
            inv[c] = i;
        }
        // Build the permuted matrix for elimination.
        let mut a: Vec<BTreeMap<usize, f64>> = vec![BTreeMap::new(); n];
        for (i, &c) in perm.iter().enumerate() {
            for (&cc, &v) in &self.rows[c] {
                a[i].insert(inv[cc], v);
            }
        }
        let mut swaps: Vec<(usize, usize)> = Vec::new();
        let mut lu: BTreeMap<(usize, usize), f64> = BTreeMap::new();
        let mut nnz = 0usize;
        for k in 0..n {
            // Partial pivoting: find the largest |a[r][k]| for r >= k.
            let mut piv = k;
            let mut piv_val = 0.0f64.abs();
            for r in k..n {
                let v = a[r].get(&k).copied().unwrap_or(0.0).abs();
                if v > piv_val {
                    piv_val = v;
                    piv = r;
                }
            }
            if piv_val < 1e-300 {
                // Singular (numerically) — leave the rest as-is; the caller's
                // finite-check handles it (the driver breaks on non-finite solves).
                continue;
            }
            if piv != k {
                swaps.push((k, piv));
                let tmp_row = std::mem::take(&mut a[piv]);
                a[piv] = std::mem::take(&mut a[k]);
                a[k] = tmp_row;
                // Swap the L multipliers of the two rows: every stored entry in
                // rows `k`/`piv` with column < k belongs to the row as a whole, so
                // the pair of rows swaps their full L history.
                let moved: Vec<((usize, usize), f64)> = lu
                    .iter()
                    .filter(|((r, c), _)| *c < k && (*r == k || *r == piv))
                    .map(|(&rc, &v)| (rc, v))
                    .collect();
                for ((r, c), v) in &moved {
                    lu.remove(&(*r, *c));
                    let r2 = if *r == k { piv } else { k };
                    lu.insert((r2, *c), *v);
                }
            }
            let pivot = a[k].get(&k).copied().unwrap_or(0.0);
            lu.insert((k, k), pivot);
            nnz += 1;
            // Store the U part of the pivot row (columns > k) so back-substitution
            // can read it from `lu`.
            let u_part: Vec<(usize, f64)> = a[k]
                .range(k + 1..n)
                .map(|(&c, &v)| (c, v))
                .collect();
            for (c, v) in u_part {
                if v != 0.0 {
                    lu.insert((k, c), v);
                    nnz += 1;
                }
            }
            for r in (k + 1)..n {
                let a_rk = match a[r].get(&k) {
                    Some(v) if *v != 0.0 => *v,
                    _ => continue,
                };
                let factor = a_rk / pivot;
                lu.insert((r, k), factor);
                nnz += 1;
                a[r].insert(k, 0.0);
                // Row update: a[r][c] -= factor * a[k][c] for c > k.
                let ks: Vec<(usize, f64)> = a[k]
                    .range(k + 1..n)
                    .map(|(&c, &v)| (c, v))
                    .collect();
                for (c, vk) in ks {
                    let cur = a[r].get(&c).copied().unwrap_or(0.0);
                    let nv = cur - factor * vk;
                    if nv == 0.0 {
                        a[r].remove(&c);
                    } else {
                        a[r].insert(c, nv);
                    }
                }
            }
        }
        Box::new(LuFactors { perm, swaps, lu, nnz })
    }

    /// Solve `A x = b` via the LU factorization. Returns `None` if the system is
    /// (numerically) singular or the solution is non-finite. The factorization
    /// is computed once and reused across solves of an unchanged matrix.
    pub fn solve(&self, b: &[f64]) -> Option<Vec<f64>> {
        let factors = self.cached_factors();
        let f = &**factors;
        let (perm, swaps, lu) = (&f.perm, &f.swaps, &f.lu);
        let n = self.n;
        // Permute RHS: b[perm[i]], then apply the partial-pivoting row swaps in
        // the same order the factorization applied them to the matrix.
        let mut bp: Vec<f64> = perm.iter().map(|&c| b[c]).collect();
        for &(k, piv) in swaps {
            bp.swap(k, piv);
        }
        // Forward substitution for L y = bp (L is unit lower; lu[(r,k)] for r>k is L).
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            let mut s = bp[i];
            for k in 0..i {
                if let Some(&l) = lu.get(&(i, k)) {
                    s -= l * y[k];
                }
            }
            y[i] = s;
        }
        // Back substitution for U x_perm = y.
        let mut x = vec![0.0f64; n];
        for i in (0..n).rev() {
            let diag = lu.get(&(i, i)).copied().unwrap_or(0.0);
            if diag.abs() < 1e-300 {
                return None;
            }
            let mut s = y[i];
            for k in (i + 1)..n {
                if let Some(&u) = lu.get(&(i, k)) {
                    s -= u * x[k];
                }
            }
            x[i] = s / diag;
        }
        if !x.iter().all(|v| v.is_finite()) {
            return None;
        }
        // Unpermute: x_perm[perm[i]] = x[i]  ->  x_orig[c] where perm[pos]=c.
        let mut xo = vec![0.0f64; n];
        for (i, &c) in perm.iter().enumerate() {
            xo[c] = x[i];
        }
        Some(xo)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solves_small_symmetric_system() {
        // 3x3 SPD system with known solution.
        let mut a = Sparse::new(3);
        // [[4,1,0],[1,3,1],[0,1,2]] x = [1,2,3]
        a.add(0, 0, 4.0);
        a.add(0, 1, 1.0);
        a.add(1, 0, 1.0);
        a.add(1, 1, 3.0);
        a.add(1, 2, 1.0);
        a.add(2, 1, 1.0);
        a.add(2, 2, 2.0);
        let x = a.solve(&[1.0, 2.0, 3.0]).expect("should solve");
        // Verify by direct multiplication.
        for i in 0..3 {
            let mut s = 0.0;
            for j in 0..3 {
                s += a.get(i, j) * x[j];
            }
            assert!((s - [1.0, 2.0, 3.0][i]).abs() < 1e-9, "row {i}: {s}");
        }
    }

    #[test]
    fn solves_laplacian_with_pin() {
        // 1-D chain Laplacian (path 0-1-2-3) with vertex 0 pinned to 5.
        // M = [[1,0,0,0],[1,-2,1,0],[0,1,-2,1],[0,0,1,-2]] ... actually the driver
        // assembles the symmetric form; build the pinned chain:
        // free rows: -2 u_i + u_{i-1} + u_{i+1} = 0; pinned row: u_0 = 5.
        let mut a = Sparse::new(4);
        a.add(0, 0, 1.0); // pin
        a.add(1, 1, -2.0);
        a.add(1, 0, 1.0);
        a.add(1, 2, 1.0);
        a.add(2, 2, -2.0);
        a.add(2, 1, 1.0);
        a.add(2, 3, 1.0);
        a.add(3, 3, -2.0);
        a.add(3, 2, 1.0);
        let b = [5.0, 0.0, 0.0, 0.0];
        let x = a.solve(&b).expect("should solve");
        // The chain interpolates: x = [5, ?, ?, ?]. With the pinned row u0=5 and the
        // (indefinite) chain rows, the solution is a straight line extended: [5, 5, 5, 5]
        // is NOT a solution (row 1: -2*5+5+5=0 yes; row 3: -2*5+5 = -5 != 0). The
        // system as written is not the standard Dirichlet Laplacian; just check it
        // solves consistently (A x = b).
        for i in 0..4 {
            let mut s = 0.0;
            for j in 0..4 {
                s += a.get(i, j) * x[j];
            }
            assert!((s - b[i]).abs() < 1e-9, "row {i}: {s} vs {}", b[i]);
        }
    }

    #[test]
    fn duplicate_entries_are_summed() {
        let mut a = Sparse::new(1);
        a.add(0, 0, 3.0);
        a.add(0, 0, 2.0);
        assert!((a.get(0, 0) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn factorization_is_reused_and_invalidated() {
        // Repeated solves of an unchanged matrix must agree; adding an entry
        // must drop the stale factors (the result changes).
        let mut a = Sparse::new(2);
        a.add(0, 0, 4.0);
        a.add(1, 1, 2.0);
        let b = [8.0, 6.0];
        let x1 = a.solve(&b).unwrap();
        let x2 = a.solve(&b).unwrap();
        assert_eq!(x1, x2);
        assert!((x1[0] - 2.0).abs() < 1e-12 && (x1[1] - 3.0).abs() < 1e-12);
        a.add(0, 1, 1.0); // now coupled: 4x + y = 8, 2y = 6 → x = 1.25
        let x3 = a.solve(&b).unwrap();
        assert!((x3[0] - 1.25).abs() < 1e-12, "x0 = {}", x3[0]);
        // The public lu() also round-trips through the cache.
        let (perm, _swaps, _lu, _nnz) = a.lu();
        assert_eq!(perm.len(), 2);
    }

    #[test]
    fn solves_a_banded_grid_system() {
        // A 5x5 interior-grid Poisson system (banded, like the driver's free
        // block on a grid chart): (4 on the diagonal, −1 to the 4-neighborhood).
        let n = 25;
        let mut a = Sparse::new(n);
        for i in 0..n {
            let (r, c) = (i / 5, i % 5);
            a.add(i, i, 4.0);
            for (dr, dc) in [(0i32, 1), (0, -1), (1, 0), (-1, 0)] {
                let (rr, cc) = (r as i32 + dr, c as i32 + dc);
                if (0..5).contains(&rr) && (0..5).contains(&cc) {
                    a.add(i, (rr * 5 + cc) as usize, -1.0);
                }
            }
        }
        let b: Vec<f64> = (0..n).map(|i| (i as f64) * 0.1 + 1.0).collect();
        let x = a.solve(&b).expect("grid system solves");
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..n {
                s += a.get(i, j) * x[j];
            }
            assert!((s - b[i]).abs() < 1e-8, "row {i}: {s} vs {}", b[i]);
        }
    }

    #[test]
    fn pivot_swap_permutes_rhs() {
        // Pure permutation matrix A = [[0,0,1],[0,1,0],[1,0,0]]: A x = b has the
        // exact solution x = [b2, b1, b0]. Step 0 has a[0][0] = 0, forcing a
        // partial-pivoting row swap (0 <-> 2) — which exercises the swap
        // recording in `lu()` and the matching RHS permutation in `solve()`.
        // (Before the swap bookkeeping, `solve` returned [1, 2, 3] here.)
        let mut a = Sparse::new(3);
        a.add(0, 2, 1.0);
        a.add(1, 1, 1.0);
        a.add(2, 0, 1.0);
        let b = [1.0, 2.0, 3.0];
        let x = a.solve(&b).expect("should solve");
        assert!((x[0] - 3.0).abs() < 1e-12, "x0 = {}", x[0]);
        assert!((x[1] - 2.0).abs() < 1e-12, "x1 = {}", x[1]);
        assert!((x[2] - 1.0).abs() < 1e-12, "x2 = {}", x[2]);
    }

    #[test]
    fn pivot_swap_in_spd_system() {
        // SPD 3x3 where the Markowitz ordering reorders the variables AND
        // elimination performs a row swap:
        // [[0, 2, 0],
        //  [2, 5, 1],
        //  [0, 1, 4]]  (diagonal [0,5,4] — column 0's max is off-diagonal)
        // x = [1, -2, 3]; verify by direct multiplication with the ORIGINAL A.
        let mut a = Sparse::new(3);
        a.add(0, 1, 2.0);
        a.add(1, 0, 2.0);
        a.add(1, 1, 5.0);
        a.add(1, 2, 1.0);
        a.add(2, 1, 1.0);
        a.add(2, 2, 4.0);
        let x_true = [1.0, -2.0, 3.0];
        let b: Vec<f64> = (0..3)
            .map(|i| (0..3).fold(0.0, |s, j| s + a.get(i, j) * x_true[j]))
            .collect();
        let x = a.solve(&b).expect("should solve");
        for i in 0..3 {
            assert!((x[i] - x_true[i]).abs() < 1e-9, "x{i} = {} (want {})", x[i], x_true[i]);
        }
    }
}
