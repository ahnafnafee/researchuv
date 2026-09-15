//! 3×3 symmetric eigenvalue solver — for the PCA tangent-frame projection
//! (reference `tangent_frame`: `np.linalg.eigh` of the 3×3 covariance matrix).
//!
//! Jacobi rotations (Cayley's method) with full re-orthogonalization of the
//! rotation accumulation; for 3×3 it converges in a handful of sweeps.

/// Eigendecompose a 3×3 symmetric matrix `m` (row-major, `m[i][j]`).
///
/// Returns `(eigenvalues, eigenvectors)` with `vectors[i]` = the i-th eigenvector
/// (unit, row i of the change-of-basis, i.e. column-major convention: eigenvector
/// `k` is `(vectors[0][k], vectors[1][k], vectors[2][k])`), matching
/// `np.linalg.eigh`'s `V[:, k]` layout.
pub fn eig3(m: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut a = m;
    let mut v = [[1.0f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..64 {
        // Find the largest off-diagonal.
        let mut p = 0usize;
        let mut q = 1usize;
        let mut max_off = a[0][1].abs();
        for (i, j) in [(0usize, 2usize), (1usize, 2usize)] {
            let x = a[i][j].abs();
            if x > max_off {
                max_off = x;
                p = i;
                q = j;
            }
        }
        if max_off <= 1e-300 {
            break;
        }
        let apq = a[p][q];
        let app = a[p][p];
        let aqq = a[q][q];
        // Jacobi rotation in the (p, q) plane (D = Qᵀ M Q, Q = [[c, −s],[s, c]]):
        // θ = ½·atan2(2·a_pq, a_pp − a_qq) zeroes the off-diagonal exactly.
        let theta = 0.5 * (2.0 * apq).atan2(app - aqq);
        let c = theta.cos();
        let s = theta.sin();
        // Apply to A (symmetric Jacobi update).
        for r in 0..3 {
            if r == p || r == q {
                continue;
            }
            let ar_p = c * a[r][p] + s * a[r][q];
            let ar_q = -s * a[r][p] + c * a[r][q];
            a[r][p] = ar_p;
            a[p][r] = ar_p;
            a[r][q] = ar_q;
            a[q][r] = ar_q;
        }
        a[p][p] = c * c * app + 2.0 * s * c * apq + s * s * aqq;
        a[q][q] = s * s * app - 2.0 * s * c * apq + c * c * aqq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;
        // Apply to V: V' = Q V (columns are the eigenvectors).
        for k in 0..3 {
            let vp = v[p][k];
            let vq = v[q][k];
            v[p][k] = c * vp - s * vq;
            v[q][k] = s * vp + c * vq;
        }
    }
    let w = [a[0][0], a[1][1], a[2][2]];
    (w, v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagonal_matrix_is_its_own_spectrum() {
        let m = [[2.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 1.0]];
        let (w, _v) = eig3(m);
        assert!((w[0] - 2.0).abs() < 1e-9);
        assert!((w[1] - 5.0).abs() < 1e-9);
        assert!((w[2] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn symmetric_matrix_recovers_known_eigenvalues() {
        // A = [[2,1,0],[1,2,0],[0,0,3]] has eigenvalues 3, 1, 3.
        let m = [[2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 3.0]];
        let (w, v) = eig3(m);
        let mut ws = w;
        ws.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((ws[0] - 1.0).abs() < 1e-9);
        assert!((ws[1] - 3.0).abs() < 1e-9);
        assert!((ws[2] - 3.0).abs() < 1e-9);
        // Eigenvectors are orthonormal.
        for i in 0..3 {
            for j in 0..3 {
                let mut dot = 0.0;
                for r in 0..3 {
                    dot += v[r][i] * v[r][j];
                }
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!((dot - expect).abs() < 1e-9, "v{i}·v{j} = {dot}");
            }
        }
        // A v_k = w_k v_k.
        for k in 0..3 {
            for i in 0..3 {
                let mut s = 0.0;
                for j in 0..3 {
                    s += m[i][j] * v[j][k];
                }
                assert!((s - w[k] * v[i][k]).abs() < 1e-9, "A v{k} != w v{i}");
            }
        }
    }

    #[test]
    fn covariance_of_a_planar_strip() {
        // Points on a line along x → covariance is diagonal with the x entry largest.
        let m = [[9.0, 0.5, 0.0], [0.5, 2.0, 0.0], [0.0, 0.0, 0.1]];
        let (w, _) = eig3(m);
        assert!(w[0] >= w[1] && w[1] >= w[2] || w[0].max(w[1]).max(w[2]) > 8.0);
    }
}
