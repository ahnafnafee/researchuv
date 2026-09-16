// Conjugate-gradient kernels for the SPD conformal system.
// Compiled to PTX at build time (nvcc -ptx); launched through the CUDA
// driver API. All entry points are extern "C" with C types.

extern "C" {

__global__ void fill0(double* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 0.0;
}

// One thread per row (CSR scalar kernel — deterministic row sums).
__global__ void spmv(const double* __restrict__ vals,
                     const int* __restrict__ cols,
                     const int* __restrict__ row_ptr,
                     const double* __restrict__ x,
                     double* y,
                     int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double s = 0.0;
        int lo = row_ptr[i], hi = row_ptr[i + 1];
        for (int k = lo; k < hi; ++k) s += vals[k] * x[cols[k]];
        y[i] = s;
    }
}

// Block-wise partial dot products (the host sums the per-block partials).
__global__ void dot(const double* __restrict__ a,
                    const double* __restrict__ b,
                    double* partials,
                    int n) {
    __shared__ double sh[256];
    int t = threadIdx.x;
    double s = 0.0;
    for (int i = blockIdx.x * blockDim.x + t; i < n; i += gridDim.x * blockDim.x)
        s += a[i] * b[i];
    sh[t] = s;
    __syncthreads();
    for (int w = 128; w > 0; w >>= 1) {
        if (t < w) sh[t] += sh[t + w];
        __syncthreads();
    }
    if (t == 0) partials[blockIdx.x] = sh[0];
}

// y += alpha * x
__global__ void axpy(double alpha, const double* __restrict__ x, double* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += alpha * x[i];
}

// y = x + alpha * y
__global__ void xpay(double alpha, const double* __restrict__ x, double* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] + alpha * y[i];
}


// dst = src.
__global__ void copy(double* dst, const double* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// ---- Preconditioning ------------------------------------------------------------------

// z = d . x (Jacobi apply with the precomputed inverse diagonal).
__global__ void mul(const double* __restrict__ d,
                    const double* __restrict__ x,
                    double* z,
                    int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) z[i] = d[i] * x[i];
}

// One level of a triangular solve L y = r. The rows listed in `rows`
// [0, count) are mutually independent (same dependency level); each row's
// diagonal is stored as its LAST CSR entry. Reads y[k] only for k in
// earlier levels (already launched and complete).
__global__ void tri_level(const double* __restrict__ vals,
                          const int* __restrict__ cols,
                          const int* __restrict__ row_ptr,
                          const int* __restrict__ rows,
                          int count,
                          const double* __restrict__ r,
                          double* y) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count) return;
    int i = rows[t];
    int end = row_ptr[i + 1] - 1;  // diagonal is last
    double s = r[i];
    for (int k = row_ptr[i]; k < end; ++k) s -= vals[k] * y[cols[k]];
    y[i] = s / vals[end];
}

// Indexed gather: dst[i] = src[idx[i]] (permutations for colored IC(0)).
__global__ void gather(double* dst, const double* __restrict__ src,
                       const int* __restrict__ idx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[idx[i]];
}

// Per-aggregate deterministic sum: out[j] = sum of in[members(j)].
// One thread per aggregate, members summed in list order (no atomics —
// bitwise deterministic).
__global__ void agg_sum(const double* __restrict__ in,
                        double* out,
                        const int* __restrict__ members,
                        const int* __restrict__ m_ptr,
                        int n_agg) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_agg) return;
    double s = 0.0;
    for (int k = m_ptr[j]; k < m_ptr[j + 1]; ++k) s += in[members[k]];
    out[j] = s;
}

// ---- GPU packing heuristic -------------------------------------------------------------

__device__ unsigned long long sm64(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// Deterministic uniform in [0, 1) from a counter-based hash.
__device__ double rnd01(unsigned long long s) {
    return (double)(sm64(s) >> 11) * (1.0 / 9007199254740992.0);
}

// Multi-restart stochastic box relocation. One block per restart; islands
// are visited in a seed-derived order and each block tests candidate anchors
// in parallel (threads round-robin), keeping moves that strictly improve the
// start-corner distance without violating containment or clearance.
__global__ void heuristic_restarts(const double* __restrict__ w,
                                   const double* __restrict__ h,
                                   const double* __restrict__ init_x,
                                   const double* __restrict__ init_y,
                                   double* px, double* py,
                                   double* scores,
                                   int n, int passes, int n_anchors,
                                   double tx0, double ty0, double tx1, double ty1,
                                   double margin,
                                   unsigned long long seed) {
    __shared__ double sh_score[256];
    __shared__ double sh_x[256];
    __shared__ double sh_y[256];
    __shared__ double sh_gain[256];
    int r = blockIdx.x;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    double* rx = px + (size_t)r * n;
    double* ry = py + (size_t)r * n;
    for (int i = tid; i < n; i += nthreads) {
        rx[i] = init_x[i];
        ry[i] = init_y[i];
    }
    __syncthreads();
    for (int pass = 0; pass < passes; ++pass) {
        for (int ii = 0; ii < n; ++ii) {
            // Deterministic visit order per (restart, pass).
            int i = (int)(sm64(seed + 0x1000ull * (unsigned long long)r +
                               0x100ull * (unsigned long long)pass +
                               (unsigned long long)ii) % (unsigned long long)n);
            double wi = w[i], hi = h[i];
            double ei = wi > hi ? wi : hi;
            double cur_score =
                (rx[i] - tx0) + (ry[i] - ty0);  // start corner = BL
            double best_gain = -1e300, bx = rx[i], by = ry[i];
            for (int c = tid; c < n_anchors; c += nthreads) {
                unsigned long long cs = seed + 0x9E37ull * (unsigned long long)r +
                                        0x70ull * (unsigned long long)pass +
                                        0x11ull * (unsigned long long)i +
                                        (unsigned long long)c;
                double ax = tx0 + rnd01(cs) * (tx1 - tx0 - wi);
                double ay = ty0 + rnd01(cs + 0xABCDull) * (ty1 - ty0 - hi);
                // Containment border (relative margin on the island extent).
                double b = margin * ei;
                if (ax < tx0 + b || ay < ty0 + b || ax + wi > tx1 - b ||
                    ay + hi > ty1 - b)
                    continue;
                bool ok = true;
                for (int j = 0; j < n && ok; ++j) {
                    if (j == i) continue;
                    double ej = w[j] > h[j] ? w[j] : h[j];
                    double g = margin * (ei > ej ? ei : ej);
                    if (ax + wi + g <= rx[j] || rx[j] + w[j] + g <= ax ||
                        ay + hi + g <= ry[j] || ry[j] + h[j] + g <= ay)
                        continue;  // clear
                    ok = false;
                }
                if (!ok) continue;
                double sc = (ax - tx0) + (ay - ty0);
                double gain = cur_score - sc;
                if (gain > best_gain) {
                    best_gain = gain;
                    bx = ax;
                    by = ay;
                }
            }
            sh_gain[tid] = best_gain;
            sh_x[tid] = bx;
            sh_y[tid] = by;
            __syncthreads();
            for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    // Max gain wins; deterministic tie-break by lane index
                    // (earlier lane keeps).
                    if (sh_gain[tid + stride] > sh_gain[tid]) {
                        sh_gain[tid] = sh_gain[tid + stride];
                        sh_x[tid] = sh_x[tid + stride];
                        sh_y[tid] = sh_y[tid + stride];
                    }
                }
                __syncthreads();
            }
            if (tid == 0 && sh_gain[0] > 1e-12) {
                rx[i] = sh_x[0];
                ry[i] = sh_y[0];
            }
            __syncthreads();
        }
    }
    // Layout score: total start-corner distance.
    double s = 0.0;
    for (int i = tid; i < n; i += nthreads)
        s += (rx[i] - tx0) + (ry[i] - ty0);
    sh_score[tid] = s;
    __syncthreads();
    for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sh_score[tid] += sh_score[tid + stride];
        __syncthreads();
    }
    if (tid == 0) scores[r] = sh_score[0];
}

}  // extern "C"
