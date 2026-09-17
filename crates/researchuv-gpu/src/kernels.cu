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
                       const int* idx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[idx[i]];
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

// ---- Free-space rasterizer packer ------------------------------------------------------

// Rasterize one polygon ring into a bitmask with even-odd scanline fill.
// One thread per (row, word) of the target; points are in cell coordinates
// relative to the target origin. Rings XOR-accumulate, so outer + hole
// rings compose to the correct even-odd coverage.
__global__ void rst_raster(const double* __restrict__ px,
                           const double* __restrict__ py,
                           int npts,
                           unsigned int* out,
                           int words_per_row, int rows) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= words_per_row * rows) return;
    int word = t % words_per_row;
    int y = t / words_per_row;
    double yc = y + 0.5;
    unsigned int bits = 0;
    for (int b = 0; b < 32; ++b) {
        double xc = word * 32 + b + 0.5;
        int parity = 0;
        for (int e = 0; e < npts; ++e) {
            int e2 = (e + 1 == npts) ? 0 : e + 1;
            double ya = py[e], yb = py[e2];
            if ((ya <= yc && yc < yb) || (yb <= yc && yc < ya)) {
                double xa = px[e], xb = px[e2];
                double xi = xa + (yc - ya) * (xb - xa) / (yb - ya);
                if (xi <= xc) parity ^= 1;
            }
        }
        if (parity) bits |= (1u << b);
    }
    out[y * words_per_row + word] ^= bits;
}

// Horizontal dilation of every row by `radius` cells (bitblock shifts; the
// source-word sweep with the `off` window covers |radius| < 96).
__global__ void rst_dilate_h(const unsigned int* __restrict__ in,
                             unsigned int* out,
                             int words, int rows, int radius) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= rows) return;
    const unsigned int* rin = in + (size_t)y * words;
    unsigned int* rout = out + (size_t)y * words;
    for (int w = 0; w < words; ++w) rout[w] = 0;
    for (int d = -radius; d <= radius; ++d) {
        for (int w = 0; w < words; ++w) {
            for (int ws = w - 3; ws <= w + 3; ++ws) {
                if (ws < 0 || ws >= words) continue;
                int off = d + 32 * (ws - w);
                unsigned int v = rin[ws];
                if (off >= 0 && off < 32) rout[w] |= (v << off);
                else if (off < 0 && off > -32) rout[w] |= (v >> (-off));
            }
        }
    }
}

// Vertical dilation: one thread per (row, word).
__global__ void rst_dilate_v(const unsigned int* __restrict__ in,
                             unsigned int* out,
                             int words, int rows, int radius) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= words * rows) return;
    int w = t % words;
    int y = t / words;
    unsigned int v = 0;
    for (int d = -radius; d <= radius; ++d) {
        int yy = y + d;
        if (yy >= 0 && yy < rows) v |= in[(size_t)yy * words + w];
    }
    out[(size_t)y * words + w] = v;
}

// Placement search: one thread per candidate anchor (x, y) in cell units.
// Valid iff the island mask ANDs to zero against EVERY grid in the table
// (the per-class margin grids for this candidate's extent class) at every
// mask row; the unaligned window comes from two occupancy words.
// mode 0 = square/auto (x+y), 1 = side-to-side vertical (y dominant),
// 2 = side-to-side horizontal (x dominant).
__global__ void rst_find_best(const unsigned long long* __restrict__ grid_ptrs,
                              int n_grids,
                              const unsigned int* __restrict__ mask,
                              int g_words, int m_words, int m_rows,
                              int cand_w, int cand_h,
                              int mode, int tile_cells, int tile_cols,
                              double* scores) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= cand_w * cand_h) {
        return;
    }
    int x = t % cand_w;
    int y = t / cand_w;
    int sh = x & 31;
    int wbase = x >> 5;
    bool ok = true;
    for (int g = 0; g < n_grids && ok; ++g) {
        const unsigned int* occ =
            reinterpret_cast<const unsigned int*>(grid_ptrs[g]);
        for (int my = 0; my < m_rows && ok; ++my) {
            const unsigned int* orow = occ + (size_t)(y + my) * g_words + wbase;
            const unsigned int* mrow = mask + (size_t)my * m_words;
            for (int mw = 0; mw < m_words; ++mw) {
                unsigned int m = mrow[mw];
                if (!m) continue;
                unsigned int window = orow[mw] >> sh;
                if (sh > 0) window |= orow[mw + 1] << (32 - sh);
                if (m & window) {
                    ok = false;
                    break;
                }
            }
        }
    }
    double score;
    if (mode == 1) score = 1e6 * (double)y + (double)x;
    else if (mode == 2) score = 1e6 * (double)x + (double)y;
    else if (mode == 3) {
        // Tile-major: row-major tile index dominates, the in-tile corner
        // distance breaks ties. `tile_cells` is one tile's side in cells.
        int tx = x / tile_cells;
        int ty = y / tile_cells;
        int in_x = x - tx * tile_cells;
        int in_y = y - ty * tile_cells;
        score = 1e6 * (double)(ty * tile_cols + tx) + (double)(in_x + in_y);
    } else score = (double)x + (double)y;
    scores[t] = ok ? score : 1e300;
}

}  // extern "C"