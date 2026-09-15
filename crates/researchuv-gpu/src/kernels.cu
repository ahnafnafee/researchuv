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

}  // extern "C"
