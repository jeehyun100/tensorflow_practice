__global__ void saxpy(int n, float a, float *x, float *y) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx < n) {
        y[idx] = a*x[idx] + y[idx];
    }
}
