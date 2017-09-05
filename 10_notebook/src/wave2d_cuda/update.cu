__global__ void update(int nx, int ny, float *f, float *g) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int i, j;

    i = idx/ny;
    j = idx%ny;
    if (i > 0 && i < nx-1 && j > 0 && j < ny-1) {
        f[idx] = 0.25*(g[idx-ny] + g[idx+ny] + g[idx-1] + g[idx+1] - 4*g[idx]) + 2*g[idx] - f[idx];
    }
}



__global__ void update_src(int nx, int ny, int tstep, float *g) {
    g[(nx/2)*ny + (ny/2)] = sin(0.1*tstep);
}
