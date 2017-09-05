void update(int nx, int ny, float *f, float *g) {
    int i, j, idx;

	for (i=1; i<nx-1; i++) {
	    for (j=1; j<ny-1; j++) {
			idx = i*ny + j;
        	f[idx] = 0.25*(g[idx-ny] + g[idx+ny] + g[idx-1] + g[idx+1] - 4*g[idx]) + 2*g[idx] - f[idx];
    	}
    }
}
