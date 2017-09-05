import numpy as np
import matplotlib.pyplot as plt

# setup
nx, ny = 100, 80
tmax = 50

# allocation
f = np.zeros((nx, ny), 'f4')
g = np.zeros((nx, ny), 'f4')

# plot
imag = plt.imshow(f, vmin=-0.1, vmax=0.1)
plt.colorbar()

# time loop
for tstep in range(1, tmax+1):
    g[nx//2, ny//2] = np.sin(0.4*tstep)

    for i in range(1, nx-1):
        for j in range(1, ny-1):
            f[i,j] = 0.25*(g[i-1,j] + g[i+1,j] + g[i,j-1] + g[i,j+1] - 4*g[i,j]) + 2*g[i,j] - f[i,j]

    for i in range(1, nx-1):
        for j in range(1, ny-1):
            g[i,j] = 0.25*(f[i-1,j] + f[i+1,j] + f[i,j-1] + f[i,j+1] - 4*f[i,j]) + 2*f[i,j] - g[i,j]

    if tstep%10 == 0:
        print('tstep={}'.format(tstep))
        imag.set_array(f)
        plt.savefig('png/wave2d_{:03d}.png'.format(tstep))
        #plt.draw()

plt.show()
