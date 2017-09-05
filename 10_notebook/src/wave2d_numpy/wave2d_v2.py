import numpy as np
import matplotlib.pyplot as plt

# setup
nx, ny = 1000, 800
tmax = 500

# allocation
f = np.zeros((nx, ny), 'f4')
g = np.zeros((nx, ny), 'f4')

# plot
imag = plt.imshow(f, vmin=-0.1, vmax=0.1)
plt.colorbar()

# time loop
for tstep in range(1, tmax+1):
    g[nx//2, ny//2] = np.sin(0.1*tstep)
    f[1:-1,1:-1] = 0.25*(g[:-2,1:-1] + g[2:,1:-1] + g[1:-1,:-2] + g[1:-1,2:] - 4*g[1:-1,1:-1]) + 2*g[1:-1,1:-1] - f[1:-1,1:-1]
    g[1:-1,1:-1] = 0.25*(f[:-2,1:-1] + f[2:,1:-1] + f[1:-1,:-2] + f[1:-1,2:] - 4*f[1:-1,1:-1]) + 2*f[1:-1,1:-1] - g[1:-1,1:-1]

    if tstep%100 == 0:
        print('tstep={}'.format(tstep))
        imag.set_array(f)
        plt.savefig('png/wave2d_{:03d}.png'.format(tstep))
        #plt.draw()

plt.show()
