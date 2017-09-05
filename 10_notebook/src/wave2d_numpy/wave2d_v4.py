import numpy as np
import matplotlib.pyplot as plt


def update(f, g):
    sl = slice(1,-1)
    f[sl,sl] = 0.25*(g[:-2,sl] + g[2:,sl] + g[sl,:-2] + g[sl,2:] - 4*g[sl,sl]) + 2*g[sl,sl] - f[sl,sl]


def main():
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
        update(f, g)
        update(g, f)

        if tstep%100 == 0:
            print('tstep={}'.format(tstep))
            imag.set_array(f)
            #plt.savefig('png/wave2d_{:03d}.png'.format(tstep))
            #plt.draw()

    plt.show()


if __name__ == '__main__':
    main()
