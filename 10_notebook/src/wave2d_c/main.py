'''

abstract : 2D wave simulation

history :
  2017-09-04  Ki-Hwan Kim  start

'''

from __future__ import print_function, division
from ctypes import c_int, c_float
from datetime import datetime

import numpy as np
import numpy.ctypeslib as npct
import matplotlib.pyplot as plt
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal



def update_numpy(f, g): 
    sl = slice(1, -1)
    f[sl,sl] = 0.25*(g[:-2,sl] + g[2:,sl] + g[sl,:-2] + g[sl,2:] - 4*g[sl,sl]) + 2*g[sl,sl] - f[sl,sl]



class WAVE2D_C(object):
    def __init__(self):
        # load the library using numpy
        libm = npct.load_library('update', './')

        # set the arguments and retun types
        arr_f4 = npct.ndpointer(ndim=1, dtype='f4')
        libm.update.argtypes = [c_int, c_int, arr_f4, arr_f4]
        libm.update.rettype = None

        # set public
        self.libm = libm

    def update_c(self, nx, ny, x, y):
        self.libm.update(nx, ny, x, y)



def main(): 
    nx, ny = 1000, 800
    tmax = 500

    # allocation
    f = np.zeros((nx, ny), 'f4')
    g = np.zeros((nx, ny), 'f4')
    f2 = np.zeros_like(f)
    g2 = np.zeros_like(f)

    # time loop
    # numpy version
    t1 = datetime.now()
    for tstep in range(1, tmax+1):
        g[nx//2, ny//2] = np.sin(0.1*tstep)
        update_numpy(f, g)
        update_numpy(g, f)
    dt_numpy = datetime.now() - t1

    # C version
    obj = WAVE2D_C()
    t2 = datetime.now()
    for tstep in range(1, tmax+1):
        g2[nx//2, ny//2] = np.sin(0.1*tstep)
        obj.update_c(nx, ny, f2.ravel(), g2.ravel())
        obj.update_c(nx, ny, g2.ravel(), f2.ravel())
    dt_cuda = datetime.now() - t2

    print('\nnx={}, ny={}, tmax={}'.format(nx, ny, tmax))
    print('numpy: {}'.format(dt_numpy))
    print('cuda : {}'.format(dt_cuda))

    # check results
    aa_equal(f, f2, 6)
    print('Check result: OK!')

    # plot
    plt.imshow(f.T, cmap='hot', origin='lower', vmin=-0.1, vmax=0.1)
    plt.colorbar()
    plt.show()



if __name__ == '__main__':
    main()
