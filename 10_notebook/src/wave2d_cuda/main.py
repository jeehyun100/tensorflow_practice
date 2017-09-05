'''

abstract : 2D wave simulation

history :
  2017-09-04  Ki-Hwan Kim  start

'''

from __future__ import print_function, division
from datetime import datetime
import atexit

import numpy as np
import matplotlib.pyplot as plt
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal



#------------------------------------------------------------------------------
# CUDA initialize
#------------------------------------------------------------------------------
cuda.init()
device = cuda.Device(0)
context = device.make_context()
atexit.register(context.pop)

# CUDA info
print("CUDA Compute Capability: {}.{}".format(*device.compute_capability()))
print("Device: {}".format(device.name()))
#------------------------------------------------------------------------------



def update_numpy(f, g): 
    sl = slice(1, -1)
    f[sl,sl] = 0.25*(g[:-2,sl] + g[2:,sl] + g[sl,:-2] + g[sl,2:] - 4*g[sl,sl]) + 2*g[sl,sl] - f[sl,sl]



class WAVE2D_CUDA(object):
    def __init__(self):
        # read a CUDA kernel file
        with open('update.cu', 'r') as f:
            mod = SourceModule(f.read())
            self.update = mod.get_function('update')
            self.update_src = mod.get_function('update_src')


    def update_cuda(self, nx, ny, f_gpu, g_gpu):
        self.update(np.int32(nx),
                    np.int32(ny),
                    f_gpu,
                    g_gpu,
                    block=(512,1,1), grid=((nx*ny)//512+1,1))


    def update_src_cuda(self, nx, ny, tstep, g_gpu):
        self.update_src(np.int32(nx),
                        np.int32(ny),
                        np.int32(tstep),
                        g_gpu,
                        block=(1,1,1), grid=(1,1))



def main(): 
    nx, ny = 1000, 800
    tmax = 500

    # allocation
    f = np.zeros((nx, ny), 'f4')
    g = np.zeros((nx, ny), 'f4')
    f_gpu = cuda.to_device(f)
    g_gpu = cuda.to_device(g)
    f2 = np.zeros((nx,ny), 'f4')

    # time loop
    # numpy version
    t1 = datetime.now()
    for tstep in range(1, tmax+1):
        g[nx//2, ny//2] = np.sin(0.1*tstep)
        update_numpy(f, g)
        update_numpy(g, f)
    dt_numpy = datetime.now() - t1

    # cuda version
    obj = WAVE2D_CUDA()
    t2 = datetime.now()
    for tstep in range(1, tmax+1):
        obj.update_src_cuda(nx, ny, tstep, g_gpu)
        obj.update_cuda(nx, ny, f_gpu, g_gpu)
        obj.update_cuda(nx, ny, g_gpu, f_gpu)
    cuda.memcpy_dtoh(f2, f_gpu)
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
