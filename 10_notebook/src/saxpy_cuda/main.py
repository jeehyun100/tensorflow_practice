'''

abstract : single precision y=ax+y

history :
  2017-09-04  Ki-Hwan Kim  start

'''

from __future__ import print_function, division
from datetime import datetime
import atexit

import numpy as np
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



def saxpy_numpy(a, x, y): 
    y[:] = a*x + y



class SAXPY_CUDA(object):
    def __init__(self):
        # read a CUDA kernel file
        with open('saxpy.cu', 'r') as f:
            mod = SourceModule(f.read())
            self.saxpy = mod.get_function('saxpy')


    def saxpy_cuda(self, n, a, x_gpu, y_gpu):
        self.saxpy(np.int32(n),
                   np.float32(a),
                   x_gpu,
                   y_gpu,
                   block=(512,1,1), grid=(n//512+1,1))



def main(): 
    n = 2**25

    a = np.float32(np.random.rand())
    x = np.random.rand(n).astype('f4')
    y = np.random.rand(n).astype('f4')
    x_gpu = cuda.to_device(x)
    y_gpu = cuda.to_device(y)
    y2 = np.zeros(n, 'f4')

    t1 = datetime.now()
    saxpy_numpy(a, x, y)
    dt_numpy = datetime.now() - t1

    obj = SAXPY_CUDA()
    t2 = datetime.now()
    obj.saxpy_cuda(n, a, x_gpu, y_gpu) 
    cuda.memcpy_dtoh(y2, y_gpu)
    dt_cuda = datetime.now() - t2

    print('n={}'.format(n))
    print('numpy: {}'.format(dt_numpy))
    print('cuda : {}'.format(dt_cuda))

    aa_equal(y, y2, 7)
    print('Check result: OK!')



if __name__ == '__main__':
    main()
