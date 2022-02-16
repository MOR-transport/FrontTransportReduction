from numpy import tanh,cosh
from numpy import pi, reshape
import numpy as np
from numba import njit,jit,float32,uint32, uint64,int32,prange
from time import perf_counter


###########################
# Numba: https://numba.pydata.org/numba-doc/dev/user/5minguide.html
#  * @njit - this is an alias for @jit(nopython=True) as it is so commonly used!
#
###########################

@njit(parallel=True)
def hyperreduced_front( phi, supportMeshIndices, f):
    front = lambda x: (1 + tanh(x)) * 0.5
    Nidx = len(supportMeshIndices)
    for i in prange(Nidx):#enumerate(supportMeshIndices):
        index = supportMeshIndices[i]
        f[index] = front(phi[index])

@njit(parallel=True)#(float32,float32,int32,int32,float32))
def hyperreduced_jac( phi,U, sampleMeshIndices, jac):
    dfront = lambda x: 0.5 / (cosh(x) ** 2)
    Nidx = len(sampleMeshIndices)
    for i in prange(Nidx):#,index in enumerate(sampleMeshIndices):
        index = sampleMeshIndices[i]
        df = dfront(phi[index])
        jac[i,:]=U[index,:]*df

def hyperreduced_jac_ECSW( phi,U, sampleMeshIndices, jac, alpha):
    dfront = lambda x: 0.5 / (cosh(x) ** 2)
    Nidx = len(sampleMeshIndices)
    for i in prange(Nidx):#,index in enumerate(sampleMeshIndices):
        index = sampleMeshIndices[i]
        df = dfront(phi[index])*alpha[index]
        jac[i,:]=U[index,:]*df