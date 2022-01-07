from numba import jit
import numpy as np
import time
from numba import jit
import numpy as np
import time
import scipy.sparse as sp
import random


def derivative(N, h, coefficient, boundary="periodic"):
    """
    Compute the discrete derivative for periodic BC
    """
    dlow = -(np.size(coefficient) -1 ) // 2
    dup = - dlow + 1

    diagonals = []
    offsets = []
    for k in np.arange(dlow, dup):
        diagonals.append(coefficient[k - dlow] * np.ones(N - abs(k)))
        offsets.append(k)
        if k > 0:
            diagonals.append(coefficient[k - dlow] * np.ones(abs(k)))
            offsets.append(-N + k)
        if k < 0:
            diagonals.append(coefficient[k - dlow] * np.ones(abs(k)))
            offsets.append(N + k)

    return sp.diags(diagonals, offsets) / h



@jit(nopython=True,parallel=True)
def diff_numba(x, stencil, rows, col, Nrows):
    k = 0
    diff = np.zeros(Nrows)
    for ir in range(Nrows):
        i = 8*rows[ir]
        for j in range(8):
         diff[ir] += stencil[i+j] * x[col[i+j]]
        k += 1
    return diff



if __name__ == '__main__':
    Nsize = 2**8*2**8
    x = np.random.rand(Nsize)
    stencil_x = np.asarray([1 / 280, -4 / 105, 1 / 5, -4 / 5, 0, 4 / 5, -1 / 5, 4 / 105, -1 / 280])
    stencil = derivative(Nsize, 1, stencil_x)

    random.seed(22123)
    sampleMeshSize = Nsize//100
    sampleMeshIndices = random.sample(range(1, Nsize), sampleMeshSize)
    sampleMeshIndices = np.append(sampleMeshIndices, [0, Nsize-1])
    # sort for convenience, not necessarily needed
    sampleMeshIndices = np.sort(sampleMeshIndices)

    diff = np.zeros([sampleMeshSize+3])

    c = stencil.tocsr()
    rows, cols = c.nonzero()
    dat = np.asarray(c[rows,cols]).flatten()
    B = diff_numba(x, dat,np.unique(rows), cols, Nrows=Nsize)
    start = time.time()
    B = diff_numba(x, dat,np.unique(rows), cols,Nrows = Nsize)
    print(time.time() - start,np.sum(B[:]),'Numba')


    start = time.time()
    B=stencil@x
    print(time.time() - start,np.sum(B[:]), 'sparse')
