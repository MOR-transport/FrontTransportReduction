import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from time import perf_counter
from matplotlib import rc
import torch as to
import pandas as pd

from scipy.integrate import odeint
# if run from within a build of mypressio, need to append to python path
import pathlib, sys
file_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(file_path) + "/..")         # to access doFom

from .numba_ftr_mapper import *


class FomAdapter:
    def __init__(self, fom_params):
        """ initialize as you want/needed by your application
         # e.g. mesh, inputs, bc, commandline aguments, etc.
        """

        self.params = fom_params
        # we test how many entries the output of the rhs has.
        # Note: the size is not always given bei self.params.fom_size,
        # since we may use hyperreduction in the rhs
        self.rhs_size = fom_params.sampleMeshSize

    def velocity(self, y, t, f):
        """
      # compute velocity, f(y,t;...), for a given state, y
      :param y: state
      :param t: time
      :return f: velocity f(y,t;...)
      """
        self.params.rhs(self.params,y, t, f)

    def applyJacobian(self, y, B, t, A):
        """
       given current state y(t):
       compute A=df/dy*B, where B is a skinny dense matrix
       Note that we just require the *action* of the Jacobian.
      :param y: state
      :param B:
      :param t: time point
      :return A
      """

        rom_size = np.size(A,1)

        if self.params.pde == "advection":
            if not to.is_tensor(y):
                y = to.tensor(y, requires_grad=True)
            if not to.is_tensor(B):
                B = to.tensor(B)
            dfdy = self.params.rhs_advection2D_periodic_torch(y, t)
            for k in range(rom_size):
                dfdy.backward(B[:,k], retain_graph=True)
                A[:, k] = y.grad.detach().numpy()
        else:
            for k in range(rom_size):
                 self.params.rhs_jacobian(self.params, y, t, B[:, k], A[:, k])

                # create f(y,t,...)
    def createVelocity(self):
        return np.zeros(self.rhs_size)  # say N is the total number of of unknowns

    # create result of df/dy*B, B is typically a skinny dense matrix
    def createApplyJacobianResult(self, B):
        return np.zeros([self.rhs_size, np.size(B,1)])

# ----------------------------------------------------------------------
# this linear solver is used at each gauss-newton iteration
class MyLinSolver:
        def __init__(self): pass

        def solve(self, A, b, x):
            lumat, piv, info = linalg.lapack.dgetrf(A, overwrite_a=True)
            x[:], info = linalg.lapack.dgetrs(lumat, piv, b, 0, 0)
# ----------------------------------------------------------------------
class MyMasker:
  def __init__(self, indices):
    self.rows_ = indices
    self.sampleMeshSize_ = len(indices)

  def createApplyMaskResult(self, operand):
    if (operand.ndim == 1):
      return np.zeros(self.sampleMeshSize_)
    else:
      return np.zeros((self.sampleMeshSize_, operand.shape[1]))

  def applyMask(self, operand, time, result):
    if (operand.ndim == 1):
      result[:] = np.take(operand, self.rows_)
    else:
      result[:] = np.take(operand, self.rows_, axis=0)
# ----------------------------------------------------------------------

class MyCustomMapper:
    def __init__(self, fom_size, rom_size):
      # attention: the jacobian of the mapping must be column-major oder
      # so that pressio can view it without deep copying it, this enables
      # to keep only one jacobian object around and to call the update
      # method below correctly
        self.jacobian_ = np.zeros((fom_size,rom_size), order='F')
        self.mapping = lambda x: x

        self.fomState0 = np.zeros(fom_size)
        self.fomState1 = np.zeros(fom_size)
 #       U, S, VT = np.linalg.svd(phi_matrix,full_matrices=False)
  #      self.mapping = lambda x: U[:,:rom_size]@np.diag(S[:rom_size])@x
   #     self.acoef = VT[:rom_size,:]
        self.numModes_ = rom_size

    def numModes(self): return self.numModes_
    def jacobian(self): return self.jacobian_

    def applyMapping(self, romState, fomState=None):
        if fomState is None:
            return self.mapping(romState)
        fomState[:] = self.mapping(romState)
#         fomState[:] = self.mapping(romState)

    def updateJacobian(self, romState):
        romStateLocal = romState.copy()
        # finite difference to approximate jacobian of the mapping
        self.applyMapping(romStateLocal, self.fomState0)
        eps = 0.001
        for i in range(self.numModes_):
            romStateLocal[i] += eps
            self.applyMapping(romStateLocal, self.fomState1)
            self.jacobian_[:, i] = (self.fomState1 - self.fomState0) / eps
            romStateLocal[i] -= eps


    def updateJacobian_fd(self, romState):
        romStateLocal = romState.copy()
        # finite difference to approximate jacobian of the mapping
        self.applyMapping(romStateLocal, self.fomState0)
        eps = 0.001
        for i in range(self.numModes_):
            romStateLocal[i] += eps
            self.applyMapping(romStateLocal, self.fomState1)
            self.jacobian_[:, i] = (self.fomState1 - self.fomState0) / eps
            romStateLocal[i] -= eps

# this is an auxiliary class that can be passed to solve
# to monitor the rom state.
class RomStateObserver:
        def __init__(self, rom_size, nsteps):
            self.romState_history = np.zeros([rom_size, nsteps+1])
        def __call__(self, timeStep, time, state):
            pass
            self.romState_history[:,timeStep] = state
            print(f"[LSPG] timeStep: {timeStep}, time: {time}")
        def get_romStates(self):
            return self.romState_history


################################################################################################################

def manifold_galerkin_rhs(params, romState, time, customDecoder, fom_rhs):

    U = customDecoder.U
    supportMeshIndices = params.diff_operators.supportMeshIndices
    front = customDecoder.f
    dfront = customDecoder.df
    Mr = params.sampleMeshSize
    M,rank = np.shape(U)
    Jf = np.zeros([Mr,rank],dtype=np.float32)
    f = np.zeros([M,1], dtype=np.float32)
    #customDecoder.updateJacobian( romState )
    phi = U @ romState
    #df = dfront(phi)
    #sampleMeshIndices = np.argpartition(-np.abs(df), Mr)[:Mr]
    # select samplemeshindizes with smallest phi values
    sampleMeshIndices = np.argpartition(np.abs(phi), Mr)[:Mr]
    #Jf = np.take(customDecoder.jacobian(),sampleMeshIndices, axis = 0)

    supportMeshIndices_now = supportMeshIndices[sampleMeshIndices,:]
    supportMeshIndices_now = pd.unique(supportMeshIndices_now.flatten())

    #df = customDecoder.df(phi,sampleMeshIndices)
    hyperreduced_jac( phi, U, sampleMeshIndices,  Jf)
    hyperreduced_front( phi, supportMeshIndices_now, f)

    #x = np.zeros_like(romState)
    b = fom_rhs(np.squeeze(f), time, sampleMeshIndices)
    #dat = np.linalg.lstsq(Jf.T@Jf,Jf.T@b)
    dat = np.linalg.lstsq(Jf, b)
    #PJf = projector @ Jf
    return dat[0]





def galerkin_odeint(rom_params, romState0, time, mapping, t0 =0 , fomReferenceState= None, sampleMeshIndices = None):
    # equivalent to odeint
    from scipy.integrate import solve_ivp as ode45
    print('#'*30+"\n Manifold Galerkin \n" + '#'*30)
    Nt = len(time)
    print(f"t0 ={t0}, Nsteps = {Nt}")
    customDecoder = mapping
    FOM = rom_params.galerkin.fom
    fom_rhs = lambda fomstate, t, sampleMeshIndices: FOM.params.rhs(fomstate,t,sampleMeshIndices= sampleMeshIndices, rhs = None)
    rhs = lambda time, romState : manifold_galerkin_rhs(FOM.params, romState, time, customDecoder, fom_rhs)
#    rhs = lambda romstate, t : manifold_galerkin_rhs(t, romstate, customDecoder, fom_rhs, sampleMeshIndices)

    ret = ode45(rhs, [time[0],time[-1]], romState0, t_eval =  time)
    romStates = ret.y.T
    print("Nr rhs calls:", ret.nfev)
    rom_params.info_dict["num_rhs_calls"] = ret.nfev
    #romStates = timesteper(rhs, [time[0], time[-1]], romState0, t_vec=time, tau = time[1]/8)

    return romStates





