import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from time import perf_counter
from matplotlib import rc
import torch as to
from scipy.integrate import odeint
# if run from within a build of mypressio, need to append to python path
import pathlib, sys
file_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(file_path) + "/..")         # to access doFom

rc('font',**{'family':'serif','serif':['Helvetica'],'size': 24})
from lib.FOM import *
from scipy.linalg import pinv
from lib.FTR import construct_mapping, simple_FTR
from pressio4py import rom as pressio_rom, logger
from pressio4py import solvers as solvers
from lib.numba_ftr_mapper import *
import pandas as pd


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
# LSPG to monitor the rom state.
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


def manifold_galerkin_ECSW_rhs(params, romState, time, customDecoder, fom_rhs, network):

    U = customDecoder.U
    supportMeshIndices = params.diff_operators.supportMeshIndices
    front = customDecoder.f
    dfront = customDecoder.df
    Mr = params.sampleMeshSize
    M,rank = np.shape(U)

    #customDecoder.updateJacobian( romState )
    phi = U @ romState
    alpha = np.squeeze(network(romState))
    sampleMeshIndices = np.nonzero(alpha > 0)[0]
    Mr = len(sampleMeshIndices)
    Jf = np.zeros([Mr, rank], dtype=np.float32)
    f = np.zeros([M, 1], dtype=np.float32)
    # select samplemeshindizes with smallest phi values
    #sampleMeshIndices = np.argpartition(np.abs(alpha), Mr)[:Mr]
    #Jf = np.take(customDecoder.jacobian(),sampleMeshIndices, axis = 0)


    supportMeshIndices_now = supportMeshIndices[sampleMeshIndices,:]
    supportMeshIndices_now = pd.unique(supportMeshIndices_now.flatten())

    #df = customDecoder.df(phi,sampleMeshIndices)
    hyperreduced_jac_ECSW( phi, U, sampleMeshIndices,  Jf, alpha)
    hyperreduced_front( phi, supportMeshIndices_now, f)

    #x = np.zeros_like(romState)
    b = fom_rhs(np.squeeze(f), time, sampleMeshIndices)
    dat = np.linalg.lstsq(Jf.T@Jf,Jf.T@b)
    #dat = np.linalg.lstsq(Jf, b)
    #PJf = projector @ Jf
    return dat[0]



def galerkin_odeint(rom_params, romState0, time, mapping, t0 =0 , fomReferenceState= None, sampleMeshIndices = None):
    # equivalent to odeint
    from scipy.integrate import solve_ivp as ode45
    print('#'*30+"\n Manifold Galerkin \n" + '#'*30)
    Nt = len(time)
    print(f"t0 ={t0}, Nsteps = {Nt}")
    customDecoder = mapping
    FOM = rom_params.lspg.fom
    fom_rhs = lambda fomstate, t, sampleMeshIndices: FOM.params.rhs(fomstate,t,sampleMeshIndices= sampleMeshIndices, rhs = None)

    if rom_params.ECSW.network:
        rhs = lambda time, romState: manifold_galerkin_ECSW_rhs(FOM.params, romState, time, customDecoder, fom_rhs, rom_params.ECSW.network)
    else:
        rhs = lambda time, romState : manifold_galerkin_rhs(FOM.params, romState, time, customDecoder, fom_rhs)
#    rhs = lambda romstate, t : manifold_galerkin_rhs(t, romstate, customDecoder, fom_rhs, sampleMeshIndices)

    ret = ode45(rhs, [time[0],time[-1]], romState0, t_eval =  time)
    romStates = ret.y.T
    print("Nr rhs calls:", ret.nfev)
    rom_params.info_dict["num_rhs_calls"] = ret.nfev
    #romStates = timesteper(rhs, [time[0], time[-1]], romState0, t_vec=time, tau = time[1]/8)

    return romStates






#################################################################################################################
def runhyperLspg(FOM, romState0, dt, nsteps, mapping, t0 =0 , nlsTol = 1e-7, nlsMaxIt = 100, fomReferenceState= None, sampleMeshIndices = None):

  logger.initialize(logger.logto.terminal, "null")
  logger.setVerbosity([logger.loglevel.info])
  # fom reference state: here it is zero
  if fomReferenceState == None:
      fomReferenceState = np.zeros(np.prod(FOM.params.geom.N))

  # create ROM state by projecting the fom initial condition
  romState = romState0

  # 1. create a decoder
  customDecoder = pressio_rom.Decoder(mapping, 'MyCustomMapper')

  # 2. create the masked lspg problem with Euler Backward
  problem = pressio_rom.lspg.unsteady.hyperreduced.ProblemEuler(FOM, customDecoder,
                                                  romState, fomReferenceState,
                                                  sampleMeshIndices)
  # linear and non linear solver
  lsO = MyLinSolver()
  nlsO = solvers.GaussNewton(problem, romState, lsO)
  nlsO.setTolerance(nlsTol)
  #nlsO.setResidualRelativeTolerance(nlsTol)
  #nlsO.setMaxIterations(nlsMaxIt)
  nlsO.setStoppingCriterion(solvers.stop.whenCorrectionAbsoluteNormBelowTolerance)

  # create object to monitor the romState at every iteration
  myObs = RomStateObserver(np.size(romState), int(nsteps))

  # solve the problem
  pressio_rom.lspg.solveNSequentialMinimizations(problem, romState, t0, dt, int(nsteps), myObs, nlsO)

  return myObs.get_romStates()

#################################################################################################################
def runLspg(FOM, romState0, dt, nsteps, mapping, t0 =0 , nlsTol = 1e-7, nlsMaxIt = 100, fomReferenceState= None):

    # ----------------------------------------
    # create a custom decoder using the mapper passed as argument
    #customDecoder = rom.Decoder(np.ndarray([np.prod(params.geom.N), Mapper.numModes()], order='F') )
    logger.initialize(logger.logto.terminal, "null")
    logger.setVerbosity([logger.loglevel.info])

    customDecoder = pressio_rom.Decoder(mapping, 'MyCustomMapper')
    # fom reference state: here it is zero
    if not fomReferenceState:
        fomReferenceState = np.zeros(FOM.fom.fom_size)

    # create ROM state by projecting the fom initial condition
    romState = romState0

    # create LSPG problem
    problem = pressio_rom.lspg.unsteady.default.ProblemEuler(FOM, customDecoder,
                                                             romState,fomReferenceState)


    # create the Gauss-Newton solver
    nonLinSolver = solvers.GaussNewton(problem, romState, MyLinSolver())
    # set tolerance and convergence criteria
    nonLinSolver.setMaxIterations(nlsMaxIt)
    nonLinSolver.setStoppingCriterion(solvers.stop.whenCorrectionAbsoluteNormBelowTolerance)

    # create object to monitor the romState at every iteration
    myObs = RomStateObserver(np.size(romState),int(nsteps))
    # solve problem
    pressio_rom.lspg.solveNSequentialMinimizations(problem, romState, t0, dt, int(nsteps), myObs, nonLinSolver)

    # after we are done, use the reconstructor object to reconstruct the fom state
    # get the reconstructor object: this allows to map romState to fomState
    #fomRecon = problem.fomStateReconstructor()
    return myObs.get_romStates()


if __name__ == "__main__":
    from lib.cases import my_cases
    from lib.ROM import FTR_Mapper

    logger.initialize(logger.logto.terminal, "null")
    logger.setVerbosity([logger.loglevel.info])

    case = "pacman"
    rom_size = 6
    dim = 2
    params = my_cases(case)
    # --- 1. FOM ---#
    time_odeint = perf_counter()
    params, q = solve_FOM(params)
    time_odeint = perf_counter() - time_odeint
    q = np.reshape(q, [*params.geom.N[:dim], -1])

    X = params.geom.Xgrid
    plt.pcolormesh(q[:, :, len(params.time.t) // 2])
    # --- 2. train a ROM ---#
    params.forward = lambda x: to.sigmoid(x*100)
    params.front = lambda x: 1/(1 + np.exp(-x*100))
    phi = simple_FTR( q, params.front, rank=rom_size, max_iter=10, dt=0.07, nt_show=80, plot_step=50, print_step=50)
    phi_matrix = np.reshape(phi,[-1,len(params.time.t)]) # reshape to snapshot matrix
    p = np.zeros( np.prod(params.geom.N) )
    # --- 3. LSPG ROM ---#
    romTimeStepSize = params.time.dt/10
    romNumberOfSteps = int(params.time.T/ romTimeStepSize)
    mapping = FTR_Mapper(params.fom_size, rom_size, phi_matrix, front_fun=params.front, dfront=params.front)

    params.rom.fom.rhs = lambda params, qvals, time, rhs: rhs_fun(params, qvals, time,
                                                                  sampleMeshIndices=sampleMeshIndices, rhs=rhs)
    params.rom.fom.rhs_jacobian = lambda params, qvals, time, dq, jac: jac_fun(params, qvals, time, dq,
                                                                               sampleMeshIndices=sampleMeshIndices,
                                                                               jac=jac)
    params.rom.set_up_lspg(params.rom.fom, mapping)
    #lspg_params = lspg(fom_params, fom_size=np.prod(fom_params.geom.N), rom_size=rom_size, front=fom_params.forward, phi_matrix= phi_matrix)
    romState_hist = runLspg(params.rom.lspg.fom, mapping.acoef[:,0], romTimeStepSize, nsteps=romNumberOfSteps, mapping=mapping, t0=0)
#    approximatedState, romState = runLspg(fom_params, dt = romTimeStepSize, nteps = romNumberOfSteps, Mapper = lspg_params.mapping, romState0 = lspg_params.mapping.acoef[:,0])

    qtilde = np.reshape(approximatedState,params.geom.N)
    fomFinalState = np.reshape(q[..., -1], params.geom.N)
    fig,axs = plt.subplots(1,2)
    axs[0].pcolormesh(qtilde)
    axs[1].pcolormesh(fomFinalState)
    # compute l2-error between fom and approximate state
    #fomFinalState = np.reshape(q[...,-1],-1)
