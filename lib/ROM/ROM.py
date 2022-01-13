
from time import perf_counter
import numpy as np
import random
import scipy.sparse as sp
from lib.lspg_utils import FomAdapter, MyCustomMapper, runLspg, runhyperLspg, galerkin_odeint
from lib.FTR import construct_mapping
from numpy.linalg import svd
from scipy.linalg import qr
from scipy.interpolate import interp1d
from lib.fourier_koopman import fourier
from lib.quadtree import give_quadtree_adapted_samples, plot_quadtree_with_samples

class rom_params_class:
    """ ROM PARAMETERS
    All parameters are listed here needed for the reduced order models
    The Class also inherits the (Full Order Model) FOM parameters since it is needed for
    setting up the parameters of the RHS of the FOM
    """
    def __init__(self,fom_params, rom_size, fom_size, time_points, online_prediction_method = "lspg", mapping_type = "POD"):
        self.case = fom_params.case
        self.online_prediction_method = online_prediction_method
        self.mapping_type = mapping_type
        self.rom_size = rom_size
        self.fom_size = fom_size
        self.time.train_time_points = time_points[::2]
        self.time.test_time_points = time_points[1::2]
        self.time.time_points = time_points
        self.train_samples_idx = np.arange(0,len(time_points),2)
            # list of indices in snapshot matrix used for training the rom, i.e. producing reduced map
        self.time.nsteps = len(time_points)-1
        self.time.dt = time_points[1] - time_points[0]
        if online_prediction_method == "fourier-koopman":
            self.fourier_koopman.num_freqs = rom_size * 2
        if online_prediction_method[:4] == "lspg" or online_prediction_method[:8]=="galerkin":
            self.fom = fom_params
        if online_prediction_method == "lspg-hyper":
            self.fom.sampleMeshSize = 20  #default
        self.info_dict ={}
    class cleaning:
        """
        This class defines parameters for cleaning the data, before computing a ROM.
        This is often needed since numerical errors can propagate through the ROM procedure or
        some algorithms (like NN) assume that the data is rescaled between [0,1] .
        For example it may happen that the state is bounded from below by 0 (for example density).
        However due to numerical error this bounds can be exceeded.
        """
        method = "cut_and_rescale" #["cut_and_rescale", "rescale"]
        min_value_cut = 0.0 # minimal an maximal value the data will have after cut_and_rescale
        max_value_cut = 1.0
    class fourier_koopman:
        num_freqs = -1
        num_iterations = 1000
    def set_num_frequencies(self,num_freqs):
        """ 
        :param num_freqs: number of frequencies used for forecasting
        :return: number of frequencies set in params structure
        """
        self.fourier_koopman.num_freqs = num_freqs
    class DEIM:
        rhs = None
    class lspg:
        NonLinSolvTol = 1e-7       # tolerance of the gauss newton nonlinear solver
        NonLinSolveMaxIt = 100      # maximal number of iterations the solver runs
        mapping = None
        fom = None
        rom_timesteps_per_fom_timestep = 1 # is the number of lspg timesteps done for on dt of the full oder model
                                            # this is not the number of predicted timesteps in between
        sampling_method = "random"   # sample random or only the front
    class ftr:
        tol = 1e-7
        max_iter = 100
        opt_step_width = 4
        offset = 0              # offset of the levelset function when initializing the algorithm
    class time:
        pass
    class fom:
        pass
    class ECSW:
        network = None

    def set_up_lspg(self, fom_params, mapping):
        self.lspg.fom = FomAdapter(fom_params)
        self.lspg.mapping = mapping
        if self.mapping_type == "POD" or self.mapping_type == "FTR":
            self.sampled_rom_states = mapping.acoef

    def get_initial_state(self, t_test):
        """
        :return: Initial state of the reduced order model is the closest state of the sampled FOM states
        """
        sampled_time_points = self.time.train_time_points
        idx_left = np.searchsorted(sampled_time_points,t_test)-1
        closest_sampled_rom_state = self.sampled_rom_states[:, idx_left]
        return closest_sampled_rom_state, sampled_time_points[idx_left]

    def get_sampleMeshIndices(self):
        if self.lspg.sampling_method == "random":
            random.seed(22123)
            sampleMeshSize = self.fom.sampleMeshSize
            sampleMeshIndices = random.sample(range(1, self.fom_size - 1), sampleMeshSize)
            #sampleMeshIndices = np.append(sampleMeshIndices, [0, self.fom_size - 1])
            # sort for convenience, not necessarily needed
            sampleMeshIndices = np.sort(sampleMeshIndices)
            self.fom.sampleMeshIndices = sampleMeshIndices
        elif self.lspg.sampling_method == "front":
            print("h")
            # snapshot_approx = self.lspg.mapping.applyMapping(rom_state)
            # snapshot_approx = snapshot_approx.reshape(self.fom.geom.N)
            # sampleMeshIndices = give_quadtree_adapted_samples(snapshot_approx, self.fom.sampleMeshSize)
            # plot_quadtree_with_samples(self.fom.geom.Xgrid[0], self.fom.geom.Xgrid[1], snapshot_approx , sampleMeshIndices)
        else:
            assert(True), "Method %s not implemented"%self.spg.sampling_method

        return sampleMeshIndices

"""
The following mappers are subclasses of the motherclass MyCustomMapper defined in lspg_utils.py
Since there are subclasses they only implement the relevant differences to the motherclass.
For example the different mapping and its jacobian.
"""
class FTR_Mapper(MyCustomMapper):
    def __init__(self, fom_size, rom_size, phi_matrix, front_fun, dfront):
        super(FTR_Mapper, self).__init__(fom_size =fom_size, rom_size = rom_size)
        self.mapping, self.acoef = construct_mapping(phi_matrix, front_fun, rank=rom_size, substract_mean=False)
        self.df=dfront
        self.f = front_fun
        self.phi = phi_matrix
        U = np.linalg.svd(phi_matrix, full_matrices=False)[0]
        self.U = U[:,:rom_size]
    def applyMapping(self, romState, fomState=None):
        if fomState is None:
            return self.mapping(romState)
        fomState[:] = self.mapping(romState)
    def updateJacobian_(self, romState, df = None):
        if df is None:
            df = self.df(self.U @ romState)
        else:
            df[:] = self.df(self.U @ romState)
        jacobi = (df * self.U.T).T
        self.jacobian_[:,:]= jacobi[:,:]

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



class POD_Mapper(MyCustomMapper):
    def __init__(self, fom_size, rom_size, fom_snapshots):
        super(POD_Mapper, self).__init__(fom_size=fom_size, rom_size=rom_size)
        [U, S, VT] = svd(fom_snapshots, full_matrices=False)
        basis = U[:,:rom_size]@np.diag(S[:rom_size])
        self.mapping = lambda romstate: basis@romstate
        self.acoef = VT[:rom_size,:]
        self.sigvals = S

"""
All functions needed for this module
"""
def clean_and_split_data(rom_params, fom_states):
    """
    This function prepares your data for the reduced order model:
        + cleans the data according to ROMs constraints (e.g. normalizes to [0,1] or smooths data)
        + splits data in to train and validation/test sets
    :param rom_params:
    :param q:
    :return:
    """
    fom_states = clean_data(rom_params.cleaning.method, fom_states, min_val=rom_params.cleaning.min_value_cut,
                           max_val=rom_params.cleaning.max_value_cut)
    sample_points = np.arange(0,np.size(fom_states,1))
    train_points = sample_points[rom_params.train_samples_idx]
    test_points = [item for item in sample_points if item not in train_points]
    fom_states_train, fom_states_test = split_data(fom_states, fom_sample_points=sample_points,
                                                   train_sample_points = train_points,
                                                   test_sample_points = test_points)
    return fom_states_train, fom_states_test

def clean_data(method, fom_states, min_val = 0.0, max_val = 1.0):
    """
    This function cleans the data for you, before computing a ROM from it.
    This is often needed since numerical errors can propagate through the ROM procedure or
    some algorithms (like NN) assume that the data is rescaled between [0,1] .
    For example it may happen that the state is bounded from below by 0 (for example density).
    However due to numerical error this bounds can be exceeded.
    :param method: either "cut_and_rescale" or "rescale"
    :param fom_states: states you want to clean. should be a 2d array of size fom_state_size x number_of_snaphots
    :param min_val (optional): minimum value for cutting, default is 0.0
    :param max_val (optional): maximum value for cutting, default is 1.0
    :return:
    """
    clean_fom_states = []
    for state in fom_states.T:
        if method == "cut_and_rescale":
            state = np.where(state > min_val, state, min_val)
            state = np.where(state < max_val, state, max_val)
            state = state - np.min(state.flatten())
            state = state / np.max(state.flatten())
        elif method == "rescale":  # normalizes to 1
            state = state - np.min(state.flatten())
            state = state / np.max(state.flatten())
        else:  # do nothing
            pass
        clean_fom_states.append(state)
    return np.asarray(clean_fom_states)

def split_data(fom_states, fom_sample_points, train_sample_points, test_sample_points):
    """
    splits the fom_states in training and validation/test samples
    according to the given training and test parameters.
    Note currently the sample points are simply time points of the sampled PDE solution
    :param fom_states: [q(mu_1), q(mu_2), ..., q(mu_N)] ... q is the solution of a PDE and mu is the sampled parameter, N the number of samples
    :param fom_sample_points: all samples mu_1, ..., mu_N (e.g time points, reynolds numbers, etc.)
    :param train_sample_points: a subset of fom_sample_points used for samples the rom needs for dimension reduction
    :param test_sample_points: a subset of fom_sample_points used for testing the predictions of the rom
    :return:
    """
    # all sampled data points
    fom_sample_points = list(fom_sample_points)
    # parameters which are given to the rom as offline stage samples
    train_sample_points = list(train_sample_points)
    # points which we want to predict with the ROM
    test_sample_points = list(test_sample_points)
    # first check if the union of the train and test set is a subset of the sampled parameters
    test_train_union = test_sample_points + train_sample_points
    assert(set(test_train_union).issubset(fom_sample_points)), "Mr or Mrs. programmer, " + \
                                                               " the test and train samples are not in the set of all samples!!"
    test = []
    train = []
    for sample_point, state in zip(fom_sample_points, fom_states):
        if sample_point in train_sample_points:
            train.append(state)
        else:
            test.append(state)

    return np.asarray(train).T, np.asarray(test).T


def generate_reduced_map(rom_params, fom_snapshots, data_dir = None):
        from lib.FTR import simple_FTR as FTR

        import torch as to
        from scipy.io import savemat
        rom_size = rom_params.rom_size
        if rom_params.mapping_type == "POD":
            mapping = POD_Mapper(rom_params.fom_size, rom_size, fom_snapshots)
        elif rom_params.mapping_type == "FTR":
            width = 100

            front = lambda x: (1-np.tanh(x))*0.5
            dfront = lambda x:  -0.5/(np.cosh(x)**2)
            #front_torch = lambda x: to.sigmoid(x * width)
            Nsamples = np.size(fom_snapshots,-1)
            phi = FTR( np.reshape(fom_snapshots,[*rom_params.fom.geom.N, -1]), f=front, rank=rom_size,
                       tol=rom_params.ftr.tol, max_iter=rom_params.ftr.max_iter, dt=rom_params.ftr.opt_step_width,
                        nt_show=Nsamples//2, plot_step=2000, print_step=1, offset= rom_params.ftr.offset)
            phi_matrix = np.reshape(phi, [rom_params.fom_size, -1])  # reshape to snapshot matrix
            mapping = FTR_Mapper(rom_params.fom_size, rom_size, phi_matrix, front_fun = front, dfront =dfront )
            if data_dir is not None:
                #savemat(data_dir + '/' + rom_params.case + '_FTR_mapping-%dDofs.mat'%rom_size,
                #                             {'phi_matrix': phi_matrix, 'rom_size' : rom_size, 'front_width': width})
                fname = data_dir + '/' + rom_params.case + '_FTR_mapping-%dDofs.mat' % rom_size
                with open(fname, 'wb') as f:
                    np.save(f, phi_matrix)
                    np.save(f, rom_size)
        elif rom_params.mapping_type == "sPOD":
            print("implement me")
        else:
            pass

        return mapping


def load_reduced_map(rom_params, data_dir):
    from scipy.io import loadmat
    rom_size = rom_params.rom_size
    if rom_params.mapping_type == "POD":
        assert(False), "not implemented"
    elif rom_params.mapping_type == "FTR":
        fname = data_dir + '/' + rom_params.case + '_FTR_mapping-%dDofs.mat' % rom_size
        with open(fname, 'rb') as f:
            phi_matrix = np.load(f)
        #dat = loadmat(data_dir + '/' + rom_params.case + '_FTR_mapping-%dDofs.mat' % rom_size )
        #phi_matrix = dat["phi_matrix"]
        front = lambda x: (1 - np.tanh(x)) * 0.5
        dfront = lambda x: -0.5 / (np.cosh(x) ** 2)

        mapping = FTR_Mapper(rom_params.fom_size, rom_size, phi_matrix, front_fun=front, dfront = dfront)
    return mapping



def get_proj_err(snapshots, mapping):
    #from lib.numba_ftr_mapper import *
    from scipy.optimize import least_squares
    from numpy.linalg import norm
    acoef = mapping.acoef
    train_snapshots = np.asarray([mapping.applyMapping(a) for a in acoef.T])
    test_snapshots = []
    for snapshot in snapshots.T:
        fitfun = lambda alpha: snapshot - mapping.applyMapping(alpha)
        # look for the closest training snapshot to determine starting point of minimization
        res_min = 1
        for idx, qtrain in enumerate(train_snapshots):
            res = norm(qtrain - snapshot)/norm(snapshot)
            if res <= res_min:
                idx_min = idx
                res_min = res
        res = least_squares(fitfun, acoef[:,idx_min])
        qtilde = mapping.applyMapping(res.x)
        test_snapshots.append(qtilde)
    snapshots_approx = np.asarray(test_snapshots).T
    return norm(snapshots_approx-snapshots)/norm(snapshots)

def galerkin_timesteper(rom_params, mapping):
    """
    :param rom_params: params of the reduced order ODE system
    :return: returns the integrated romStates
    """
    from lib.rhs_advection_reaction_diffusion import \
        rhs_advection_reaction_diffusion_2D_periodic_hyperreduced_numba_wrapper as rhs_fun
    from lib.rhs_advection_reaction_diffusion import \
        jacobian_rhs_advection_reaction_diffusion_2D_periodic_hyperreduced_numba_wrapper as jac_fun
    if rom_params.online_prediction_method=="galerkin":
        sampleMeshIndices = np.arange(0, rom_params.fom_size)
        rom_params.fom.sampleMeshIndices = sampleMeshIndices
        rom_params.fom.sampleMeshSize = rom_params.fom_size
    elif rom_params.online_prediction_method == "galerkin-hyper":
        sampleMeshIndices = rom_params.get_sampleMeshIndices()


    rom_params.set_up_lspg(rom_params.fom, mapping)
    mapper = rom_params.lspg.mapping

    dt_min = rom_params.fom.time.dt/rom_params.lspg.rom_timesteps_per_fom_timestep
    romStates_test = []
    rom_params.sampled_rom_states = mapping.acoef
    rom_params.lspg.mapping = mapping
    #initial_rom_state, t0 = rom_params.get_initial_state(dt_min)
    initial_rom_state = rom_params.sampled_rom_states[:, 0]
    t0 = 0
    romStates_test = galerkin_odeint(rom_params, initial_rom_state, rom_params.time.test_time_points, mapper, t0=t0, fomReferenceState=None, sampleMeshIndices = sampleMeshIndices)
    return np.asarray(romStates_test).T




def lspg_timesteper_(rom_params, mapping):
    """
    :param rom_params: params of the reduced order ODE system
    :return: returns the integrated romStates
    """
    from lib.rhs_advection_reaction_diffusion import \
        rhs_advection_reaction_diffusion_2D_periodic_hyperreduced_numba_wrapper as rhs_fun
    from lib.rhs_advection_reaction_diffusion import \
        jacobian_rhs_advection_reaction_diffusion_2D_periodic_hyperreduced_numba_wrapper as jac_fun
    if rom_params.online_prediction_method=="lspg":
        sampleMeshIndices = np.arange(0, rom_params.fom_size)
        rom_params.fom.sampleMeshIndices = sampleMeshIndices
        rom_params.fom.sampleMeshSize = rom_params.fom_size
    elif rom_params.online_prediction_method == "lspg-hyper":
        sampleMeshIndices = rom_params.get_sampleMeshIndices()

    rom_params.fom.rhs = lambda params, qvals, time, rhs: rhs_fun(params, qvals, time,
                                                                      sampleMeshIndices=sampleMeshIndices, rhs=rhs)
    rom_params.fom.rhs_jacobian = lambda params, qvals, time, dq, jac: jac_fun(params, qvals, time, dq,
                                                                                   sampleMeshIndices=sampleMeshIndices,
                                                                                   jac=jac)
    rom_params.set_up_lspg(rom_params.fom, mapping)
    mapper = rom_params.lspg.mapping

    Tol = rom_params.lspg.NonLinSolvTol
    MaxIt = rom_params.lspg.NonLinSolveMaxIt
    dt_min = rom_params.fom.time.dt/rom_params.lspg.rom_timesteps_per_fom_timestep
    romStates_test = []
    rom_params.sampled_rom_states = mapping.acoef
    rom_params.lspg.mapping = mapping
    for t_test in rom_params.time.test_time_points:
        initial_rom_state, t_train = rom_params.get_initial_state(t_test)
        delta_t = t_test - t_train
        t0 = t_train
        assert ( delta_t > 0 ), "Oh no! We should only step forward in time! (dt < 0)"
        if delta_t > dt_min:
            # if the time interval is larger then the smallest allowed stepsize,
            # then we need to do more timesteps
            nsteps = np.round(delta_t/dt_min)
            dt = delta_t/nsteps
        else:
            dt = delta_t
            nsteps = 1
        if rom_params.online_prediction_method=="lspg-hyper":
            romStates = runhyperLspg(rom_params.lspg.fom, initial_rom_state, dt, nsteps, mapper, t0=t0, nlsTol=Tol,
                                    nlsMaxIt=MaxIt, fomReferenceState=None, sampleMeshIndices = sampleMeshIndices)
        else:
            romStates = runLspg(rom_params.lspg.fom, initial_rom_state, dt, nsteps, mapper, t0=t0, nlsTol=Tol, nlsMaxIt=MaxIt, fomReferenceState=None)

        romStates_test.append(romStates[:,-1])
    return np.asarray(romStates_test).T



def lspg_timesteper(rom_params, mapping):
    """
    :param rom_params: params of the reduced order ODE system
    :return: returns the integrated romStates
    """
    from lib.rhs_advection_reaction_diffusion import \
        rhs_advection_reaction_diffusion_2D_periodic_hyperreduced_numba_wrapper as rhs_fun
    from lib.rhs_advection_reaction_diffusion import \
        jacobian_rhs_advection_reaction_diffusion_2D_periodic_hyperreduced_numba_wrapper as jac_fun
    if rom_params.online_prediction_method=="lspg":
        sampleMeshIndices = np.arange(0, rom_params.fom_size)
        rom_params.fom.sampleMeshIndices = sampleMeshIndices
        rom_params.fom.sampleMeshSize = rom_params.fom_size
    elif rom_params.online_prediction_method == "lspg-hyper":
        sampleMeshIndices = rom_params.get_sampleMeshIndices()

    rom_params.fom.rhs = lambda params, qvals, time, rhs: rhs_fun(params, qvals, time,
                                                                      sampleMeshIndices=sampleMeshIndices, rhs=rhs)
    rom_params.fom.rhs_jacobian = lambda params, qvals, time, dq, jac: jac_fun(params, qvals, time, dq,
                                                                                   sampleMeshIndices=sampleMeshIndices,
                                                                                   jac=jac)
    rom_params.set_up_lspg(rom_params.fom, mapping)
    mapper = rom_params.lspg.mapping

    Tol = rom_params.lspg.NonLinSolvTol
    MaxIt = rom_params.lspg.NonLinSolveMaxIt
    dt_min = rom_params.fom.time.dt/rom_params.lspg.rom_timesteps_per_fom_timestep
    romStates_test = []
    rom_params.sampled_rom_states = mapping.acoef
    rom_params.lspg.mapping = mapping
    initial_rom_state = rom_params.sampled_rom_states[:, 0]
    t0 = 0
    dt = rom_params.fom.time.dt
    nsteps = len(rom_params.fom.time.t)-1
    romStates = runhyperLspg(rom_params.lspg.fom, initial_rom_state, dt, nsteps, mapper, t0=t0, nlsTol=Tol,
                                    nlsMaxIt=MaxIt, fomReferenceState=None, sampleMeshIndices = sampleMeshIndices)
    return romStates

def solve_ROM(params, mapping):
    """
    This function implements the different methods for reduced order modeling
    :param params:
    :return:
    """
    print("Solving Reduced Order Model")

    time_odeint = perf_counter()
    rom_params = params.rom
    if rom_params.online_prediction_method == "fourier-koopman":
        N_snapshots = rom_params.time.nsteps
        rom_params.set_num_frequencies(rom_params.rom_size*2)
        f = fourier(num_freqs=rom_params.fourier_koopman.num_freqs)
        rom_state_timeseries = mapping.acoef.T
        f.fit(rom_state_timeseries, iterations=rom_params.fourier_koopman.num_iterations)
        dt = np.diff(rom_params.time.train_time_points)
        assert(np.all(np.abs(dt - dt[0])<1e-14)), "I dont agree master, time step between train samples should be equal!!!"
        timepoints_predict = rom_params.time.test_time_points/ dt[0]

        #np.arange(1, N_snapshots) + 0.5
        romState_hist = f.predict(timepoints=timepoints_predict).T
    elif rom_params.online_prediction_method[0:4] == "lspg":
        romState_hist = lspg_timesteper(rom_params, mapping)
    elif rom_params.online_prediction_method[:8] == "galerkin":
        romState_hist = galerkin_timesteper(rom_params, mapping)
    elif rom_params.online_prediction_method == "POD-DEIM" and rom_params.mapping_type == "POD":
        romState_hist = DEIM_timesteper(params, mapping)

    qtilde_hist = np.squeeze(np.asarray([ mapping.applyMapping(romState) for romState in romState_hist.T])).T
    time_odeint = perf_counter()-time_odeint
    print("t_cpu = %1.3f"%time_odeint)

    return qtilde_hist, romState_hist




def POD_DEIM(params_list, fom_snapshots, ndeim = None):
    from scipy.integrate import solve_ivp as ode45
    from lib.rhs_advection_reaction_diffusion import rhs_advection_reaction_diffusion_2D_periodic as rhs_ARD_2D

    params = params_list[0]
    rom_size = params.rom.rom_size

    if not ndeim:
        ndeim = rom_size

    [U, S, VT] = svd(fom_snapshots, full_matrices=False)
    U = U[:, :rom_size]
    UT = U.T



    if params.case == "reaction1D":
        # linear term
        Dxx = params.diff_operators.Dxx_mat
        L = UT @ Dxx @ U
        # nonlinear term nll(u) = 8 / delta. ^ 2 * u. ^ 2. * (1 - u)
        NLL = lambda q:    q**2 * (1 - q)
        NLL_mat = NLL(fom_snapshots)
        [XI, S_NL, W] = svd(NLL_mat, full_matrices=0)
        # [_, _, pivot] = qr(NLL_mat.T, pivoting=True)
        # P = np.zeros([params.fom_size,ndeim])
        # for i,p in enumerate(pivot[:ndeim+1]):
        #     P[p,i]=1
        # XI_m = XI[:, : ndeim+1]
        # P_NLL = UT@ (XI_m@np.linalg.inv(P.T@XI_m))
        # PU = P.T@U
        nmax = np.argmax(np.abs(XI[:, 0]))
        n = params.fom_size
        r = ndeim
        XI_m = XI[:, 0].reshape(n, 1)
        z = np.zeros((n, 1))
        P = np.copy(z)
        P[nmax] = 1

        # DEIM points 2 to r
        for jj in range(1, r):
            c = np.linalg.solve(P.T @ XI_m, P.T @ XI[:, jj].reshape(n, 1))
            res = XI[:, jj].reshape(n, 1) - XI_m @ c
            nmax = np.argmax(np.abs(res))
            XI_m = np.concatenate((XI_m, XI[:, jj].reshape(n, 1)), axis=1)
            P = np.concatenate((P, z), axis=1)
            P[nmax, jj] = 1
        P_NL = U.T @ (XI_m @ np.linalg.inv(P.T @ XI_m))  # nonlinear projection
        PU = P.T @ U

        a0 = UT@params.inicond
        initial_rom_state = a0
        time = params.time.t

        acoef_list =[]
        cpu_time = perf_counter()
        for parameter in params_list:
            rhs_rom_POD_DEIM_eff = lambda t, a: L @ a + 8 / parameter.reaction_const ** 2 * P_NL @ NLL(PU @ a)
            ret = ode45(rhs_rom_POD_DEIM_eff, [time[0], time[-1]], initial_rom_state, t_eval=time)
            acoef = ret.y
            acoef_list.append(acoef)

        cpu_time = (perf_counter() - cpu_time)

    elif params.case == "pacman":
        Nt = len(params.time.t)
        time = params.time.t

        L2r = np.zeros([params.rom.rom_size, params.rom.rom_size])
        Dx = params.diff_operators.Dx_mat
        Dy = params.diff_operators.Dy_mat
        Dxx = params.diff_operators.Dxx_mat
        Dyy = params.diff_operators.Dyy_mat
        Nint = 1000
        L1r = np.zeros([params.rom.rom_size, params.rom.rom_size, Nint])
        time_int = np.linspace(-0.1,time[-1]+0.1,Nint)
        DxU = Dx@U
        DyU = Dy@U
        # for it, t in enumerate(time_int):
        #     [ux, uy] = params.velocity_field(t)
        #     L1 = sp.diags(ux) @ DxU + sp.diags(uy) @ DyU
        #     L1r[:, :, it] = UT @ L1
        # L2r = UT @ (Dxx + Dyy) @ U
        # L1r = interp1d(time_int, L1r)
        #
        # # nonlinearity
        # NLL = lambda q: q ** 2 * (q - 1)
        #
        # NLL_mat = NLL(fom_snapshots)
        # [XI, _, _] = svd(NLL_mat, full_matrices=0)
        # #XI_m = XI[:,:ndeim]
        # # _, _, pivot = qr(NLL_mat.T, pivoting=True)
        # # P = np.zeros([params.fom_size, ndeim])
        # # for i in range(ndeim):
        # #     P[pivot[i], i] = 1
        #
        # nmax = np.argmax(np.abs(XI[:, 0]))
        # n = params.fom_size
        # r = ndeim
        # XI_m = XI[:, 0].reshape(n, 1)
        # z = np.zeros((n, 1))
        # P = np.copy(z)
        # P[nmax] = 1
        #
        # # DEIM points 2 to r
        # for jj in range(1, r):
        #     c = np.linalg.solve(P.T @ XI_m, P.T @ XI[:, jj].reshape(n, 1))
        #     res = XI[:, jj].reshape(n, 1) - XI_m @ c
        #     nmax = np.argmax(np.abs(res))
        #     XI_m = np.concatenate((XI_m, XI[:, jj].reshape(n, 1)), axis=1)
        #     P = np.concatenate((P, z), axis=1)
        #     P[nmax, jj] = 1
        #
        # P_NL = U.T @ (XI_m @ np.linalg.inv(P.T @ XI_m))  # nonlinear projection
        # PU = P.T @ U

        a0 = UT @ params.inicond
        initial_rom_state = a0


        acoef_list = []
        cpu_time = perf_counter()
        for parameter in params_list:
            diff = parameter.diffusion_const
            react = parameter.reaction_const

            #rhs_rom_POD_DEIM_eff = lambda t, a: (-L1r(t) + diff*L2r) @ a - react * P_NL @ NLL(PU @ a)
            rhs_rom_POD_DEIM_eff = lambda t, a: UT @ rhs_ARD_2D(parameter, U @ a, t)
            ret = ode45(rhs_rom_POD_DEIM_eff, [time[0], time[-1]], initial_rom_state, rtol=1e-7 ,t_eval=time)
            acoef = ret.y
            acoef_list.append(acoef)

        cpu_time = (perf_counter() - cpu_time)


    acoefs = np.concatenate(acoef_list, axis=1)

    return acoefs, U, cpu_time