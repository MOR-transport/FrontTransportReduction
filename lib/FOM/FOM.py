import pathlib, sys

file_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(file_path) + "/..")
import numpy as np
from numpy import linspace, pi,meshgrid, reshape, sin, cos
import torch as to
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from time import perf_counter
from scipy.io import savemat
from numpy.fft import fft,ifft,fft2,ifft2
from lib.plot_utils import show_animation
from lib.rhs_advection_reaction_diffusion import \
    rhs_advection_reaction_diffusion_2D_periodic_hyperreduced_numba_wrapper, \
    rhs_advection_reaction_diffusion_2D_periodic
from lib.rhs_react_1D import \
    rhs_advection_reaction_diffusion_1D_periodic_hyperreduced_numba_wrapper, \
    rhs_advection_reaction_diffusion_1D_periodic

# choose dimension
dim = 2


def derivative(N, h, coefficient, boundary="periodic"):
    """
    Compute the discrete derivative for periodic BC
    """
    dlow = -(np.size(coefficient) - 1) // 2
    dup = - dlow+1

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


def give_primary_observables(params, q):
    '''
    :param params: parameter structure of nstokes
    :param q: statevector at given time
    :return: dictionary with primary observables
    '''
    q = np.reshape(q, [*params.geom.N, params.Neqn]).squeeze()
    rho = q[..., 0]
    u = q[..., 1]
    p = q[..., 2]
    Y = q[..., 3]
    return {'rho': rho, 'u': u, 'p': p, 'Y': Y}


def rk4(tau, t, x, rhs):
    k1 = tau * rhs( t, x)
    k2 = tau * rhs( t + tau / 2, x + k1 / 2)
    k3 = tau * rhs( t + tau / 2, x + k2 / 2)
    k4 = tau * rhs( t + tau, x + k3)
    return x + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

def timesteper(rhs,t_interval, q0, t_vec, tau):
    q = q0
    qhist = [q0]
    t0 = t_interval[0]
    tend = t_interval[1]
    T = tend - t0
    Nsteps = int(np.round(T/tau))
    dt_write = np.diff(t_vec)
    assert( np.all(np.abs(dt_write - dt_write[0])<1e-12) ), "distance between time samples should be equal and multiple of tau"
    nt_write = int(dt_write[0]/tau)
    assert(np.abs(dt_write[0]-tau*nt_write)<1e-12), "distance between time samples is not a multiple of tau"
    t = t0
    for nt in range(Nsteps):
        t +=tau
        q = rk4(tau,t,q,rhs)
        if nt % nt_write == 0:
            qhist.append(q)
    return np.asarray(qhist)


######################################################
# PARAMETERS
######################################################
class params_class:
    def __init__(self, pde = "advection",dim = 2 ,L=[1,1] ,N = [2**7, 2**6], T=0.1, Nt=300 , case="bunsen_flame"):
        # init geometry
        self.geom.L = np.asarray(L) # domain size
        self.geom.N = np.asarray(N) # number of spatial points in each dimension
        self.time.T = T
        self.time.dt = T/Nt
        self.geom.dX = self.geom.L/self.geom.N
        self.geom.X = [np.arange(0, self.geom.L[d],self.geom.dX[d]) for d in range(dim)]
        self.geom.K = [np.fft.fftfreq(self.geom.N[k],d=self.geom.dX[k]) for k in range(dim)]
        self.geom.Xgrid = meshgrid(*self.geom.X, indexing='ij')
        # init time
        self.time.t = np.arange(0,self.time.T,self.time.dt)
        #self.advection_speed= 0.1#0.1 # velocity of moving vortex
        self.advection_speed = 10 # 0.1 # velocity of moving vortex
        #self.w0 = [45, 45]   # initial vorticity
        self.w0 = [1e3, 1e3]  # initial vorticity
        self.r0 = 0.0005    # initial size of vortex
        self.decay_constant = T*3#0.2 # decay of the initial vorticity
        # init advection reaction diffusion
        self.diffusion_const = 1e-3 # adjust the front width ( larger value = larger front)
        self.reaction_const = 1e1#80 # adjust the propagation of the flame (larger value = faster propagation)
        self.dim = dim
        self.shape = [*self.geom.N] + [len(self.time.t)]
        self.fom_size = np.prod(self.geom.N)
        self.pde = pde
        self.diff_operators=self.diff_operators(self.geom.N,self.geom.dX, dim = dim)
        self.case = case
        self.info_dict ={}
        if pde == "advection":
            if dim == 2:
                self.rhs = lambda qvals, time: self.rhs_advection2D_periodic( qvals,time)
            else:
                self.rhs = lambda qvals, time: self.rhs_advection1D_periodic(qvals)
        elif pde == "burgers":
            if dim == 2:
                self.rhs = lambda qvals, time: self.rhs_burgers2D_periodic( qvals)
            else:
                self.rhs = lambda qvals, time: self.rhs_burgers1D_periodic(qvals)
        elif pde == "react":
            if dim == 2:
                #self.rhs = lambda qvals, time: rhs_advection_reaction_diffusion_2D_periodic(self, qvals, time)
                self.rhs = lambda qvals, time, **kwargs: rhs_advection_reaction_diffusion_2D_periodic_hyperreduced_numba_wrapper(self, qvals, time,**kwargs )
            else:
                #self.rhs = lambda qvals, time: self.rhs_advection_reaction_diffusion_1D_periodic(qvals)
                self.rhs = lambda qvals, time, **kwargs: rhs_advection_reaction_diffusion_1D_periodic_hyperreduced_numba_wrapper(self, qvals, time, **kwargs)
        elif pde == "react":
            if dim == 2:
                self.rhs = lambda qvals, time: self.rhs_low_mach2d( qvals, time)
            else:
                pass
                #self.rhs = lambda qvals, time: self.rhs_low_mach1d(qvals)
        elif pde == "compNS":
            if dim == 2:
                print("implement me!")
            else:
                self.rhs = lambda qvals, time: self.rhs_compressible_navier_stokes_1d(qvals,time)
    class diff_operators:
        def __init__(self, Ngrid, dX, dim,  order = 6):
            if dim == 1:

                stencil_x = np.asarray([1 / 280, -4 / 105, 1 / 5, -4 / 5, 0, 4 / 5, -1 / 5, 4 / 105, -1 / 280])
                self.Dx_mat = derivative(Ngrid[0], dX[0], stencil_x)
                Dx_mat = self.Dx_mat.tocsr()
                rows, stencil_x_col = Dx_mat.nonzero()
                vals = np.asarray(Dx_mat[rows, stencil_x_col]).flatten()
                stencil_x_size = sum(rows==0)
                self.stencil_csr_x = [vals, stencil_x_col,stencil_x_size]

                #stencil_xx = np.asarray([-1 / 560, 8 / 315, -1 / 5, 8 / 5, -205 / 72, 8 / 5, -1 / 5, 8 / 315, -1 / 560]) # 8th order
                #stencil_xx = np.asarray([1/90,	-3/20, 	3/2, 	-49/18, 	3/2, 	-3/20, 	1/90]) # 6th order
                stencil_xx = np.asarray([-1/12, 4/3, -5/2, 4/3,-1/12]) # 4th order
                #stencil_xx = np.asarray([1, -2, 1])  # 4th order
                self.Dxx_mat = derivative(Ngrid[0],dX[0]**2, stencil_xx)
                Dxx_mat = self.Dxx_mat.copy()
                Dxx_mat = Dxx_mat.tocsr()
                rows, stencil_xx_col = Dxx_mat.nonzero()
                stencil_xx_size = sum(rows == 0)
                vals = np.asarray(Dxx_mat[rows, stencil_xx_col]).flatten()
                self.stencil_csr_xx = [vals, stencil_xx_col,stencil_xx_size]

                ############################################################################################
                # for hyperreduction we need to finde the set of meshpoints on which the rhs is evaluated.
                # this includes not only the sampled meshpoints of the rhs, but also the stencils.
                # therefore at an samplemeshindex i, the matrix row supportMeshIndices[i,:] contains all support
                # points needed to evaluate the rhs at index i
                ############################################################################################
                # gather all meshindices of stencils
                M = np.prod(Ngrid)
                supportMeshIndices = [[i] for i in np.arange(0, M)]

                for index in range(M):
                    for j in range(stencil_x_size):  # 8 is the stencil size
                        supportMeshIndices[index].append(stencil_x_col[stencil_x_size * index + j])
                    for j in range(stencil_xx_size):  # 9 is the stencil size
                        supportMeshIndices[index].append(stencil_xx_col[stencil_xx_size * index + j])
                    # in this step we kick out double accuring indices
                    supportMeshIndices[index] = sorted(list(set(supportMeshIndices[index])))

                self.supportMeshIndices = np.asarray(supportMeshIndices)
                ###########################################################################################

            else:
                Ix = sp.eye(Ngrid[0])
                Iy = sp.eye(Ngrid[1])
                # stencilx = np.asarray( [-1/60,	3/20, 	-3/4, 	0, 	3/4, 	-3/20, 	1/60])
                stencil_x = np.asarray([1 / 280, -4 / 105, 1 / 5, -4 / 5, 0, 4 / 5, -1 / 5, 4 / 105, -1 / 280])
                # stencil_x = np.asarray([-0.5,0,0.5])
                # stencil_xx = np.asarray([1/90,	-3/20, 	3/2, 	-49/18, 	3/2, 	-3/20, 	1/90])
                stencil_xx = np.asarray([-1 / 560, 8 / 315 ,-1 / 5, 8 / 5 ,-205 / 72, 8 / 5, -1 / 5, 8 / 315, -1 / 560 ])
                self.Dx_mat = sp.kron(derivative(Ngrid[0], dX[0], stencil_x), Iy)
                self.Dy_mat = sp.kron(Ix,derivative(Ngrid[1], dX[1], stencil_x))
                self.Dxx_mat = sp.kron(derivative(Ngrid[0], dX[0 ]**2, stencil_xx ), Iy)
                self.Dyy_mat = sp.kron(Ix,derivative(Ngrid[1], dX[1 ]**2, stencil_xx ))
                Dx_mat = self.Dx_mat.copy()
                Dx_mat = Dx_mat.tocsr()
                rows, stencil_x_col = Dx_mat.nonzero()
                stencil_x_size = sum(rows == 0)
                vals = np.asarray(Dx_mat[rows, stencil_x_col]).flatten()
                self.stencil_csr_x = [vals,stencil_x_col,stencil_x_size]

                Dxx_mat = self.Dxx_mat.copy()
                Dxx_mat = Dxx_mat.tocsr()
                rows, stencil_xx_col = Dxx_mat.nonzero()
                stencil_xx_size = sum(rows == 0)
                vals = np.asarray(Dxx_mat[rows, stencil_xx_col]).flatten()
                self.stencil_csr_xx = [vals,stencil_xx_col, stencil_xx_size ]

                Dy_mat = self.Dy_mat.copy()
                Dy_mat = Dy_mat.tocsr()
                rows, stencil_y_col = Dy_mat.nonzero()
                stencil_y_size = sum(rows == 0)
                vals = np.asarray(Dy_mat[rows, stencil_y_col]).flatten()
                self.stencil_csr_y = [vals, stencil_y_col,stencil_y_size]

                Dyy_mat = self.Dyy_mat.copy()
                Dyy_mat = Dyy_mat.tocsr()
                rows, stencil_yy_col = Dyy_mat.nonzero()
                stencil_yy_size = sum(rows == 0)
                vals = np.asarray(Dyy_mat[rows, stencil_yy_col]).flatten()
                self.stencil_csr_yy = [vals, stencil_yy_col, stencil_yy_size]

                ############################################################################################
                # for hyperreduction we need to finde the set of meshpoints on which the rhs is evaluated.
                # this includes not only the sampled meshpoints of the rhs, but also the stencils.
                # therefore at an samplemeshindex i, the matrix row supportMeshIndices[i,:] contains all support
                # points needed to evaluate the rhs at index i
                ############################################################################################
                # gather all meshindices of stencils
                M = np.prod(Ngrid)
                supportMeshIndices = [[i] for i in np.arange(0,M)]

                for index in range(M):
                    for j in range(stencil_x_size): # 8 is the stencil size
                        supportMeshIndices[index].append(stencil_x_col[stencil_x_size * index + j])
                    for j in range(stencil_y_size):
                        supportMeshIndices[index].append(stencil_y_col[stencil_y_size * index + j])
                    # dq_dxx and dq_dyy
                    for j in range(stencil_xx_size): # 9 is the stencil size
                        supportMeshIndices[index].append(stencil_xx_col[stencil_xx_size * index + j])
                    for j in range(stencil_yy_size):  # 9 is the stencil size
                        supportMeshIndices[index].append(stencil_yy_col[stencil_yy_size * index + j])
                    # in this step we kick out double accuring indices
                    supportMeshIndices[index] = sorted(list(set(supportMeshIndices[index])))

                self.supportMeshIndices = np.asarray(supportMeshIndices)
            ###########################################################################################
        # def Dx(self, q):
        #     c = self.Dx_mat.tocsr()
        #     rows, cols = c.nonzero()
        #     dat = np.asarray(c[rows, cols]).flatten()
        #
        #     dq = diff_numba(q, dat, np.unique(rows), cols, Nrows=np.size(self.Dx_mat,0))
        #     return dq
        def Dx(self, q):
           return self.Dx_mat @ q
        def Dxx(self, q):
            return self.Dxx_mat @ q
        def Dy(self, q):
            return self.Dy_mat@q
        def Dyy(self, q):
            return self.Dyy_mat@q



    def get_stencil_in_sparse_format(self):
        """
        https://rushter.com/blog/scipy-sparse-matrices/
        :return:
        """
        if self.dim ==2:
            return self.diff_operators.stencil_csr_x, self.diff_operators.stencil_csr_y, self.diff_operators.stencil_csr_xx, self.diff_operators.stencil_csr_yy
        else:
            return self.diff_operators.stencil_csr_x, self.diff_operators.stencil_csr_xx

    class material:
            schmidt_number = 1
            reynolds_number = 200
            gamma = 1.197
            R = 8.314462
            W = 1
            cp= 1004.7029
            reaction_const =  4 # adjust the propagation of the flame (larger value = faster propagation) Y
            reaction_const_h =  0.1 # adjust energy transfer to NS  (larger value = faster propagation)
            mu0 = 3.6e-7*1950**0.7#3.6e-0                 # dynamic viscosity
            W1 = 1
            dh1 = 3.5789e6
            preexponent = 2.8e9
            activation_temperature = 2.0130e4
            Le = 1                       # Lewis Number
            prandtl_number = 0.720  # Prandtl Number
    class geom:
        L = np.asarray([1.0, 1.0])
        N = np.asarray([2**7, 2**7])
    class rom:
        pass
    class odeint:
        timestep = 0.05
    class time:
        T = 1
        dt = 0.05
    class penalisation:
        eta = 100
        case = "bunsen_flame"
        def init_mask(self):

            if self.penalisation.case == "bunsen_flame":
                X, Y = self.geom.Xgrid[0], self.geom.Xgrid[1]
                smoothwidth = self.geom.dX[0]*1
                softstep = lambda x,d: (1+np.tanh(x/d))*0.5
                weights = softstep(X,smoothwidth)-softstep(X-0.2*self.geom.L[0], smoothwidth)
                q_target = (softstep(X-0*self.geom.L[0], smoothwidth/10)-softstep(X-0.3*self.geom.L[0], smoothwidth))#*\
                          #(softstep(Y-0.4*self.geom.L[1], smoothwidth/10)-softstep(Y-0.6*self.geom.L[1], smoothwidth/10))
                q_target = 1-q_target
            else:
                q_target    = np.zeros(self.geom.N)
                weights = np.zeros(self.geom.N)
            self.penalisation.q_target = np.reshape(q_target,-1)
            self.penalisation.weights = np.reshape(weights,-1)

    def velocity_field(self, t, sampleMeshIndices = None):

        # dipole vortex benchmark
        c = self.advection_speed
        tau = self.decay_constant
        [X, Y] = self.geom.Xgrid
        L = self.geom.L
        T = self.time.T
        we = np.asarray(self.w0)
        r0 = self.r0
        ri = lambda xi, yi: (X - xi) ** 2 + (Y - yi) ** 2
        if self.case == "pacman":
            (x1, y1) = (0.6 - c * t, 0.49 * L[1])
            (x2, y2) = (0.6 - c * t, 0.51 * L[1])
            distance1 = (ri(x1, y1)/r0)**2
            distance2 = (ri(x2, y2)/r0)**2
            ux = -we[0] * (Y - y1) * np.exp(-(distance1)) + we[1] * (Y - y2) * np.exp(-(distance2))
            uy = we[0] * (X - x1) * np.exp(-(distance1)) - we[1] * (X - x2) * np.exp(-(distance2) )
            ux *= np.exp(-(t / tau) ** 2)
            uy *= np.exp(-(t / tau) ** 2)
        elif self.case == "bunsen_flame":
            inv_2pi = 1/(2*np.pi)
            my_sawtooth = lambda x: (x*inv_2pi)%1
            (x1, y1) = (0.1 * L[0] + 0.8*L[0] * my_sawtooth(2*pi/T*t * 5), 0.46*L[1])
            (x2, y2) = (0.1 * L[0] + 0.8*L[0] * my_sawtooth(2*pi/T*t * 5), 0.54*L[1])
            ux = + we[0] * (Y - y1) * np.exp(-(ri(x1, y1) / r0) ** 2) - we[1] * (Y - y2) * np.exp(-(ri(x2, y2) / r0) ** 2)
            uy = - we[0] * (X - x1) * np.exp(-(ri(x1, y1) / r0) ** 2) + we[1] * (X - x2) * np.exp(-(ri(x2, y2) / r0) ** 2)
            #ux *= sin(2*pi/T*t*4)**2#np.exp(-(t / tau) ** 2)
            #uy *= sin(2*pi/T*t*4)**2#np.exp(-(t / tau) ** 2)

            smoothwidth = self.geom.dX[0] * 1
            softstep = lambda x, d: (1 + np.tanh(x / d)) * 0.5
            ux += (0.05)*(softstep(Y-0.3*self.geom.L[1], 2*smoothwidth)-softstep(Y-0.7*self.geom.L[1], 2*smoothwidth))*softstep(X-0.05*self.geom.L[0], smoothwidth)#*params.penalisation.weights
            #ux += c
        else:
            pass

        ux = np.reshape(ux, -1)
        uy = np.reshape(uy, -1)
        if sampleMeshIndices is not None:
            ux = np.take(ux, sampleMeshIndices, axis=0)
            uy = np.take(uy, sampleMeshIndices, axis=0)

        return np.reshape(ux,-1), np.reshape(uy,-1)

    def set_inicond(self,case=""):
        dx = np.min(self.geom.dX)
        L  = np.min(self.geom.L)
        delta = 1 * dx
        self.front = lambda x: 1/(1 + np.exp(-x/delta))
        self.forward=lambda x: to.sigmoid(x/delta)
        if self.dim == 2:
            if case== "pacman":
                [X,Y] = self.geom.Xgrid
                phi0 = np.sqrt((X-L*(0.4))**2+(Y-L*(0.5))**2) - 0.2*L
                q0 = self.front(phi0)
                return 1-reshape(q0,-1)
            if case == "bunsen_flame":
                return np.reshape((1-self.penalisation.weights),-1)
            else:
                return np.reshape(np.zeros(self.geom.N),-1)
        else:
            if case == "reaction1D":
                self.front = lambda x: 0.5 * (1 - np.tanh(x / 2))
                return self.front(2*(np.abs(self.geom.X[0] - self.geom.L[0] * 0.5)-2)/self.reaction_const)
            else:
                return self.front(self.geom.X[0]-self.geom.L[0]*0.05)-self.front(self.geom.X[0]-self.geom.L[0]*0.15)

    def rhs_advection1D_periodic(self, q):
        c = self.advection_speed
        K = self.geom.K[0]
        qhat = fft(q)
        qhat_x = K * qhat * (1j)
        return -c*np.real(ifft(qhat_x))

    def rhs_advection_reaction_diffusion_1D_periodic(self, q):
        a = self.diffusion_const
        b = self.reaction_const
        c = self.advection_speed

        K = self.geom.K[0]
        qhat = fft(q)
        qhat_x = np.real(ifft(K * qhat * (1j)))
        qhat_xx= np.real(ifft(-K*K*qhat))

        return a * qhat_xx - b * q * (q-1) - c * qhat_x

    def rhs_lowmach1d(self,q, time):
        """Not working yet"""
        from cases import give_reaction_rates
        Re = self.material.Reynolds_number
        Pr = self.material.Prandtl_number
        Sc = self.material.Schmidt_number
        gamma = self.material.gamma
        p0 = 1
        beta = self.material.compressibility

        rho, u, p, Y = give_primary_observables(self, q)
        T = p0/rho
        urho_x = self.diff_operators.Dx @ (rho*u)
        u_x = self.diff_operators.Dx @ u
        u_xx = self.diff_operators.Dxx @ u
        p_x = self.diff_operators.Dx @ p
        uY_x = self.diff_operators.Dx @ (u*Y)
        Y_xx = self.diff_operators.Dxx @ Y
        T_xx = self.diff_operators.Dxx @ T

        # calculate reaction rate and heat:
        w_T_dot,w_dot = give_reaction_rates(params, rho*Y, T)
        Q = 1 / (p0 * Pr * Re) * T_xx - 1 / (gamma * p0) * w_T_dot

        rhs = np.zeros(self.geom.N, self.geom.Nvar)
        ## mass equation
        rho_dot = - urho_x
        ## momentum equation
        u_dot = - u*u_x - 1/rho*p_x + 1/(Re*rho)*u_xx
        ## artificial pressure
        p_dot = beta* (-u_x + Q)
        ## reaction equation
        Y_dot = - uY_x - 1/(Re*Sc)*Y_xx + w_dot
        rhs = np.asarray([rho_dot,u_dot,p_dot,Y_dot])

        return np.reshape(rhs,-1)


    # def rhs_advection_reaction_diffusion_2D_periodic(self, q, t):
    #
    #     a = self.diffusion_const
    #     b = self.reaction_const
    #     chi = self.penalisation.eta
    #     weights = self.penalisation.weights
    #     q_target = self.penalisation.q_target
    #     # first derivative
    #     dq_dx = self.diff_operators.Dx(q)
    #     dq_dy = self.diff_operators.Dy(q)
    #     # second derivative
    #     ddq_ddx = self.diff_operators.Dxx(q)
    #     ddq_ddy = self.diff_operators.Dyy(q)
    #
    #     ux, uy = self.velocity_field(t)
    #     # right hand side
    #     rhs = - ux * dq_dx -  uy * dq_dy + a * (ddq_ddx + ddq_ddy) - b * q*(q-1) -chi * weights * (q-q_target)
    #
    #     return reshape(rhs,-1)

    def jacobian_rhs_advection_reaction_diffusion_2D_periodic(self, q, t, dq):
        """
        This function computes the derivative of the rhs with respect to q in the direction of dq

        :param q: state at time t
        :param t: time
        :param dq: direction of the derivative dq
        :return: drhs(q,t)/dq * dq
        """

        a = self.diffusion_const
        b = self.reaction_const
        chi = self.penalisation.eta
        weights = self.penalisation.weights
        #K = self.geom.K
       # [kx, ky] = meshgrid(K[0], K[1])
        # first derivative
        dq_dx = self.diff_operators.Dx(q)
        dq_dy = self.diff_operators.Dy(q)
        # second derivative
        ddq_ddx = self.diff_operators.Dxx(q)
        ddq_ddy = self.diff_operators.Dyy(q)

        ux, uy = self.velocity_field(t)
        # right hand side
        rhs_dq = - ux * dq_dx - uy * dq_dy + a * (ddq_ddx + ddq_ddy) - b* (2*q - 1)*dq - chi * weights * dq

        return reshape(rhs_dq, -1)

    def rhs_advection2D_periodic(self,q,time):
        q = reshape(q,self.geom.N)
        L = self.geom.L
        K = self.geom.K

        v_x = L[0]* 10 * ( -sin(2 * pi * time))
        v_y = L[1] * 10 * (  cos(2 * pi * time))
        [kx,ky] = meshgrid(K[0],K[1])
        qhat = fft2(q)
        dq_dx = np.real(ifft2(kx * qhat * (1j)))
        dq_dy = np.real(ifft2(ky * qhat * (1j)))
        return reshape(- (v_x*dq_dx + v_y *dq_dy),-1)

    def rhs_advection2D_periodic_torch(self,q,time):
        if not to.is_tensor(q):
            q = to.tensor(q)

        q = to.reshape(q,self.geom.N.tolist())
        L = self.geom.L
        K = self.geom.K

        v_x = L[0]* 10 * ( -sin(2 * pi * time))
        v_y = L[1] * 10 * (  cos(2 * pi * time))
        [kx,ky] = to.meshgrid(to.tensor(K[0]),to.from_numpy(K[1]))
        qhat = to.rfft(q, signal_ndim=2,onesided=False)
        dq = to.zeros_like(qhat)
        dq[:,:,0] = -kx * qhat[:,:,1].clone() # kx * imag(qhat)*i²
        dq[:,:,1] = kx * qhat[:,:,0].clone() # kx * real(qhat)*i
        dq_dx = to.irfft(dq ,signal_ndim=2,onesided=False).clone()
        dq = to.zeros_like(qhat)
        dq[:, :, 0] = -ky * qhat[:, :,1].clone()  # kx * imag(qhat)*i²     is real
        dq[:, :, 1] = ky * qhat[:, :, 0].clone()  # kx * real(qhat)*i       is complex
        dq_dy = to.irfft(dq,signal_ndim=2, onesided=False).clone()
        return to.reshape( -(v_y*dq_dx + v_x *dq_dy),[-1])

    def rhs_burgers1D_periodic(self,q):

        nu = self.diffusion_const
        K = self.geom.K[0]

        # spectral derivatives
        qhat = fft(q)
        dq = ifft(K * qhat * (1j))
        ddq = ifft(- K**2 *qhat)

        # right hand side
        rhs = -q * dq + nu * ddq
        return np.real( rhs)

    def rhs_burgers2D_periodic(self,q):

        q = reshape(q, params.geom.N)
        nu = self.diffusion_const
        K = self.geom.K
        [kx, ky] = meshgrid(K[0], K[1])

        # spectral derivatives
        qhat = fft2(q)
        # first derivative
        dq_dx = np.real(ifft2(kx * qhat * (1j)))
        dq_dy = np.real(ifft2(ky * qhat * (1j)))
        # second derivative
        ddq_ddx = np.real(ifft2(-kx**2 * qhat ))
        ddq_ddy = np.real(ifft2(-ky**2 * qhat))

        # right hand side
        rhs = -q * dq_dx - q * dq_dy + nu * (ddq_ddx + ddq_ddy)

        return reshape(rhs,-1)


######################################################
# 1.) solve ODE spectral or with Finite Diffs
######################################################

def solve_FOM(params):
    from scipy.integrate import solve_ivp as ode45
    print("Solving Full Order Model")
    q0 =params.inicond

    time_odeint = perf_counter() # save timing
    #q = odeint(params.rhs,q0,params.time.t).transpose()
    rhs = lambda state, t: params.rhs(t,state)
    time = params.time.t
    #q = timesteper(rhs, [time[0], time[-1]], q0, t_vec=time, tau=params.odeint.timestep).T
    #nrhs = 4*len(time)
    ret = ode45(rhs, [time[0], time[-1]], q0, method="RK45",rtol=1e-7 , t_eval=time)
    q = ret.y
    nrhs = ret.nfev
    time_odeint = perf_counter()-time_odeint
    print("t_cpu = %1.3f #rhs calls: %d"%(time_odeint,nrhs))
    params.info_dict["num_rhs_calls"] = nrhs
    # reshape to data field!

    q = reshape(q, [*params.geom.N[:dim], -1])

    return params, q

if __name__ == "__main__":
    from cases import my_cases
    # choose pde [burgers,advection]
    pde = "react"
    case = "bunsen_flame"
    params = my_cases(case=case)
    params,q = solve_FOM(params)

    #savemat('../popey.mat', {'data': q, 'x': params.geom.X[0], 'y': params.geom.X[0]})


    if params.dim ==1:
        ntime = q.shape[-1]
        cycles = 1
        fig, ax = plt.subplots()
        line, = ax.plot(params.geom.X[0],q[...,0])
        for t in range(cycles * ntime):
            line.set_ydata(q[...,t % ntime])
            plt.draw()
            plt.pause(0.01)
        plt.close()
    else:
        show_animation(q,Xgrid = params.geom.Xgrid, frequency = 5)
