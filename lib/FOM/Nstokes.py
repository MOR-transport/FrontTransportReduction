
import numpy as np
from numpy import linspace, pi,meshgrid, reshape, sin, cos
import torch as to
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.integrate import RK45
from scipy.optimize import least_squares
from time import perf_counter
# pyfftw is a lot faster then numpys fft,
# so if pyfftw is available we make use of it
try:
    import pyfftw
    from pyfftw.interfaces.numpy_fft import fft,ifft,fft2,ifft2
except ImportError:
    print("!!! Warning: Slow version !!!\n Install pyfftw to speed up FTRs fft algorithm. !!!! Warning !!!!")
    from numpy.fft import fft,ifft,fft2,ifft2

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Helvetica'],'size'   : 12})
#
rc('text', usetex=True)

pic_dir = "../imgs"
plt.close("all")

class params_class:
    def __init__(self, pde = "advection",dim = 2 ,L=[1]*2 ,N = [2**8]*2, T=0.005, Nt=100 ):
        # init geometry
        self.geom.L = np.asarray(L[:dim]) # domain size
        self.geom.N = [1,1] # number of spatial points in each dimension
        self.geom.N[:dim] = np.asarray(N[:dim])
        self.time.T = T
        self.Neqn = 4
        self.geom.dX = self.geom.L/self.geom.N
        self.geom.X = [np.arange(0, self.geom.L[d],self.geom.dX[d]) for d in range(dim)]
        self.geom.Xgrid = np.meshgrid(self.geom.X)
        self.geom.K = [np.fft.fftfreq(self.geom.N[k],d=self.geom.dX[k]) for k in range(dim)]
        # init time
        self.time.t = np.linspace(0,self.time.T,Nt)
        self.time.Nt = Nt
        self.w0 = 3.5   # initial vorticity
        self.r0 = 0.02    # initial size of vortex
        # init advection reaction diffusion
        self.reaction_const = 5000#0.1 # adjust the propagation of the flame (larger value = faster propagation)
        self.dim = dim
        self.shape = [*self.geom.N] + [len(self.time.t)]
        self.pde = pde
        self.eqn_name = [r"$\rho$",r"$\rho u$", r"$p$", r"$\rho Y$"]
        c = np.sqrt(1.4 * self.init.p_oo / self.init.rho_oo);
        cfl = 1
        self.time.dt_max = cfl * min(self.geom.dX)/c
        print("cfl is ",cfl)
        print("dt_max is ", self.time.dt_max)
    class geom:
            L = np.asarray([1.0, 1.0])
            N = np.asarray([2**7, 2**7])
    class time:
            T = 1
            dt = 0.05
    class init:
        case = "pressure_blob"
        rho_oo = 1.204
        u_oo = 0
        v_oo = 0
        p_oo = 1e5
        T_oo = 273.15
    class material:
            gamma = 1.4
            R = 8.314462
            W = 0.0289644
            cp= 1004.7029
            Pr = 0.71                    # Prandtl Number
            mu0 = 3.6e-7                 # dynamic viscosity
            Le = 1                       # Lewis Number
            diffusion_const = mu0/Pr/Le  # diffusion constant: adjusts the front width ( larger vlaue = larger front)
    class diff_operators:
        def Dx(q,params):
            qhat = fft(q)
            K = params.geom.K[0]
            return np.real(ifft(K * qhat * (1j)))




def rhs_compressible_navier_stokes_1d(params, q, time):

        gamma = params.material.gamma
        Dx = lambda x: params.diff_operators.Dx(x,params)

        q = reshape(q, [*params.geom.N, 4]).squeeze()
        sqrho = q[..., 0]
        rho = sqrho**2
        u   = q[..., 1]/sqrho
        p   = q[..., 2]
        rhoY   = q[..., 3]
        pref  = p #- params.init.p_oo

        pref_x = Dx(pref)
        u_x = Dx(u)
        ################
        # mass equation
        ################
        rho_u = rho*u
        dot_sqrho = -Dx(rho_u)/(2*sqrho)

        #####################
        # momentum equation
        #####################
        if False:
            pass
            print("Implement Friction terms")
        else:
            Fric_u = 0
            Fric_E = 0
            heatflux = 0

        Drhou_u = Dx(rho_u*u) + rho_u * u_x
        dot_sqrhou = - (Drhou_u*0.5 + pref_x - Fric_u)/sqrho

        #####################
        # enerty equation
        #####################
        u_grad_p = u*pref_x
        div_u_p = Dx(u*p)
        dot_p =  -( gamma*div_u_p  +
                    (gamma-1)*( -u_grad_p +  u * Fric_u  - Fric_E - heatflux)
                   )

        ######################
        # reaction term
        ######################
        Dconst = params.material.diffusion_const
        b = params.reaction_const
        Y = rhoY/rho
        dot_rhoY = Dconst * Dx(rho*Dx(Y)) - b * Y * (Y-1) - Dx(rhoY*u)

        rhs = np.zeros_like(q)
        rhs[:,  0] = dot_sqrho
        rhs[:,  1] = dot_sqrhou
        rhs[:,  2] = dot_p
        rhs[:,  3] = dot_rhoY

        return np.reshape(rhs,-1)




def inicond(params):
    rho_oo = params.init.rho_oo
    u_oo = params.init.u_oo
    p_oo = params.init.p_oo
    gamma = params.material.gamma
    L = params.geom.L[0]
    delta = 0.005 * L
    dx = params.geom.dX[0]
    x = np.reshape(params.geom.Xgrid[0],params.geom.N)
    x_0 = 0.5*params.geom.L[0]
    front = lambda x: 1/(1 + np.exp(-(x / delta)))
    if params.init.case == "pressure_blob":
        q0 = np.zeros([*params.geom.N, params.Neqn])
        rho0 = np.ones(params.geom.N) * rho_oo
        u0   = np.ones(params.geom.N) * u_oo
        p0   = np.ones(params.geom.N) * p_oo + 0.05*p_oo *np.exp(-((x - x_0)**2)/((5*(dx)/2)**2))
        rho0 = rho0 * (p0 / p_oo) ** (1 / gamma)
        Y0 = front(x - (x_0 - L * 0.1  )) - front(x - (x_0 + L * 0.1))
        q0[:, :, 0] =  np.sqrt(rho0)
        q0[:, :, 1] = np.sqrt(rho0) * u0
        q0[:, :, 2] = p0
        q0[:, :, 3] = rho0 * Y0

    return reshape(q0,-1)
######################################################
# 1.) solve ODE spectral or with Finite Diffs
######################################################
def RK4(q, dt, t, rhs):

    k1 = rhs(q, t)
    k2=rhs(q+dt/2* k1 , t+dt/2)
    k3=rhs(q+dt/2* k2 , t+dt/2)
    k4=rhs(q+dt* k3   , t+dt  )

    return q+dt/6 * (k1 + 2*k2 +2*k3+k4)

def timesteper(q0, params):

    dt = params.time.dt
    t= 0
    q = q0
    qhist = [q0]
    for nt in range(params.time.Nt):
        t = t + dt
        q = RK4(q,dt,t,params.rhs)
        if nt%100 ==0:
            qhist.append(q)
    return np.asarray(qhist).T



def solve_FOM(params):

    print("Solving Full Order Model")
    q0 = inicond(params)
    time_odeint = perf_counter() # save timing
    params.rhs = lambda *args: rhs_compressible_navier_stokes_1d(params,*args)

    # #q = RK45(params.rhs, params.time.t[0], q0, params.time.T)
    q = odeint(params.rhs, q0, params.time.t).transpose()
    time_odeint = perf_counter()-time_odeint

    print("t_cpu = %1.3f"%time_odeint)
    # reshape to data field!
    q = reshape(q, [*params.geom.N[:dim],params.Neqn,-1])

    return params, q

if __name__ == "__main__":

    # choose pde [burgers,advection,react]
    pde = "react"
    dim = 1
    params = params_class(pde, dim)
    params, q = solve_FOM(params)


    if params.dim ==1:
        ntime = q.shape[-1]
        cycles = 2
        fig, ax = plt.subplots(params.Neqn,1,sharex=True)
        line=[0]*params.Neqn
        for i in range(params.Neqn):
            line[i], = ax[i].plot(params.geom.X[0],q[...,i,0])
            qflat = q[...,i,:].flatten()
            ax[i].set_ylim([min(qflat),max(qflat)])
            ax[i].set_ylabel(params.eqn_name[i])
        ax[i].set_xlabel(r"$x$")
        for t in range(cycles * ntime):
            for i in range(params.Neqn):
                line[i].set_ydata( q[..., i, t % ntime])
            plt.draw()
            plt.pause(0.01)
        plt.close()
    else:
        ntime = q.shape[-1]
        cycles = 1
        fig, ax = plt.subplots()
        h = ax.pcolormesh(q[...,0])
        for t in range(cycles * ntime):
            h.set_array(q[...,t % ntime].ravel())
            fig.savefig("imgs/taylor_%3.3d.png" % t)
            plt.draw()
            plt.pause(0.05)
        plt.close()