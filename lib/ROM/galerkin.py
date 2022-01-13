import numpy as np
from numpy import linspace, pi,meshgrid, reshape, sin, cos
import torch as to
from time import perf_counter
from numpy.fft import fft,ifft,fft2,ifft2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.integrate import odeint
from scipy.optimize import least_squares
from scipy.linalg import pinv
from time import perf_counter
from mpl_toolkits.mplot3d import axes3d


def reduced_residual_MLSPG(a,mapping, rhs,q_n,dt,tdt): # formula 3.27 and 3.26 in Carlbergs paper
    # tdt means t+dt (for implicit euler)
    # residual of discreticed implicit euler
    if not to.is_tensor(a):
        a = to.tensor(a,requires_grad=True)
    if not to.is_tensor(q_n):
        q_n = to.tensor(q_n)

    res = mapping(a) - q_n - dt * rhs(mapping(a), tdt)
    res.backward(res, retain_graph=True)
    d_res = a.grad  # is exactly the formula: <d res/da, res>=0 of carlbergs paper

    return d_res.detach().numpy()#, d_res.detach().numpy()


def lspg_odeint(rhs_torch, q0, a_coef, mapping, t):
    # equivalent to odeint
    print('#'*30+"\n Manifold Galerkin \n" + '#'*30)
    q_n = to.tensor(q0)
    a_n = a_coef[:,0]
    delta_t = np.diff(t)
    Nt = len(t) # number of snapshots
    a_galerkin = np.zeros([a_n.size, Nt])
    q_tilde = np.zeros([q0.size, Nt])
    a_galerkin[:,0] = a_coef[:,0]
    q_tilde[:,0] = q0
    for it in range(Nt-1):
        tstart = perf_counter()
        time = t[it]
        dt   = delta_t[it]
        # make a step with implicit euler (hidden in the residual)
        res = least_squares(reduced_residual_MLSPG, a_n,
                            args=(mapping, rhs_torch, q_n, dt, time+dt))
        a_n = res.x
        # map back to full space
        q_n = mapping(a_n)
        # save for output
        a_galerkin[:, it+1] = a_n
        q_tilde[:, it+1] = q_n.detach().numpy()
        print(f"[iter={it:2d} in {perf_counter() - tstart:.2f}s] Galerkin residual: {res.cost:.2e}")


    return q_tilde,a_galerkin

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

