import pathlib, sys
file_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(file_path) + "/../ALM_FTR_Galerkin_python")

import numpy as np
import torch
from NeuralFTR.moving_disk_data import make_grid, generate_data
from lib.FTR import *
from NeuralFTR.ftr_net import FTR_Dec_1Lay
from ALM_FTR_Galerkin_python.lib.FOM import params_class
import matplotlib.pyplot as plt
from compare_compression import *
import scipy
from scipy.io import savemat
from lib.plot_utils import *
from os.path import expanduser
from matplotlib import rc
latexfont = {'size'   : 34}
rc('font',**latexfont)
rc('text',usetex=True)


home = expanduser("~")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Load network
#decoder_variants = [FTR_Dec, FTR_Dec_1Lay]
plt.close("all")

## Load data
Nx = 129
Ny = 129
X, Y = make_grid(Nx, Ny, 10, 10)
Radius = 2.2
lam = [0.1]
n_samples = 100
time = np.linspace(1/n_samples, 1, n_samples)
R, t, lam = np.meshgrid([Radius], time, lam)
tRlam = np.concatenate([t.flatten()[:, None], R.flatten()[:, None], lam.flatten()[:, None]], 1)
data_set, _ = generate_data(X, Y, tRlam[:, 0], tRlam[:, 1], tRlam[:, 2], to_torch=True, device='cpu')
dofs = 3
## FTR with augmented lagrangian:
# For FTR approach q = f(phi):
#           q ... snapshot data
#           f ... front
#         phi ... smooth low rank field

# a) compute phi as a result of an optimization procedure solved with augmented lagrangians
q = np.squeeze(data_set.numpy())
q = np.moveaxis(q,0,-1)


params=params_class(N = [Nx,Ny],T=1, Nt=data_set.shape[0])
delta = lam.flatten()[0]#1/(0.002 * np.min(params.geom.L))
params.front = lambda x: (np.tanh(x*delta)+1)*0.5#1/(1 + np.exp(-(x * delta))) #sigmoid
#params.front = lambda x: 1/(1 + np.exp(-(x * delta)))
#phi_ftr = augmented_lagrange_multiplier(params, q, f=params.front, tol=1e-7, max_iter=10000, mu=0.4, dt=0.0001, offset=0.001, plot_step=1000)
phi_ftr = simple_FTR(q, f=params.front, tol=1e-7, max_iter=10000, mu=0.4, dt=100,  plot_step=1000, print_step = 100, rank = 4)

# %%
phi_mat = np.reshape(phi_ftr,[-1,n_samples])
phi_svd = np.linalg.svd(phi_mat, full_matrices=False)

fig, ax = plt.subplots(2,3,sharex=True)
basis = np.asarray([-(X-5)**2-(Y-5)**2-R.flatten()[0]**2,-Y,X])
basis = np.moveaxis(basis,0,-1)
modes = np.reshape(phi_svd[0],[Nx,Ny,-1])
for col in range(3):
    p = ax[1, col].pcolormesh(X,Y,modes[...,col])
    cbar = plt.colorbar(p,ax=ax[1, col])
    cbar.formatter.set_powerlimits((0, 0))
    p = ax[0, col].pcolormesh(X,Y,basis[..., col])
    ax[0, col].set_aspect('equal', 'box')
    ax[1, col].set_aspect('equal', 'box')
    plt.colorbar(p,ax=ax[0, col])
    ax[0, col].set_title(r'$u_%d(x,y)$'%col)
    ax[1, col].set_title(r'$\tilde{u}_%d(x,y)$' % col)
    ax[1,col].set_yticks([])
    ax[1,col].set_xticks([])
    ax[0,col].set_yticks([])
    ax[0,col].set_xticks([])
    ax[1,col].set_yticklabels([])
    ax[1,col].set_xticklabels([])
    ax[0,col].set_yticklabels([])
    ax[0,col].set_xticklabels([])


    ax[1,col].set_xlabel(r"$x$")

ax[0,0].set_ylabel(r"$y$")
ax[1,0].set_ylabel(r"$y$")

# draw arrows
ax[0,1].quiver(5,5,0,-1,angles='xy', scale_units='xy', scale=0.2, pivot="middle",minshaft=2,width=0.2)
ax[0,2].quiver(5,5,1,0,angles='xy', scale_units='xy', scale=0.2, pivot="middle",minshaft=2,width=0.2)


u1 = modes[...,1]
ux = np.diff(u1,axis=0,append=1.0)
uy = np.diff(u1,axis=1, append =1.0)
denomi = np.sqrt(ux**2+uy**2)
uyn=np.where(denomi> 0, np.divide(uy, denomi+1e-12), 0)
uxn=np.where(denomi> 0, np.divide(ux, denomi+1e-12), 0)
mask = ((X-5)**2+(Y-5)**2)**0.5<2*Radius
uxn_mean = np.nanmean(uxn*mask)
uyn_mean = np.nanmean(uyn*mask)
ax[1,1].quiver(5,5,uxn_mean,uyn_mean,angles='xy', scale_units='xy', scale=0.1, pivot="middle",minshaft=2,width=0.2)

u2 = modes[...,2]
ux = np.diff(u2,axis=0,append=1.0)
uy = np.diff(u2,axis=1, append =1.0)
denomi = np.sqrt(ux**2+uy**2)
uyn=np.where(denomi> 0, np.divide(uy, denomi+1e-12), 0)
uxn=np.where(denomi> 0, np.divide(ux, denomi+1e-12), 0)
mask = ((X-5)**2+(Y-5)**2)**0.5<2*Radius
uxn_mean = np.nanmean(uxn*mask)
uyn_mean = np.nanmean(uyn*mask)
ax[1,2].quiver(5,5,uxn_mean,uyn_mean,angles='xy', scale_units='xy', scale=0.1, pivot="middle",minshaft=2,width=0.2)
save_fig("../ALM_FTR_Galerkin_python/imgs/disc_transport_map.png",strict=True)

plt.figure(4)
a_coef = (np.diag(phi_svd[1])@phi_svd[2])
plt.plot(time,a_coef[:3,:].T)
plt.xlabel(r"time $t$")
plt.ylabel(r"$\tilde{a}_i(t)$")
plt.legend([r"$i=0$",r"$i=1$",r"$i=2$"],loc=4)
save_fig("../ALM_FTR_Galerkin_python/imgs/disc_transport_map_time.png")


