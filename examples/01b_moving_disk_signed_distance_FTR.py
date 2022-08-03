from globs import *
import numpy as np
import os
import torch
from NeuralFTR.synthetic_data import make_grid, generate_data, f
from NeuralFTR.main import train_NN
import matplotlib.pyplot as plt
from plot_utils import save_fig
from compare_compression import *
from ROM.FTR import FTR_ranks, FTR_SD
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib
from os.path import expanduser
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
plt.close("all")
home = expanduser("~")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Using: "+ torch.cuda.get_device_name(0))
##########################################################################
# %% Load / Generate the data
#########################################################################
case = "moving_disk"
smoothness = 1e-7   # this is the smoothness parameter for regularizing the neural network
rank_list = [1,2,3,4,5,6,7,8,9,10,15,20,30]
skip_existing = True# resumbe backup ?
Nx, Ny = 129, 129
X, Y = make_grid(Nx, Ny, 10, 10)
Radius = 2.2
lambd = 0.1
n_samples = 200
t = np.linspace(1/n_samples, 1, n_samples)
R, t, lam = np.meshgrid([Radius], t, [lambd])
tRlam = np.concatenate([t.flatten()[:, None], R.flatten()[:, None], lam.flatten()[:, None]], 1)
data_set, phi_field = generate_data(X, Y, tRlam[:, 0], tRlam[:, 1], tRlam[:, 2], to_torch=True, type="moving_disk", device=DEVICE)

# split data in train and test sets
data_train = data_set[::2]
data_test = data_set[1::2]
Ntime = np.size(data_test, 0)

# generate snapshot matrix
q = np.squeeze(data_set.cpu().numpy())
q = np.moveaxis(q,0,-1)
qtrain = np.reshape(np.squeeze(np.moveaxis(data_train.cpu().numpy(), 0, -1)), [-1, Ntime])
qtest = np.reshape(np.squeeze(np.moveaxis(data_test.cpu().numpy(), 0, -1)), [-1, Ntime])


###################################################################
# %% Evals NNs for all dofs
###################################################################
import skfmm
from scipy.interpolate import interp1d as interp1d
threshold = 0.5
q_boundary = [0,1]
N_interp = int(np.sum(np.shape(X))/2)
phi_max = np.max([X,Y]) # maximal value the SD can have is limitted by the domain size
dx = [X[1,0]-X[0,0],Y[0,1]-Y[0,0]]
Nt = np.size(q,-1)
phi_SD = np.zeros_like(q)
for it in range(Nt):
    phi_SD[...,it] = skfmm.distance(q[...,it]-threshold,dx=dx)

phi_flat = phi_SD.flatten()
q_flat = q.flatten()

# make sure to exclude double accuring phi values
phi_flat , idx_unique = np.unique(phi_flat,return_index=True)
q_flat = q_flat[idx_unique]

phi_bounds = [np.min(phi_flat), np.max(phi_flat)]
phi_interval = np.linspace(phi_bounds[0],phi_bounds[1],N_interp)
inter = interp1d(phi_flat,q_flat)
q_interval = inter(phi_interval)

N_interval = np.size(q_interval)

q_interval = np.insert(q_interval,[0,N_interval],q_boundary)
phi_interval = np.insert(phi_interval,[0,N_interval],[-phi_max,phi_max])

N_interval = np.size(q_interval)

front = interp1d(phi_interval,q_interval)

# %%
it =10
plt.pcolormesh(front(phi_SD[...,it])-q[...,it])

err = np.linalg.norm(np.reshape(front(phi_SD)-q,-1))/np.linalg.norm(np.reshape(q,-1))
print("err: ",err)

phi_SD2,front2 = FTR_SD(q, dx = dx)

err2 = np.linalg.norm(np.reshape(front2(phi_SD2)-q,-1))/np.linalg.norm(np.reshape(q,-1))
print("err: ",err2)


err_phi = np.linalg.norm(np.reshape(phi_SD2-phi_SD,-1))/np.linalg.norm(np.reshape(phi_SD,-1))
print("err_phi: ",err_phi)