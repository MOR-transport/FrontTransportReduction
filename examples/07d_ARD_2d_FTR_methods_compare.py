import globs
from FOM.FOM import *
from cases import my_cases
from plot_utils import autolabel
from tabulate import tabulate
from paths import init_dirs
import os
import numpy as np
from cases import my_cases
from numpy.linalg import norm
import matplotlib.pyplot as plt
from ROM.ROM import get_proj_err,  load_reduced_map
from plot_utils import save_fig
from NeuralFTR.ftr_net import *
from NeuralFTR.main import train_NN
from compare_compression import *
from plot_utils import bound_exceed_cmap, save_fig
from compare_compression import *
from ROM.FTR import truncate, convergence_plot, FTR_ranks, FTR_SD
from os.path import expanduser
from plot_utils import save_fig
import torch
from numpy.linalg import svd
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Using: "+ torch.cuda.get_device_name(0))

# %%
case = "pacman"
smoothness = 0
react = 100
dirs = {'data':'data/pacman-compare-offline', 'images': 'imgs'}
data_folder, pic_dir = init_dirs(dirs['data'], dirs['images'])
solve_ODE = True
skip_existing = True
max_rank = 30
rank_list = np.concatenate([np.arange(2,16),np.asarray([20,25,30])])
# %%
params = my_cases(case)
params.reaction_const = react
params, q = solve_FOM(params)
q = np.clip(q,0,1)
data_set = np.moveaxis(q, -1, 0)
data_set = torch.tensor(data_set[:,np.newaxis,:,:],dtype=torch.float32).to(DEVICE)

# ODE dim
M = np.prod(params.geom.N)
# %% split data in train and test sets

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
data_folder= '../data/training_results/'+case+"/"
train_results_list=[ {"name": "FTR-NN" , "folder": data_folder+'/FTR-NN/lambda_%.0e/'%smoothness,"decoder": FTR_Dec_1Lay},
                     {"name": "NN" , "folder": data_folder+"/NN/lambda_%.0e/"%smoothness,"decoder": FTR_Dec}]

# train FTR-NN
X = params.geom.Xgrid[0]
Y = params.geom.Xgrid[1]
train_NN(data_train, data_test, train_results_list[0], X, Y, smooth = smoothness,
         skip_existing = skip_existing, dofs = rank_list)
errFTR_NN, rank_list_FTR_NN = give_NN_errors(data_set[1::2], [train_results_list[0]])
phiNN, qtildeNN = give_NN_fields(data_set[1::2], [train_results_list[0]], dofs = 3)

# train NN
train_NN(data_train[...,::2,::2], data_test[...,::2,::2], train_results_list[1], X[::2], Y[::2] , smooth = smoothness,
         skip_existing = skip_existing, dofs = rank_list)
errNN, rank_list_ = give_NN_errors(data_set[1::2,:,::2,::2], [train_results_list[1]])


# %% FTR results

data_folder_FTR = data_folder+"/FTR/"
if not os.path.exists(data_folder_FTR):
    os.makedirs(data_folder_FTR)

Nt_train = data_train.size(0)
matrizise = lambda fun: fun.reshape(-1, Nt_train)
q_train = np.squeeze(np.moveaxis(data_train.cpu().numpy(),0,-1))
qmat = matrizise(q_train)
lambd = 1
front = lambda x:  (np.tanh(x/lambd) + 1) * 0.5
max_iter = np.ones([max_rank])*5000

# FTR thresholding
phi_ftr_list, q_ftr_list, errFTR_offline =  FTR_ranks(qmat, rank_list, front, save_fname = data_folder_FTR+case,
                                                      max_iter = max_iter, print_step = 10,
                                                      stop_if_residual_increasing = True, skip_existing = skip_existing)

#FTR signed distance
phi_ftr_SD, front_SD = FTR_SD(np.reshape(qmat,[*params.geom.N,-1]),dx = params.geom.dX)

phi_sd_mat = matrizise(phi_ftr_SD)

qsvd = np.linalg.svd(qmat, full_matrices=False)
phi_sd_svd = np.linalg.svd(phi_sd_mat, full_matrices=False)
norm_q = norm(qmat,ord='fro')
errors = np.empty((3, len(rank_list)))

for k,r in enumerate(rank_list):

    # POD
    q_trunc = qsvd[0][:, :r] @ np.diag(qsvd[1][:r]) @ qsvd[2][:r, :]
    err = norm(q_trunc - qmat, ord='fro') / norm_q
    print("POD       :", err)
    errors[0, k] = err

    # FTR
    q_trunc = matrizise(front(phi_ftr_list[k]))
    err = norm(q_trunc - qmat, ord='fro') / norm_q
    print("err-FTR   :", err)
    errors[1, k] = err

    # FTR-SD
    q_trunc = front_SD(phi_sd_svd[0][:, :r] @ np.diag(phi_sd_svd[1][:r]) @ phi_sd_svd[2][:r, :])
    err = norm(q_trunc - qmat, ord='fro') / norm_q
    print("err-FTR-SD:", err)
    errors[2, k] = err

np.save(data_folder_FTR+"/FTR-POD-errors.npy", errors)
# %% Plot errors
plt.figure(3)
plt.semilogy(rank_list,errors[0,:],'x', label="POD")
plt.semilogy(rank_list,errNN,'<', label="NN")
plt.semilogy(rank_list_FTR_NN,errFTR_NN,'o', mfc='none',label="FTR-NN")
plt.semilogy(rank_list,errors[1,:],'*', label = "FTR")
plt.semilogy(rank_list,errors[2,:],'+', label="FTR-SD")
#plt.xlim([0,max_rank-0.5])
#plt.xticks(np.arange(0,max_rank,1))
plt.xlabel(r"degrees of freedom $r$")
plt.ylabel(r"relativ error")
plt.tight_layout()
plt.axis()
plt.legend()
save_fig(pic_dir + "/pacman_offline_err.png")
