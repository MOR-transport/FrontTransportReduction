import globs
from numpy.linalg import norm
from ROM.FTR import *
from plot_utils import *
from NeuralFTR.synthetic_data import make_grid, generate_data, f
from NeuralFTR.main import train_NN
from compare_compression import *
import os
import numpy as np
import torch

##################################################################################################
# %% settings
case = "topo_change"
pic_dir="../imgs/"
data_folder= '../data/training_results/'+case+'/'
smoothness = 1e-6
skip_existing = True

###################################################################################################
# %% generate data
###################################################################################################
max_rank = 10
rank_list = np.arange(1,max_rank)
L = [4, 4]

N = [2**8, 2**8]
Nt = 101
R = 0
X, Y = make_grid(*N, *L)
lambd = 1
data_set, phi_dat = generate_data(X, Y, 1, R, lam=1, to_torch=True, type=case, device=DEVICE)

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
train_NN(data_train, data_test, train_results_list[0], X, Y, smooth = smoothness,
         skip_existing = skip_existing, dofs = rank_list)
errFTR_NN, rank_list = give_NN_errors(data_set[1::2], [train_results_list[0]])
phiNN, qtildeNN = give_NN_fields(data_set[1::2], [train_results_list[0]], dofs = 3)

# train NN
train_NN(data_train, data_test, train_results_list[1], X, Y , smooth = smoothness,
         skip_existing = skip_existing, dofs = rank_list)
errNN, rank_list = give_NN_errors(data_set[1::2], [train_results_list[1]])

# %% FTR results

data_folder_FTR = data_folder+"/FTR/"
if not os.path.exists(data_folder_FTR):
    os.makedirs(data_folder_FTR)

Nt_train = data_train.size(0)
matrizise = lambda fun: fun.reshape(-1, Nt_train)
q_train = np.squeeze(np.moveaxis(data_train.cpu().numpy(),0,-1))
qmat = matrizise(q_train)
front = lambda x:  (np.tanh(x/lambd) + 1) * 0.5
max_iter = np.ones([max_rank])*3000
phi_ftr_list, q_ftr_list, errFTR_offline =  FTR_ranks(qmat, rank_list, front, save_fname = data_folder_FTR+case,
                                                      max_iter = max_iter, print_step = 2,
                                                      stop_if_residual_increasing = True, skip_existing = skip_existing)

qsvd = np.linalg.svd(qmat, full_matrices=False)
norm_q = norm(qmat,ord='fro')
errors = np.empty((2, len(rank_list)))

for k,r in enumerate(rank_list):

    # POD
    q_trunc = qsvd[0][:, :r] @ np.diag(qsvd[1][:r]) @ qsvd[2][:r, :]
    errors[0, k] = norm(q_trunc - qmat, ord='fro') / norm_q

    # FTR
    q_trunc = matrizise(front(phi_ftr_list[k]))
    err = norm(q_trunc - qmat, ord='fro') / norm_q
    print("err:", err)
    errors[1, k] = err



# %% Plot errors
plt.figure(3)
plt.semilogy(rank_list,errors[0,:],'x')
plt.semilogy(rank_list,errNN,'<')
plt.semilogy(rank_list,errFTR_NN,'o', mfc='none')
plt.semilogy(rank_list,errors[1,:],'*')
plt.xlim([0,max_rank-0.5])
plt.xticks(np.arange(0,max_rank,1))
plt.xlabel(r"degrees of freedom $r$")
plt.ylabel(r"relativ error")
plt.tight_layout()
plt.legend(["POD","NN","FTR-NN","FTR"])
save_fig(pic_dir + "/topo_change_err.png")
# for i in range(0, np.size(q, 0), 2):
#     plt.pcolormesh(np.squeeze(q[i, ...]))
#     plt.draw()
#     plt.pause(0.01)
# %%
plt.figure(4)
levelset_surface_plot(q[40,0,...],phi_dat[40,...], [X,Y], figure_number=2)

