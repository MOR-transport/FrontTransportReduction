import numpy as np
import torch
from NeuralFTR.moving_disk_data import make_grid, generate_data
from NeuralFTR.ftr_net import FTR_Dec_1Lay
from ALM_FTR_Galerkin_python.lib.FOM import params_class
import matplotlib.pyplot as plt
from ALM_FTR_Galerkin_python.lib.plot_utils import save_fig
from compare_compression import *
import scipy
from scipy.io import savemat
from os.path import expanduser
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
from matplotlib import rc
latexfont = {'size'   : 24}
rc('font',**latexfont)
rc('text',usetex=True)
imagepath="../imgs/"
#plt.style.use('dark_background')
tikzpath ="invfigs/FTR/"
m2tikzOPT={'tex_relative_path_to_data':tikzpath,\
            'axis_height' : '\\figureheight',\
            'axis_width' : '\\figurewidth',\
            'strict' : False,\
             'override_externals': True}

home = expanduser("~")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Load network
#decoder_variants = [FTR_Dec, FTR_Dec_1Lay]
plt.close("all")

## Load data
Nx = 129
Ny = 129
X, Y = make_grid(Nx, Ny, 10, 10)
R = [2.2]
lam = [0.1]
n_samples = 100
t = np.linspace(1/n_samples, 1, n_samples)
R, t, lam = np.meshgrid(R, t, lam)
tRlam = np.concatenate([t.flatten()[:, None], R.flatten()[:, None], lam.flatten()[:, None]], 1)
data_set, phi_field = generate_data(X, Y, tRlam[:, 0], tRlam[:, 1], tRlam[:, 2], to_torch=True, type="moving_disc", device='cpu')
#data_set = moving_disk(X, Y, tRlam[:, 0], tRlam[:, 1], tRlam[:, 2], to_torch=True, device='cpu')
q = np.squeeze(data_set.numpy())
q = np.moveaxis(q,0,-1)
params=params_class(N = [Nx,Ny],T=1, Nt=data_set.shape[0])

# data preparation
data_set = np.asarray(data_set)
Ntime = np.size(data_set[::2], 0)
qtrain = np.reshape(np.squeeze(np.moveaxis(data_set[::2], 0, -1)), [-1, Ntime])
qtest = np.reshape(np.squeeze(np.moveaxis(data_set[1::2], 0, -1)), [-1, Ntime])


###################################################################
# %% Evals NNs for all dofs
###################################################################
data_folder= home+ "/tubcloud/FTR/01_NeuronalFTR/training_results/"
#data_folder= 'D:/Arbeit/FrontTransportReduction//NeuralFTR/training_results_local/NewTrainings/'
train_results_list=[#{"name": "FNL" , "folder": data_folder+"FullDec/NoSmoothing/LearnFrontWidth/","decoder": FTR_Dec},
                     # {"name": "FN" , "folder": data_folder+"FullDec/NoSmoothing/NoLearnFrontWidth/","decoder": FTR_Dec},
                     # {"name": "FSL" , "folder": data_folder+"FullDec/Smoothing/LearnFrontWidth/","decoder": FTR_Dec},
                     # {"name": "NN" , "folder": data_folder+"FullDec/Smoothing/NoLearnFrontWidth/","decoder": FTR_Dec},
                     # {"name": "L1NL" , "folder": data_folder+"1LayDec/NoSmoothing/LearnFrontWidth/","decoder": FTR_Dec_1Lay},
                     # {"name": "L1N" , "folder": data_folder+"1LayDec/NoSmoothing/NoLearnFrontWidth/","decoder": FTR_Dec_1Lay},
                    #{"name": "L1SL" , "folder": data_folder+"1LayDec/Smoothing/LearnFrontWidth/","decoder": FTR_Dec_1Lay},
                      {"name": "NN" , "folder": data_folder+"1LayDec/Smoothing/NoLearnFrontWidth/","decoder": FTR_Dec_1Lay},
                     #{"name": "phiNN", "folder": data_folder+"/Phi/2020_11_07__22-35/", "decoder": Phi}
                    ]
errNN, rank_list = give_NN_errors(data_set, train_results_list, periodic=True)
phiNN, qtildeNN = give_NN_fields(data_set, train_results_list, dofs = 3, periodic=True)

train_results_list=[ {"name": "FN" , "folder": data_folder+"FullDec/NoSmoothing/NoLearnFrontWidth/","decoder": FTR_Dec}]
errNN2, rank_list = give_NN_errors(data_set, train_results_list, periodic=True)
phiNN2, qtildeNN2 = give_NN_fields(data_set, train_results_list, dofs = 3, periodic=True)
###################################################################
# %% FTR for all ranks
###################################################################
phi_ftr_list = []
q_ftr_list = []
front = lambda x: (np.tanh(x/lam[0][0][0]) + 1) * 0.5
file_name_results = "../data/FTR-error-moving-discb.txt"
errFTR = []
with open(file_name_results, 'w') as f:
    f.write("moving disc\n")
    f.write("rank\toffline-err\tonline-err\n")
for rank in rank_list:
    # offline
    phi_ftr = simple_FTR(qtrain, front, max_iter=30000, rank=rank, print_step=500, stop_if_residual_increasing=True)
    qtilde_FTR = front(phi_ftr)
    q_ftr_list.append(qtilde_FTR)
    phi_ftr_list.append(phi_ftr)
    err = np.linalg.norm(qtilde_FTR - qtrain, "fro") / np.linalg.norm(qtrain, "fro")
    # online
    phisvd = np.linalg.svd(phi_ftr, full_matrices=False)
    a_coef_phi = np.diag(phisvd[1][:rank]) @ phisvd[2][:rank, :]
    t_train = np.arange(1, Ntime + 1)
    dt = t_train[1] - t_train[0]
    if Ntime % 2 == 0:
        t_test = t_train[:] + 0.5
    else:
        t_test = t_train[:-1] + 0.5

    a = koopman_prediction(a_coef_phi, dt, t_test, iterations=500)
    qtest_ftr = front(phisvd[0][:, :rank] @ a)
    err_online = np.linalg.norm(qtest_ftr - qtest, "fro") / np.linalg.norm(qtest, "fro")
    errFTR.append(err)
    print("rank: %d, error offline: %.4f online: %.4f" % (rank, err, err_online))
    with open(file_name_results, 'a') as f:
        f.write("%d \t %.1e \t %.1e\n" % (rank, err, err_online))

###################################################################
# %% POD for all ranks
###################################################################
q_pod_list = []
errPOD = []
file_name_results = "../data/POD-error-moving-disc.txt"
with open(file_name_results, 'w') as f:
    f.write("flame pinch off\n")
    f.write("rank\toffline-err\tonline-err\n")
for rank in rank_list:
    # offline
    phisvd = np.linalg.svd(qtrain, full_matrices=False)
    a_coef_train = np.asarray(np.diag(phisvd[1][:rank]) @ phisvd[2][:rank, :],dtype=np.float64)
    t_train = np.arange(1, Ntime + 1)
    dt = t_train[1] - t_train[0]
    if Ntime % 2 == 0:
        t_test = t_train[:] + 0.5
    else:
        t_test = t_train[:-1] + 0.5

    a_coef_test = koopman_prediction(a_coef_train, dt, t_test, iterations=500)
    qtest_pod  = phisvd[0][:, :rank] @ a_coef_test
    qtrain_pod = phisvd[0][:, :rank] @ a_coef_train
    err_offline =  np.linalg.norm(qtrain_pod - qtrain, "fro") / np.linalg.norm(qtrain, "fro")
    err_online = np.linalg.norm(qtest_pod - qtest, "fro") / np.linalg.norm(qtest, "fro")
    errPOD.append(err_offline)
    print("rank: %d, error offline: %.4f online: %.4f" % (rank, err_offline, err_online))
    with open(file_name_results, 'a') as f:
        f.write("%d \t %.1e \t %.1e\n" % (rank, err_offline, err_online))

# %% compare results
import matplotlib
fig = plt.figure(33)
plt.semilogy(rank_list, errPOD, 'x', label="POD")
plt.semilogy(rank_list, errNN, 'o', label="NN")
plt.semilogy(rank_list, errNN2, '<', label="FTR-NN")
plt.semilogy(rank_list, errFTR,'*',  label="FTR")
plt.legend(loc="upper right",frameon=False)
plt.xlabel("degrees of freedom $r$")
plt.ylabel("relative error")
plt.yscale("log")
ax = plt.gca()
y_major = matplotlib.ticker.LogLocator(base = 10.0, numticks = 5)
ax.yaxis.set_major_locator(y_major)
y_minor = matplotlib.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
ax.yaxis.set_minor_locator(y_minor)
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

plt.grid(which='minor', linestyle='--')
plt.grid(which='major', linestyle='-')
plt.xlim([0,33])
plt.ylim([1e-5,1])
save_fig("../imgs/moving-disc-rel-offline-err.tikz",fig)
# %%  compares FTR, NN, POD in a pcolor plot
nt_show=10
pcol_list = []

figure, axes = plt.subplots(nrows=3, ncols=3, sharey="all", sharex="all", num=2,
                            figsize=(6, 6*len(train_results_list)))
n=0
Ngrid = q.shape[:-1]
q_tilde_ftr = front(phi_ftr)
df = q_tilde_ftr - qtrain
axes[n, 0].pcolormesh(phi_ftr[...,nt_show].reshape(Ngrid))
axes[n, 1].pcolormesh(q_tilde_ftr[...,nt_show].reshape(Ngrid))
pcol_list.append(axes[n, 2].pcolormesh(df[...,nt_show].reshape(Ngrid)))
axes[n, 0].set_ylabel("FTR", size='large')
# %
# plot FTR-NN results

axes[-2, 0].pcolormesh(phiNN[nt_show,0,...].cpu())
axes[-2, 1].pcolormesh(qtildeNN[nt_show,0,...]Proper.cpu())
pcol_list.append(axes[-2, 2].pcolormesh(qtrain[...,nt_show].reshape(Ngrid) - qtildeNN[nt_show,0,...].cpu().numpy()))
axes[-2, 0].set_ylabel("FTR-NN", size='large')
#figure.suptitle(str(dofs) + " DOFs")

# %
# plot NN results

axes[-1, 0].pcolormesh(phiNN2[nt_show,0,...].cpu())
axes[-1, 1].pcolormesh(qtildeNN2[nt_show,0,...].cpu())
pcol_list.append(axes[-1, 2].pcolormesh(qtrain[...,nt_show].reshape(Ngrid) - qtildeNN2[nt_show,0,...].cpu().numpy()))
axes[-1, 0].set_ylabel("NN", size='large')

# # %%
# qtilde = POD_predict["q"][dofs][..., nt_show]
# axes[-1, 0].pcolormesh(qtilde)
# axes[-1, 1].pcolormesh(qtilde)
# pcol_list.append(axes[-1, 2].pcolormesh(q_test[..., nt_show].reshape(qtilde.shape) - qtilde))
# # axes[-1, 0].set_title("Data", size='large')
# axes[-1, 0].set_ylabel("POD", size='large')
# # axes[-1, 2].set_title("Data-POD", size='large')
# figure.suptitle(str(dofs) + " DOFs")
# # remove all ticks and set column labels
for ax in axes.flatten():
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box')
    gs = ax.get_gridspec()
    gs.update(wspace=0.0, hspace=0.0)
for ax, col in zip(axes[0], [r"$\phi$", r"$f(\phi)$", r"$q-f(\phi)$"]):
    ax.set_title(col)

cmin = min([pcol_list[n].get_clim()[0] for n in range(len(pcol_list))])
cmax = max([pcol_list[n].get_clim()[1] for n in range(len(pcol_list))])
for pcol in pcol_list:
    pcol.set_clim([cmin, cmax])

save_fig("../imgs/moving-disc-FTR-approx.tikz.tex",figure,strict=True)

# %% POD
rank = 3
from mpl_toolkits.axes_grid1 import make_axes_locatable

figure, axes = plt.subplots(nrows=1, ncols=2, sharey="all", sharex="all", num=3)
qsvd = np.linalg.svd(qtrain, full_matrices=False)
qPOD = np.asarray(qsvd[0][:,:rank]@np.diag(qsvd[1][:rank]) @ qsvd[2][:rank, :], dtype=np.float64)
handl = [0,0]
handl[0] = axes[0].pcolormesh(qtrain[...,nt_show].reshape(Ngrid))
axes[0].set_title(r"data $q(\boldsymbol{x},t)$")
handl[1] = axes[1].pcolormesh(qPOD[...,nt_show].reshape(Ngrid))
axes[1].set_title(r"POD $\tilde{q}(\boldsymbol{x},t)$")
for k,ax in enumerate(axes.flatten()):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box')
    gs = ax.get_gridspec()
    gs.update(wspace=0.1, hspace=0.0)

    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes('bottom', size='5%', pad=0.05)
    # figure.colorbar(handl[k], cax=cax, orientation='horizontal')
# cmin = min([handl[n].get_clim()[0] for n in range(len(handl))])
# cmax = max([handl[n].get_clim()[1] for n in range(len(handl))])
# for pcol in handl:
#     pcol.set_clim([cmin, cmax])

save_fig("../imgs/moving-disc-pod-approx.tikz.tex",figure,strict=True)