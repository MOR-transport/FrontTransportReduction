from globs import *
import numpy as np
import os
import torch
from NeuralFTR.synthetic_data import make_grid, generate_data, f
from NeuralFTR.ftr_net import FTR_Dec_1Lay, FTR_Dec
from NeuralFTR.main import train_NN
import matplotlib.pyplot as plt
from plot_utils import save_fig
from compare_compression import *
from ROM.FTR import FTR_ranks
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
data_folder= '../data/training_results/'+case+'/'
train_results_list=[ {"name": "FTR-NN" , "folder": data_folder+'/FTR-NN/lambda_%.0e/'%smoothness,"decoder": FTR_Dec_1Lay},
                     {"name": "NN" , "folder": data_folder+"/NN/lambda_%.0e/"%smoothness,"decoder": FTR_Dec}]

# train FTR-NN
train_NN(data_train, data_test, train_results_list[0], X, Y, smooth = smoothness,
         skip_existing = skip_existing, dofs = rank_list)
errNN, rank_list = give_NN_errors(data_set[1::2], [train_results_list[0]])
phiNN, qtildeNN = give_NN_fields(data_set[1::2], [train_results_list[0]], dofs = 3)

# train NN
train_NN(data_train, data_test, train_results_list[1], X, Y , smooth = smoothness,
         skip_existing = skip_existing, dofs = rank_list)
errNN2, rank_list = give_NN_errors(data_set[1::2], [train_results_list[1]])
phiNN2, qtildeNN2 = give_NN_fields(data_set[1::2], [train_results_list[1]], dofs = 3)
###################################################################
# %% FTR for all ranks
###################################################################
rank_list = [1,2,3,4,5,6,7,8,9,10,15,20,30]
data_folder_FTR = data_folder+"/FTR/"
if not os.path.exists(data_folder_FTR):
    os.makedirs(data_folder_FTR)

front = lambda x:  (np.tanh(x/lambd) + 1) * 0.5
max_iter = np.ones_like(rank_list)*40000
max_iter[0] = 400

phi_ftr_list, q_ftr_list, errFTR_offline =  FTR_ranks(qtrain, rank_list, front, save_fname = data_folder_FTR+case,
                                                      max_iter = max_iter, print_step = 10000, dt = 0.3, offset = 1,
                                                      stop_if_residual_increasing = True, skip_existing = skip_existing)
file_name_results = data_folder + "/FTR-error-moving-discb.txt"
errFTR = []
with open(file_name_results, 'w') as f:
    f.write("moving disc\n")
    f.write("rank\toffline-err\tonline-err\n")
for k,rank in enumerate(rank_list):
    # offline
    phi_ftr, qtilde, err = phi_ftr_list[k], q_ftr_list[k], errFTR_offline[k]
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
file_name_results = data_folder + "/POD-error-moving-disc.txt"
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


###################################################################
# %% compare results quantitative
###################################################################
fig = plt.figure(33)
plt.semilogy(rank_list, errPOD, 'x', label="POD")
plt.semilogy(rank_list, errNN2, '<', label="NN")
plt.semilogy(rank_list, errNN, 'o', label="FTR-NN",  mfc='none')
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


###################################################################
# %%  compares FTR, NN, POD in a pcolor plot
###################################################################

nt_show=1
pcol_list = []
pcol_phi_list = []

figure, axes = plt.subplots(nrows=3, ncols=3, sharey="all", sharex="all", num=34,
                            figsize=(8, 8))
rank = 3
if rank_list[2]==rank:
    phi_ftr = phi_ftr_list[2]
else:
    assert True, "rank should be 3 for this example"

n=0
Ngrid = q.shape[:-1]
q_tilde_ftr = front(phi_ftr)
df = q_tilde_ftr - qtrain
pcol_phi_list.append(axes[n, 0].pcolormesh(phi_ftr[...,nt_show].reshape(Ngrid)))
axes[n, 1].pcolormesh(q_tilde_ftr[...,nt_show].reshape(Ngrid))
pcol_list.append(axes[n, 2].pcolormesh(df[...,nt_show].reshape(Ngrid)))
axes[n, 0].set_ylabel("FTR", size='large')
# %
# plot FTR-NN results

pcol_phi_list.append(axes[-2, 0].pcolormesh(phiNN[nt_show,0,...].cpu()))
axes[-2, 1].pcolormesh(qtildeNN[nt_show,0,...].cpu())
pcol_list.append(axes[-2, 2].pcolormesh(qtrain[...,nt_show].reshape(Ngrid) - qtildeNN[nt_show,0,...].cpu().numpy()))
axes[-2, 0].set_ylabel("FTR-NN", size='large')
#figure.suptitle(str(dofs) + " DOFs")

# %
# plot NN results

pcol_phi_list.append(axes[-1, 0].pcolormesh(phiNN2[nt_show,0,...].cpu()))
pcol_approx = axes[-1, 1].pcolormesh(qtildeNN2[nt_show,0,...].cpu())
pcol_list.append(axes[-1, 2].pcolormesh(qtrain[...,nt_show].reshape(Ngrid) - qtildeNN2[nt_show,0,...].cpu().numpy()))
axes[-1, 0].set_ylabel("NN", size='large')



#
# axins1 = inset_axes(axes[-1,2],
#                     width="80%",  # width = 50% of parent_bbox width
#                     height="5%",  # height : 5%
#                     loc='lower left')
# fig.colorbar(pcol_list[-1], cax=axins1, orientation="horizontal")
# axins1.xaxis.set_ticks_position("bottom")
# # %%
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

# colorbar of errors
box = axes[-1,2].get_position()
axColor = plt.axes([box.x0+box.width * 0.1, box.y0-0.1*box.height, box.width * 0.8, 0.01])
plt.colorbar(pcol_list[-1], cax = axColor, orientation="horizontal", ticks = [-0.4,0,0.3])
plt.show()

# # colorbar of approximation
# box = axes[-1,1].get_position()
# axColor = plt.axes([box.x0+box.width * 0.1, box.y0-0.1*box.height, box.width * 0.8, 0.01])
# plt.colorbar(pcol_approx, cax = axColor, orientation="horizontal")
# plt.show()
#
# # colorbar for phi
# for k,pcol in enumerate(pcol_phi_list):
#     # colorbar of errors
#     box = axes[k,0].get_position()
#     axColor = plt.axes([box.x0+box.width * 0.1, box.y0+0.05*box.height, box.width * 0.8, 0.01])
#     cb = plt.colorbar(pcol_phi_list[k], cax = axColor, orientation="horizontal")
#     cb.ax.xaxis.set_ticks_position("top")
#     plt.show()


save_fig("../imgs/moving-disc-FTR-approx.tikz.tex",figure,strict=True)

# %% POD
rank = 10
phi_ftr = phi_ftr_list[2]
from mpl_toolkits.axes_grid1 import make_axes_locatable

figure, axes = plt.subplots(nrows=1, ncols=2, sharey="all", sharex="all", num=35)
qsvd = np.linalg.svd(qtrain, full_matrices=False)
qPOD = np.asarray(qsvd[0][:,:rank]@np.diag(qsvd[1][:rank]) @ qsvd[2][:rank, :], dtype=np.float64)
handl = [0,0]
handl[0] = axes[0].pcolormesh(qtrain[...,nt_show].reshape(Ngrid))
axes[0].set_title(r"data $q(\vec{x},t)$")
handl[1] = axes[1].pcolormesh(qPOD[...,nt_show].reshape(Ngrid))
axes[1].set_title(r"POD $\tilde{q}(\vec{x},t)$")
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


# %% interpolation koopman forecasting

# --------------------- POD:
rank = 3
phisvd = np.linalg.svd(qtrain, full_matrices=False)
a_coef_train = np.asarray(np.diag(phisvd[1][:rank]) @ phisvd[2][:rank, :], dtype=np.float64)
t_train = np.arange(1, Ntime + 1)
dt = t_train[1] - t_train[0]
a_coef_test = koopman_prediction(a_coef_train, dt, t_test, iterations=500)
qtest_pod = phisvd[0][:, :rank] @ a_coef_test
qtrain_pod = phisvd[0][:, :rank] @ a_coef_train
#a_coef_train
#a_coef_test

time = t.flatten()
ttrain = time[::2]
ttest = time[1::2]
figure = plt.figure(36)
plt.plot(ttrain,a_coef_train.T,'-*', label="train")
plt.plot(ttest,a_coef_test.T,'ko',mfc='none', label = "test")


legend_elements = [Line2D([0], [0], color='k', marker='x', label='Train'),
                   Line2D([0], [0], marker='o', color='k', label='Test',linestyle="",mfc='none')]
plt.legend(handles=legend_elements, loc='center left')
plt.xlabel(r"time $t$")
plt.ylabel(r"coefficient $\vec{a}(t)=\mathbf{A}\vec{\Omega}(t)$")
plt.title(r"POD $\vec{q}(t) = \hat{\mathbf{U}}\mathbf{A}\vec{\Omega}(t)$")
save_fig("../imgs/moving-disc-POD-acoefs-forecast.tikz.tex",figure,strict=True)




fig,axes = plt.subplots(2,2,sharex=True, sharey=True,num=37,figsize=(10,10))

for k,ax in enumerate(axes.flatten()):
    it = 9+ k*10
    ax.pcolormesh(qtest_pod[:,it].reshape(Ngrid))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(r"$t=%.1f$"%ttest[it])

save_fig("../imgs/moving-disc-POD-forecast.tikz.tex",fig,strict=True)


# %% --------------------- FTR: fourier koopman predictions
rank = 3
if rank_list[2]==rank:
    phi_ftr = phi_ftr_list[2]
else:
    assert True, "rank should be 3 for this example"

phisvd = np.linalg.svd(phi_ftr, full_matrices=False)
a_coef_train = np.asarray(np.diag(phisvd[1][:rank]) @ phisvd[2][:rank, :], dtype=np.float64)
t_train = np.arange(1, Ntime + 1)
dt = t_train[1] - t_train[0]
a_coef_test = koopman_prediction(a_coef_train, dt, t_test, iterations=500)
qtest_ftr = front(phisvd[0][:, :rank] @ a_coef_test)
qtrain_ftr = front(phisvd[0][:, :rank] @ a_coef_train)
#a_coef_train
#a_coef_test

time = t.flatten()
ttrain = time[::2]
ttest = time[1::2]
figure = plt.figure(38)
plt.plot(ttrain,a_coef_train.T,'-*', label="train")
plt.plot(ttest,a_coef_test.T,'ko',mfc='none', label = "test")

legend_elements = [Line2D([0], [0], color='k', marker='x', label='Train'),
                   Line2D([0], [0], marker='o', color='k', label='Test',linestyle="",mfc='none')]

plt.legend(handles=legend_elements, loc='upper left')
plt.xlabel(r"time $t$")
plt.ylabel(r"coefficient $\vec{a}(t)=A\vec{\Omega}(t)$")
plt.title(r"FTR $\vec{q}(t) = f(\mathbf{U}\mathbf{A}\vec{\Omega}(t))$")
save_fig("../imgs/moving-disc-FTR-acoefs-forecast.tikz.tex",figure,strict=True)


fig,axes = plt.subplots(2,2,sharex=True, sharey=True,num=7,figsize=(10,10))

for k,ax in enumerate(axes.flatten()):
    it = 9+ k*10
    ax.pcolormesh(qtest_ftr[:,it].reshape(Ngrid))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(r"$t=%.1f$"%ttest[it])

save_fig("../imgs/moving-disc-FTR-forecast.tikz.tex", fig, strict=True)


#########################################################################################
# %% Plot transport field
#########################################################################################
rank = 3
if rank_list[2]==rank:
    phi_ftr = phi_ftr_list[2]
else:
    assert True, "rank should be 3 for this example"

phi_mat = phi_ftr
phi_svd = np.linalg.svd(phi_mat, full_matrices=False)

fig, ax = plt.subplots(2,3,sharex=True,num = 39,figsize=(16,8))
basis = np.asarray([-(X-5)**2-(Y-5)**2-R.flatten()[0]**2,Y,X])
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
    ax[0, col].set_title(r'$\psi_%d(x,y)$'%(col+1),  fontsize=30)
    ax[1, col].set_title(r'$\tilde{\psi}_%d(x,y)$' %(col+1), fontsize=30)
    ax[1,col].set_yticks([])
    ax[1,col].set_xticks([])
    ax[0,col].set_yticks([])
    ax[0,col].set_xticks([])
    ax[1,col].set_yticklabels([])
    ax[1,col].set_xticklabels([])
    ax[0,col].set_yticklabels([])
    ax[0,col].set_xticklabels([])


    ax[1,col].set_xlabel(r"$x$", fontsize=30)

ax[0,0].set_ylabel(r"$y$", fontsize=30)
ax[1,0].set_ylabel(r"$y$", fontsize=30)

# draw arrows
ax[0,1].quiver(5,5,0,1,angles='xy', scale_units='xy', scale=0.2, pivot="middle",minshaft=2,width=0.2)
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
save_fig("../imgs/disc_transport_map.png",strict=True)

plt.figure(4)
a_coef = (np.diag(phi_svd[1])@phi_svd[2])
plt.plot(time[::2],a_coef[:3,:].T)
plt.xlabel(r"time $t$")
plt.ylabel(r"$\tilde{a}_i(t)$")
plt.legend([r"$i=1$",r"$i=2$",r"$i=3$"],loc=4)
save_fig("../imgs/disc_transport_map_time.png")


# %%
smooth_range = [1e-3,1e-6,1e-9]
fig, axes = plt.subplots(2,len(smooth_range), sharey=True,num=43, figsize=(11,8))
h1_list = []
h2_list = []
for k, smooth in enumerate(smooth_range):

    train_results_list=[ {"name": "FTR-NN" , "folder": data_folder+'/FTR-NN/lambda_%.0e/'%smooth,"decoder": FTR_Dec_1Lay},
                         {"name": "NN" , "folder": data_folder+"/NN/lambda_%.0e/"%smooth,"decoder": FTR_Dec}]


    train_NN(data_train, data_test, train_results_list[0], X, Y, smooth = smooth,
             skip_existing = skip_existing, dofs = rank_list)
    train_NN(data_train, data_test, train_results_list[1], X, Y, smooth = smooth,
             skip_existing = skip_existing, dofs = rank_list)
    phiNN2, qtildeNN2 = give_NN_fields(data_set[1::2], [train_results_list[1]], dofs = 3)
    phiNN, qtildeNN = give_NN_fields(data_set[1::2], [train_results_list[0]], dofs = 3)

    h1_list.append(axes[0, k].pcolormesh(phiNN[nt_show, 0, ...].cpu()))
    h2_list.append(axes[1, k].pcolormesh(phiNN2[nt_show, 0, ...].cpu()))

    axes[0,k].set_xticks([])
    axes[0,k].set_yticks([])
    # axes[0,k].set_xlabel(r"$x$")
    # axes[0,k].set_ylabel(r"$y$")

    axes[1,k].set_xticks([])
    axes[1,k].set_yticks([])
    # axes[1,k].set_xlabel(r"$x$")
    # axes[1,k].set_ylabel(r"$y$")



cmin = min([h1_list[n].get_clim()[0] for n in range(len(pcol_list))])
cmax = max([h1_list[n].get_clim()[1] for n in range(len(pcol_list))])
for pcol in h1_list:
    pcol.set_clim([cmin, cmax])


cmin = min([h2_list[n].get_clim()[0] for n in range(len(pcol_list))])
cmax = max([h2_list[n].get_clim()[1] for n in range(len(pcol_list))])
for pcol in h2_list:
    pcol.set_clim([cmin, cmax])

for ax, col in zip(axes[0], [r"$\lambda_{\mathrm{smooth}}=10^{-3}$", r"$\lambda_{\mathrm{smooth}}=10^{-6}$", r"$\lambda_{\mathrm{smooth}}=10^{-9}$"]):
    ax.set_title(col)

axes[0, 0].set_ylabel("FTR-NN", size='large')
axes[1, 0].set_ylabel("NN", size='large')


box = axes[0,k].get_position()
axColor = plt.axes([box.x0+box.width *1.05, box.y0+0.1*box.height, 0.01, 0.8*box.height])
plt.colorbar(h1_list[-1], cax = axColor, orientation="vertical")

box = axes[1,k].get_position()
axColor = plt.axes([box.x0+box.width *1.05, box.y0+0.1*box.height, 0.01, 0.8*box.height])
plt.colorbar(h2_list[-1], cax = axColor, orientation="vertical")

save_fig("../imgs/disc_smoothness.png")
