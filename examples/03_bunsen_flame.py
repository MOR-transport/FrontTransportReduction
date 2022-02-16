import globs
import numpy as np
import torch
from numpy.linalg import svd
import matplotlib.pyplot as plt
from openfoam_io import read_data
from NeuralFTR.ftr_net import FTR_AE, FTR_Dec_1Lay, FTR_Dec
from NeuralFTR.main import train_NN
from compare_compression import *
from plot_utils import bound_exceed_cmap, save_fig
from compare_compression import koopman_prediction
from ROM.FTR import truncate, convergence_plot, FTR_ranks
from os.path import expanduser
import os
from glob import glob
import re
import scipy.io
from matplotlib import rc
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Using: "+ torch.cuda.get_device_name(0))
latexfont = {'size' : 24}
rc('font',**latexfont)
home = expanduser("~")

case = "flame_pinch_off"

if __name__ == "__main__":
    plt.close("all")
    max_iter = 10
    dofs = 30 # degrees of freedom used by the rom
    ####################################################################
    # load the data
    ####################################################################
    rank_list = [2,4,6,8,10,12,15]
    qty_name = "CH4"
    case = "FlamePinchOff"
    smoothness = 1e-7
    skip_existing = True
    folder = home + "/tubcloud/FTR/04_FlamePinchOff/"  # Name of the folder
    cleaning_method = "Normalize"
    file = qty_name + "_" + cleaning_method + "_smalldomain.mat"
    q, X, Y = read_data(folder + file)

    data_set = q[50::, ...]
    Nt_train = np.size(data_set[::2],0)
    Nt_test = np.size(data_set,0)-Nt_train
    Nt = Nt_train + Nt_test
    time = np.linspace(0.01,0.05,Nt)
    ttrain = time[::2]
    ttest = time[1::2]
    qtrain =  np.reshape(np.squeeze(np.moveaxis(data_set[::2], 0, -1)),[-1, Nt_train])
    qtest = np.reshape(np.squeeze(np.moveaxis(data_set[1::2], 0, -1)), [-1, Nt_train-1])

    data_train = torch.tensor(data_set[::2],dtype=torch.float32).to(DEVICE)
    data_test = torch.tensor(data_set[1::2],dtype=torch.float32).to(DEVICE)
    Nt_train = np.size(data_test, 0)
    ###################################################################
    # %% Evals NNs for all dofs
    ###################################################################
    data_folder = '../data/training_results/'+case+'/'
    train_results_list = [
        {"name": "FTR-NN", "folder": data_folder + '/FTR-NN/lambda_%.0e/' % smoothness, "decoder": FTR_Dec_1Lay},
        {"name": "NN", "folder": data_folder + "/NN/lambda_%.0e/" % smoothness, "decoder": FTR_Dec}]

    # train FTR-NN
    train_NN(data_train, data_test, train_results_list[0], X, Y, smooth=smoothness,
             skip_existing=skip_existing, dofs=rank_list)
    errNN, rank_list = give_NN_errors(data_test, [train_results_list[0]])


    # train NN
    # train_NN(data_train, data_test, train_results_list[1], X, Y, smooth=smoothness,
    #          skip_existing=skip_existing, dofs=rank_list)
    # errNN2, rank_list = give_NN_errors(data_test, [train_results_list[1]])
    # phiNN2, qtildeNN2 = give_NN_fields(data_test, [train_results_list[1]], dofs=3)
    ###################################################################
    #%% FTR for all ranks
    ###################################################################
    front = lambda x: (np.tanh(x) + 1) * 0.5
    data_folder_FTR = data_folder + "/FTR/"
    if not os.path.exists(data_folder_FTR):
        os.makedirs(data_folder_FTR)

    max_iter = np.ones_like(rank_list) * 40000
    max_iter[0] = 400

    phi_ftr_list, q_ftr_list, errFTR_offline = FTR_ranks(qtrain, rank_list, front, save_fname=data_folder_FTR + case,
                                                         max_iter=max_iter, print_step=10000, dt=0.3, offset=1,
                                                         stop_if_residual_increasing=True, skip_existing=skip_existing)
    file_name_results = data_folder + "/FTR-error.txt"
    qtest_FTR_list =[]
    with open(file_name_results, 'w') as f:
        f.write("flame pinch off\n")
        f.write("rank\toffline-err\tonline-err\n")
    for k,rank in enumerate(rank_list):
        # offline
        phi_ftr = phi_ftr_list[k]
        err = errFTR_offline[k]
        # online
        phisvd = np.linalg.svd(phi_ftr, full_matrices=False)
        a_coef_phi = np.diag(phisvd[1][:rank]) @ phisvd[2][:rank,:]
        t_train = np.arange(1, Nt_train + 1)
        dt = t_train[1] - t_train[0]
        if Nt_train % 2 == 0:
            t_test = t_train[:] + 0.5
        else:
            t_test = t_train[:-1] + 0.5

        a = koopman_prediction(a_coef_phi, dt, t_test, iterations = 30)
        qtest_ftr = front(phisvd[0][:,:rank]@a)
        qtest_FTR_list.append(qtest_ftr)
        err_online = np.linalg.norm(qtest_ftr - qtest, "fro") / np.linalg.norm(qtest, "fro")
        print("rank: %d, error offline: %.4f online: %.4f" % (rank, err, err_online))
        with open(file_name_results, 'a') as f:
            f.write("%d \t %.1e \t %.1e\n" % (rank, err, err_online))

    np.save(data_folder+"/FlamePinchOff_FTR_phi_list",phi_ftr_list)

    ###################################################################
    # %% POD for all ranks
    ###################################################################
    qtest_pod_list = []
    Nt_train = np.size(data_set[::2], 0)
    file_name_results = data_folder+"/POD-error.txt"
    with open(file_name_results, 'w') as f:
        f.write("flame pinch off\n")
        f.write("rank\toffline-err\tonline-err\n")
    for rank in rank_list:
        # offline
        phisvd = np.linalg.svd(qtrain, full_matrices=False)
        a_coef_train = np.diag(phisvd[1][:rank]) @ phisvd[2][:rank, :]
        t_train = np.arange(1, Nt_train + 1)
        dt = t_train[1] - t_train[0]
        if Nt_train % 2 == 0:
            t_test = t_train[:-1] + 0.5
        else:
            t_test = t_train[:-1] + 0.5

        a_coef_test = koopman_prediction(a_coef_train, dt, t_test)
        qtest_pod  = phisvd[0][:, :rank] @ a_coef_test
        qtrain_pod = phisvd[0][:, :rank] @ a_coef_train
        qtest_pod_list.append(qtest_pod)
        err_offline =  np.linalg.norm(qtrain_pod - qtrain, "fro") / np.linalg.norm(qtrain, "fro")
        err_online = np.linalg.norm(qtest_pod - qtest, "fro") / np.linalg.norm(qtest, "fro")
        print("rank: %d, error offline: %.4f online: %.4f" % (rank, err_offline, err_online))
        with open(file_name_results, 'a') as f:
            f.write("%d \t %.1e \t %.1e\n" % (rank, err_offline, err_online))


# %% Pcolor FTR vs POD vs DATA
    figure, axes = plt.subplots(num = 2,nrows=3, ncols=1, sharey="all", sharex="all")#,
                                #figsize=(10, (len(train_results_list)+4)))
    imagepath = "../imgs/"+case
    if not os.path.exists(imagepath):
        os.makedirs(imagepath)

    nt_show = 66
    ir = 3
    rank = rank_list[ir]
    Ngrid = np.shape(X)
    phiNN, qtilde_NN = give_NN_fields(data_test, [train_results_list[0]], dofs=rank_list[ir])
    qtilde_NN = np.reshape(np.squeeze(np.moveaxis(qtilde_NN.to("cpu").numpy(), 0, -1)), [-1, Nt_train-1])
    qtilde_FTR = qtest_FTR_list[ir]
    qtilde_POD = qtest_pod_list[ir]
    print("rank: ", rank)

    newcmp = bound_exceed_cmap(Index_upper=65, Index_lower=16)

    vmin = np.min(qtilde_POD[...,nt_show])
    vmax = np.max(qtilde_POD[...,nt_show])

    axes[0].set_title(r"$t=%.2f$"%ttrain[nt_show])
    axes[0].imshow(np.reshape(qtest[...,nt_show],Ngrid), vmin = vmin, vmax = vmax, cmap=newcmp)
    axes[0].set_ylabel("data")
    im = axes[1].imshow(np.reshape(qtilde_POD[..., nt_show],Ngrid),vmin = vmin, vmax = vmax,cmap=newcmp)
    axes[1].set_ylabel("POD")
    axes[2].imshow(np.reshape(qtilde_FTR[..., nt_show],Ngrid),  vmin = vmin, vmax = vmax,cmap=newcmp)
    axes[2].set_ylabel("FTR")

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', adjustable='box')
        gs = ax.get_gridspec()

    #figure.subplots_adjust(right=1.1)
    cbar_ax = figure.add_axes([0.85, 0.15, 0.05, 0.7])
    figure.colorbar(im, cax=cbar_ax,cmap=newcmp)

    save_fig(imagepath + "/FTR-online-bunsen02.png", figure)

    figure, axes = plt.subplots(num = 61,nrows=3, ncols=1, sharey="all", sharex="all")#,
                                #figsize=(10, (len(train_results_list)+4)))
    nt_show = 10
    ir = 3
    rank = rank_list[ir]
    Ngrid = np.shape(X)
    phiNN, qtilde_NN = give_NN_fields(data_test, [train_results_list[0]], dofs=rank_list[ir])
    qtilde_NN = np.reshape(np.squeeze(np.moveaxis(qtilde_NN.to("cpu").numpy(), 0, -1)), [-1, Nt_train-1])
    qtilde_FTR = qtest_FTR_list[ir]
    qtilde_POD = qtest_pod_list[ir]
    print("rank: ", rank)

    newcmp = bound_exceed_cmap(Index_upper=40, Index_lower=32)

    vmin = np.min(qtilde_POD[...,nt_show])
    vmax = np.max(qtilde_POD[...,nt_show])

    axes[0].set_title(r"$t=%.2f$"%ttrain[nt_show])
    axes[0].imshow(np.reshape(qtest[...,nt_show],Ngrid), vmin = vmin, vmax = vmax, cmap=newcmp)
    axes[0].set_ylabel("data")
    im = axes[1].imshow(np.reshape(qtilde_POD[..., nt_show],Ngrid),vmin = vmin, vmax = vmax,cmap=newcmp)
    axes[1].set_ylabel("POD")
    axes[2].imshow(np.reshape(qtilde_FTR[..., nt_show],Ngrid),  vmin = vmin, vmax = vmax,cmap=newcmp)
    axes[2].set_ylabel("FTR")

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', adjustable='box')
        gs = ax.get_gridspec()

    #figure.subplots_adjust(right=1.1)
    cbar_ax = figure.add_axes([0.85, 0.15, 0.05, 0.7])
    figure.colorbar(im, cax=cbar_ax,cmap=newcmp)

    save_fig(imagepath+"/FTR-online-bunsen01.png", figure)

# %% time predictions


    phi_ftr = phi_ftr_list[ir]
    # online
    phisvd = np.linalg.svd(phi_ftr, full_matrices=False)
    a_coef_train =  np.diag(phisvd[1][:rank]) @phisvd[2][:rank, :]

    t_train = np.arange(1, Nt_train+1)
    dt = t_train[1] - t_train[0]
    if Nt_train % 2 == 0:
        t_test = t_train[:-1] + 0.5
    else:
        t_test = t_train[:-1] + 0.5

    a_coef_test = koopman_prediction(a_coef_train, dt, t_test,iterations=20)
    qtest_ftr = front(phisvd[0][:, :rank] @ a_coef_test)

    figure = plt.figure(38)
    htrain = plt.plot(ttrain,a_coef_train[:3,:].T,'-*', label="train")
    htest = plt.plot(ttest,a_coef_test[:3,:].T,'ko',mfc='none', label = "test")
    plt.xlabel(r"time $t$")
    plt.ylabel(r"coefficients $\vec{a}(t)$")
    plt.legend([htrain[0],htrain[1],htrain[2],htest[0]],[r'training $a_1(t)$',r'training $a_2(t)$',r'training $a_3(t)$','test'])
    save_fig(imagepath+"/FTR-online-bunsen_time.png", figure)

