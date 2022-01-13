import numpy as np
import torch
from numpy.linalg import svd
import matplotlib.pyplot as plt
from ALM_FTR_Galerkin_python.openfoam_io import read_data
from NeuralFTR.ftr_net import FTR_AE, FTR_Dec_1Lay, FTR_Dec
from NeuralFTR.utils import to_torch
from time import perf_counter
from compare_compression import koopman_prediction
from ALM_FTR_Galerkin_python.lib.FTR import truncate, convergence_plot, simple_FTR
from os.path import expanduser
import os
from glob import glob
import re
from PaperPlots.utils import natural_keys
import scipy.io
from matplotlib import rc

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#plt.style.use('dark_background')
latexfont = {'size' : 24}
rc('font',**latexfont)
home = expanduser("~")

if __name__ == "__main__":
    plt.close("all")
    max_iter = 10
    dofs = 30 # degrees of freedom used by the rom
    ####################################################################
    # load the data
    ####################################################################
    rank_list = [2,4,6,8,10,12,15]
    qty_name = "CH4"
    folder = home + "/tubcloud/FTR/04_FlamePinchOff/"  # Name of the folder
    cleaning_method = "Normalize"
    file = qty_name + "_" + cleaning_method + "_smalldomain.mat"
    q, X, Y = read_data(folder + file)
    data_set = q[50::, ...]

    ###################################################################
    #%% FTR for all ranks
    ###################################################################
    front = lambda x: (np.tanh(-x) + 1) * 0.5
    phi_ftr_list = []
    q_ftr_list = []
    Ntime = np.size(data_set[::2],0)
    qtrain =  np.reshape(np.squeeze(np.moveaxis(data_set[::2], 0, -1)),[-1, Ntime])
    qtest = np.reshape(np.squeeze(np.moveaxis(data_set[1::2], 0, -1)), [-1, Ntime-1])
    file_name_results = "../data/FTR-error.txt"
    with open(file_name_results, 'w') as f:
        f.write("flame pinch off\n")
        f.write("rank\toffline-err\tonline-err\n")
    for rank in rank_list:
        # offline
        phi_ftr = simple_FTR(qtrain, front, max_iter=800, rank=rank, print_step=10, stop_if_residual_increasing=True)
        qtilde_FTR = front(phi_ftr)
        q_ftr_list.append(qtilde_FTR)
        phi_ftr_list.append(phi_ftr)
        err = np.linalg.norm(qtilde_FTR - qtrain, "fro") / np.linalg.norm(qtrain, "fro")
        # online
        phisvd = np.linalg.svd(phi_ftr, full_matrices=False)
        a_coef_phi = np.diag(phisvd[1][:rank]) @ phisvd[2][:rank,:]
        t_train = np.arange(1, Ntime + 1)
        dt = t_train[1] - t_train[0]
        if Ntime % 2 == 0:
            t_test = t_train[:-1] + 0.5
        else:
            t_test = t_train[:-1] + 0.5

        a = koopman_prediction(a_coef_phi, dt, t_test, iterations = 100)
        qtest_ftr = front(phisvd[0][:,:rank]@a)
        err_online = np.linalg.norm(qtest_ftr - qtest, "fro") / np.linalg.norm(qtest, "fro")
        print("rank: %d, error offline: %.4f online: %.4f" % (rank, err, err_online))
        with open(file_name_results, 'a') as f:
            f.write("%d \t %.1e \t %.1e\n" % (rank, err, err_online))

    np.save("../data/FlamePinchOff_FTR_phi_list",phi_ftr_list)

    ###################################################################
    # %% POD for all ranks
    ###################################################################
    q_pod_list = []
    Ntime = np.size(data_set[::2], 0)
    file_name_results = "../data/POD-error.txt"
    with open(file_name_results, 'w') as f:
        f.write("flame pinch off\n")
        f.write("rank\toffline-err\tonline-err\n")
    for rank in rank_list:
        # offline
        phisvd = np.linalg.svd(qtrain, full_matrices=False)
        a_coef_train = np.diag(phisvd[1][:rank]) @ phisvd[2][:rank, :]
        t_train = np.arange(1, Ntime + 1)
        dt = t_train[1] - t_train[0]
        if Ntime % 2 == 0:
            t_test = t_train[:-1] + 0.5
        else:
            t_test = t_train[:-1] + 0.5

        a_coef_test = koopman_prediction(a_coef_train, dt, t_test)
        qtest_pod  = phisvd[0][:, :rank] @ a_coef_test
        qtrain_pod = phisvd[0][:, :rank] @ a_coef_train
        err_offline =  np.linalg.norm(qtrain_pod - qtrain, "fro") / np.linalg.norm(qtrain, "fro")
        err_online = np.linalg.norm(qtest_pod - qtest, "fro") / np.linalg.norm(qtest, "fro")
        print("rank: %d, error offline: %.4f online: %.4f" % (rank, err_offline, err_online))
        with open(file_name_results, 'a') as f:
            f.write("%d \t %.1e \t %.1e\n" % (rank, err_offline, err_online))

    ###################################################################
    # %% NN-FTR
    ###################################################################

    data_folder = home + "/tubcloud/FTR/01_NeuronalFTR/training_results/FlamePinchOff/Training_with_half_snapshots/"
    data_folder = home + "/tubcloud/FTR/01_NeuronalFTR/training_results/FlamePinchOff/Training_with_allSnapshots/"
    # data_folder= 'D:/Arbeit/FrontTransportReduction//NeuralFTR/training_results_local/NewTrainings/'
    train_results_list = [
        # {"name": "FNL", "folder": data_folder + "FullDec/NoSmoothing/LearnFrontWidth/", "decoder": FTR_Dec},
        # {"name": "FN", "folder": data_folder + "FullDec/NoSmoothing/NoLearnFrontWidth/", "decoder": FTR_Dec},
        # {"name": "FSL", "folder": data_folder + "FullDec/Smoothing/LearnFrontWidth/", "decoder": FTR_Dec},
        #{"name": "NN1", "folder": data_folder + "FullDec/Smoothing/NoLearnFrontWidth/", "decoder": FTR_Dec},
        # {"name": "L1NL", "folder": data_folder + "1LayDec/NoSmoothing/LearnFrontWidth/", "decoder": FTR_Dec_1Lay},
        # {"name": "L1N", "folder": data_folder + "1LayDec/NoSmoothing/NoLearnFrontWidth/", "decoder": FTR_Dec_1Lay},
        # {"name": "L1SL", "folder": data_folder + "1LayDec/Smoothing/LearnFrontWidth/", "decoder": FTR_Dec_1Lay},
        {"name": "NN", "folder": data_folder + "1LayDec/Smoothing/NoLearnFrontWidth/", "decoder": FTR_Dec_1Lay}
    ]

    q = np.squeeze(np.moveaxis(data_set, 0, -1))
    AE_data_set = to_torch(data_set)

    for nt, train_dict in enumerate(train_results_list):
        n = nt + 1
        name, train_results_path, dec = train_dict.values()
        print("\n" + name + "\n")
        folder_list = [train_results_path + f for f in os.listdir(train_results_path) if '_Latents_' in f]
        folder_list.sort(key=natural_keys)
        if len(folder_list) == 0:
            print("WARNING! Folder not found or empty!")
            continue
        lam = "/LearnFrontWidth/" in train_results_path
        err = []
        dof = []
        for folder in folder_list:
            '''
            loops over all folders,
            to study dependence on the number of degrees of freedom (DOF)
            '''
            NN_folder = folder + '/net_weights/'
            NN_files = glob(NN_folder + "*.pt")
            NN_files.sort(key=natural_keys)
            best_NN_file = [NN_f for NN_f in NN_files if 'best_results.pt' in NN_f]
            NN_files = best_NN_file if best_NN_file else NN_files
            match = re.match(r"(\d+)_Latents.*", folder.split("/")[-1])
            a = int(match.group(1))
            print("DOFs:", a, " Reading:", "/".join(folder.split("/")[-4:]))
            frontwidth = '/LearnFrontWidth/' in folder
            error = 100  # maximal relative error possible
            for NN_file in NN_files:
                '''
                 loop over all training checkpoint files,
                 to find the one with best results
                '''
                print(NN_file.split("/")[-1])
                AE = FTR_AE(
                    decoder=dec(n_alpha=a, learn_frontwidth=frontwidth, spatial_shape=AE_data_set.shape[-2:]),
                    n_alphas=a, learn_periodic_alphas=False,
                    alpha_act=False,
                    spatial_shape=AE_data_set.shape[-2:]).to(DEVICE)
                AE.load_net_weights(NN_file)

                AE.eval()
                with torch.no_grad():
                    code, phi, qtilde = AE(AE_data_set, return_code=True, return_phi=True)
                AE.train()

            if a == dofs:
                phi = phi.cpu()
                qtilde_NN = qtilde.cpu()
                break

    qtilde_NN = np.squeeze(qtilde_NN.cpu()).numpy()
    qtilde_NN = np.moveaxis(qtilde_NN, 0, -1)

    Ngrid = data_set.shape[2:]
    Ntime = data_set.shape[0]
    qtilde_POD = truncate(q.reshape(-1, Ntime), dofs).reshape([*Ngrid, Ntime])

    phi_ftr = simple_FTR(q, front, max_iter=500, rank=dofs, print_step=1)
    qtilde_FTR = front(phi_ftr)

    imagepath = "imgs/flame_pinchoff/"
# %%
    figure, axes = plt.subplots(num = 2,nrows=4, ncols=1, sharey="all", sharex="all")#,
                                #figsize=(10, (len(train_results_list)+4)))
    nt_show = 66
    q = np.squeeze(np.moveaxis(data_set, 0, -1))

    axes[0].imshow(q[...,nt_show], vmin = 0, vmax = 1)
    axes[0].set_ylabel("data")
    axes[1].imshow(qtilde_POD[..., nt_show],  vmin = 0, vmax = 1)
    axes[1].set_ylabel("POD")
    axes[2].imshow(qtilde_FTR[..., nt_show],  vmin = 0, vmax = 1)
    axes[2].set_ylabel("FTR")
    im = axes[3].imshow(qtilde_NN[..., nt_show],  vmin = 0, vmax = 1)
    axes[3].set_ylabel("FTR-NN")

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', adjustable='box')
        gs = ax.get_gridspec()

    #figure.subplots_adjust(right=1.1)
    cbar_ax = figure.add_axes([0.75, 0.15, 0.05, 0.7])
    figure.colorbar(im, cax=cbar_ax)

