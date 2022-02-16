import globs
from hyper_FTR import hyper_FTR
from cases import my_cases
from plot_utils import autolabel
from tabulate import tabulate
from paths import init_dirs
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from ROM.ROM import get_proj_err,  load_reduced_map
from plot_utils import save_fig
from pathlib import Path
from ROM.ROM import POD_DEIM
from plot_utils import save_fig
# %%
case = "pacman"
reac_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
idx_train_params = [0, 2, 4, 6, 9]
#idx_train_params = [0, 8]
idx_test_params = list(set(np.arange(0,len(reac_list)))-set(idx_train_params))
dirs = {'data':'data', 'images': 'imgs'}
data_dir, pic_dir = init_dirs(dirs['data'], dirs['images'])
solve_ODE = True
construct_mapping = False
rank_list = np.arange(2,15)
hyp_red_frac = [0.1,0.2,0.5,1]
info_dict_list =[]
dict_lists = []
projection_error = []

for ir, rank in enumerate(rank_list):
        params_list = []
        for mu in reac_list:
            params = my_cases(case)
            params.reaction_const = mu
            params.rom.rom_size = rank
            params_list.append(params)

        fname = data_dir + '/' + case + '_fomP.mat'
        # q = loadmat(data_dir + '/' + case + '_fomP.mat')['fom_solution']
        # number_train_snapshots = loadmat(data_dir + '/' + case + '_fomP.mat')['number_train_snapshots'][0,0]
        with open(fname, 'rb') as f:
            q = np.load(f)
            number_train_snapshots = np.load(f)
        q_train = q[:, :number_train_snapshots]
        q_test = q[:, number_train_snapshots:]

        mapping = load_reduced_map(params.rom, data_dir)
        rel_err = get_proj_err(q_test, mapping)
        projection_error.append(rel_err)
        print("rank: %d, rel_proj_error: %f" % (rank, rel_err))

fname = data_dir + '/' + case + '_proj_err'
np.save(fname,projection_error)


# %% POD DEIM approximation
online_err = []
offline_err = []
proj_err = []
tcpu = []

for i,rank in enumerate(rank_list):
        params_list = []
        for mu in reac_list:
            params = my_cases(case)
            params.reaction_const = mu
            params.rom.rom_size = rank
            params_list.append(params)
        # parameters used for training. all others are used for testing
        train_params_list = [params_list[idx] for idx in idx_train_params]
        test_params_list = [params for params in params_list if params not in train_params_list]

        a, U, t_cpu = POD_DEIM(test_params_list, q_train, rank)
        tcpu.append(t_cpu)
        online_err.append( norm(q_test-U@a)/norm(q_test,'fro'))
        proj_err.append( norm(U@(U.T@q_test) - q_test, "fro") / norm(q_test, 'fro'))
        offline_err.append(  norm(U@(U.T@q_train) - q_train, "fro") / norm(q_train, 'fro'))
        print("rank: %d offline_err= %.2e, online_err= %.2e, proj_err= %.2e tcpu= %.2e" % (rank, offline_err[i],
                                                                                           online_err[i], proj_err[i], tcpu[i]))

pod_err_dict = {"offline_error": offline_err, "online_error": online_err, "proj_err": proj_err }
fsave = data_dir + '/' + case + '_pod_err_dict.npy'
np.save(fsave, pod_err_dict)