import globs
from hyper_FTR import hyper_FTR
import numpy as np
from numpy.linalg import norm
from cases import my_cases
from plot_utils import autolabel
import matplotlib.pyplot as plt
from tabulate import tabulate
from paths import init_dirs
from ROM.ROM import get_proj_err,  load_reduced_map
from pathlib import Path
from ROM.ROM import POD_DEIM
from plot_utils import save_fig
dirs = {'data':'../data', 'images': '../imgs'}
data_dir, pic_dir = init_dirs(dirs['data'], dirs['images'])

# %%
case = "reaction1D"
reac_list = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
#idx_train_params = [0, 2, 4, 6, 8]
idx_train_params = [0, 8]
idx_test_params = list(set(np.arange(0,len(reac_list)))-set(idx_train_params))

solve_ODE = True
construct_mapping = False
solve_ROM = True
skip_existing = True
rank_list = np.arange(2,10)
hyp_red_frac = [0.1,0.2,0.5,1]
info_dict_list =[]
dict_lists = []
for idxf , frac in enumerate(hyp_red_frac):
    info_dict_list = []
    print("----------------------------------------------------------")
    print("(%d) Sample mesh point fraction: %.2f"%(idxf,frac))
    print("----------------------------------------------------------")
    for rank in rank_list:
        params_list = []
        for mu in reac_list:
            params = my_cases(case)
            params.rom.fom.sampleMeshSize = min(int(frac * params.fom_size),params.fom_size-2) # 2 for the boundaries
            params.reaction_const = mu
            params.inicond = params.set_inicond(case=case)
            params.rom.rom_size = rank
            params_list.append(params)
        # parameters used for training. all others are used for testing
        train_params_list = [params_list[idx] for idx in idx_train_params]
        test_params_list = [params for params in params_list if params not in train_params_list]
        q_train, q_test, sol_dict, koopman_sol_dict, info_dict = hyper_FTR(train_params_list, test_params_list,
                                                                              solve_ODE=solve_ODE, construct_reduced_map=construct_mapping,
                                                                              solve_galerkin=solve_ROM, solve_koopman=False)

        solve_ODE = False # after first iteration we can stop solving the FOM
        info_dict_list.append(info_dict)
    dict_lists.append(info_dict_list)
    construct_mapping=False

# %% calc FTR projection error
data_dir = "../data/"
fname = data_dir + '/' + case + '_proj_err'
if Path(fname).is_file() and skip_existing:
    porjection_error = np.load(fname)
else:
    projection_error = []
    for ir, rank in enumerate(rank_list):
        params.rom.rom_size = rank
        mapping = load_reduced_map(params.rom, data_dir)
        rel_err = get_proj_err(q_test, mapping)
        projection_error.append(rel_err)
        print("rank: %d, rel_proj_error: %f" % (rank, rel_err))

    np.save(fname, projection_error)

# %% calc POD - DEIM offline and online errors
#rank_list = np.arange(2,100,1)
tcpu = []
online_err =[]
offline_err = []
proj_err = []
for i,rank in enumerate(rank_list):

    for params in test_params_list:
        params.rom.rom_size=rank

    a, U, t_cpu = POD_DEIM(test_params_list, q_train, rank)
    tcpu.append(t_cpu)
    online_err.append( norm(q_test-U@a)/norm(q_test,'fro'))
    proj_err.append( norm(U@U.T@q_test - q_test, "fro") / norm(q_test, 'fro'))
    offline_err.append(  norm(U@U.T@q_train - q_train, "fro") / norm(q_train, 'fro'))
    print("rank: %d offline_err= %.2e, online_err= %.2e, proj_err= %.2e tcpu= %.2e" % (rank, offline_err[i], online_err[i], proj_err[i], tcpu[i]))

POD_err_dict = {"online_err": online_err, "proj_err": proj_err, "offline_err": offline_err}
fname = data_dir + '/' + case + '_POD_err_dict.npy'
np.save(fname,POD_err_dict)

# %% plot rank vs error
info_dict_list = dict_lists[-1]
offline_error = [info_dict["rel_offline_error"] for info_dict in info_dict_list]
online_error = [info_dict["rel_online_error"] for info_dict in info_dict_list]

plt.figure(33)
plt.semilogy(rank_list, offline_error, '--*', label="FTR-offline")
plt.semilogy(rank_list, online_error, '--+', label="FTR-online")
plt.plot(rank_list, projection_error, '--o',label= "FTR-projection")
plt.legend()
plt.xlabel(r"degrees of freedom $r$")
plt.ylabel("relative error")
plt.tight_layout()

# %%  make table as well
fname = data_dir + '/' + case + '_POD_err_dict.npy'
POD_err_dict = np.load(fname, allow_pickle=True).flatten()[0]
pod_online = POD_err_dict["online_err"]
pod_proj = POD_err_dict["proj_err"]
table = np.stack((rank_list,np.asarray(offline_error),np.asarray(online_error),np.asarray(projection_error),np.asarray(pod_online),np.asarray(pod_proj)),axis=1)
print(tabulate(table,headers=["rank","FTR-offine error", "FTR-online error", "FTR-projection error", "POD-online error", "POD-projection error"], tablefmt='latex',floatfmt=(".0f", ".1e", ".1e", ".1e", ".1e", ".1e")))
# %% plot number of time steps vs mu for full and reduced order model
fom_rhs_calls = dict_lists[0][0]["num_rhs_calls_fom"]
rom_rhs_calls = info_dict_list[0]["num_rhs_calls_rom"]
mu_test_list = [reac_list[idx] for idx in idx_test_params]

fig, ax = plt.subplots(num=34)
x = np.arange(len(mu_test_list))  # the label locations
width = 0.35  # the width of the bars

rect_FOM=plt.bar(x-width/2,fom_rhs_calls, label="FOM")
rect_ROM=plt.bar(x+width/2,rom_rhs_calls, label="ROM")

autolabel(rect_FOM,ax, fmt= "%df")
autolabel(rect_ROM,ax, fmt= "%df")

ax.set_xlabel(r"$\mu$")
ax.set_ylabel("\# rhs calls")
ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels(mu_test_list)

ax.legend()

# %% speedup vs error
fom_time = dict_lists[0][0]["cpu_time"]["FOM"]

sample_frac_list =[]
fig, ax = plt.subplots(num=22)
markers=["o","*","x",'+']
for id,dicts in enumerate(dict_lists):
    min_err = dicts[0]["rel_online_error"]
    rank = rank_list[0]
    error_list = []
    ranks_list = []
    speedup_list = []
    for ir,r in enumerate(rank_list):
        err = dicts[ir]["rel_online_error"]
        speedup = dicts[ir]["cpu_time"]["ROM-Galerkin"]
        speedup_list.append(speedup)
        error_list.append(err)
        ranks_list.append(r)
        sample_frac_list.append(hyp_red_frac[id])
        if err<min_err:
            min_err = err
            rank = r
    for i in range(len(speedup_list)):
        valign ="bottom" if id <3 else "top"
        ax.annotate("(%d)" % (ranks_list[i]), (speedup_list[i], error_list[i]), fontsize=14,
                    horizontalalignment='center', va=valign)
    ax.scatter(speedup_list, error_list,marker=markers[id], label=r"$M_p/M=%.1f$"%hyp_red_frac[id])

ax.axvline(fom_time,linestyle='--',color="k")
ax.annotate("FOM",(fom_time,ax.get_ylim()[1] ),va="top",ha="right")
ax.set_yscale('log')
ax.set_xscale('log')


plt.legend(loc="upper left")
plt.xlabel("cpu-time [s]")
plt.ylabel("rel. error")
save_fig("../imgs/1D-ARD-speedup-error.tikz.tex", figure=fig)
