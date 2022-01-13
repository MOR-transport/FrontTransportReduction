import pathlib, sys

file_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(file_path) + "/../ALM_FTR_Galerkin_python")
sys.path.append(str(file_path) + "/../NeuralFTR")

from numpy.linalg import norm
from lib.FOM import *
from lib.FTR import *
from lib.plot_utils import *
from NeuralFTR import *
from compare_compression import *

pic_dir="../ALM_FTR_Galerkin_python/imgs/"

L = [4, 4]
max_rank = 15
N = [2**8, 2**8]
Nt = 51
R = 0
X, Y = make_grid(*N, *L)
q, phi_dat = generate_data(X, Y, 1, R, 1, to_torch=False, type="dw", device='cpu')
# %% neurla networks
data_folder= home+ "/tubcloud/FTR/01_NeuronalFTR/training_results/topo_change/"
train_results_list=[ {"name": "FN" , "folder": data_folder+"FullDec/","decoder": FTR_Dec}]
errNN, rank_list = give_NN_errors(q[1::2], train_results_list, periodic=False)

train_results_list=[ {"name": "FN" , "folder": data_folder+"1LayDec/","decoder": FTR_Dec}]
errFTR_NN, rank_list = give_NN_errors(q[1::2], train_results_list, periodic=False)


# %%
matrizise = lambda fun: fun.reshape(-1, Nt)
q_train = np.squeeze(np.moveaxis(q,0,-1))
front = lambda x: (np.tanh(x)+1)*0.5#np.squeeze(f(x,0.078))

qmat = matrizise(q_train)
qsvd = np.linalg.svd(qmat, full_matrices=False)
phi_dat_mat = matrizise(np.squeeze(np.moveaxis(phi_dat,0,-1)))
phi_dat_svd = np.linalg.svd(phi_dat_mat, full_matrices=False)
norm_q = norm(qmat,ord='fro')
errors = np.empty((2, len(rank_list)))
max_iter = np.ones([max_rank])*2000
max_iter[0] = 5
max_iter[1] = 250
for k,r in enumerate(rank_list):

    q_trunc = qsvd[0][:, :r] @ np.diag(qsvd[1][:r]) @ qsvd[2][:r, :]
    errors[0, k] = norm(q_trunc - qmat, ord='fro') / norm_q

    phi = simple_FTR(q_train, f=front, rank=r,
            tol=1e-6, max_iter=max_iter[r],
           nt_show=20, plot_step=30000, print_step=2)
    q_trunc = matrizise(front(phi))
    err = norm(q_trunc - qmat, ord='fro') / norm_q
    print("err:", err)
    errors[1, k] = err



# %% Plot everything
#savemat("../ALM_FTR_Galerkin_python/data/phi_topo.mat",{"phi":phi})
plt.semilogy(rank_list,errors[0,:],'+')
plt.semilogy(rank_list,errors[1,:],'*')
plt.semilogy(rank_list,errFTR_NN,'o')
plt.xlim([0,max_rank+0.5])
plt.xticks(np.arange(0,max_rank,2))
plt.xlabel(r"degrees of freedom $r$")
plt.ylabel(r"relativ error")
plt.tight_layout()
plt.legend(["POD","FTR"])
save_fig(pic_dir + "/topo_change_err.png")
# for i in range(0, np.size(q, 0), 2):
#     plt.pcolormesh(np.squeeze(q[i, ...]))
#     plt.draw()
#     plt.pause(0.01)
# %%
levelset_surface_plot(q[40,0,...],phi_dat[40,...], [X,Y], figure_number=2)

