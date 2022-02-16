import globs
import numpy as np
from cases import my_cases
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
latexfont = {'size'   : 28}
matplotlib.rc('font',**latexfont)
matplotlib.rc('text',usetex=True)

from paths import init_dirs
from ROM.ROM import get_proj_err,  load_reduced_map, clean_and_split_data
from FOM.rhs_advection_reaction_diffusion import \
    rhs_advection_reaction_diffusion_2D_periodic as rhs
from plot_utils import save_fig
# %%
case = "pacman"
reac_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
idx_train_params = [0, 2, 4, 6, 9]
rank = 3
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

####################################################
# 2.) Postprocess clean data and split in train and
# vaildation set
####################################################
mapping = load_reduced_map(params.rom, data_dir)

# %%
nt = 805
Nt = len(params.time.t)
Mp = params.fom_size//10
phi = np.reshape(mapping.phi, [*params.geom.N,-1])
phi_vec = mapping.phi[:,nt]
sampleMeshIndices = np.argpartition(np.abs(phi_vec), Mp)[:Mp]

R = rhs(params,q_train[:,nt],params.time.t[nt%Nt])
Rf = np.reshape(R,params.geom.N)
rf_min = np.min(Rf)/4
rf_max = np.max(Rf)/4

Xgrid = params.geom.Xgrid
my_alpha = 0.05 # transparency
fig,ax = plt.subplots(1,2, num = 3,figsize=(12,4), sharey=True)

#ax[0,0].set_title(r"$\boldsymbol{F}(\boldsymbol{q},t,\mu)$")


h = ax[0].pcolormesh(Xgrid[0], Xgrid[1], Rf)
Xvec, Yvec = Xgrid[0].flatten(), Xgrid[1].flatten()
Xsample, Ysample = np.take(Xvec, sampleMeshIndices), np.take(Yvec, sampleMeshIndices)
ax[0].scatter(Xsample, Ysample, color='red', s = 0.1, alpha = my_alpha)
h.set_clim(rf_min, rf_max)
ax[0].axis("image")
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_xlim([min(Xvec), max(Xvec)])
ax[0].set_ylim([min(Yvec), max(Yvec)])
ax[0].annotate(r"$t=%.2f$"%params.time.t[nt%Nt], xy = (0.05,0.05))
#ax[0,1].set_title("sample points")




nt = 990

Mp = params.fom_size//10
phi = np.reshape(mapping.phi, [*params.geom.N,-1])
phi_vec = mapping.phi[:,nt]
sampleMeshIndices = np.argpartition(np.abs(phi_vec), Mp)[:Mp]

R = rhs(params,q_train[:,nt],params.time.t[nt%Nt])
Rf = np.reshape(R,params.geom.N)
rf_min = min(np.min(Rf),rf_min)
rf_max = max(np.max(Rf),rf_max)
Xgrid = params.geom.Xgrid

#plt.figure(10)
#fig,ax = plt.subplots(1,2, num = 5)
# h = ax[1,0].pcolormesh(Xgrid[0], Xgrid[1], Rf)
# h.set_clim(rf_min, rf_max)
# ax[1,0].set_xticks([])
# ax[1,0].set_yticks([])
# ax[1,0].annotate(r"$t=%.2f$"%params.time.t[nt%Nt], xy = (0.05,0.05))
# ax[1,0].axis("image")
#ax[1,0].set_title(r'$\bm{F}(\bm{q},t,\mu)$')


h = ax[1].pcolormesh(Xgrid[0], Xgrid[1], Rf)
Xvec, Yvec = Xgrid[0].flatten(), Xgrid[1].flatten()
Xsample, Ysample = np.take(Xvec, sampleMeshIndices), np.take(Yvec, sampleMeshIndices)
ax[1].scatter(Xsample, Ysample, color='red', s = 0.1, alpha = my_alpha)
h.set_clim(rf_min, rf_max)
ax[1].axis("image")
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_xlim([min(Xvec), max(Xvec)])
ax[1].set_ylim([min(Yvec), max(Yvec)])
ax[1].annotate(r"$t=%.2f$"%params.time.t[nt%Nt], xy = (0.05,0.05))
#ax[1,1].set_title("sample points")
cbar = fig.colorbar(h, ax=ax.ravel().tolist())
cbar.ax.set_yticks([-15, 0,15])
axins = zoomed_inset_axes(ax[1], 8, loc=1) # zoom = 3
h= axins.pcolormesh(Xgrid[0], Xgrid[1], Rf)
h.set_clim(rf_min, rf_max)
axins.scatter(Xsample, Ysample, color='red', s = 2, alpha =0.6)
# sub region of the original image
x1, x2, y1, y2 = 0.78, 0.83, 0.3, 0.35
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
plt.xticks(visible=False)
plt.yticks(visible=False)

# draw a bbox of the region of the inset axes in the parent axes and
# connecting lines between the bbox and the inset axes area
mark_inset(ax[1], axins, loc1=3, loc2=4, fc="none", ec="0")



save_fig(pic_dir+"/sample_points",fig)
