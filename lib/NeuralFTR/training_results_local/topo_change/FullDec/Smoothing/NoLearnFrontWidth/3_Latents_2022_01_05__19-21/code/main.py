import numpy as np
import torch
import os
from NeuralFTR.moving_disk_data import make_grid, generate_data
from NeuralFTR.ftr_net import FTR_AE, FTR_Enc, FTR_Dec, FTR_Dec2, FTR_Dec_1Lay, Phi, Phi2
from NeuralFTR.trainer import Trainer
from NeuralFTR.utils import to_torch
import scipy.io

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

case = "topo_change"

if case == "data":
    
    fname = 'D:\Arbeit\FrontTransportReduction\Data\FlamePinchOff\CH4_Normalize_smalldomain.mat'
    data = scipy.io.loadmat(fname)
    X = data['xgrid']
    Y = data['ygrid']
    train_set = to_torch(data['data'][50:100:2, ...])
    test_set = to_torch(data['data'][100:150:2, ...])
elif case=="moving_disk":
    Nx = 129
    Ny = 129
    X, Y = make_grid(Nx, Ny, 10, 10)
    
    R = [2.2]
    lam = [0.1]
    n_samples = 100
    t = np.linspace(1/n_samples, 1, n_samples)
    R, t, lam = np.meshgrid(R, t, lam)
    tRlam = np.concatenate([t.flatten()[:, None], R.flatten()[:, None], lam.flatten()[:, None]], 1)
    data_set, phi_field = generate_data(X, Y, tRlam[:, 0], tRlam[:, 1], tRlam[:, 2], to_torch=True, type="moving_disc",
                                        device=DEVICE)
    train_set = data_set[::2, ...]
    test_set = data_set[1::2, ...]
elif case=="topo_change":
    L = [4, 4]
    max_rank = 10
    N = [2 ** 8, 2 ** 8]
    Nt = 51
    R = 0
    X, Y = make_grid(*N, *L)
    data_set, phi_dat = generate_data(X, Y, 1, R, 1, to_torch=True, type="dw", device=DEVICE)
    train_set = data_set[::2, ...]
    test_set = data_set[1::2, ...]

base_bame = './training_results_local/'+case+'/'
# log_folder = base_bame + 'Phi/'
# log_name = ''
# net = Phi()
# trainer = Trainer(net, X, Y, train_set=train_set, test_set=test_set, lr=0.0025, smooth_phi=10, sparse_reg=0, sparsity_measure='nuc',
#                   log_folder=log_folder)
#
# trainer.training(trainsteps=1e4, test_every=5e2, save_every=1e3, batch_size=50, test_batch_size=None, resume=True, log_base_name=log_name)

learn_periodic_alphas_dummy = case == 'moving_disk'

skip_existing = True
decoder_variants = {'FullDec': FTR_Dec}# '1LayDec': FTR_Dec_1Lay}
learn_lambda = [False]
smooth = [0.0000001]
alpha = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 30]
for a in alpha:
    for decname in decoder_variants.keys():
        for smoo in smooth:
            for lam in learn_lambda:
                dec = decoder_variants[decname]
                smooth_name = 'NoSmoothing/' if smoo == 0 else 'Smoothing/'
                learn_lam_name = 'NoLearnFrontWidth/' if not lam else 'LearnFrontWidth/'
                log_folder = base_bame + decname + '/' + smooth_name + learn_lam_name
                log_name = str(a) + '_Latents_'
                has_a_latents = 0
                if os.path.isdir(log_folder):
                    has_a_latents = sum([log_name in dirname for dirname in os.listdir(log_folder)])
                if ((a == 1 and lam) or has_a_latents) and skip_existing:
                    continue
                print(log_folder + log_name)
                AE = FTR_AE(decoder=dec(n_alpha=a, learn_frontwidth=lam, spatial_shape=train_set.shape[-2:]), n_alphas=a, alpha_act=False,
                            spatial_shape=train_set.shape[-2:], learn_periodic_alphas=learn_periodic_alphas_dummy)
                trainer = Trainer(AE, X, Y, train_set=train_set, test_set=test_set, lr=0.0025, smooth_phi=smoo, sparse_reg=0,
                                  log_folder=log_folder)
                trainer.training(trainsteps=1.5e4, test_every=5e2, save_every=5e4, batch_size=50, test_batch_size=None, resume=True,
                                 log_base_name=log_name)
