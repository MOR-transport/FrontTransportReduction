import numpy as np
import torch
import os
from NeuralFTR.moving_disk_data import make_grid, generate_data
from NeuralFTR.ftr_net import FTR_AE, FTR_Dec, FTR_Dec_1Lay
from NeuralFTR.trainer import Trainer
from NeuralFTR.utils import to_torch
import scipy.io

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

case = "topo_change"
preciscion = 'single'

dtype=torch.float32 if preciscion=='single' else torch.float64
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
                                        device=DEVICE, dtype=dtype)
    train_set = data_set[::2, ...]
    test_set = data_set[1::2, ...]
elif case=="topo_change":
    L = [4, 4]
    max_rank = 10
    N = [2 ** 8, 2 ** 8]
    Nt = 51
    R = 0
    X, Y = make_grid(*N, *L)
    data_set, phi_dat = generate_data(X, Y, 1, R, 1, to_torch=True, type="dw", device=DEVICE, dtype=dtype)
    train_set = data_set[::2, ...]
    test_set = data_set[1::2, ...]

base_bame = './training_results_local/'+case+'/'

skip_existing = True
decoder_variants = {'1LayDec': FTR_Dec_1Lay, 'FullDec': FTR_Dec, }
# decoder_variants = {'1LayDec': FTR_Dec_1Lay}
# decoder_variants = {'FullDec': FTR_Dec, }
smooth = 1e-7
alpha = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for a in alpha:
    for decname in decoder_variants.keys():
        dec = decoder_variants[decname]
        log_folder = base_bame + decname + '/'
        log_name = str(a) + '_Latents_'
        has_a_latents = 0
        if os.path.isdir(log_folder):
            has_a_latents = sum([log_name in dirname for dirname in os.listdir(log_folder)])
        if has_a_latents and skip_existing:
            continue
        print(log_folder + log_name)
        AE = FTR_AE(decoder=dec(n_alpha=a, spatial_shape=train_set.shape[-2:], dtype=dtype), n_alphas=a,
                    spatial_shape=train_set.shape[-2:], dtype=dtype)
        trainer = Trainer(AE, X, Y, train_set=train_set, test_set=test_set, lr=0.0025, lr_min=0.00002, smooth_phi=smooth,
                          log_folder=log_folder, dtype=dtype)
        trainer.training(trainsteps=2e4, test_every=5e2, save_every=5e4, batch_size=50, test_batch_size=None, log_base_name=log_name)
