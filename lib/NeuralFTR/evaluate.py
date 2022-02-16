import numpy as np
import torch
from NeuralFTR.moving_disk_data import make_grid, moving_disk
from NeuralFTR.ftr_net import FTR_AE, FTR_Dec, FTR_Dec2, FTR_Dec_1Lay, FTR_Enc, FTR_Enc2
from NeuralFTR.trainer import show_video
from NeuralFTR.utils import to_torch
import scipy.io

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

case = "moving_disk"

if case != "moving_disk":
    
    fname = 'D:\Arbeit\FrontTransportReduction\Data\FlamePinchOff\CH4_Normalize_smalldomain.mat'
    data = scipy.io.loadmat(fname)
    X = data['xgrid']
    Y = data['ygrid']
    train_set = to_torch(data['data'][50:100:2, ...])
    test_set = to_torch(data['data'][100:150:2, ...])
else:
    Nx = 129
    Ny = 129
    X, Y = make_grid(Nx, Ny, 10, 10)
    
    R = [2.2]
    lam = [0.1]
    n_samples = 100
    t = np.linspace(1 / n_samples, 1, n_samples)
    R, t, lam = np.meshgrid(R, t, lam)
    tRlam = np.concatenate([t.flatten()[:, None], R.flatten()[:, None], lam.flatten()[:, None]], 1)
    data_set = moving_disk(X, Y, tRlam[:, 0], tRlam[:, 1], tRlam[:, 2], to_torch=True, device=DEVICE)
    train_set = data_set[::2, ...]
    test_set = data_set[1::2, ...]

learn_periodic_alphas_dummy = case == 'moving_disk'

base_name = 'D:/Arbeit/FrontTransportReduction/NeuralFTR/training_results/FlamePinchOff/Training_with_allSnapshots/'
base_name = 'D:/Arbeit/FrontTransportReduction/NeuralFTR/training_results/'
weights = '/1LayDec/NoSmoothing/NoLearnFrontWidth/3_Latents_2020_11_01__17-30/net_weights/best_results.pt'
a = 3
dec = FTR_Dec

# weights = '1LayDec/Smoothing/NoLearnFrontWidth/10_Latents_2020_11_23__21-57/net_weights/best_results.pt'
# a = 10
# dec = FTR_Dec_1Lay

enc = FTR_Enc
AE = FTR_AE(encoder=enc(n_alphas=a, spatial_shape=train_set.shape[-2:]),
            decoder=dec(n_alpha=a, learn_frontwidth=False, spatial_shape=train_set.shape[-2:]),
            n_alphas=a,
            alpha_act=False,
            learn_periodic_alphas=learn_periodic_alphas_dummy,
            spatial_shape=train_set.shape[-2:]).to(DEVICE)
AE.load_net_weights(base_name + weights)

show_video(test_set, AE, reps=3, pause=0.5)