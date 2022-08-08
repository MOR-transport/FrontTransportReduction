import numpy as np
import torch
import os,sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from NeuralFTR.synthetic_data import make_grid, generate_data
from NeuralFTR.ftr_net import FTR_AE, FTR_Dec, FTR_Dec_1Lay
from NeuralFTR.trainer import Trainer
from NeuralFTR.utils import to_torch
import scipy.io

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_NN(train_set, test_set, train_dict, X, Y, dofs = np.arange(1,10), smooth = 1e-6, skip_existing = True,
             init_zero=True):

    backup_folder = train_dict["folder"]
    dec = train_dict["decoder"]

    for a in dofs:
            log_folder = backup_folder + '/'
            log_name = str(a) + '_dof_'
            has_a_latents = 0
            if os.path.isdir(log_folder):
                has_a_latents = sum([log_name in dirname for dirname in os.listdir(log_folder)])
            if has_a_latents and skip_existing:
                continue
            print(log_folder + log_name)

            Decoder = dec(dof=a, spatial_shape=train_set.shape[-2:], init_zero=init_zero)
            AE = FTR_AE(decoder=Decoder, dof=a, spatial_shape=train_set.shape[-2:])
            trainer = Trainer(AE, X, Y, train_set=train_set, test_set=test_set, lr=0.0025, lr_min=0.00002,
                              smooth_phi=smooth, log_folder=log_folder)
            trainer.training(trainsteps=2e4, test_every=5e2, save_every=5e4, log_base_name=log_name)
            #trainer.training(trainsteps=5e3, test_every=5e2, save_every=5e2, log_base_name=log_name)


if __name__ == '__main__':
    case = "topo_change"
    if case == "data":

        fname = '../data/FlamePinchOff/CH4_Normalize_smalldomain.mat'
        data = scipy.io.loadmat(fname)
        X = data['xgrid']
        Y = data['ygrid']
        train_set = to_torch(data['data'][50:100:2, ...])
        test_set = to_torch(data['data'][100:150:2, ...])

    elif case == "moving_disk":
        Nx = 129
        Ny = 129
        X, Y = make_grid(Nx, Ny, 10, 10)

        R = [2.2]
        lam = [0.1]
        n_samples = 100
        t = np.linspace(1/n_samples, 1, n_samples)
        R, t, lam = np.meshgrid(R, t, lam)
        tRlam = np.concatenate([t.flatten()[:, None], R.flatten()[:, None], lam.flatten()[:, None]], 1)
        data_set, phi_field = generate_data(X, Y, tRlam[:, 0], tRlam[:, 1], tRlam[:, 2], to_torch=True,
                                            type="moving_disk", device=DEVICE)
        train_set = data_set[::2, ...]
        test_set = data_set[1::2, ...]

    elif case == "topo_change":
        L = [4, 4]
        max_rank = 10
        N = [2 ** 8, 2 ** 8]
        Nt = 51
        R = 0
        X, Y = make_grid(*N, *L)
        data_set, phi_dat = generate_data(X, Y, 1, R, 1, to_torch=True, type="dw", device=DEVICE)
        train_set = data_set[::2, ...]
        test_set = data_set[1::2, ...]

    # Train FTR-NN
    train_dict = {"decoder": FTR_Dec_1Lay, "folder": "./training_results_local/"+case+"/FTR-NN"}
    train_NN(train_set, test_set, train_dict, X, Y, skip_existing= False, init_zero=True)

    # Train NN
    train_dict = {"decoder": FTR_Dec, "folder": "./training_results_local/" + case + "/NN"}
    train_NN(train_set, test_set, train_dict, X, Y, skip_existing= False)
