import numpy as np
import torch
from numpy.linalg import svd
import matplotlib.pyplot as plt
#from ALM_FTR_Galerkin_python.openfoam_io import read_data
from NeuralFTR.ftr_net import FTR_AE, FTR_Dec_1Lay, FTR_Dec
from NeuralFTR.utils import to_torch, natural_keys
from time import perf_counter
from ROM.FTR import truncate, convergence_plot, simple_FTR
from os.path import expanduser
import os
from glob import glob
import re
import scipy.io
from matplotlib import rc
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
from plot_utils import save_fig

#plt.style.use('dark_background')
latexfont = {'size'   : 24}
rc('font',**latexfont)
home = expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def give_NN_errors(data_set, train_results_list):


    Ngrid = data_set.shape[2:]
    Ntime = data_set.shape[0]
    q = np.squeeze(np.moveaxis(data_set.cpu().numpy(),0,-1))
    AE_data_set = to_torch(data_set)
    norm_q = np.linalg.norm(q.flatten())
    for nt, train_dict in enumerate(train_results_list):
        n = nt + 1
        name, train_results_path, dec = train_dict.values()
        print("\n" + name + "\n")
        if name == "phiNN":
            NN_file = train_results_path + '/net_weights/step_9999.pt'
            AE = dec().to('cpu')
            AE.load_net_weights(NN_file)

            AE.eval()
            with torch.no_grad():
                phi, qtilde = AE(AE_data_set, apply_f=True, return_phi=True)
            AE.train()
            err = []
        else:
            folder_list = [train_results_path + f for f in os.listdir(train_results_path) if '_dof_' in f]
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
                match = re.match(r"(\d+)_dof.*", folder.split("/")[-1])
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
                    AE = FTR_AE(decoder=dec(dof=a, spatial_shape=AE_data_set.shape[-2:]),
                                dof=a,
                                spatial_shape=AE_data_set.shape[-2:]).to(DEVICE)
                    AE.load_net_weights(NN_file)

                    AE.eval()
                    with torch.no_grad():
                        code, phi, qtilde = AE(AE_data_set, return_code=True, return_phi=True)
                    AE.train()

                    df = AE_data_set.reshape(Ntime, -1) - qtilde.reshape(Ntime, -1)
                    df = df.cpu()
                    this_error = np.linalg.norm(df, ord='fro') / norm_q
                    if error > this_error:
                        error = this_error
                err.append(error)
                dof.append(a)

    return err, dof


def give_NN_fields(data_set, train_results_list, dofs, periodic = False):


    Ngrid = data_set.shape[2:]
    Ntime = data_set.shape[0]
    q = np.squeeze(np.moveaxis(data_set.cpu().numpy(),0,-1))
    AE_data_set = to_torch(data_set)
    norm_q = np.linalg.norm(q.flatten())
    for nt, train_dict in enumerate(train_results_list):
        n = nt + 1
        name, train_results_path, dec = train_dict.values()
        print("\n" + name + "\n")
        if name == "phiNN":
            NN_file = train_results_path + '/net_weights/step_9999.pt'
            AE = dec().to('cpu')
            AE.load_net_weights(NN_file)

            AE.eval()
            with torch.no_grad():
                phi, qtilde = AE(AE_data_set, apply_f=True, return_phi=True)
            AE.train()
            err = []
        else:
            folder_list = [train_results_path + f for f in os.listdir(train_results_path) if '_dof_' in f]
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
                match = re.match(r"(\d+)_dof.*", folder.split("/")[-1])
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
                    AE = FTR_AE(decoder=dec(dof=a, spatial_shape=AE_data_set.shape[-2:]),
                                dof=a, spatial_shape=AE_data_set.shape[-2:]).to(DEVICE)
                    AE.load_net_weights(NN_file)

                    AE.eval()
                    with torch.no_grad():
                        code, phi, qtilde = AE(AE_data_set, return_code=True, return_phi=True)
                    AE.train()
                if a == dofs:
                    break

    return phi, qtilde

def compare_compression(data_set, train_results_list, phi_ftr, front, dofs=7, nt_show=66, max_rank=50, periodic = False, max_iter = 300):


    # marker styles
    mStyles = [ "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "+", "x", "X", "D", "d",
               "|", "_", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    Ngrid = data_set.shape[2:]
    Ntime = data_set.shape[0]
    q = np.squeeze(np.moveaxis(data_set,0,-1))
    AE_data_set = to_torch(data_set)
    front = lambda x: (1+np.tanh(-x))*0.5
    fig_err = plt.figure(num=145)
    #fig_err, errors_POD, errors_FTR = convergence_plot(q , phi_ftr, f=front, max_rank=max_rank,max_iter = max_iter)

    ax_err = fig_err.gca()
    fig_svd, ax_svd = plt.subplots()
    sigma_svd = svd(q.reshape(-1, Ntime), full_matrices=False, compute_uv=False)
    ax_svd.semilogy(sigma_svd / sigma_svd[0], '>', label=r"$\sigma(q)$")
    sigma_svd = svd(phi_ftr.reshape(-1, AE_data_set.shape[0]), full_matrices=False, compute_uv=False)
    ax_svd.semilogy(sigma_svd / sigma_svd[0], 'D', label=r"$\sigma_\mathrm{FTR}(\phi)$")
    figure, axes = plt.subplots(nrows=len(train_results_list)+2, ncols=3, sharey="all", sharex="all",
                                figsize=(3, len(train_results_list)))
    pcol_list = []
    norm_q = np.linalg.norm(q.flatten())
    for nt, train_dict in enumerate(train_results_list):
        n = nt + 1
        name, train_results_path, dec = train_dict.values()
        print("\n" + name + "\n")
        if name == "phiNN":
            NN_file = train_results_path + '/net_weights/step_9999.pt'
            AE = dec().to('cpu')
            AE.load_net_weights(NN_file)

            AE.eval()
            with torch.no_grad():
                phi, qtilde = AE(AE_data_set, apply_f=True, return_phi=True)
            AE.train()
            Unn, Snn, Vtnn = svd(phi.flatten(1).T, full_matrices=False, compute_uv=True)
            ax_svd.semilogy(Snn / Snn[0], label=r"$\sigma(\phi_\mathrm{NN})$", marker=mStyles[-1] )
            err = []
            for r in range(max_rank):
                phi_tilde = torch.tensor(Unn[:, :r] @ np.diag(Snn[:r]) @ Vtnn[:r, :]).T.reshape(AE_data_set.shape)
                qtilde = torch.sigmoid(phi_tilde)
                df = AE_data_set.flatten(1) - qtilde.flatten(1)
                err.append(np.linalg.norm(df, ord='fro') / norm_q)
                print("r=", r, "   err=", err[-1])
                if r == dofs:
                    # plot images for phi, h(phi),q-h(phi) of snapshot 10
                    axes[n, 0].pcolormesh(phi_tilde[nt_show, 0, ::]);
                    axes[n, 1].pcolormesh(qtilde[nt_show, 0, ::]);
                    pcol_list.append(axes[n, 2].pcolormesh(df[nt_show, :].reshape(Ngrid)))
                    axes[n, 0].set_ylabel(name, size='large')
            dof = np.arange(0,r+1)
        else:
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
                    AE = FTR_AE(decoder=dec(n_alpha=a, learn_frontwidth=frontwidth, spatial_shape=AE_data_set.shape[-2:]),
                                n_alphas=a, learn_periodic_alphas=periodic,
                                alpha_act=False,
                                spatial_shape=AE_data_set.shape[-2:]).to(DEVICE)
                    AE.load_net_weights(NN_file)

                    AE.eval()
                    with torch.no_grad():
                        code, phi, qtilde = AE(AE_data_set, return_code=True, return_phi=True)
                    AE.train()

                    df = AE_data_set.reshape(Ntime, -1) - qtilde.reshape(Ntime, -1)
                    df = df.cpu()
                    this_error = np.linalg.norm(df, ord='fro') / norm_q
                    if error > this_error:
                        error = this_error
                err.append(error)
                dof.append(a)

                if a == dofs:
                    phi = phi.cpu()
                    qtilde = qtilde.cpu()
                    # plot images for phi, h(phi),q-h(phi) of snapshot 10
                    axes[n, 0].pcolormesh(phi[nt_show, 0, ::]);
                    axes[n, 1].pcolormesh(qtilde[nt_show, 0, ::]);
                    pcol_list.append(axes[n, 2].pcolormesh(df[nt_show, :].reshape(Ngrid)))
                    axes[n, 0].set_ylabel(name, size='large')
                    sigma_svd = svd(phi.numpy().reshape(AE_data_set.shape[0], -1), full_matrices=False, compute_uv=False)
                    ax_svd.semilogy(sigma_svd / sigma_svd[0], marker=mStyles[n],
                                    label=r"$\sigma_\mathrm{" + name.replace("_", "\_") + "}(\phi_r)$")
        ax_err.plot(dof, err, linestyle="none", marker=mStyles[n],
                    label=r"$q-f_\mathrm{" + name.replace("_", "\_") + "}(\phi_r)$")

    # plot FTR results
    qtilde = truncate(q.reshape(-1, Ntime), dofs).reshape([*Ngrid, Ntime])
    #axes[-1, 0].pcolormesh(qtilde[..., nt_show])
    axes[0, 1].pcolormesh(qtilde[..., nt_show])
    axes[0, 0].pcolormesh(qtilde[..., nt_show])
    pcol_list.append(axes[0, 2].pcolormesh(q[..., nt_show].reshape(*Ngrid) - qtilde[..., nt_show]))
    axes[0, 0].set_ylabel("POD", size='large')

    phi_ftr = simple_FTR(q, front, max_iter=max_iter, rank=dofs,  print_step = 1)
    axes[-1, 0].pcolormesh(phi_ftr[...,nt_show])
    qtilde = front(phi_ftr[...,nt_show])
    axes[-1, 1].pcolormesh(qtilde)
    pcol_list.append(axes[-1, 2].pcolormesh(q[...,nt_show].reshape(qtilde.shape)-qtilde))
    axes[-1, 0].set_ylabel("FTR", size='large')
    figure.suptitle(r"$r="+str(dofs)+"$ DOFs")
    ax_err.legend(frameon=False)
    ax_err.set_xlabel(r'degrees of freedom $r$')
    ax_err.set_ylabel(r'relative error')
    ax_svd.legend(loc=1)
    ax_svd.set_xlabel(r'degrees of freedom $r$')
    ax_svd.set_ylabel(r'singular values $\sigma_r/\sigma_0$')

    # remove all ticks and set column labels
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', adjustable='box')
        gs = ax.get_gridspec()
        gs.update(wspace=0.0, hspace=0.0)
    for ax, col in zip(axes[0], [r"$\phi_r$", r"$f(\phi_r)$", r"$q-f(\phi_r)$"]):
        ax.set_title(col)

    cmin = min([pcol_list[n].get_clim()[0] for n in range(len(pcol_list))])
    cmax = max([pcol_list[n].get_clim()[1] for n in range(len(pcol_list))])
    for pcol in pcol_list:
        pcol.set_clim([cmin, cmax])
    #return errors_POD, errors_FTR, err


def koopman_prediction(a_coef,dt,t_predict, iterations=1000):
        from fourier_koopman import fourier
        dofs = np.size(a_coef,1)
        time_fourier = perf_counter()
        f = fourier(num_freqs=min(100,2*dofs))
        f.fit(a_coef.T, iterations = iterations)
        time_fourier = perf_counter() - time_fourier
        a_tilde_f = f.predict(timepoints=t_predict/dt).T
        print ( "Reduced Order Model (dofs=%.3d) (koopman-fourier) solved in %.3f s"% (dofs,time_fourier))
        return a_tilde_f

def compare_interpolation(data_set, train_results_list, phi_ftr, front, dofs=7, nt_show=133, max_rank=50):

    # marker styles
    mStyles = [ "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "+", "x", "X", "D", "d",
               "|", "_", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    Ngrid = data_set.shape[2:]
    Ntime = data_set.shape[0]
    q = np.squeeze(np.moveaxis(data_set,0,-1))
    data_train = to_torch(data_set[::2,...])
    data_test = to_torch(data_set[1::2, ...])



    def convergence_plot(q, phi, f, max_rank=None):
        import matplotlib.pyplot as plt
        # norm = lambda x: np.max(abs(x))
        Ngrid = q.shape[:-1]
        matrizise = lambda fun: fun.reshape(np.prod(Ngrid), -1)
        q = matrizise(q)
        phi = matrizise(phi)
        norm = np.linalg.norm
        q_train = q[:, ::2]
        q_test = q[:, 1::2]
        ntime = q_train.shape[-1]
        if max_rank is None:
            max_rank = ntime
        errors = np.empty((2, max_rank))
        qsvd = np.linalg.svd(q_train, full_matrices=False)
        phisvd = np.linalg.svd(phi, full_matrices=False)
        a_coef_q = np.diag(qsvd[1]) @ qsvd[2]
        a_coef_phi = np.diag(phisvd[1]) @ phisvd[2]
        t_train = np.arange(1,ntime+1)
        dt = t_train[1]-t_train[0]
        if ntime%2==0:
            t_test = t_train[:-1]  +0.5
        else:
            t_test = t_train[:-1]+0.5


        a_coef_predict_phi = koopman_prediction(a_coef_phi, dt, t_test)
        a_coef_predict_q = koopman_prediction(a_coef_q, dt, t_test)
        # plt.plot(t_train, a_coef_q[2, :], '-x',t_test, a_coef_predict_q[2, :],'k.')
        # plt.xlabel("time $t$"); plt.ylabel(" $a_{%d}(t)$"%2)
        # plt.legend(["input","prediction"], loc = 1)
        for r in range(3):
            h = plt.plot(t_train, a_coef_phi[r, :], '-x', t_test, a_coef_predict_phi[r, :], 'k.')
            plt.text(103, a_coef_phi[r, 100], r"$a_%d(t)$" % (r + 1), color=h[0].get_color())

        plt.xlabel("time $t$");
        plt.ylabel(r"$\mathbf{a}(t)$")
        plt.legend(["input", "prediction"], loc=2)
        plt.tight_layout()
        plt.savefig("predict-amplitudes", format='png',transparent=True,dpi=300)



        #plt.plot(t_train, a_coef_q[10, :], 'x', t_test, a_coef_predict_q[10, :])
        norm_q = norm(q_test,ord="fro")

        POD_predict = {"r" : [], "q" : [] }
        FTR_predict = {"r": [], "q": [], "phi":[]}
        for r in range(max_rank):

            q_trunc = qsvd[0][:, :r] @ a_coef_predict_q[:r,:]
            POD_predict["r"].append(r)
            POD_predict["q"].append(q_trunc.reshape(*Ngrid,-1))
            errors[0, r] = norm(q_trunc - q_test, ord='fro') / norm_q

            phi_trunc = phisvd[0][:, :r] @ a_coef_predict_phi[:r,:]
            q_trunc = f(phi_trunc)
            FTR_predict["r"].append( r )
            FTR_predict["phi"].append(phi_trunc.reshape(*Ngrid,-1))
            FTR_predict["q"].append( q_trunc.reshape(*Ngrid, -1))
            errors[1, r] = norm(q_trunc - q_test, ord='fro') / norm_q
            print("r: %4d, error q-q_r: %1.2e, error q-f(phi_r): %1.2e" % (r, errors[0, r], errors[1, r]))
        fig = plt.figure(428)
        plt.plot(errors[0], '<', label=r"$q-q_{r}$")
        # plt.plot(phisvd[1][:max_rank]/phisvd[1][0], 's', label=r"$\phi-\lfloor \phi\rfloor_{\,r}$")
        plt.plot(errors[1], 'D', label=r"$q-f_{\mathrm{FTR}}(\phi_r)$")
        # plt.xlim([0,max_rank])
        plt.yscale('log')
        plt.legend()
        plt.xlabel('truncation rank')
        plt.ylabel('error')
        plt.show()
        return fig,POD_predict,FTR_predict,t_train, t_test

    fig_err,POD_predict,FTR_predict,t_train, t_test = convergence_plot(q , phi_ftr, f=front, max_rank=max_rank)
    ax_err = fig_err.gca()
    figure, axes = plt.subplots(nrows=len(train_results_list)+2, ncols=3, sharey="all", sharex="all",
                                figsize=(3, len(train_results_list)))
    pcol_list = []
    norm_q = np.linalg.norm(data_test.cpu().flatten(1),ord="fro")
    NN_predict = {"phi": [], "q": [], "name": []}
    for n, train_dict in enumerate(train_results_list):
        name, train_results_path, dec = train_dict.values()
        print("\n" + name + "\n")
        if name == "phiNN":
            NN_file = train_results_path + '/net_weights/step_9999.pt'
            AE = dec().to('cpu')
            AE.load_net_weights(NN_file)

            AE.eval()
            with torch.no_grad():
                phi, qtilde = AE(data_test, apply_f=True, return_phi=True)
            AE.train()
            Unn, Snn, Vtnn = svd(phi.flatten(1).T, full_matrices=False, compute_uv=True)
            err = []
            for r in range(max_rank):
                phi_tilde = torch.tensor(Unn[:, :r] @ np.diag(Snn[:r]) @ Vtnn[:r, :]).T.reshape(data_test.shape)
                qtilde = torch.sigmoid(phi_tilde)
                df = data_test.flatten(1) - qtilde.flatten(1)
                err.append(np.linalg.norm(df, ord='fro') / norm_q)
                print("r=", r, "   err=", err[-1])
                if r == dofs:
                    # plot images for phi, h(phi),q-h(phi) of snapshot 10
                    axes[n, 0].pcolormesh(phi_tilde[nt_show, 0, ::])
                    axes[n, 1].pcolormesh(qtilde[nt_show, 0, ::])
                    pcol_list.append(axes[n, 2].pcolormesh(df[nt_show, :].reshape(Ngrid)))
                    axes[n, 0].set_ylabel(name, size='large')
            dof = np.arange(0,r+1)
        else:
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
                    AE = FTR_AE(decoder=dec(n_alpha=a, learn_frontwidth=frontwidth, spatial_shape=data_train.shape[-2:]),
                                n_alphas=a, learn_periodic_alphas=False,
                                alpha_act=False,
                                spatial_shape=data_train.shape[-2:]).to(DEVICE)
                    AE.load_net_weights(NN_file)

                    AE.eval()
                    with torch.no_grad():
                        a_train_NN, phi, qtilde = AE(data_train, return_code=True, return_phi=True)
                        a_train_NN = a_train_NN.cpu()
                        phi = phi.cpu()
                        qtilde = qtilde.cpu()
                    a_test_NN = koopman_prediction(np.asarray(a_train_NN.T,dtype="float64"), t_train[1] - t_train[0], t_test)
                    with torch.no_grad():
                        phi_test, qtilde_test = AE.decoder(to_torch(a_test_NN.T),return_phi=True)
                        phi_test = phi_test.cpu()
                        qtilde_test = qtilde_test.cpu()
                    AE.train()


                    df = data_test.cpu().flatten(1) - qtilde_test.flatten(1)
                    this_error = np.linalg.norm(df, ord='fro') / norm_q
                    if error > this_error:
                        error = this_error
                err.append(error)
                dof.append(a)

                if a == dofs:
                    # plot images for phi, h(phi),q-h(phi) of snapshot 10
                    NN_predict["name"].append(name)
                    NN_predict["phi"].append(phi_test)
                    NN_predict["q"].append(qtilde_test)
                    axes[n, 0].pcolormesh(phi_test[nt_show, 0, ::]);
                    axes[n, 1].pcolormesh(qtilde_test[nt_show, 0, ::]);
                    pcol_list.append(axes[n, 2].pcolormesh(df[nt_show, :].reshape(Ngrid)))
                    axes[n, 0].set_ylabel(name, size='large')
        ax_err.plot(dof, err, linestyle="none", marker=mStyles[n],
                    label=r"$q-h_\mathrm{" + name.replace("_", "\_") + "}(\phi_r)$")

    # plot FTR results
    q_test = q[...,1::2]
    phi_ftr = FTR_predict["phi"][dofs]
    axes[-2, 0].pcolormesh(phi_ftr[...,nt_show])
    qtilde = front(phi_ftr[...,nt_show])
    axes[-2, 1].pcolormesh(qtilde)
    pcol_list.append(axes[-2, 2].pcolormesh(q_test[...,nt_show].reshape(qtilde.shape)-qtilde))
    axes[-2, 0].set_ylabel("FTR", size='large')
    figure.suptitle(str(dofs)+" DOFs")

    qtilde = POD_predict["q"][dofs][...,nt_show]
    axes[-1, 0].pcolormesh(qtilde)
    axes[-1, 1].pcolormesh(qtilde)
    pcol_list.append(axes[-1, 2].pcolormesh(q_test[...,nt_show].reshape(qtilde.shape)-qtilde))
    #axes[-1, 0].set_title("Data", size='large')
    axes[-1, 0].set_ylabel("POD", size='large')
    #axes[-1, 2].set_title("Data-POD", size='large')
    figure.suptitle(str(dofs)+" DOFs")

    ax_err.legend(frameon=False)
    ax_err.set_xlabel(r'degrees of freedom $r$')
    ax_err.set_ylabel(r'relative error')
    # remove all ticks and set column labels
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', adjustable='box')
        gs = ax.get_gridspec()
        gs.update(wspace=0.0, hspace=0.0)
    for ax, col in zip(axes[0], [r"$\phi$", r"$f(\phi)$", r"$q-f(\phi)$"]):
        ax.set_title(col)

    cmin = min([pcol_list[n].get_clim()[0] for n in range(len(pcol_list))])
    cmax = max([pcol_list[n].get_clim()[1] for n in range(len(pcol_list))])
    for pcol in pcol_list:
        pcol.set_clim([cmin, cmax])

    return POD_predict, FTR_predict, NN_predict

if __name__ == "__main__":
    plt.close("all")
    dofs = 15 # degrees of freedom used by the rom
    ####################################################################
    # load the data
    ####################################################################
    qty_name = "CH4"
    folder = home + "/tubcloud/FTR/04_FlamePinchOff/"  # Name of the folder
    cleaning_method = "Normalize"
    file = qty_name + "_" + cleaning_method + "_smalldomain.mat"
    q, X, Y = read_data(folder + file)
    ####################################################################
    # load ftr field
    ####################################################################
    #fname = home + "/tubcloud/FTR/04_FlamePinchOff/FTR_results.mat"  # Name of the folder
    fname = home + "/tubcloud/FTR/04_FlamePinchOff/FTR_results_20210302.mat"  # Name of the folder
    data = scipy.io.loadmat(fname)
    delta = data["delta"]
    front = lambda x: (np.tanh(x * delta) + 1) * 0.5
    # delta = 1 / (0.05)
    # front = lambda x: 1 / (1 + np.exp(-(x * delta)))

    phi = data["phi"]
    phi_ftr = phi

    ###################################################################
    # Plots
    ###################################################################

    #data_folder = home + "/tubcloud/FTR/01_NeuronalFTR/training_results/FlamePinchOff/Training_with_half_snapshots/"
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

    data_set = q[50::,...]
    #errPOD, errFTR, errNN  = compare_compression(data_set[::2,...], train_results_list, phi_ftr, front= front, dofs=dofs, nt_show=66, max_rank=20)

    POD_predict, FTR_predict, NN_predict = compare_interpolation(data_set, train_results_list, phi_ftr, front= front, dofs=dofs, nt_show=83, max_rank=50)

    # %%
    nt_show = 133//2
    imagepath = "imgs/flame_pinchoff/"
    figure, axes = plt.subplots(nrows=len(train_results_list)+2, ncols=3, sharey="all", sharex="all",
                                figsize=(10, (len(train_results_list)+4)))
    q = np.squeeze(np.moveaxis(data_set, 0, -1))
    q_test = q[..., 1::2]
    for nt_show in range(q_test.shape[-1]):
        n = 0
        pcol_list = []
        for name,phi_test,qtilde_test in zip(NN_predict["name"],NN_predict["phi"],NN_predict["q"]):
                    axes[n, 0].pcolormesh(phi_test[nt_show, 0, ::])
                    axes[n, 1].pcolormesh(qtilde_test[nt_show, 0, ::])
                    pcol_list.append(axes[n, 2].pcolormesh(q_test[...,nt_show]-qtilde_test[nt_show, 0, ::].detach().numpy()))
                    axes[n, 0].set_ylabel(name, size='large')
                    n = n +1

        phi_ftr = FTR_predict["phi"][dofs]
        axes[-2, 0].pcolormesh(phi_ftr[...,nt_show])
        qtilde = front(phi_ftr[...,nt_show])
        axes[-2, 1].pcolormesh(qtilde)
        pcol_list.append(axes[-2, 2].pcolormesh(q_test[...,nt_show].reshape(qtilde.shape)-qtilde))
        axes[-2, 0].set_ylabel("FTR", size='large')
        figure.suptitle(str(dofs)+" DOFs")

        qtilde = POD_predict["q"][dofs][...,nt_show]
        axes[-1, 0].pcolormesh(qtilde)
        axes[-1, 1].pcolormesh(qtilde)
        pcol_list.append(axes[-1, 2].pcolormesh(q_test[...,nt_show].reshape(qtilde.shape)-qtilde))
        #axes[-1, 0].set_title("Data", size='large')
        axes[-1, 0].set_ylabel("POD", size='large')
        #axes[-1, 2].set_title("Data-POD", size='large')
        figure.suptitle(str(dofs)+" DOFs")

        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal', adjustable='box')
            gs = ax.get_gridspec()
            gs.update(wspace=0.0, hspace=0.0)
        for ax, col in zip(axes[0], [r"$\phi$", r"$f(\phi)$", r"$q-f(\phi)$"]):
            ax.set_title(col)

        cmin = min([pcol_list[n].get_clim()[0] for n in range(len(pcol_list))])
        cmax = max([pcol_list[n].get_clim()[1] for n in range(len(pcol_list))])
        for pcol in pcol_list:
            pcol.set_clim([cmin, cmax])
        plt.show()
        figure.savefig(imagepath+"fp_%3.3d.png"%(nt_show), format='png',transparent=True,dpi=300)


    # %%
    qtilde_POD = POD_predict["q"][dofs]
    qtilde_FTR = FTR_predict["q"][dofs]
    qtilde_NN = np.squeeze(NN_predict["q"][0])
    figure, axes = plt.subplots(num = 2,nrows=4, ncols=1, sharey="all", sharex="all")#,
                                #figsize=(10, (len(train_results_list)+4)))

    minval = -0.2
    maxval = 1.1
    # colormap:
    viridis = cm.get_cmap('viridis', 256)
    newcolors = viridis(np.linspace(0, 1, 256))
    Ir = 51
    Ib = 25
    reds = np.ones([Ir, 4])
    blacks = np.ones([Ib, 4])
    for i in range(Ir):
        reds[i, :] = [1 - i / Ir, 0, 0, 1]
    for i in range(Ib):
        blacks[i, :] = [i / Ib, i / Ib, i / Ib, 1]
    newcolors = np.concatenate([reds, newcolors, blacks])
    newcmp = ListedColormap(newcolors)

    nt_show = 66
    q = np.squeeze(np.moveaxis(data_set, 0, -1))
    h = [0]*len(axes)
    h[0]=axes[0].imshow(q_test[...,nt_show], cmap = newcmp ,vmin =minval, vmax = maxval)
    axes[0].set_ylabel("data")
    h[1]=axes[1].imshow(qtilde_POD[..., nt_show], cmap = newcmp)
    axes[1].set_ylabel("POD")
    h[2]=axes[2].imshow(qtilde_FTR[..., nt_show], cmap = newcmp)
    axes[2].set_ylabel("FTR")
    h[3] = axes[3].imshow(qtilde_NN[nt_show,...], cmap = newcmp)
    axes[3].set_ylabel("FTR-NN")



    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', adjustable='box')
        gs = ax.get_gridspec()
    cmin = min([h[n].get_clim()[0] for n in range(len(h))])
    cmax = max([h[n].get_clim()[1] for n in range(len(h))])
    for im in h:
        im.set_clim([cmin, cmax])
    #figure.subplots_adjust(right=1.1)
    cbar_ax = figure.add_axes([0.75, 0.15, 0.05, 0.7])
    figure.colorbar(h[3], cax=cbar_ax)
    save_fig("./FTR-online-bunsen.tikz", figure=figure)
