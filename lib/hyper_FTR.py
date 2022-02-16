# =============================================
#    Advection and Burgers Equation in 1D/2D
# ---------------------------------------------
#    My testbed for nonlinear MOR methods
#    FTR and NN
# =============================================

import numpy as np
from numpy import reshape
from FOM.FOM import *
from cases import my_cases
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from time import perf_counter
from scipy.io import loadmat, savemat
from numpy.linalg import norm
from ROM.ROM import generate_reduced_map, solve_ROM, load_reduced_map,clean_data, set_up_galerkin
from plot_utils import *
from paths import init_dirs
from pathlib import Path



########################################################################################################################
########################################################################################################################
########################################################################################################################

def hyper_FTR(train_params_list, test_params_list, dirs={'data': 'data', 'images': 'imgs'}, solve_ODE=True, construct_reduced_map=True,
          solve_galerkin=True, solve_koopman=True, visualize=True):
    """
    REDUCED ORDER MODELLING PROCEDURE
    - This routine computes the Full Order Model Solution (FOM)
    - Splits the data into odd/even snapshots (in the time and parameter index) for train and test data
    - Creates a reduced map on the train data
    - With the reduced map a Reduced Order Model (ROM) is created using galerkin projections

    Remark:
    - This procedure only "interpolates" between different parameters! For interpolation for different parameters look at ROMsy

    """
    info_dict = {}
    params_list = train_params_list + test_params_list
    params = params_list[0]
    case = params.case
    # make a first simple check
    assert np.all([p.case == case for p in params_list]), "ROMsy cant do magic! Cases need to be equal."

    number_of_params = len(params_list)
    ntime = len(params.time.t)
    params.rom.time.test_time_points= params.time.t
    params.rom.time.train_time_points = params.time.t
    data_dir, pic_dir = init_dirs(dirs['data'], dirs['images'])
    cpu_time_dict = {"FOM": -10.0, "ROM-Galerkin": -10.0, "ROM-Koopman": -10.0}
    ######################################################
    # 1.) solve ODE spectral or with Finite Diffs
    ######################################################
    if solve_ODE:
        q_train_params_list = []
        for k,params in enumerate(train_params_list):
            print("\nTrain Parameter Set",k)
            params, qfield = solve_FOM(params)
            q = reshape(qfield, [params.fom_size, -1])
            q_train_params_list.append(q)

        time_odeint = perf_counter()
        q_test_params_list = []
        for k,params in enumerate(test_params_list):
            print("\nTest Parameter Set",k)
            params, qfield = solve_FOM(params)
            q = reshape(qfield, [params.fom_size, -1])
            q_test_params_list.append(q)
        cpu_time_dict["FOM"] = (perf_counter() - time_odeint)
        X = params.geom.Xgrid
        q_train = np.concatenate(q_train_params_list,axis=1)
        q_test = np.concatenate(q_test_params_list,axis=1)
        number_train_snapshots = np.size(q_train,1)
        q = np.concatenate([q_train,q_test],axis=1)
        qfield = np.reshape(q,[*params.geom.N,-1])
        fname = data_dir + '/' + case + '_fomP.mat'
        # savemat(fname, {'fom_solution': q,'number_train_snapshots': number_train_snapshots})
        with open(fname, 'wb') as f:
            np.save(f, q)
            np.save(f, number_train_snapshots)

        if visualize and params.dim == 2:
            pass
            #show_animation(qfield, Xgrid=X, frequency=8, figure_number=34)
        else:
            plt.figure(34)
            plt.pcolormesh(qfield)
            plt.show()
            plt.pause(0.5)
    else:
        fname = data_dir + '/' + case + '_fomP.mat'
        #q = loadmat(data_dir + '/' + case + '_fomP.mat')['fom_solution']
        #number_train_snapshots = loadmat(data_dir + '/' + case + '_fomP.mat')['number_train_snapshots'][0,0]
        with open(fname, 'rb') as f:
            q = np.load(f)
            number_train_snapshots = np.load(f)
        q_train = q[:,:number_train_snapshots]
        q_test = q[:, number_train_snapshots:]


    ####################################################
    # 2.) Postprocess clean data
    ####################################################
    q_test = clean_data(params.rom.cleaning.method, q_test, min_val=params.rom.cleaning.min_value_cut,
                             max_val=params.rom.cleaning.max_value_cut).T
    q_train = clean_data(params.rom.cleaning.method, q_train, min_val=params.rom.cleaning.min_value_cut,
                         max_val=params.rom.cleaning.max_value_cut).T
    # ####################################################
    # 3.) Postprocess learn neuronal net or
    # compute FTR (Front Transport Reduction)
    ####################################################
    fname = data_dir + '/' + params.rom.case + '_FTR_mapping-%dDofs.mat' % params.rom.rom_size
    if construct_reduced_map or not Path(fname).is_file():
        mapping = generate_reduced_map(params.rom, q_train, data_dir)
    else:
        mapping = load_reduced_map(params.rom, data_dir)
    # offline error:
    info_dict["rel_offline_error"] = norm(q_train-mapping.applyMapping(mapping.acoef))/norm(q_train)

    ##################################################
    # 4.) simulate trajectory using mainfold galerkin [LeeCarlberg2019]:
    ##################################################
    if params.rom.online_prediction_method == "POD-DEIM":
        for k, params in enumerate(test_params_list):
            params.rom.DEIM.rhs = give_POD_DEIM_rhs(params,q_train)
    if solve_galerkin:
        q_tilde_list = []
        rom_state_list = []

        set_up_galerkin(train_params_list, test_params_list, mapping)
        time_galerkin = perf_counter()
        for k, params in enumerate(test_params_list):
            print("\nTest Parameter Set", k)
            q_tilde, romState_hist = solve_ROM(params, mapping)
            q_tilde_list.append(q_tilde)
            rom_state_list.append(romState_hist)
        cpu_time_dict["ROM-Galerkin"] = perf_counter() - time_galerkin
        q_tilde =np.concatenate(q_tilde_list , axis=1)
        romState_hist =np.concatenate(rom_state_list , axis=1)
        rom_sol_dict = {'fomState_solution': np.reshape(q_tilde,[*params.geom.N,-1]), 'romState_solution': romState_hist}
        filename = data_dir + '/' + case + '_' + params.rom.mapping_type + '_%ddofs_rom_lspg.mat' % (
            params.rom.rom_size)
        savemat(filename, rom_sol_dict)
    else:
        filename = data_dir + '/' + case + '_' + params.rom.mapping_type + '_%ddofs_rom_lspg.mat' % (
            params.rom.rom_size)
        rom_sol_dict = loadmat(filename)
        q_tilde = rom_sol_dict["fomState_solution"]

    ##################################################
    # 4.) alternative koopman-fourier
    ##################################################
    if solve_koopman:
        params.rom.online_prediction_method = "fourier-koopman"
        time_fourier = perf_counter()
        q_tilde_f, romState_hist = solve_ROM(params.rom, mapping)
        cpu_time_dict["ROM-Koopman"] = perf_counter() - time_fourier
        q_tilde_f = np.reshape(q_tilde_f, [*params.geom.N, -1])
        koopman_sol_dict = {'fomState': q_tilde_f, 'romState': romState_hist}
        filename = data_dir + '/' + case + '_' + params.rom.mapping_type + '_%ddofs_rom_fourier-koopman.mat' % (
            params.rom.rom_size)
        savemat(filename, koopman_sol_dict)
    else:
        filename = data_dir + '/' + case + '_' + params.rom.mapping_type + '_%ddofs_rom_fourier-koopman.mat' % (
            params.rom.rom_size)
        #koopman_sol_dict = loadmat(filename)
        koopman_sol_dict =[]# loadmat(filename)
    print("\n------------------------------")
    print("cpu-time: \nFOM %.3f s \nGalerkin %.3f \nKoopman %.3f" % (
    cpu_time_dict["FOM"], cpu_time_dict["ROM-Galerkin"], cpu_time_dict["ROM-Koopman"]))
    print("------------------------------\n")
    info_dict["rom_size"]=params.rom.rom_size
    info_dict["cpu_time"] = cpu_time_dict
    if np.size(q_tilde,1) == np.size(q_tilde,1):
        info_dict["rel_online_error"] = norm(q_tilde - q_test)/norm(q_test)
    else:
        info_dict["rel_online_error"] = np.NaN
    #info_dict["rel_proj_error"] = get_proj_err(q_test , mapping)
    if solve_ODE == True:
        info_dict["num_rhs_calls_fom"] = [params.info_dict["num_rhs_calls"] for params in test_params_list]
    if solve_galerkin == True:
        info_dict["num_rhs_calls_rom"] = [params.rom.info_dict["num_rhs_calls"] for params in test_params_list]
    print("rank = %d"%params.rom.rom_size)
    print("offline error: %.5f " % (info_dict["rel_offline_error"]))
    print("online error: %.5f"%info_dict["rel_online_error"])
    #print("projection error: %.5f"%info_dict["rel_proj_error"])
    return q_train, q_test, rom_sol_dict, koopman_sol_dict, info_dict



#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################

if __name__ == '__main__':
    #case = "bunsen_flame"
    case = "pacman"
    reac_list = [10,20,30,40,50,60,70, 80, 90, 100]
    idx_train_params = [0, 2, 4, 6, 9]
    #case = "reaction1D"
    #reac_list = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    idx_train_params = [0, 2, 4, 6, 8]
    params_list = []
    for r in reac_list:
        params = my_cases(case)
        params.reaction_const = r
        params_list.append(params)
     # parameters used for training. all others are used for testing
    train_params_list = [params_list[idx] for idx in idx_train_params]
    test_params_list = [params for params in params_list if params not in train_params_list]
    q_train, q_test, sol_dict, koopman_sol_dict,  cpu_time_dict = hyper_FTR(train_params_list, test_params_list,
                                                                                    solve_ODE=False, construct_reduced_map=False,
                                                                                    solve_galerkin = True, solve_koopman = False)

    q_tilde = sol_dict["fomState_solution"]
    print("err: ", np.linalg.norm(q_test - np.reshape(q_tilde,np.shape(q_test))) / np.linalg.norm(q_test))

    ###################################################
    #       visualize
    ###################################################
    mapping = train_params_list[0].rom.lspg.mapping
    # %%
    from lib.ECSW import *
    fom_states = q_train
    rom_states = mapping.acoef
    mu_train = [reac_list[idx] for idx in idx_train_params]
    G, b = build_ECSW_system(mapping, rom_states, fom_states, train_params_list)
    network, test_set = solve_NN_EECSW(G, b, rom_states, mapping.U, mu_train, idx_train=[0,2,3,4], maxiter=10000,
                                       batch_size=400,  it_start = 3)
# %%
    plt.figure(12)
    qdat=q_train[:,100:200]
    it_start = 3
    with to.set_grad_enabled(False):
        plt.pcolormesh(qdat.T)
        rel_err = []
        for i in range(it_start, np.size(qdat, 1) - it_start):
            (G, b, mu, x) = map(lambda x: x.to(device), test_set[i])
            alpha = network(x.reshape([1, -1]), mu.reshape([1, -1]))
            # alpha = to.pinverse(G) @ b
            idx = np.where(alpha.detach().cpu().numpy() > 1e-5)
            #alpha = to.zeros_like(alpha)
            #alpha[idx]= 1
            #alpha[0][0] = 0
            err = b - (G @ alpha.T).T
            norm = to.sum(b ** 2).cpu().detach().numpy()
            rel_err.append(to.sum(err ** 2).cpu().detach().numpy() / norm)
            plt.plot(idx, np.ones_like(idx) * i, 'b*')

        plt.figure(33)
        plt.plot(rel_err)
        plt.xlabel("time")
        plt.ylabel(r"rel error: rhs(a(t)) - rhs-EECSW(a(t))")
        print("mean err: ", np.mean(rel_err))

    # %% surface plots
    params = train_params_list[3]
    params.rom.ECSW.network = lambda a: network(to.tensor(a,dtype=to.float32).reshape([1,-1]),
                                                to.tensor(params.reaction_const,dtype=to.float32).reshape([1,-1])).detach().cpu().numpy()
    q_tilde, romState_hist = solve_ROM(params.rom, mapping)
    plt.pcolormesh(q_tilde)

    # from lib.FTR import simple_FTR as FTR
    # width = 100
    # front = lambda x: 1 / (1 + np.exp(-x * width))
    # q_train = np.reshape(q_train, [*params.rom.fom.geom.N, -1])
    # phi = FTR(q_train, f=front, rank=params.rom.rom_size,
    #           tol=params.rom.ftr.tol, max_iter=params.rom.ftr.max_iter, dt=params.rom.ftr.opt_step_width,
    #           nt_show=80, plot_step=50, print_step=50)
    # n=0
    # for nt in range(0, np.size(phi, -1), 5):
    #     fig, ax = levelset_surface_plot(q_train[..., nt], phi[..., nt], params.geom.Xgrid, figure_number=1)
    #     fpath = pic_dir + "/levelset_%.3d" % n + ".png"
    #     print("saved image: ", fpath)
    #     fig.savefig(fpath, dpi=300, transparent=True)
    #     n = n + 1
    #     fig.clf()
    # %%

    if params.dim ==2 :
        X = params.geom.Xgrid
        q_test = np.reshape(q_test,[*params.geom.N,-1])
        # plot some snapshots
        N = len(params.rom.time.test_time_points)
        step = N//3
        i = 0
        fig=plt.figure(2)
        fig.suptitle("FOM")
        for k in range(0, N, step):
            time = params.rom.time.test_time_points[k]
            i = i + 1
            fig.add_subplot(2, 2, i)
            plt.pcolormesh(X[0],X[1],q_test[:,:,k])
            plt.xticks([])
            plt.yticks([])
            plt.xlabel("$x$")
            plt.ylabel("$y$")
            plt.title("$t = %1.2f $"% time)
        plt.tight_layout()
        plt.savefig(pic_dir + "disc_FOM.png", dpi=600, transparent=True, bbox_inches='tight')

        if 'rom_sol_dict' in locals():
            fig=plt.figure(3)
            fig.suptitle("LSPG")
            i=0
            q_tilde = rom_sol_dict["fomState_solution"]
            for k in range(0, N, step):
                time = params.rom.time.test_time_points[k]
                i = i + 1
                fig.add_subplot(2, 2, i)
                plt.pcolormesh(X[0], X[1], q_tilde[:, :, k])
                plt.xticks([])
                plt.yticks([])
                plt.xlabel("$x$")
                plt.ylabel("$y$")
                plt.title("$t = %1.2f $" % time)
            plt.tight_layout()

        print("err: ", np.linalg.norm(q_test-q_tilde)/np.linalg.norm(q_test))

        # if 'koopman_sol_dict' in locals():
        #     fig=plt.figure(4)
        #     q_tilde_f = koopman_sol_dict["fomState"]
        #     fig.suptitle("Koopman")
        #     i=0
        #     for k in range(0, N, step):
        #         time = params.rom.time.test_time_points[k]
        #         i = i + 1
        #         fig.add_subplot(2, 2, i)
        #         plt.pcolormesh(X[0], X[1], q_tilde_f[:, :, k])
        #         # plt.xlabel("$x$")
        #         # plt.ylabel("$y$")
        #         plt.xticks([], [])
        #         plt.yticks([], [])
        #         plt.axis("square")
        #         plt.title("$t = %1.2f $" % (time))
        #     plt.savefig(pic_dir + "disc_prediction.png", dpi=600, transparent=True, bbox_inches='tight')
        #
        # from numpy.linalg import svd
        #
        #
        # fig = plt.figure(5)
        # phi = q_test
        # u,s,vt = svd(np.reshape(phi,[params.fom_size,-1]), full_matrices=False)
        # basis = np.reshape(u,params.shape)
        # for r in range(4):
        #     plt.subplot(2,2,r+1)
        #     plt.pcolor(X[0],X[1],basis[...,r])
        #     # plt.xlabel("$x$")
        #     # plt.ylabel("$y$")
        #     plt.axis('off')
        #     plt.title("$\\tilde{\psi}_%d(x,y)$"%(r+1))
        # #plt.tight_layout()
        # plt.savefig(pic_dir + "disc_mode_phi.png", dpi=600, transparent=True, bbox_inches='tight')
        # #fig.suptitle("\\textbf{Ansatz:} $\displaystyle q(x,y,t)\\approx \\mathbf{f}(\sum_{i=1}^r a_i(t) \psi_i(x,y))$",fontsize=22)
        # plt.show()
        # fig = plt.figure(6)
        # u,s,vt = svd(np.reshape(q_train,[params.fom_size,-1]), full_matrices=False)
        # a_coef_q = np.diag(s)@vt
        # basis = np.reshape(u,params.shape)
        # for r in range(4):
        #     plt.subplot(2,2,r+1)
        #     plt.pcolormesh(X[0],X[1],basis[...,r])
        #     # plt.xlabel("$x$")
        #     # plt.ylabel("$y$")
        #     plt.axis('off')
        #     plt.title("$\psi_%d(x,y)$"%(r+1))
        # #plt.tight_layout()
        # plt.savefig(pic_dir + "disc_mode_q.png", dpi=600, transparent=True, bbox_inches='tight')
        # #fig.suptitle("\\textbf{Ansatz:} $\displaystyle q(x,y,t) \\approx \sum_{i=1}^r a_i(t) \psi_i(x,y)$",fontsize=22)

        # plt.figure(7)
        # plt.plot(params.time.t,a_coef[:4].T,marker='+')
        # plt.xlabel("time $t$",fontsize=22)
        # plt.ylabel("coefficients $a_i(t)$",fontsize=22)
        # plt.savefig(pic_dir + "disc_acoef_phi.pdf", dpi=300, transparent=True, bbox_inches='tight')
        #
        # plt.figure(8)
        # plt.plot(params.time.t,a_coef_q[:4].T,marker='*')
        # plt.xlabel("time $t$",fontsize=22)
        # plt.ylabel("coefficients $a_i(t)$",fontsize=22)
        # plt.savefig(pic_dir + "disc_acoef_q.pdf", dpi=300, transparent=True, bbox_inches='tight')

        from matplotlib.lines import Line2D
        plt.figure(99)
        legend_elements = [Line2D([0], [0], marker='o',label='FOM'),
                           Line2D([0], [0],linestyle='', marker='x',color='k', label='LSPG-ROM'),
                           Line2D([0], [0],linestyle='', marker='+',color='k', label='Koopman-ROM')]
        a_coef = mapping.acoef
        a_coef_tilde = rom_sol_dict["romState_solution"]
        a_coef_tilde_f = koopman_sol_dict["romState"]
        for r in range(2):
            #plt.subplot(2, 2, r + 1)
            h=plt.plot(params.rom.time.train_time_points, a_coef[r].T, '-o')
            plt.plot(params.rom.time.test_time_points, a_coef_tilde[r].T, 'kx')
            plt.plot(params.rom.time.test_time_points, a_coef_tilde_f[r].T, 'k+')#color=h[0].get_color())
            plt.xlabel("time $t$")#, fontsize=12)
            plt.ylabel("amplitudes $a_%d(t)$"%r)#, fontsize=12)

        plt.legend(handles=legend_elements,loc="upper right",frameon=False)
        #plt.savefig(pic_dir + "disc_acoef_phi_predict.pdf", dpi=300, transparent=True, bbox_inches='tight')
    else:
        fig = plt.figure(1)
        ax = fig.add_subplot(111,projection='3d')
        q_plot = q_train[:,0:-1:10]
        for it in range(np.size(q_plot,1)):
            ys = it*np.ones(np.size(q_plot,0))
            ax.plot(X[0],ys,q_plot[:,it],color=cm.Blues(300-it*20))
        plt.xlabel("space $x$")
        plt.ylabel("time $t$")
        plt.title("$q(x,t)$")

        fig = plt.figure(2)
        ax = fig.add_subplot(111)
        ax.pcolormesh(q_train)

        plt.set_cmap('jet_r')
        plt.xlabel("space $t$")
        plt.ylabel("time $x$")

    plt.tight_layout() # makes labels not overlap
    plt.show()

    #################################################################
    # % show sample mesh points
    #################################################################
    qapprox = np.reshape(rom_sol_dict['fomState_rom_solution'],[*params.geom.N,-1])
    show_samplepoints(qapprox[...,100], params.geom.Xgrid, params.sampleMeshIndices, figure_number = 84)
    plt.savefig(pic_dir + "sampleMeshPonts_"+params.case+".pdf", dpi=300, transparent=True, bbox_inches='tight')