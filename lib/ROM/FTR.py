# =============================================
#    Heat adjuster for jumps and bumps
# ---------------------------------------------
#    Heat adjoint problem for rank reduction
#    of transports with sharp gradients.
# =============================================
import numpy as np
from sklearn.utils.extmath import randomized_svd as rsvd

from time import perf_counter
import matplotlib.pyplot as plt
from pathlib import Path


# =============================================
#    IMPLEMENTATION
# =============================================

# def truncate(fun, rank):
#     assert rank >= 0
#     u,s,vt = np.linalg.svd(fun, full_matrices=False)
#     return (u[:,:rank]@np.diag(s[:rank])@vt[:rank,:])

def truncate(fun, rank):
    assert rank >= 0
    u,s,vt = rsvd(fun, rank)
    return (u@np.diag(s)@vt)

def cut_zero_weights(field, weights):
    # delete columns with zeros
    field = np.delete(field, np.where(~weights.any(axis=0))[0], axis=1)
    weights = np.delete(weights, np.where(~weights.any(axis=0))[0], axis=1)
    #delete rows with zeros
    field = np.delete(field, np.where(~weights.any(axis=1))[0], axis=0)
    return field


def normalise(q):
    # normalized data
    min_q = np.min(q)
    range_q = np.max(q) - min_q
    q = (q - min_q) / range_q
    return q

def animate(fun, cycles=1):
    import matplotlib.pyplot as plt
    x = np.linspace(0, 1, fun.shape[0])
    y = np.linspace(0, 1, fun.shape[1])
    ntime = fun.shape[-1]
    for t in range(cycles * ntime):
        plt.contourf(x, y, fun[..., t % ntime])
        plt.draw()
        plt.pause(0.005)
    plt.close()

def convergence_plot( q, phi, f, max_rank = None, max_iter = 100):
    import matplotlib.pyplot as plt
    # norm = lambda x: np.max(abs(x))
    matrizise = lambda fun: fun.reshape(-1, q.shape[-1])
    q = matrizise(q)
    phi = matrizise(phi)
    norm = np.linalg.norm

    norm_q = norm(q,ord='fro')
    ntime = q.shape[-1]
    if max_rank is None:
        max_rank = ntime
    errors = np.empty((2, max_rank))
    qsvd = np.linalg.svd(q, full_matrices=False)
    phisvd = np.linalg.svd(phi, full_matrices=False)
    for r in range(max_rank):

        q_trunc = qsvd[0][:,:r]@np.diag(qsvd[1][:r])@qsvd[2][:r,:]
        errors[0, r] = norm(q_trunc - q,ord='fro') / norm_q

        phi_trunc = simple_FTR(q, f, max_iter=max_iter, rank=r,  print_step = 1)
        q_trunc = f(phi_trunc)
        errors[1, r] = norm(q_trunc - q, ord='fro') / norm_q
        print("r: %4d, error q-q_r: %1.2e, error q-f(phi_r): %1.2e"%(r,errors[0,r],errors[1,r]))
    fig = plt.figure(428)
    plt.plot(errors[0], '<', label=r"$q- q_{r}$")
    #plt.plot(phisvd[1][:max_rank]/phisvd[1][0], 's', label=r"$\phi-\lfloor \phi\rfloor_{\,r}$")
    plt.plot(errors[1], 'D', label=r"$q-f_{\mathrm{FTR}}(\phi_{\,r})$")
    #plt.xlim([0,max_rank])
    plt.yscale('log')
    plt.legend()
    plt.xlabel('truncation rank')
    plt.ylabel('error')
    plt.show()
    return fig, errors[0], errors[1]


def construct_mapping(phi, front, rank = 1, substract_mean=True):
    if substract_mean:
        mean=np.mean(phi,1) # mean over all
        offset = mean
        mean=np.reshape(np.repeat(mean,np.size(phi,1)), phi.shape)
        phi = phi - mean
    else:
        offset = np.zeros([np.size(phi,0)])

    u,s,vt = np.linalg.svd(phi,full_matrices=False)
    basis = u[:,:rank]
    a_coef=np.diag(s[:rank])@vt[:rank,:]

    # ------------------------------------------------------
    def mapping(a):

            phi = basis@a
            if a.ndim == 2:
                for i in range(np.size(a,1)):
                    # if a is not only a vector but a matrix containing vectors at different times in its columns,
                    # we have to add the offset to each column
                    phi[:,i] += offset

            return front(phi)
    # ------------------------------------------------------

    return mapping,a_coef

def bgfs_FTR( q, f, df, tol=1e-7, max_iter=100, mu= 0.01, dt = 1.0, rank=3,
                                  offset = 0,plot_step=100, phi_0=None, weights = None, nt_show = None, print_step = 100):
    from scipy.optimize import minimize as minimize

        # fun, x0, args=(), method='L-BFGS-B', jac=None, bounds=None, tol=None, callback=None,
        #                     options={'disp': None, 'maxcor': 10, 'ftol': 2.220446049250313e-09, 'gtol': 1e-05,
        #                              'eps': 1e-08, 'maxfun': 15000, 'maxiter': 15000, 'iprint': - 1, 'maxls': 20,
        #                              'finite_diff_rel_step': None})
    """
    INPUTS:
    q  ... data field
    f  ... front function for pointwise evaluation of tensors

    Optional:
    tol       ... stopping criteria of error
    max_itrer ... maximal number of iterations
    mu        ... regularizing parameter of nuclear norm
    dt        ... stepsize of gradient
    offset    ... constant to fix after laplace is solved
    plot_step ... number of steps between plotting
    phi_0     ... restart from previous phi
    weights   ... weightmatrix in case of nonperiodic BC (use 0 padding for boundaries)
    nt_show   ... time slice shown in the plots

    OUTPUT:
    phi       ... a low rank levelset function to approximate q with q\approx f(phi)
    """

    shape = q.shape
    if weights is None:
        weights = np.ones(shape)
    if phi_0 is not None:
        phi_n = np.reshape(phi_0,np.shape(q))
    else:
        phi_n = offset*np.ones_like(q)
    if nt_show is not None:
        nt = nt_show
    else:
        nt = np.random.randint(0, shape[-1])

    tensorise = lambda fun: fun.reshape(shape)
    matrizise = lambda fun: fun.reshape(-1, shape[-1])
    vectorise = lambda fun: fun.reshape(-1)
    SVT = lambda dat, thresh_perc: soft_threshold( dat, thresh_perc)

    q_norm = np.linalg.norm(vectorise(q))

    it = 0
    Jfit = Jfitold = np.inf
    u_n = np.zeros_like(q)
    q_cut = cut_zero_weights(q, weights)
    qtilde_cut = cut_zero_weights(f(phi_n), weights)

    if plot_step < max_iter:
        fig,ax = plt.subplots(1, 2, sharey=True , num=9)
        h=[0,0]
        h[0] = ax[0].pcolormesh(q_cut[..., nt], shading='gouraud')
        ax[0].set_title(r"$\phi(x,y,t_n)$")
        ax[1].set_title(r"error $q-f(\phi)$")
        h[1] = ax[1].pcolormesh(q_cut[..., nt]-qtilde_cut[..., nt], shading='gouraud')
        fig.colorbar(h[1], ax=ax[1])

    while Jfit > tol and it < max_iter: #and Jfitold >= Jfit:
        tstart = perf_counter()
        #################################################
        # FTR - Algorithm
        #################################################
        fun =lambda phi: (np.linalg.norm(vectorise(weights) * (f(vectorise(phi)) - vectorise(q)))**2*0.5,  f(vectorise(phi))-vectorise(q))
        sol = minimize(fun,vectorise(phi_n),jac=True,method='Newton-CG',options={'maxiter': 1, 'disp':True})
        # threshold singular values
        #phi_n = tensorise(SVT(matrizise(phi_n), mu))
        phi_n = tensorise(truncate(matrizise(sol.x), rank))
        #################################################

        if it % plot_step == 0 and it > 0:
            phi_cut = cut_zero_weights(phi_n, weights)
            h[0].set_array(phi_cut[...,nt].ravel())
            h[0].set_clim(min(phi_cut.flatten()), max(phi_cut.flatten()))
            res = q-f(phi_n)
            res_cut = cut_zero_weights(res, weights)
            h[1].set_array(np.abs((res_cut[...,nt])).ravel())
            h[1].set_clim(0, max(res_cut.flatten()))
            plt.draw(), plt.pause(1e-3)
        it += 1

        if it % print_step ==0:
            Jfitold = Jfit
            # relative fitting error
            Jfit = np.linalg.norm(vectorise((f(phi_n) - q))) / q_norm
            # lagrangian we are trying to minimize
            print(f"[iter={it:2d} in {perf_counter() - tstart:.2f}s] rel.error: {Jfit:.4f}")

    phi_cut = cut_zero_weights(tensorise(phi_n),weights)
    return phi_cut





def simple_FTR( q, f, tol=1e-7, max_iter=100, mu= 0.01, dt = 1.0, rank=3, stop_if_residual_increasing = False,
                                  offset = 0,plot_step=1000000, phi_0=None, weights = None, nt_show = None, print_step = 100):
    """
    INPUTS:
    q  ... data field
    f  ... front function for pointwise evaluation of tensors

    Optional:
    tol       ... stopping criteria of error
    max_itrer ... maximal number of iterations
    mu        ... regularizing parameter of nuclear norm
    dt        ... stepsize of gradient
    offset    ... constant to fix after laplace is solved
    plot_step ... number of steps between plotting
    phi_0     ... restart from previous phi
    weights   ... weightmatrix in case of nonperiodic BC (use 0 padding for boundaries)
    nt_show   ... time slice shown in the plots

    OUTPUT:
    phi       ... a low rank levelset function to approximate q with q\approx f(phi)
    """

    shape = q.shape
    if weights is None:
        weights = np.ones(shape)
    if phi_0 is not None:
        phi_n = np.reshape(phi_0,np.shape(q))
    else:
        phi_n = offset*np.ones_like(q)
    if nt_show is not None:
        nt = nt_show
    else:
        nt = np.random.randint(0, shape[-1])

    tensorise = lambda fun: fun.reshape(shape)
    matrizise = lambda fun: fun.reshape(-1, shape[-1])
    vectorise = lambda fun: fun.reshape(-1)

    q_norm = np.linalg.norm(vectorise(q))

    it = 0
    Jfit = Jfitold = np.inf
    q_cut = cut_zero_weights(q, weights)
    qtilde_cut = cut_zero_weights(f(phi_n), weights)

    if plot_step < max_iter:
        fig,ax = plt.subplots(1, 2, sharey=True , num=9)
        h=[0,0]
        if q_cut.ndim==2:
            h[0] = ax[0].pcolormesh(q_cut, shading='gouraud')
            ax[0].set_title(r"$\phi(x,y,t_n)$")
            ax[1].set_title(r"error $q-f(\phi)$")
            h[1] = ax[1].pcolormesh(q_cut - qtilde_cut, shading='gouraud')
            fig.colorbar(h[1], ax=ax[1])

        else:
            h[0] = ax[0].pcolormesh(q_cut[..., nt], shading='gouraud')
            ax[0].set_title(r"$\phi(x,y,t_n)$")
            ax[1].set_title(r"error $q-f(\phi)$")
            h[1] = ax[1].pcolormesh(q_cut[..., nt]-qtilde_cut[..., nt], shading='gouraud')
            fig.colorbar(h[1], ax=ax[1])

    while Jfit > tol and it < max_iter: #and Jfitold >= Jfit:
        tstart = perf_counter()
        #################################################
        # FTR - Algorithm
        #################################################
        res = weights * ( q - f(phi_n))
        phi_n = phi_n + dt * res
        # threshold singular values
        #phi_n = tensorise(SVT(matrizise(phi_n), mu))
        phi_n = tensorise(truncate(matrizise(phi_n), rank))
        #################################################

        if it % plot_step == 0 and it > 0:
            phi_cut = cut_zero_weights(phi_n, weights)
            res = q - f(phi_n)
            res_cut = cut_zero_weights(res, weights)
            if phi_cut.ndim == 2:
                phi_plt = phi_cut
                res_plt = res_cut
            else:
                phi_plt = phi_cut[...,nt]
                res_plt = res_cut[...,nt]
            h[0].set_array(phi_plt.ravel())
            h[0].set_clim(min(phi_cut.flatten()), max(phi_cut.flatten()))
            h[1].set_array(np.abs(res_plt).ravel())
            h[1].set_clim(0, max(res_cut.flatten()))
            plt.draw(), plt.pause(1e-3)
        it += 1
        if Jfitold < Jfit and stop_if_residual_increasing:
            break
        if it % print_step ==0:
            Jfitold = Jfit
            # relative fitting error
            Jfit = np.linalg.norm(vectorise((f(phi_n) - q))) / q_norm
            # lagrangian we are trying to minimize
            print(f"[iter={it:2d} in {perf_counter() - tstart:.2f}s] rel.error: {Jfit:.2e}")

    phi_cut = cut_zero_weights(tensorise(phi_n),weights)
    return phi_cut




def simple_FTR_torch( q, f, tol=1e-7, max_iter=100, mu= 0.01, dt = 1.0, rank=3, stop_if_residual_increasing = False,
                                  offset = 0,plot_step=1000000, phi_0=None, weights = None, nt_show = None, print_step = 100):
    """
    INPUTS:
    q  ... data field
    f  ... front function for pointwise evaluation of tensors

    Optional:
    tol       ... stopping criteria of error
    max_itrer ... maximal number of iterations
    mu        ... regularizing parameter of nuclear norm
    dt        ... stepsize of gradient
    offset    ... constant to fix after laplace is solved
    plot_step ... number of steps between plotting
    phi_0     ... restart from previous phi
    weights   ... weightmatrix in case of nonperiodic BC (use 0 padding for boundaries)
    nt_show   ... time slice shown in the plots

    OUTPUT:
    phi       ... a low rank levelset function to approximate q with q\approx f(phi)
    """

    shape = q.shape
    if weights is None:
        weights = np.ones(shape)
    if phi_0 is not None:
        phi_n = np.reshape(phi_0,np.shape(q))
    else:
        phi_n = offset*np.ones_like(q)
    if nt_show is not None:
        nt = nt_show
    else:
        nt = np.random.randint(0, shape[-1])

    tensorise = lambda fun: fun.reshape(shape)
    matrizise = lambda fun: fun.reshape(-1, shape[-1])
    vectorise = lambda fun: fun.reshape(-1)

    q_norm = np.linalg.norm(vectorise(q))

    it = 0
    Jfit = Jfitold = np.inf
    q_cut = cut_zero_weights(q, weights)
    qtilde_cut = cut_zero_weights(f(phi_n), weights)

    if plot_step < max_iter:
        fig,ax = plt.subplots(1, 2, sharey=True , num=9)
        h=[0,0]
        if q_cut.ndim==2:
            h[0] = ax[0].pcolormesh(q_cut, shading='gouraud')
            ax[0].set_title(r"$\phi(x,y,t_n)$")
            ax[1].set_title(r"error $q-f(\phi)$")
            h[1] = ax[1].pcolormesh(q_cut - qtilde_cut, shading='gouraud')
            fig.colorbar(h[1], ax=ax[1])

        else:
            h[0] = ax[0].pcolormesh(q_cut[..., nt], shading='gouraud')
            ax[0].set_title(r"$\phi(x,y,t_n)$")
            ax[1].set_title(r"error $q-f(\phi)$")
            h[1] = ax[1].pcolormesh(q_cut[..., nt]-qtilde_cut[..., nt], shading='gouraud')
            fig.colorbar(h[1], ax=ax[1])

    while Jfit > tol and it < max_iter: #and Jfitold >= Jfit:
        tstart = perf_counter()
        #################################################
        # FTR - Algorithm
        #################################################
        res = weights * ( q - f(phi_n))
        phi_n = phi_n + dt * res
        # threshold singular values
        #phi_n = tensorise(SVT(matrizise(phi_n), mu))
        phi_n = tensorise(truncate(matrizise(phi_n), rank))
        #################################################

        if it % plot_step == 0 and it > 0:
            phi_cut = cut_zero_weights(phi_n, weights)
            res = q - f(phi_n)
            res_cut = cut_zero_weights(res, weights)
            if phi_cut.ndim == 2:
                phi_plt = phi_cut
                res_plt = res_cut
            else:
                phi_plt = phi_cut[...,nt]
                res_plt = res_cut[...,nt]
            h[0].set_array(phi_plt.ravel())
            h[0].set_clim(min(phi_cut.flatten()), max(phi_cut.flatten()))
            h[1].set_array(np.abs(res_plt).ravel())
            h[1].set_clim(0, max(res_cut.flatten()))
            plt.draw(), plt.pause(1e-3)
        it += 1
        if Jfitold < Jfit and stop_if_residual_increasing:
            break
        if it % print_step ==0:
            Jfitold = Jfit
            # relative fitting error
            Jfit = np.linalg.norm(vectorise((f(phi_n) - q))) / q_norm
            # lagrangian we are trying to minimize
            print(f"[iter={it:2d} in {perf_counter() - tstart:.2f}s] rel.error: {Jfit:.2e}")

    phi_cut = cut_zero_weights(tensorise(phi_n),weights)
    return phi_cut



def FTR_ranks(snapshots, rank_list, front, save_fname = None, max_iter = 40000, print_step = 100, dt = 1.0,
              offset = 0,  stop_if_residual_increasing = False, skip_existing = True, plot_step = 1000000):
    """

    :param snapshots:
    :param rank_list:
    :param front:
    :param save_fname: filename without data type e.g. mydir/myFTRdata
    :param max_iter:
    :param print_step:
    :param dt:
    :param offset:
    :param stop_if_residual_increasing:
    :param skip_existing: if save_fname exists we load the data instead
    :return:
    """
    phi_ftr_list = []
    q_ftr_list = []
    err_list = []
    if np.size(max_iter) == 1:
        max_iter = np.ones_like(rank_list) * max_iter
    for k, rank in enumerate(rank_list):

        if save_fname:
            fname = save_fname + "_%d_dofs.npy"%rank
            if skip_existing and Path(fname).is_file():
                print("reading: "+fname)
                with open(fname, 'rb') as f:
                    phi_ftr, qtilde_FTR, err = np.load(f, allow_pickle=True).item().values()
            else:
                phi_ftr = simple_FTR(snapshots, front, max_iter=max_iter[k], rank=rank, print_step=print_step, dt=dt,
                                     offset=offset,
                                     stop_if_residual_increasing=stop_if_residual_increasing, plot_step=plot_step)
                qtilde_FTR = front(phi_ftr)
                err = np.linalg.norm(qtilde_FTR - snapshots, "fro") / np.linalg.norm(snapshots, "fro")

                print("rank: %d, error offline: %.4f" % (rank, err))
                dict_FTR = {"phi": phi_ftr, "q_tilde": qtilde_FTR, "rel_error": err}
                with open(fname, 'wb') as f:
                    np.save(f, dict_FTR, allow_pickle=True)

        q_ftr_list.append(qtilde_FTR)
        phi_ftr_list.append(phi_ftr)
        err_list.append(err)

    return phi_ftr_list, q_ftr_list, err_list