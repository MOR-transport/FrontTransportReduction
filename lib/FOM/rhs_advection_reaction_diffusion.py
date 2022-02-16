from time import perf_counter
import numpy as np
from numpy import pi, reshape
from numba import njit
from time import perf_counter


def jacobian_rhs_advection_reaction_diffusion_2D_periodic(self, q, t, dq):
    """
    This function computes the derivative of the rhs with respect to q in the direction of dq

    :param q: state at time t
    :param t: time
    :param dq: direction of the derivative dq
    :return: drhs(q,t)/dq * dq
    """

    a = self.diffusion_const
    b = self.reaction_const
    chi = self.penalisation.eta
    weights = self.penalisation.weights
    # K = self.geom.K
    # [kx, ky] = meshgrid(K[0], K[1])
    # first derivative
    dq_dx = self.diff_operators.Dx(dq)
    dq_dy = self.diff_operators.Dy(dq)
    # second derivative
    ddq_ddx = self.diff_operators.Dxx(dq)
    ddq_ddy = self.diff_operators.Dyy(dq)

    ux, uy = self.velocity_field(t)
    # right hand side
    rhs_dq = - ux * dq_dx - uy * dq_dy + a * (ddq_ddx + ddq_ddy) - b * (2 * q - 1) * dq - chi * weights * dq

    return reshape(rhs_dq, -1)




def rhs_advection_reaction_diffusion_2D_periodic(self, q, t):
    a = self.diffusion_const
    b = self.reaction_const
    chi = self.penalisation.eta
    weights = self.penalisation.weights
    q_target = self.penalisation.q_target
    # first derivative
    dq_dx = self.diff_operators.Dx(q)
    dq_dy = self.diff_operators.Dy(q)
    # second derivative
    ddq_ddx = self.diff_operators.Dxx(q)
    ddq_ddy = self.diff_operators.Dyy(q)

    ux, uy = self.velocity_field(t)
    # right hand side
    rhs = - ux * dq_dx -  uy * dq_dy + a * (ddq_ddx + ddq_ddy) - b * q**2*(q-1) -chi * weights * (q-q_target)

    return reshape(rhs,-1)







######################################################################################################################
######################################################################################################################
#
# RHS and its jacobian for hyperreduction and parallelization with NUMBA
#
######################################################################################################################
######################################################################################################################
def rhs_advection_reaction_diffusion_2D_periodic_hyperreduced_numba_wrapper(fom_params, q, t, sampleMeshIndices= None, rhs=None):
    a = fom_params.diffusion_const
    b = fom_params.reaction_const
    chi = fom_params.penalisation.eta
    weights = fom_params.penalisation.weights*chi
    q_target = fom_params.penalisation.q_target
    c = fom_params.advection_speed
    tau = fom_params.decay_constant
    we = np.asarray(fom_params.w0)
    r0 = fom_params.r0
    [X,Y] = fom_params.geom.Xgrid
    Xvec = X.flatten()
    Yvec = Y.flatten()
    L = fom_params.geom.L
    T = fom_params.time.T
    case = fom_params.case
    smoothwidth = fom_params.geom.dX[0] * 1
    stencil_list = fom_params.get_stencil_in_sparse_format()
    if sampleMeshIndices is None:
        sampleMeshSize = fom_params.fom_size
        sampleMeshIndices = np.arange(0, sampleMeshSize)
    else:
        sampleMeshSize = np.size(sampleMeshIndices)

    if rhs is None:
        rhs = np.zeros(sampleMeshSize)
        return_rhs = True
    else:
        return_rhs =False

    ux, uy = velocity_field(t, tau, T, Xvec, Yvec, L, c, we, r0, case, smoothwidth, sampleMeshSize, sampleMeshIndices)
    rhs_advection_reaction_diffusion_2D_periodic_hyperreduced_numba(q, ux, uy, a, b, weights, q_target,
                                                                    stencil_list[0][0],stencil_list[0][1],
                                                                    stencil_list[1][0], stencil_list[1][1],
                                                                    stencil_list[2][0], stencil_list[2][1],
                                                                    stencil_list[3][0], stencil_list[3][1],
                                                                    sampleMeshSize, sampleMeshIndices, rhs)
    # right hand side
    if return_rhs:
        return rhs


@njit()
def rhs_advection_reaction_diffusion_2D_periodic_hyperreduced_numba(q, ux_hreduced, uy_hreduced, a, b, weights, q_target,
                                                                    stencil_x_val, stencil_x_col,
                                                                    stencil_y_val, stencil_y_col,
                                                                    stencil_xx_val, stencil_xx_col,
                                                                    stencil_yy_val, stencil_yy_col,
                                                                    sampleMeshSize, sampleMeshIndices, rhs):

    for ir in range(sampleMeshSize):
        ux_i = ux_hreduced[ir]
        uy_i = uy_hreduced[ir]
        index = sampleMeshIndices[ir]
        q_i = q[index]
        w_i = weights[index]
        qt_i = q_target[index]
        dqx_i, dqy_i, dqxx_i, dqyy_i = 0.0, 0.0, 0.0, 0.0
        # dq_dx and dq_dy
        for j in range(8):
            dqx_i += stencil_x_val[8 * index + j] * q[stencil_x_col[8 * index + j]]
            dqy_i += stencil_y_val[8 * index + j] * q[stencil_y_col[8 * index + j]]
        # dq_dxx and dq_dyy
        for j in range(9):
            dqxx_i += stencil_xx_val[9 * index + j] * q[stencil_xx_col[9 * index + j]]
            dqyy_i += stencil_yy_val[9 * index + j] * q[stencil_yy_col[9 * index + j]]

        rhs[ir] = - ux_i * dqx_i - uy_i * dqy_i + a * (dqxx_i + dqyy_i) - b * q_i**2 * (q_i - 1) - w_i * (q_i - qt_i)

@njit()
def velocity_field(t, tau, T, Xvec, Yvec, L, c, we, r0, case, smoothwidth, sampleMeshSize, sampleMeshIndices):

        # dipole vortex benchmark
        ux = np.zeros(sampleMeshSize)
        uy = np.zeros(sampleMeshSize)
        if case == "pacman":
            (x1, y1) = (0.6 - c * t, 0.49 * L[1])
            (x2, y2) = (0.6 - c * t, 0.51 * L[1])
            for i in range(sampleMeshSize):
                idx = sampleMeshIndices[i]
                X = Xvec[idx]
                Y = Yvec[idx]
                distance1 = (X - x1) ** 2 + (Y - y1) ** 2
                distance1 = (distance1/r0)**2
                distance2 = (X - x2) ** 2 + (Y - y2) ** 2
                distance2 = (distance2/r0)**2
                ux[i] = -we[0] * (Y - y1) * np.exp(-(distance1)) + we[1] * (Y - y2) * np.exp(-(distance2))
                uy[i] = we[0] * (X - x1) * np.exp(-(distance1)) - we[1] * (X - x2) * np.exp(-(distance2) )
            ux *= np.exp(-(t / tau) ** 2)
            uy *= np.exp(-(t / tau) ** 2)
        elif case == "bunsen_flame":
            inv_2pi = 1/(2*np.pi)
            my_sawtooth = lambda x: (x*inv_2pi)%1
            (x1, y1) = (0.1 * L[0] + 0.8*L[0] * my_sawtooth(2*pi/T*t * 5), 0.46*L[1])
            (x2, y2) = (0.1 * L[0] + 0.8*L[0] * my_sawtooth(2*pi/T*t * 5), 0.54*L[1])
            softstep = lambda x, d: (1 + np.tanh(x / d)) * 0.5
            for i in range(sampleMeshSize):
                idx = sampleMeshIndices[i]
                X = Xvec[idx]
                Y = Yvec[idx]
                distance1 = (X - x1) ** 2 + (Y - y1) ** 2
                distance1 = (distance1/r0)**2
                distance2 = (X - x2) ** 2 + (Y - y2) ** 2
                distance2 = (distance2/r0)**2
                ux[i] = + we[0] * (Y - y1) * np.exp(-distance1) - we[1] * (Y - y2) * np.exp(-distance2)
                uy[i] = - we[0] * (X - x1) * np.exp(-distance1) + we[1] * (X - x2) * np.exp(-distance2)
                ux[i] += (0.05)*(softstep(Y-0.3*L[1], 2*smoothwidth)-softstep(Y-0.7*L[1], 2*smoothwidth))*softstep(X-0.05*L[0], smoothwidth)#*params.penalisation.weights
        else:
            pass

        return ux, uy


def jacobian_rhs_advection_reaction_diffusion_2D_periodic_hyperreduced_numba_wrapper(fom_params, q, t, qlin, sampleMeshIndices= None, jac=None):
    """

    :param fom_params:
    :param q:
    :param t:
    :param qlin:
    :param sampleMeshIndices:
    :param jac:
    :return:
    """
    a = fom_params.diffusion_const
    b = fom_params.reaction_const
    chi = fom_params.penalisation.eta
    weights = fom_params.penalisation.weights*chi
    c = fom_params.advection_speed
    tau = fom_params.decay_constant
    we = np.asarray(fom_params.w0)
    r0 = fom_params.r0
    [X,Y] = fom_params.geom.Xgrid
    Xvec = X.flatten()
    Yvec = Y.flatten()
    L = fom_params.geom.L
    T = fom_params.time.T
    case = fom_params.case
    smoothwidth = fom_params.geom.dX[0] * 1
    stencil_list = fom_params.get_stencil_in_sparse_format()
    if sampleMeshIndices is None:
        sampleMeshSize = fom_params.fom_size
        sampleMeshIndices = np.arange(0, sampleMeshSize)
    else:
        sampleMeshSize = np.size(sampleMeshIndices)

    if jac is None:
        jac = np.zeros(sampleMeshSize)
        return_jac = True
    else:
        return_jac =False

    ux, uy = velocity_field(t, tau, T, Xvec, Yvec, L, c, we, r0, case, smoothwidth, sampleMeshSize, sampleMeshIndices)
    jacobian_rhs_advection_reaction_diffusion_2D_periodic_hyperreduced_numba(q, qlin, ux, uy, a, b, weights,
                                                                    stencil_list[0][0],stencil_list[0][1],
                                                                    stencil_list[1][0], stencil_list[1][1],
                                                                    stencil_list[2][0], stencil_list[2][1],
                                                                    stencil_list[3][0], stencil_list[3][1],
                                                                    sampleMeshSize, sampleMeshIndices, jac)
    # right hand side
    if return_jac:
        return jac


@njit()
def jacobian_rhs_advection_reaction_diffusion_2D_periodic_hyperreduced_numba(q, qlin, ux_hreduced, uy_hreduced, a, b, weights,
                                                                    stencil_x_val, stencil_x_col,
                                                                    stencil_y_val, stencil_y_col,
                                                                    stencil_xx_val, stencil_xx_col,
                                                                    stencil_yy_val, stencil_yy_col,
                                                                    sampleMeshSize, sampleMeshIndices, jac):
    """
    Jacobian of the rhs. Be aware that this is only working for stencil size 8 yet. Has to be generalized!
    :param q:
    :param qlin:
    :param ux_hreduced:
    :param uy_hreduced:
    :param a:
    :param b:
    :param weights:
    :param stencil_x_val:
    :param stencil_x_col:
    :param stencil_y_val:
    :param stencil_y_col:
    :param stencil_xx_val:
    :param stencil_xx_col:
    :param stencil_yy_val:
    :param stencil_yy_col:
    :param sampleMeshSize:
    :param sampleMeshIndices:
    :param jac:
    :return:
    """
    for ir in range(sampleMeshSize):
        index = sampleMeshIndices[ir]
        ux_i = ux_hreduced[ir]
        uy_i = uy_hreduced[ir]
        q_i = q[index]
        qlin_i = qlin[index]
        w_i = weights[index]
        dqx_i, dqy_i, dqxx_i, dqyy_i = 0.0, 0.0, 0.0, 0.0
        # dq_dx and dq_dy
        for j in range(8):
            dqx_i += stencil_x_val[8 * index + j] * qlin[stencil_x_col[8 * index + j]]
            dqy_i += stencil_y_val[8 * index + j] * qlin[stencil_y_col[8 * index + j]]
        # dq_dxx and dq_dyy
        for j in range(9):
            dqxx_i += stencil_xx_val[9 * index + j] * qlin[stencil_xx_col[9 * index + j]]
            dqyy_i += stencil_yy_val[9 * index + j] * qlin[stencil_yy_col[9 * index + j]]

        jac[ir] = - ux_i * dqx_i - uy_i * dqy_i + a * (dqxx_i + dqyy_i) - b * (2 * q_i - 1) * qlin_i - w_i * qlin_i



######################################################################################################################
######################################################################################################################
#
# UNIT TESTING
#
######################################################################################################################
######################################################################################################################


if __name__ == "__main__":
    """
    This part is to validate/unittest the rhs and the optimized rhs. Furthermore we check the jacobians (linearized rhs) 
    and hyperreduction.
    """
    import pathlib, sys
    file_path = pathlib.Path(__file__).parent.absolute()
    sys.path.append(str(file_path) + "/..")
    import random
    from numpy.linalg import norm
    from time import time
    from lib.FOM import *
    from lib.cases import my_cases
    np.random.seed(1)
    random.seed(2)

    case = "bunsen_flame"
    params = my_cases(case)
    q0 = params.inicond
    qlin = np.random.random(np.size(q0))

    "1.) RHS Check/Benchmark"
    # run numba once to compile it (otherwise timer counts compilation time)
    rhs_numba = rhs_advection_reaction_diffusion_2D_periodic_hyperreduced_numba_wrapper(params, q0, t=0)
    # optimized rhs
    tcpu =[0,0]
    tcpu[0] = time()
    rhs_numba = rhs_advection_reaction_diffusion_2D_periodic_hyperreduced_numba_wrapper(params, q0, t=0)
    tcpu[0] = time() - tcpu[0]

    # default RHS
    tcpu[1] = time()
    rhs = rhs_advection_reaction_diffusion_2D_periodic(params, q0, t=0)
    tcpu[1] = time() - tcpu[1]
    err = norm(rhs_numba-rhs)/norm(rhs)
    print(f"1.) rel error between fom rhs and optimized rhs: {err}, speedup: {tcpu[1]/tcpu[0]}")

    "2.) Check/Benchmark hyperreduced RHS "
    sampleMeshSize = 22
    sampleMeshIndices = random.sample(range(1, params.fom_size - 1), sampleMeshSize)
    sampleMeshIndices = np.append(sampleMeshIndices, [0, params.fom_size - 1])
    # sort for convenience, not necessarily needed
    sampleMeshSize += 2
    sampleMeshIndices = np.sort(sampleMeshIndices)

    # optmized rhs
    tcpu =[0,0]
    tcpu[0] = time()
    rhs_numba = rhs_advection_reaction_diffusion_2D_periodic_hyperreduced_numba_wrapper(params, q0, t=0, sampleMeshIndices=sampleMeshIndices)
    tcpu[0] = time() - tcpu[0]
    # default RHS
    tcpu[1] = time()
    rhs = rhs_advection_reaction_diffusion_2D_periodic(params, q0, t=0)
    rhs = np.take(rhs,sampleMeshIndices)
    tcpu[1] = time() - tcpu[1]
    err = norm(rhs_numba-rhs)/norm(rhs)
    print(f"2.) rel error between opt rhs and hypreduced opt rhs: {err}, speedup: {tcpu[1]/tcpu[0]}")

    "3.) Jacobian of RHS Check/Benchmark"
    # run numba once to compile it (otherwise timer counts compilation time)
    jac_numba = jacobian_rhs_advection_reaction_diffusion_2D_periodic_hyperreduced_numba_wrapper(params, q0, 0, qlin)
    # optimized rhs
    tcpu = [0, 0]
    tcpu[0] = time()
    jac_numba = jacobian_rhs_advection_reaction_diffusion_2D_periodic_hyperreduced_numba_wrapper(params, q0, 0, qlin)
    tcpu[0] = time() - tcpu[0]

    # default RHS
    tcpu[1] = time()
    jac = jacobian_rhs_advection_reaction_diffusion_2D_periodic(params, q0, 0, qlin)
    tcpu[1] = time() - tcpu[1]
    err = norm(jac_numba - jac) / norm(jac)
    print(f"3.) rel error between fom jacobian_rhs and optimized jacobian_rhs: {err}, speedup: {tcpu[1] / tcpu[0]}")

    "4.) Check/Benchmark hyperreduced jacobian "
    # optmized rhs
    tcpu = [0, 0]
    tcpu[0] = time()
    jac_numba = jacobian_rhs_advection_reaction_diffusion_2D_periodic_hyperreduced_numba_wrapper(params, q0, 0, qlin,
                                                                                        sampleMeshIndices=sampleMeshIndices)
    tcpu[0] = time() - tcpu[0]
    # default RHS
    tcpu[1] = time()
    jac = jacobian_rhs_advection_reaction_diffusion_2D_periodic(params, q0, 0, qlin)
    jac = np.take(jac, sampleMeshIndices)
    tcpu[1] = time() - tcpu[1]
    err = norm(jac_numba - jac) / norm(jac)
    print(f"4.) rel error between opt jac and hypreduced opt jac: {err}, speedup: {tcpu[1] / tcpu[0]}")

