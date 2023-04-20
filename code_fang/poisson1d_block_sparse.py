'''investigate the effect of the fix weight, freq and lengthscale'''
import numpy as np
import kernels_new
from kernels_new import *
import random
import matplotlib.pyplot as plt
import optax
import jax
import jax.numpy as jnp
from kernel_matrix import Kernel_matrix
from jax import vmap
import pandas as pd
import utils

import time
import tqdm
import os
import copy
import init_func

np.random.seed(0)
random.seed(0)


class GPRLatent:

    # equation: u_{xx}  = f(x)
    # Xind: the indices of X_col that corresponds to training points, i.e., boundary points
    # y: training outputs
    # Xcol: collocation points
    def __init__(self, Xind, y, X_col, src_col, jitter, X_test, Y_test, trick_paras=None, fix_dict=None):
        self.Xind = Xind
        self.y = y
        self.X_col = X_col
        self.src_col = src_col
        self.jitter = jitter
        # X is the 1st and the last point in X_col
        self.X_con = X_col
        self.N = self.Xind.shape[0]
        self.N_con = self.X_con.shape[0]

        self.trick_paras = trick_paras
        self.fix_dict = fix_dict

        lr = trick_paras['lr'] if trick_paras is not None else 0.01

        self.optimizer = optax.adam(learning_rate=lr)

        self.llk_weight = trick_paras['llk_weight']

        # fix_kernel_paras =  {'log-w': np.log(1/trick_paras['Q'])*np.ones(trick_paras['Q']), 'log-ls': np.zeros(trick_paras['Q']), 'freq': np.linspace(0, 1, trick_paras['Q'])*100} if fix_dict is not None else None

        # {'log-w': np.log(1/trick_paras['Q'])*np.ones(trick_paras['Q']), 'log-ls': np.zeros(trick_paras['Q']), 'freq': np.ones(trick_paras['Q'])} if fix_dict is not None else None
        fix_kernel_paras = None

        self.num = trick_paras['Q']

        self.cov_func = trick_paras['kernel'](fix_dict, None, trick_paras['Q'])

        self.kernel_matrix = Kernel_matrix(self.jitter, self.cov_func, 'NONE')
        self.Xte = X_test
        self.yte = Y_test

        print('equation is: ', self.trick_paras['equation'])
        print('kernel is:', self.cov_func.__class__.__name__)

        self.ln_s0 = 0.0
        self.bg = 1e-5
        self.b0 = 1e-5
        self.PI = math.pi
        self.ONE = 1.0
        self.HALF = 0.5

    def KL_term(self, mu, L, K):
        Ltril = jnp.tril(L)
        hh_expt = jnp.matmul(Ltril, Ltril.T) + jnp.matmul(mu, mu.T)
        # hh_expt = jnp.matmul(Ltril, Ltril.T)
        kl = (0.5 * jnp.trace(jnp.linalg.solve(K, hh_expt)) + 0.5 *
              jnp.linalg.slogdet(K)[1] - 0.5 * jnp.sum(jnp.log(jnp.square(jnp.diag(Ltril)))))
        return kl

    @partial(jit, static_argnums=(0,))
    def loss(self, params, key,):
        u = params['u']  # function values at the collocation points
        u = u.sum(axis=1).reshape(-1, 1)  # sum over trick
        log_v = params['log_v']  # inverse variance for eq ll
        log_tau = params['log_tau']  # inverse variance for boundary ll
        kernel_paras = params['kernel_paras']

        x_p = jnp.tile(self.X_con.flatten(), (self.N_con, 1)).T
        X1_p = x_p.flatten()
        X2_p = jnp.transpose(x_p).flatten()
        # only the cov matrix of func vals
        K = self.kernel_matrix.get_kernel_matrix(X1_p, X2_p, kernel_paras)
        Kinv_u = jnp.linalg.solve(K, u)
        # log_prior = -0.5*jnp.linalg.slogdet(K)[1] - 0.5*jnp.sum(u*Kinv_u)
        log_prior = - 0.5*jnp.sum(u*Kinv_u)

        # boundary
        log_boundary_ll = 0.5 * self.N * log_tau - 0.5 * \
            jnp.exp(
                log_tau) * jnp.sum(jnp.square(u[self.Xind].reshape(-1) - self.y.reshape(-1)))

        # equation
        K_dxx1 = vmap(self.cov_func.DD_x1_kappa, (0, 0, None))(
            X1_p, X2_p, kernel_paras).reshape(self.N_con, self.N_con)
        u_xx = jnp.matmul(K_dxx1, Kinv_u)
        eq_ll = 0.5 * self.N_con * log_v - 0.5 * \
            jnp.exp(log_v) * \
            jnp.sum(jnp.square(u_xx.flatten() - self.src_col.flatten()))

        # HS prior part loss
        _tau_p, _v_p = self.update_aux(kernel_paras)
        HS_loss = self.entropy_post(
            kernel_paras) + self.exp_prior(kernel_paras, _tau_p, _v_p)

        log_joint = log_prior + log_boundary_ll * self.llk_weight + eq_ll + HS_loss
        return -log_joint

    def exp_prior(self, params_HS, _tau_p, _v_p):
        u_v = params_HS['u_v']
        ln_s_v = params_HS['ln_s_v']
        M_mu = params_HS['M_mu']
        M_U = params_HS['M_U']

        # U = jnp.matmul(jnp.tril(M_U), jnp.tril(M_U).T)
        tril_M_U = jnp.tril(M_U)
        U = jnp.matmul(tril_M_U, jnp.transpose(
            tril_M_U, axes=(0, 2, 1)))  # Q x 2 x 2

        # exp_tau = (-1.5 * M_mu[self.num:, 0] - self.ONE / _tau_p * (
        #     jnp.exp(-M_mu[self.num:, 0] + 0.5 * (jnp.diag(M_U)[self.num:]**2)))).sum()

        exp_tau = (-1.5 * M_mu[:, 0] - self.ONE / _tau_p *
                   (jnp.exp(-M_mu[:, 0] + 0.5 * (M_U[:, 0, 0] ** 2)))).sum()

        # exp_v = (-1.5 * u_v - self.ONE / _v_p *
        #          (jnp.exp(-u_v + 0.5 * jnp.exp(ln_s_v)))).sum()

        # exp_w = (-0.5 * (M_mu[:self.num, 0]**2) / jnp.exp(self.ln_s0)).sum() +
        # (-0.5 * (jnp.trace(U[:self.num, :self.num])) /
        #  jnp.exp(self.ln_s0)).sum()

        exp_w = (-0.5 * (M_mu[:, 1]**2) / jnp.exp(self.ln_s0)).sum() + \
            (-0.5 * (jnp.sum(U[:, 1, 1])) / jnp.exp(self.ln_s0)).sum()

        # return exp_tau + exp_v + exp_w
        return exp_tau + exp_w

    def update_aux(self, params_HS):
        u_v = params_HS['u_v']
        ln_s_v = params_HS['ln_s_v']
        M_mu = params_HS['M_mu']
        M_U = params_HS['M_U']

        # _tau_p = jnp.exp(-M_mu[self.num:, 0] + 0.5 *
        #                  (jnp.diag(M_U)[self.num:])**2) + 1 / self.b0**2

        _tau_p = jnp.exp(-M_mu[:, 0] + 0.5 * M_U[:, 0, 0]
                         ** 2) + 1 / self.b0**2  # Q x 1

        _v_p = jnp.exp(-u_v + 0.5 * jnp.exp(ln_s_v)) + 1 / self.bg**2

        return _tau_p, _v_p

    def entropy_post(self, params_HS):
        u_v = params_HS['u_v']
        ln_s_v = params_HS['ln_s_v']
        M_U = params_HS['M_U']

        tril_M_U = jnp.tril(M_U)
        U = jnp.matmul(tril_M_U, jnp.transpose(
            tril_M_U, axes=(0, 2, 1)))  # Q x 2 x 2

        # U = jnp.matmul(jnp.tril(M_U), jnp.tril(M_U).T)
        # ent_v = (0.5 * ln_s_v + u_v).sum()
        ent_M = 0.5 * jnp.linalg.slogdet(U)[1].sum()
        # return ent_v + ent_M
        return ent_M

    @ partial(jit, static_argnums=(0,))
    def step(self, params, opt_state, key):
        loss = self.loss(params, key)
        loss, d_params = jax.value_and_grad(self.loss)(params, key)
        updates, opt_state = self.optimizer.update(d_params, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @ partial(jit, static_argnums=(0,))
    def preds(self, params, Xte):
        ker_paras = params['kernel_paras']
        u = params['u']

        u = u.sum(axis=1).reshape(-1, 1)  # sum over trick

        x_p = jnp.tile(self.X_con.flatten(), (self.N_con, 1)).T
        X1_p = x_p.flatten()
        X2_p = jnp.transpose(x_p).flatten()
        K = self.kernel_matrix.get_kernel_matrix(X1_p, X2_p, ker_paras)

        Kinv_u = jnp.linalg.solve(K, u)

        N_te = Xte.shape[0]
        x_p11 = jnp.tile(Xte.flatten(), (self.N_con, 1)).T
        x_p22 = jnp.tile(self.X_con.flatten(), (N_te, 1)).T
        X1_p2 = x_p11.flatten()
        X2_p2 = jnp.transpose(x_p22).flatten()
        Kmn = vmap(self.cov_func.kappa, (0, 0, None))(
            X1_p2.flatten(), X2_p2.flatten(), ker_paras).reshape(N_te, self.N_con)

        preds = jnp.matmul(Kmn, Kinv_u)
        return preds,  K

    def train(self, nepoch):
        key = jax.random.PRNGKey(0)
        self.cov_func.update_key(key)

        Q = self.trick_paras['Q']  # number of basis functions

        freq_scale = self.trick_paras['freq_scale']

        # set the initial value of block-M_U as the

        M_U_init = np.zeros((Q, 2, 2))
        M_U_init[:, 0, 0] = 1
        M_U_init[:, 1, 1] = 1

        params = {
            "log_tau": 0.0,  # inv var for data ll
            "log_v": 0.0,  # inv var for eq likelihood
            "kernel_paras":
            {
                "u_v": np.array([0.0]),
                # "u_v": np.log(0.01),
                # "ln_s_v": np.array([0.0]),
                "ln_s_v": np.log(0.01),
                'M_mu': np.zeros((Q, 2)),
                'M_U': M_U_init,
                'log-ls': np.zeros(Q),
                'freq': np.linspace(0, 1, Q)*freq_scale,
            },
            # u value on the collocation points
            "u": self.trick_paras['init_u_trick'](self, self.trick_paras),
        }

        opt_state = self.optimizer.init(params)
        # self.run_lbfgs(params)

        loss_list = []
        err_list = []
        w_list = []
        freq_list = []
        ls_list = []
        epoch_list = []
        K_list = []

        min_err = 2.0
        threshold = 1e-4
        print("here")
        for i in tqdm.tqdm(range(nepoch)):
            key, sub_key = jax.random.split(key)

            self.cov_func.update_key(key)
            # _tau_p, _v_p = self.update_aux(params['kernel_paras'])

            params, opt_state, loss = self.step(params, opt_state, sub_key)

            # evluating the error with frequency epoch/20, store the loss and error in a list

            if i % (nepoch/20) == 0:

                preds,  K = self.preds(params, self.Xte)
                err = jnp.linalg.norm(
                    preds.reshape(-1) - self.yte.reshape(-1))/jnp.linalg.norm(self.yte.reshape(-1))

                if True or err < min_err:
                    if err < min_err:
                        min_err = err
                    print('loss = %g' % loss)
                    print("It ", i, '  loss = %g ' %
                          loss, " Relative L2 error", err)

                print('K.mean:', K.mean())
                print('K.trace:', jnp.trace(K))

                # u_v = params['kernel_paras']['u_v']
                # M_mu = params['kernel_paras']['M_mu']
                # s_M = M_mu
                # s_tau = jnp.exp(s_M[self.num:, 0]).reshape(1, -1)
                # s_w = s_M[:self.num, ].reshape(1, -1)
                # s_v = jnp.exp(u_v)
                # weights = (s_tau * s_v* s_w**2).reshape(-1)
                # weights = (s_tau**0.5  * s_v**0.5 * s_w).reshape(-1)

                _, sub_key1 = jax.random.split(key)
                _, sub_key2 = jax.random.split(key)
                u_v = params['kernel_paras']['u_v']
                ln_s_v = params['kernel_paras']['ln_s_v']
                M_mu = params['kernel_paras']['M_mu']
                M_U = params['kernel_paras']['M_U']
                L = jnp.tril(M_U)
                s_M = M_mu + \
                    jnp.matmul(L, jax.random.normal(
                        sub_key1, shape=(self.num, 2, 1))).squeeze()  # Q x 2

                s_tau = jnp.exp(s_M[:, 0]).reshape(1, -1)

                s_w = s_M[:, 1].reshape(1, -1)

                s_v = jnp.exp(u_v + jax.random.normal(sub_key2,
                                                      shape=(1, )) * jnp.exp(ln_s_v * 0.5))
                # weights = (s_tau * s_v * s_w**2).reshape(-1)
                # weights = (s_tau * s_w**2).reshape(-1)

                weights = (s_tau**0.5 * s_w).reshape(-1)
                weights = jnp.exp(weights)/jnp.sum(jnp.exp(weights))

                # weights = (s_tau * s_w**2).reshape(-1)
                # weights = weights / (jnp.sum(weights))

                # weights = (s_tau**0.5 * s_v**0.5 * s_w).reshape(-1)
                # weights = weights / (jnp.sum(weights))

                # step by step calculation of kernel-diag(trace)
                log_ls = params['kernel_paras']['log-ls']
                freq = params['kernel_paras']['freq']

                d = 0  # zeros((self.N_con, 1)).reshape(-1)

                matern = (1 + jnp.sqrt(5)*d*jnp.exp(log_ls) + 5/3*d**2 *
                          jnp.exp(log_ls)**2)*jnp.exp(-jnp.sqrt(5)*d*jnp.exp(log_ls))

                cosine = jnp.cos(2*jnp.pi*d*freq)

                hand_trace = (weights*matern*cosine).sum()*self.N_con

                print('hand trace: ', hand_trace)

                # weights = weights.reshape(-1)
                # weights = weights / jnp.sum(weights)

                print("\n rel_weights: ", weights)
                # print('s_v: ', s_v)
                # print('s_tau: ', s_tau.reshape(-1))
                # print('s_w: ', s_w.reshape(-1))

                # print("\n selected rel_weights ", weights[np.where(
                #     abs(weights) > 1e-3)[0]], "\n indices ", np.where(abs(weights) > 1e-3)[0])

                loss_list.append(np.log(loss) if loss > 1 else loss)
                err_list.append(err)
                w_list.append(weights)
                freq_list.append(params['kernel_paras']['freq'])
                ls_list.append(np.exp(params['kernel_paras']['log-ls']))
                epoch_list.append(i)
                K_list.append(K)

                if i > 0 and err < threshold:
                    print('get thr of relative l2 loss:  %f,  early stop at epoch %d' % (
                        threshold, i))
                    break

                log_dict = {'loss_list': loss_list, 'err_list': err_list, 'w_list': w_list,
                            'freq_list': freq_list, 'ls_list': ls_list, 'epoch_list': epoch_list, 'K_list': K_list}

                utils.save_paras(self, params, log_dict)

        print('gen fig ...')
        utils.make_fig_1d(self, params, log_dict)


# Poisson source, u_xx = f
# u should be the target function
def get_source_val(u, x_vec):
    return vmap(grad(grad(u, 0), 0), (0))(x_vec)


def test_multi_scale(trick_paras, fix_dict=None):
    # equation
    equation_dict = {
        'poisson1d-mix-sin': lambda x: jnp.sin(5*jnp.pi*x) + jnp.sin(23.7*jnp.pi*x) + jnp.cos(92.3*jnp.pi*x),
        'poisson1d-single-sin': lambda x: jnp.sin(92.3*jnp.pi*x),
        'poisson1d-single-sin-low': lambda x: jnp.sin(jnp.pi*x),
        'poisson1d-x_time_sinx': lambda x: x*jnp.sin(50*jnp.pi*x),
        'poisson1d-x2_add_sinx': lambda x: jnp.sin(72.6 * jnp.pi*x) - 2*(x-0.5)**2,
    }

    u = equation_dict[trick_paras['equation']]

    # u = lambda x: jnp.sin(5*jnp.pi*x) + jnp.sin(23.7*jnp.pi*x) + jnp.cos(92.3*jnp.pi*x)
    # test points
    M = 300
    X_test = np.linspace(0, 1, num=M).reshape(-1, 1)
    Y_test = u(X_test)
    # collocation points
    N_col = 200
    X_col = np.linspace(0, 1, num=N_col).reshape(-1, 1)
    Xind = np.array([0, X_col.shape[0]-1])
    y = jnp.array([u(X_col[Xind[0]]), u(X_col[Xind[1]])]).reshape(-1)
    src_vals = get_source_val(u, X_col.reshape(-1))

    model_PIGP = GPRLatent(Xind, y, X_col, src_vals,  1e-6,
                           X_test, Y_test, trick_paras, fix_dict)
    np.random.seed(123)
    random.seed(123)
    # nepoch = 250000
    nepoch = trick_paras['nepoch']

    model_PIGP.train(nepoch)


if __name__ == '__main__':
    # trick_paras =

    fix_dict_list = [

        # {'log-w':1, 'freq':1, 'log-ls':1},
        {'log-w': 0, 'freq': 0, 'log-ls': 0},

    ]

    trick_list = [

        {'equation': 'poisson1d-x2_add_sinx', 'init_u_trick': init_func.zeros, 'num_u_trick': 1, 'Q': 30, 'lr': 1e-2,
            'llk_weight': 200.0, 'kernel': kernels_new.Block_Sparse_Matern52_Cos_1d, 'nepoch': 100000, 'freq_scale': 100},
        # {'equation': 'poisson1d-single-sin-low', 'init_u_trick': init_func.zeros, 'num_u_trick': 1, 'Q': 30, 'lr': 1e-2,
        #     'llk_weight': 200.0, 'kernel': kernels_new.Block_Sparse_Matern52_Cos_1d, 'nepoch': 20000, 'freq_scale': 50},

    ]

    for trick_paras in trick_list:
        for fix_dict in fix_dict_list:
            test_multi_scale(trick_paras, fix_dict)
