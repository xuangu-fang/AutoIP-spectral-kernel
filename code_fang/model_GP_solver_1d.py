import numpy as np
import random
import matplotlib.pyplot as plt
import optax
import jax
import jax.numpy as jnp

import kernel_matrix
from kernel_matrix import *
from jax import vmap
import pandas as pd
import utils
import copy

import time
import tqdm
import os
import copy
import init_func


'''GP solver class for 1d dynamics with single kernel,
 now support poisson-1d and allen-cahn-1d'''


class GP_solver_1d_single(object):

    # equation: u_{xx}  = f(x)
    # Xind: the indices of X_col that corresponds to training points, i.e., boundary points
    # y: training outputs
    # Xcol: collocation points
    def __init__(self,
                 Xind,
                 y,
                 X_col,
                 src_col,
                 jitter,
                 X_test,
                 Y_test,
                 trick_paras=None,
                 fix_dict=None):
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

        self.optimizer = optax.adam(learning_rate=trick_paras['lr'])

        self.llk_weight = trick_paras['llk_weight']

        self.cov_func = trick_paras['kernel']()

        self.kernel_matrix = Kernel_matrix(self.jitter, self.cov_func)

        self.Xte = X_test
        self.yte = Y_test

        self.params = None  # to be assugned after training the mixture-GP
        self.pred_func = None  # to be assugned when starting prediction

        self.eq_type = trick_paras['equation'].split('-')[0]
        assert self.eq_type in ['poisson_1d', 'allencahn_1d']

        print('equation is: ', self.trick_paras['equation'])
        print('kernel is:', self.cov_func.__class__.__name__)

    @partial(jit, static_argnums=(0, ))
    def value_and_grad_kernel(self, params, key):
        '''compute the value of the kernel matrix, along with Kinv_u and u_xx'''

        u = params['u']  # function values at the collocation points
        kernel_paras = params['kernel_paras']
        x_p = jnp.tile(self.X_con.flatten(), (self.N_con, 1)).T
        X1_p = x_p.flatten()
        X2_p = jnp.transpose(x_p).flatten()
        # only the cov matrix of func vals
        K = self.kernel_matrix.get_kernel_matrix(X1_p, X2_p, kernel_paras)

        Kinv_u = jnp.linalg.solve(K, u)

        K_dxx1 = vmap(self.cov_func.DD_x1_kappa,
                      (0, 0, None))(X1_p, X2_p, kernel_paras).reshape(
                          self.N_con, self.N_con)
        u_xx = jnp.matmul(K_dxx1, Kinv_u)

        return K, Kinv_u, u_xx

    @partial(jit, static_argnums=(0, ))
    def boundary_and_eq_gap(self, u, u_xx):
        """compute the boundary and equation gap, to construct the training loss or computing the early stopping criteria"""
        # boundary
        boundary_gap = jnp.sum(jnp.square(u[self.Xind].reshape(-1) -
                                          self.y.reshape(-1)))
        # equation
        if self.eq_type == 'poisson_1d':

            eq_gap = jnp.sum(jnp.square(
                u_xx.flatten() - self.src_col.flatten()))

        elif self.eq_type == 'allencahn_1d':

            eq_gap = jnp.sum(jnp.square(u_xx.flatten() +
                             (u*(u**2-1)).flatten() - self.src_col.flatten()))

        else:
            raise NotImplementedError

        return boundary_gap, eq_gap

    @partial(jit, static_argnums=(0, ))
    def loss(self, params, key):
        '''compute the loss function'''
        u = params['u']  # function values at the collocation points
        log_tau = params['log_tau']
        log_v = params['log_v']

        K, Kinv_u, u_xx = self.value_and_grad_kernel(params, key)

        boundary_gap, eq_gap = self.boundary_and_eq_gap(u, u_xx)

        # prior
        log_prior = -0.5 * \
            jnp.linalg.slogdet(
                K)[1]*self.trick_paras['logdet'] - 0.5*jnp.sum(u*Kinv_u)

        # boundary
        log_boundary_ll = 0.5 * self.N * log_tau - 0.5 * \
            jnp.exp(
                log_tau) * boundary_gap
        # equation

        eq_ll = 0.5 * self.N_con * log_v - 0.5 * \
            jnp.exp(log_v) * eq_gap

        log_joint = log_prior + log_boundary_ll * self.llk_weight + eq_ll
        return -log_joint

    @partial(jit, static_argnums=(0, ))
    def step(self, params, opt_state, key):
        # loss = self.loss(params, key)
        loss, d_params = jax.value_and_grad(self.loss)(params, key)
        updates, opt_state = self.optimizer.update(d_params, opt_state, params)

        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @partial(jit, static_argnums=(0, ))
    def preds(self, params, Xte):
        ker_paras = params['kernel_paras']
        u = params['u']

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
        Kmn = vmap(self.cov_func.kappa,
                   (0, 0, None))(X1_p2.flatten(), X2_p2.flatten(),
                                 ker_paras).reshape(N_te, self.N_con)
        preds = jnp.matmul(Kmn, Kinv_u)
        return preds, K

    def train(self, nepoch):
        key = jax.random.PRNGKey(0)
        Q = self.trick_paras['Q']  # number of basis functions

        freq_scale = self.trick_paras['freq_scale']

        params = {
            "log_tau": 0.0,  # inv var for data ll
            "log_v": 0.0,  # inv var for eq likelihood
            "kernel_paras": {
                'log-w': np.log(1 / Q) * np.ones(Q),
                'log-ls': np.zeros(Q),
                'freq': np.linspace(0, 1, Q) * freq_scale,
            },
            # u value on the collocation points
            "u": np.zeros((self.N_con, 1))
        }

        # params['kernel_paras']['freq'][0] = 0.5

        opt_state = self.optimizer.init(params)

        loss_list = []
        err_list = []
        w_list = []
        freq_list = []

        ls_list = []
        epoch_list = []

        min_err = 2.0
        threshold = 1e-5
        print("here")

        self.pred_func = self.preds

        # to be assigned later
        opt_state_extra = None
        params_extra = None

        # for i in tqdm.tqdm(range(nepoch)):

        change_point = int(nepoch * self.trick_paras['change_point'])

        for i in range(nepoch):

            key, sub_key = jax.random.split(key)

            if i <= change_point:
                params, opt_state, loss = self.step(params, opt_state, sub_key)

            else:
                params_extra, opt_state_extra, loss = self.step_extra(
                    params_extra, opt_state_extra, sub_key)

            if i == change_point:

                print('start to train the matern kernel')

                self.params = copy.deepcopy(params)

                params_extra = {
                    # "log_tau": 0.0,  # inv var for data ll
                    # "log_v": 0.0,  # inv var for eq likelihood
                    "log_tau":
                    copy.deepcopy(params['log_tau']),  # inv var for data ll
                    "log_v": 0.0,  # inv var for eq likelihood
                    "kernel_paras": {
                        'log-w-matern': np.zeros(1),
                        'log-ls-matern': np.zeros(1),
                    },
                    # u value on the collocation points
                    # "u": copy.deepcopy(params['u']),
                    "u": self.trick_paras['init_u_trick'](self,
                                                          self.trick_paras)
                }

                self.pred_func = self.preds_extra

                opt_state_extra = self.optimizer_extra.init(params_extra)

            # evluating the error with frequency epoch/20, store the loss and error in a list

            if i % (nepoch / 20) == 0:

                # params = params if i <= int(nepoch / 2) else params_extra
                current_params = params if i <= int(
                    change_point) else params_extra
                preds, _ = self.pred_func(current_params, self.Xte)
                err = jnp.linalg.norm(
                    preds.reshape(-1) -
                    self.yte.reshape(-1)) / jnp.linalg.norm(
                        self.yte.reshape(-1))
                if True or err < min_err:
                    if err < min_err:
                        min_err = err
                    print('loss = %g' % loss)
                    print("It ", i, '  loss = %g ' % loss,
                          " Relative L2 error", err)

                loss_list.append(np.log(loss) if loss > 1 else loss)
                err_list.append(err)
                w_list.append(np.exp(params['kernel_paras']['log-w']))
                freq_list.append(params['kernel_paras']['freq'])
                ls_list.append(np.exp(params['kernel_paras']['log-ls']))

                matern_w_list.append(
                    np.exp(current_params['kernel_paras']['log-w-matern']))

                matern_ls_list.append(
                    np.exp(current_params['kernel_paras']['log-ls-matern']))

                epoch_list.append(i)

                if i > 0 and err < threshold:
                    print(
                        'get thr of relative l2 loss:  %f,  early stop at epoch %d'
                        % (threshold, i))
                    break

        log_dict = {
            'loss_list': loss_list,
            'err_list': err_list,
            'w_list': w_list,
            'freq_list': freq_list,
            'ls_list': ls_list,
            'epoch_list': epoch_list,
            'matern_w_list': matern_w_list,
            'matern_ls_list': matern_ls_list,
        }

        print('gen fig ...')
        # other_paras = '-extra-GP'
        other_paras = self.trick_paras[
            'other_paras'] + '-change_point-%.2f' % self.trick_paras[
                'change_point']
        utils.make_fig_1d_extra_GP(self, params_extra, log_dict, other_paras)


'''GP solver for 1d equation with a extra GP, which can accelerate the convergence by capturing the low frequency part of the solution quickly'''


class GP_solver_1d_extra(GP_solver_1d_single):

    def __init__(self, Xind, y, X_col, src_col, jitter, X_test, Y_test, trick_paras=None, fix_dict=None):
        super().__init__(Xind, y, X_col, src_col, jitter,
                         X_test, Y_test, trick_paras, fix_dict)

        print('using extra GP with kernel:',
              self.cov_func_extra.__class__.__name__)

        self.cov_func_extra = trick_paras['kernel_extra']()
        self.kernel_matrix_extra = Kernel_matrix(self.jitter,
                                                 self.cov_func_extra)
        self.optimizer_extra = optax.adam(learning_rate=trick_paras['lr'])

    @partial(jit, static_argnums=(0, ))
    def loss_extra(self, params_extra, key):

        params = self.params
        u = params['u']  # function values at the collocation points
        u = u.sum(axis=1).reshape(-1, 1)  # sum over trick
        log_v = params['log_v']  # inverse variance for eq ll
        log_tau = params['log_tau']  # inverse variance for boundary ll
        kernel_paras = params['kernel_paras']

        u_extra = params_extra[
            'u']  # function values at the collocation points
        u_extra = u_extra.sum(axis=1).reshape(-1, 1)  # sum over trick
        log_v_extra = params_extra['log_v']  # inverse variance for eq ll
        log_tau_extra = params_extra[
            'log_tau']  # inverse variance for boundary ll
        kernel_paras_extra = params_extra['kernel_paras']

        x_p = jnp.tile(self.X_con.flatten(), (self.N_con, 1)).T
        X1_p = x_p.flatten()
        X2_p = jnp.transpose(x_p).flatten()
        # only the cov matrix of func vals
        K = self.kernel_matrix.get_kernel_matrix(X1_p, X2_p, kernel_paras)
        Kinv_u = jnp.linalg.solve(K, u)

        K_extra = self.kernel_matrix_extra.get_kernel_matrix(
            X1_p, X2_p, kernel_paras_extra)
        Kinv_u_extra = jnp.linalg.solve(K_extra, u_extra)

        log_prior = -0.5 * \
            jnp.linalg.slogdet(
                K_extra)[1]*self.trick_paras['logdet'] - 0.5*jnp.sum(u_extra*Kinv_u_extra)
        # log_prior = - 0.5*jnp.sum(u*Kinv_u)

        # boundary
        log_boundary_ll = 0.5 * self.N * log_tau_extra - 0.5 * \
            jnp.exp(
                log_tau_extra) * jnp.sum(jnp.square(u[self.Xind].reshape(-1) + u_extra[self.Xind].reshape(-1) - self.y.reshape(-1)))

        # equation
        K_dxx1 = vmap(self.cov_func.DD_x1_kappa,
                      (0, 0, None))(X1_p, X2_p, kernel_paras).reshape(
                          self.N_con, self.N_con)
        u_xx = jnp.matmul(K_dxx1, Kinv_u)

        K_dxx1_extra = vmap(self.cov_func_extra.DD_x1_kappa, (0, 0, None))(
            X1_p, X2_p, kernel_paras_extra).reshape(self.N_con, self.N_con)
        u_xx_extra = jnp.matmul(K_dxx1_extra, Kinv_u_extra)

        eq_ll = 0.5 * self.N_con * log_v_extra - 0.5 * \
            jnp.exp(log_v_extra) * \
            jnp.sum(jnp.square(u_xx.flatten() +
                    u_xx_extra.flatten() - self.src_col.flatten()))

        log_joint = log_prior + log_boundary_ll * self.llk_weight + eq_ll
        return -log_joint

    @partial(jit, static_argnums=(0, ))
    def step_extra(self, params_extra, opt_state, key):
        # loss = self.loss_extra(params_extra, key)
        loss, d_params = jax.value_and_grad(self.loss_extra)(params_extra, key)
        updates, opt_state = self.optimizer_extra.update(
            d_params, opt_state, params_extra)

        params_extra = optax.apply_updates(params_extra, updates)
        return params_extra, opt_state, loss

    @partial(jit, static_argnums=(0, ))
    def preds_extra(self, params_extra, Xte):

        preds, _ = self.preds(self.params, Xte)

        ker_paras = params_extra['kernel_paras']
        u = params_extra['u']

        u = u.sum(axis=1).reshape(-1, 1)  # sum over trick

        x_p = jnp.tile(self.X_con.flatten(), (self.N_con, 1)).T
        X1_p = x_p.flatten()
        X2_p = jnp.transpose(x_p).flatten()
        K = self.kernel_matrix_extra.get_kernel_matrix(X1_p, X2_p, ker_paras)
        Kinv_u = jnp.linalg.solve(K, u)

        N_te = Xte.shape[0]
        x_p11 = jnp.tile(Xte.flatten(), (self.N_con, 1)).T
        x_p22 = jnp.tile(self.X_con.flatten(), (N_te, 1)).T
        X1_p2 = x_p11.flatten()
        X2_p2 = jnp.transpose(x_p22).flatten()
        Kmn = vmap(self.cov_func_extra.kappa,
                   (0, 0, None))(X1_p2.flatten(), X2_p2.flatten(),
                                 ker_paras).reshape(N_te, self.N_con)

        preds_extra = jnp.matmul(Kmn, Kinv_u)

        preds_all = preds + preds_extra

        return preds_all, None

    def train(self, nepoch):
        key = jax.random.PRNGKey(0)
        Q = self.trick_paras['Q']  # number of basis functions

        freq_scale = self.trick_paras['freq_scale']

        params = {
            "log_tau": 0.0,  # inv var for data ll
            "log_v": 0.0,  # inv var for eq likelihood
            "kernel_paras": {
                'log-w': np.log(1 / Q) * np.ones(Q),
                'log-ls': np.zeros(Q),
                'freq': np.linspace(0, 1, Q) * freq_scale,
                'log-w-matern': np.zeros(1),
                'log-ls-matern': np.zeros(1),
            },
            # u value on the collocation points
            "u": self.trick_paras['init_u_trick'](self, self.trick_paras),
        }

        # params['kernel_paras']['freq'][0] = 0.5

        opt_state = self.optimizer.init(params)
        # self.run_lbfgs(params)

        loss_list = []
        err_list = []
        w_list = []
        freq_list = []

        matern_w_list = []
        matern_ls_list = []

        ls_list = []
        epoch_list = []

        min_err = 2.0
        threshold = 1e-5
        print("here")

        self.pred_func = self.preds

        # to be assigned later
        opt_state_extra = None
        params_extra = None

        # for i in tqdm.tqdm(range(nepoch)):

        change_point = int(nepoch * self.trick_paras['change_point'])

        for i in range(nepoch):

            key, sub_key = jax.random.split(key)

            if i <= change_point:
                params, opt_state, loss = self.step(params, opt_state, sub_key)

            else:
                params_extra, opt_state_extra, loss = self.step_extra(
                    params_extra, opt_state_extra, sub_key)

            if i == change_point:

                print('start to train the matern kernel')

                self.params = copy.deepcopy(params)

                params_extra = {
                    # "log_tau": 0.0,  # inv var for data ll
                    # "log_v": 0.0,  # inv var for eq likelihood
                    "log_tau":
                    copy.deepcopy(params['log_tau']),  # inv var for data ll
                    "log_v": 0.0,  # inv var for eq likelihood
                    "kernel_paras": {
                        'log-w-matern': np.zeros(1),
                        'log-ls-matern': np.zeros(1),
                    },
                    # u value on the collocation points
                    # "u": copy.deepcopy(params['u']),
                    "u": self.trick_paras['init_u_trick'](self,
                                                          self.trick_paras)
                }

                self.pred_func = self.preds_extra

                opt_state_extra = self.optimizer_extra.init(params_extra)

            # evluating the error with frequency epoch/20, store the loss and error in a list

            if i % (nepoch / 20) == 0:

                # params = params if i <= int(nepoch / 2) else params_extra
                current_params = params if i <= int(
                    change_point) else params_extra
                preds, _ = self.pred_func(current_params, self.Xte)
                err = jnp.linalg.norm(
                    preds.reshape(-1) -
                    self.yte.reshape(-1)) / jnp.linalg.norm(
                        self.yte.reshape(-1))
                if True or err < min_err:
                    if err < min_err:
                        min_err = err
                    print('loss = %g' % loss)
                    print("It ", i, '  loss = %g ' % loss,
                          " Relative L2 error", err)

                loss_list.append(np.log(loss) if loss > 1 else loss)
                err_list.append(err)
                w_list.append(np.exp(params['kernel_paras']['log-w']))
                freq_list.append(params['kernel_paras']['freq'])
                ls_list.append(np.exp(params['kernel_paras']['log-ls']))

                matern_w_list.append(
                    np.exp(current_params['kernel_paras']['log-w-matern']))

                matern_ls_list.append(
                    np.exp(current_params['kernel_paras']['log-ls-matern']))

                epoch_list.append(i)

                if i > 0 and err < threshold:
                    print(
                        'get thr of relative l2 loss:  %f,  early stop at epoch %d'
                        % (threshold, i))
                    break

        log_dict = {
            'loss_list': loss_list,
            'err_list': err_list,
            'w_list': w_list,
            'freq_list': freq_list,
            'ls_list': ls_list,
            'epoch_list': epoch_list,
            'matern_w_list': matern_w_list,
            'matern_ls_list': matern_ls_list,
        }

        print('gen fig ...')
        # other_paras = '-extra-GP'
        other_paras = self.trick_paras[
            'other_paras'] + '-change_point-%.2f' % self.trick_paras[
                'change_point']
        utils.make_fig_1d_extra_GP(self, params_extra, log_dict, other_paras)


def get_source_val(u, x_vec):
    return vmap(grad(grad(u, 0), 0), (0))(x_vec)


def test_multi_scale(trick_paras, fix_dict=None):
    # equation
    equation_dict = {
        'poisson1d-mix-sin':
        lambda x: jnp.sin(5 * jnp.pi * x) + jnp.sin(23.7 * jnp.pi * x) + jnp.
        cos(92.3 * jnp.pi * x),
        'poisson1d-single-sin':
        lambda x: jnp.sin(92.3 * jnp.pi * x),
        'poisson1d-single-sin-low':
        lambda x: jnp.sin(jnp.pi * x),
        'poisson1d-x_time_sinx':
        lambda x: x * jnp.sin(50 * jnp.pi * x),
        'poisson1d-x2_add_sinx':
        lambda x: jnp.sin(72.6 * jnp.pi * x) - 2 * (x - 0.5)**2,
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
    Xind = np.array([0, X_col.shape[0] - 1])
    y = jnp.array([u(X_col[Xind[0]]), u(X_col[Xind[1]])]).reshape(-1)
    src_vals = get_source_val(u, X_col.reshape(-1))

    model_PIGP = GPRLatent(Xind, y, X_col, src_vals, 1e-6, X_test, Y_test,
                           trick_paras, fix_dict)
    np.random.seed(123)
    random.seed(123)
    # nepoch = 250000
    nepoch = trick_paras['nepoch']

    model_PIGP.train(nepoch)


if __name__ == '__main__':

    trick_list = [
        {
            'equation': 'poisson1d-x2_add_sinx',
            'init_u_trick': init_func.zeros,
            'num_u_trick': 1,
            'Q': 30,
            'lr': 1e-2,
            'llk_weight': 100.0,
            'kernel': kernels_new.Matern52_Cos_1d,
            'kernel_extra': kernels_new.Matern52_1d,
            'nepoch': 50000,
            'freq_scale': 100,
            'logdet': True,
            'other_paras': '-extra-GP',
            'change_point': 0.5,
            'fold': 5,
        },
    ]

    for trick_paras in trick_list:
        test_multi_scale(trick_paras)
