import numpy as np
import kernels_new
from kernels_new import *
import random
import matplotlib.pyplot as plt
import pickle
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

    # equation: u_{xx} + u_{yy}  = f(x,y)
    # bvals: 1d array, boundary values
    # X_col = (x_pos, y_pos), x_pos: 1d array, y_pos: 1d array
    # src_vals: source values at the collocation mesh, N1 x N2
    # X_test = (x_test_pos, y_test_pos): x_test_pos: 1d array, y_test_pos: 1d array
    # u_test:  M1 x M2
    def __init__(self, bvals, X_col, src_vals, jitter, X_test, u_test, trick_paras=None, fix_dict=None):
        self.bvals = bvals  # Nb dim
        self.X_col = X_col
        self.jitter = jitter
        self.Nb = bvals.size  # number of boundary points
        self.N1 = X_col[0].size
        self.N2 = X_col[1].size
        self.Nc = self.N1*self.N2
        self.src_vals = src_vals  # N1 by N2

        self.trick_paras = trick_paras
        self.fix_dict = fix_dict

        lr = trick_paras['lr'] if trick_paras is not None else 0.01

        self.optimizer = optax.adamw(learning_rate=lr)
        self.optimizer_extra = optax.adamw(learning_rate=lr*0.1)

        self.llk_weight = trick_paras['llk_weight']

        fix_kernel_paras = None

        self.cov_func = trick_paras['kernel'](fix_dict, None)

        self.KM_calc = Kernel_matrix(self.jitter, self.cov_func, 'NONE')

        self.cov_func_extra = trick_paras['kernel_extra'](fix_dict, None)

        self.KM_calc_extra = Kernel_matrix(self.jitter,
                                           self.cov_func_extra, 'NONE')

        self.Xte = X_test
        self.ute = u_test

        self.x_pos_tr_mesh, self.x_pos_tr_mesh_T = np.meshgrid(
            self.X_col[0], self.X_col[0], indexing='ij')
        self.y_pos_tr_mesh, self.y_pos_tr_mesh_T = np.meshgrid(
            self.X_col[1], self.X_col[1], indexing='ij')

        self.params = None  # to be assugned after training the mixture-GP
        self.pred_func = None  # to be assugned when starting prediction

        print('equation is: ', self.trick_paras['equation'])
        print('kernel is:', self.cov_func.__class__.__name__)
        print('kernel_extra is:', self.cov_func_extra.__class__.__name__)
        print('change points are:', self.trick_paras['change_point'])

    @partial(jit, static_argnums=(0,))
    def loss(self, params, key):
        U = params['U']  # function values at the collocation points, N1 X N2
        log_v = params['log_v']  # inverse variance for eq ll
        log_tau = params['log_tau']  # inverse variance for boundary ll
        # ker params for 1st dimension
        kernel_paras_x = params['kernel_paras_1']
        # ker params for 2nd dimension
        kernel_paras_y = params['kernel_paras_2']

        K1 = self.KM_calc.get_kernel_matrix(
            self.x_pos_tr_mesh, self.x_pos_tr_mesh_T, kernel_paras_x)  # N1 x N1
        K2 = self.KM_calc.get_kernel_matrix(
            self.y_pos_tr_mesh, self.y_pos_tr_mesh_T, kernel_paras_y)  # N2 x N2
        K1inv_U = jnp.linalg.solve(K1, U)  # N1 x N2
        K2inv_Ut = jnp.linalg.solve(K2, U.T)  # N2 x N1
        log_prior = -0.5*self.N2*jnp.linalg.slogdet(
            K1)[1]*self.trick_paras['logdet'] - 0.5*self.N1*jnp.linalg.slogdet(K2)[1]*self.trick_paras['logdet'] - 0.5*jnp.sum(K1inv_U*K2inv_Ut.T)

        # boundary
        u_b = jnp.hstack((U[0, :], U[-1, :], U[:, 0], U[:, -1]))
        log_boundary_ll = 0.5 * self.Nb * log_tau - 0.5 * \
            jnp.exp(log_tau) * \
            jnp.sum(jnp.square(u_b.reshape(-1) - self.bvals.reshape(-1)))

        # equation
        K_dxx1 = vmap(self.cov_func.DD_x1_kappa, (0, 0, None))(self.x_pos_tr_mesh.reshape(
            -1), self.x_pos_tr_mesh_T.reshape(-1), kernel_paras_x).reshape(self.N1, self.N1)

        U_xx = jnp.matmul(K_dxx1, K1inv_U)

        K_dyy1 = vmap(self.cov_func.DD_x1_kappa, (0, 0, None))(self.y_pos_tr_mesh.reshape(
            -1), self.y_pos_tr_mesh_T.reshape(-1), kernel_paras_y).reshape(self.N2, self.N2)

        U_yy = jnp.matmul(K_dyy1, K2inv_Ut).T

        eq_ll = 0.5 * self.Nc * log_v - 0.5 * \
            jnp.exp(log_v) * jnp.sum(jnp.square(U_xx +
                                                U_yy + U*(U**2-1) - self.src_vals))
        log_joint = log_prior + log_boundary_ll*self.llk_weight + eq_ll
        return -log_joint

    @partial(jit, static_argnums=(0,))
    def loss_extra(self, params_extra, key):

        params = self.params

        U = params['U']  # function values at the collocation points, N1 X N2
        log_v = params['log_v']  # inverse variance for eq ll
        log_tau = params['log_tau']  # inverse variance for boundary ll
        # ker params for 1st dimension
        kernel_paras_x = params['kernel_paras_1']
        # ker params for 2nd dimension
        kernel_paras_y = params['kernel_paras_2']

        K1 = self.KM_calc.get_kernel_matrix(
            self.x_pos_tr_mesh, self.x_pos_tr_mesh_T, kernel_paras_x)  # N1 x N1
        K2 = self.KM_calc.get_kernel_matrix(
            self.y_pos_tr_mesh, self.y_pos_tr_mesh_T, kernel_paras_y)  # N2 x N2
        K1inv_U = jnp.linalg.solve(K1, U)  # N1 x N2
        K2inv_Ut = jnp.linalg.solve(K2, U.T)  # N2 x N1

        # function values at the collocation points, N1 X N2
        U_extra = params_extra['U']
        log_v_extra = params_extra['log_v']  # inverse variance for eq ll
        # inverse variance for boundary ll
        log_tau_extra = params_extra['log_tau']
        # ker params for 1st dimension
        kernel_paras_x_extra = params_extra['kernel_paras_1']
        # ker params for 2nd dimension
        kernel_paras_y_extra = params_extra['kernel_paras_2']

        K1_extra = self.KM_calc_extra.get_kernel_matrix(
            self.x_pos_tr_mesh, self.x_pos_tr_mesh_T, kernel_paras_x_extra)  # N1 x N1
        K2_extra = self.KM_calc_extra.get_kernel_matrix(
            self.y_pos_tr_mesh, self.y_pos_tr_mesh_T, kernel_paras_y_extra)  # N2 x N2
        K1inv_U_extra = jnp.linalg.solve(K1_extra, U_extra)  # N1 x N2
        K2inv_Ut_extra = jnp.linalg.solve(K2_extra, U_extra.T)  # N2 x N1

        log_prior = -0.5*self.N2*jnp.linalg.slogdet(
            K1_extra)[1]*self.trick_paras['logdet'] - 0.5*self.N1*jnp.linalg.slogdet(K2_extra)[1]*self.trick_paras['logdet'] - 0.5*jnp.sum(K1inv_U_extra*K2inv_Ut_extra.T)

        # boundary
        u_b = jnp.hstack((U[0, :], U[-1, :], U[:, 0], U[:, -1])) + jnp.hstack(
            (U_extra[0, :], U_extra[-1, :], U_extra[:, 0], U_extra[:, -1]))

        log_boundary_ll = 0.5 * self.Nb * log_tau_extra - 0.5 * \
            jnp.exp(log_tau_extra) * \
            jnp.sum(jnp.square(u_b.reshape(-1) - self.bvals.reshape(-1)))

        # equation
        K_dxx1 = vmap(self.cov_func.DD_x1_kappa, (0, 0, None))(self.x_pos_tr_mesh.reshape(
            -1), self.x_pos_tr_mesh_T.reshape(-1), kernel_paras_x).reshape(self.N1, self.N1)

        U_xx = jnp.matmul(K_dxx1, K1inv_U)

        K_dyy1 = vmap(self.cov_func.DD_x1_kappa, (0, 0, None))(self.y_pos_tr_mesh.reshape(
            -1), self.y_pos_tr_mesh_T.reshape(-1), kernel_paras_y).reshape(self.N2, self.N2)

        U_yy = jnp.matmul(K_dyy1, K2inv_Ut).T

        K_dxx1_extra = vmap(self.cov_func_extra.DD_x1_kappa, (0, 0, None))(self.x_pos_tr_mesh.reshape(
            -1), self.x_pos_tr_mesh_T.reshape(-1), kernel_paras_x_extra).reshape(self.N1, self.N1)

        U_xx_extra = jnp.matmul(K_dxx1_extra, K1inv_U_extra)

        K_dyy1_extra = vmap(self.cov_func_extra.DD_x1_kappa, (0, 0, None))(self.y_pos_tr_mesh.reshape(
            -1), self.y_pos_tr_mesh_T.reshape(-1), kernel_paras_y_extra).reshape(self.N2, self.N2)

        U_yy_extra = jnp.matmul(K_dyy1_extra, K2inv_Ut_extra).T

        U_all = U + U_extra

        eq_ll = 0.5 * self.Nc * log_v_extra - 0.5 * \
            jnp.exp(log_v_extra) * jnp.sum(jnp.square(U_xx +
                                                      U_yy + U_xx_extra +
                                                      U_yy_extra + U_all*(U_all**2-1) - self.src_vals))
        log_joint = log_prior + log_boundary_ll*self.llk_weight + eq_ll
        return -log_joint

    @partial(jit, static_argnums=(0,))
    def step(self, params, opt_state, key):
        # loss = self.loss(params, key)
        loss, d_params = jax.value_and_grad(self.loss)(params, key)
        updates, opt_state = self.optimizer.update(d_params, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @partial(jit, static_argnums=(0,))
    def step_extra(self,  params_extra, opt_state, key):
        # loss = self.loss(params, key)
        loss, d_params = jax.value_and_grad(self.loss_extra)(params_extra, key)
        updates, opt_state = self.optimizer_extra.update(
            d_params, opt_state, params_extra)
        params_extra = optax.apply_updates(params_extra, updates)
        return params_extra, opt_state, loss

    @partial(jit, static_argnums=(0,))
    def preds(self, params):
        ker_paras_x = params['kernel_paras_1']
        ker_paras_y = params['kernel_paras_2']
        U = params['U']
        K1 = self.KM_calc.get_kernel_matrix(
            self.x_pos_tr_mesh, self.x_pos_tr_mesh_T, ker_paras_x)
        K1inv_U = jnp.linalg.solve(K1, U)  # N1 x N2

        x_te_cross_mh, x_tr_cross_mh = np.meshgrid(
            self.Xte[0], self.X_col[0], indexing='ij')
        Kmn = vmap(self.cov_func.kappa, (0, 0, None))(x_te_cross_mh.reshape(-1),
                                                      x_tr_cross_mh.reshape(-1), ker_paras_x).reshape(self.Xte[0].size, self.N1)

        M1 = jnp.matmul(Kmn, K1inv_U)

        K2 = self.KM_calc.get_kernel_matrix(
            self.y_pos_tr_mesh, self.y_pos_tr_mesh_T, ker_paras_y)
        M2 = jnp.linalg.solve(K2, M1.T)

        y_te_cross_mh, y_tr_cross_mh = np.meshgrid(
            self.Xte[1], self.X_col[1], indexing='ij')
        Kmn2 = vmap(self.cov_func.kappa, (0, 0, None))(y_te_cross_mh.reshape(-1),
                                                       y_tr_cross_mh.reshape(-1), ker_paras_y).reshape(self.Xte[1].size, self.N2)
        U_pred = jnp.matmul(Kmn2, M2).T
        return U_pred, None

    @partial(jit, static_argnums=(0,))
    def preds_extra(self, params_extra):

        U_pred, _ = self.preds(self.params)

        # U_pred__extra, _ = self.preds(params_extra)

        ker_paras_x_extra = params_extra['kernel_paras_1']
        ker_paras_y_extra = params_extra['kernel_paras_2']
        U = params_extra['U']
        K1 = self.KM_calc_extra.get_kernel_matrix(
            self.x_pos_tr_mesh, self.x_pos_tr_mesh_T, ker_paras_x_extra)
        K1inv_U = jnp.linalg.solve(K1, U)  # N1 x N2

        x_te_cross_mh, x_tr_cross_mh = np.meshgrid(
            self.Xte[0], self.X_col[0], indexing='ij')
        Kmn = vmap(self.cov_func_extra.kappa, (0, 0, None))(x_te_cross_mh.reshape(-1),
                                                            x_tr_cross_mh.reshape(-1), ker_paras_x_extra).reshape(self.Xte[0].size, self.N1)

        M1 = jnp.matmul(Kmn, K1inv_U)

        K2 = self.KM_calc_extra.get_kernel_matrix(
            self.y_pos_tr_mesh, self.y_pos_tr_mesh_T, ker_paras_y_extra)
        M2 = jnp.linalg.solve(K2, M1.T)

        y_te_cross_mh, y_tr_cross_mh = np.meshgrid(
            self.Xte[1], self.X_col[1], indexing='ij')
        Kmn2 = vmap(self.cov_func_extra.kappa, (0, 0, None))(y_te_cross_mh.reshape(-1),
                                                             y_tr_cross_mh.reshape(-1), ker_paras_y_extra).reshape(self.Xte[1].size, self.N2)
        U_pred__extra = jnp.matmul(Kmn2, M2).T

        return U_pred + U_pred__extra, None

    def train(self, nepoch):
        key = jax.random.PRNGKey(0)
        Q = self.trick_paras['Q']  # number of basis functions
        # Q = 100
        # Q = 20
        freq_scale = self.trick_paras['freq_scale']
        params = {
            "log_tau": 0.0,  # inv var for data ll
            "log_v": 0.0,  # inv var for eq likelihood
            # "kernel_paras": {'log-w': np.zeros(Q), 'log-ls': np.zeros(Q), 'freq': np.linspace(0, 1, Q)*100},
            "kernel_paras_1": {'log-w': np.log(1/Q)*np.ones(Q), 'log-ls': np.zeros(Q), 'freq': np.linspace(0.05, 1, Q)*freq_scale, 'log-w-matern': np.zeros(1), 'log-ls-matern': np.zeros(1)},
            "kernel_paras_2": {'log-w': np.log(1/Q)*np.ones(Q), 'log-ls': np.zeros(Q), 'freq': np.linspace(0.05, 1, Q)*freq_scale, 'log-w-matern': np.zeros(1), 'log-ls-matern': np.zeros(1)},
            # u value on the collocation points
            "U": np.zeros((self.N1, self.N2)),
        }

        # params['kernel_paras_1']['freq'][0] = 1/(2*jnp.pi)
        # params['kernel_paras_2']['freq'][0] = 1/(2*jnp.pi)

        opt_state = self.optimizer.init(params)

        loss_list = []
        err_list = []
        w_list_k1 = []
        ls_list_k1 = []
        freq_list_k1 = []

        w_list_k2 = []
        ls_list_k2 = []
        freq_list_k2 = []

        matern_w_list_k1 = []
        matern_ls_list_k1 = []

        matern_w_list_k2 = []
        matern_ls_list_k2 = []

        epoch_list = []

        min_err = 2.0
        min_err_epoch = -1
        threshold = 1e-5

        self.pred_func = self.preds

        # to be assigned later
        opt_state_extra = None
        params_extra = None

        change_point = int(nepoch * self.trick_paras['change_point'])

        print("here")

        for i in tqdm.tqdm(range(nepoch)):
            key, sub_key = jax.random.split(key)

            if i <= change_point:
                params, opt_state, loss = self.step(params, opt_state, sub_key)

            else:
                params_extra, opt_state_extra, loss = self.step_extra(
                    params_extra, opt_state_extra, sub_key)

            if i == change_point:
                print('start to train the matern kernel')
                self.params = copy.deepcopy(params)
                self.pred_func = self.preds_extra

                params_extra = {
                    "log_tau": 0.0,  # inv var for data ll
                    "log_v": 0.0,  # inv var for eq likelihood
                    "kernel_paras_1": {'log-w-matern': np.zeros(1), 'log-ls-matern': np.zeros(1)},
                    "kernel_paras_2": {'log-w-matern': np.zeros(1), 'log-ls-matern': np.zeros(1)},
                    # u value on the collocation points
                    "U": np.zeros((self.N1, self.N2)),
                }

                opt_state_extra = self.optimizer_extra.init(params_extra)

            # evluating the error with epoch/20, store the loss and error in a list
            if i % (nepoch/20) == 0:

                current_params = params if i <= change_point else params_extra

                preds, _ = self.pred_func(current_params)
                err = jnp.linalg.norm(
                    preds.reshape(-1) - self.ute.reshape(-1))/jnp.linalg.norm(self.ute.reshape(-1))
                if True or err < min_err:
                    if err < min_err:
                        min_err = err
                    print('loss = %g' % loss)
                    print("It ", i, '  loss = %g ' %
                          loss, " Relative L2 error", err)

                loss_list.append(np.log(loss) if loss > 1 else loss)
                err_list.append(err)
                epoch_list.append(i)

                w_list_k1.append(np.exp(params['kernel_paras_1']['log-w']))
                ls_list_k1.append(np.exp(params['kernel_paras_1']['log-ls']))
                freq_list_k1.append(params['kernel_paras_1']['freq'])

                w_list_k2.append(np.exp(params['kernel_paras_2']['log-w']))
                ls_list_k2.append(np.exp(params['kernel_paras_2']['log-ls']))
                freq_list_k2.append(params['kernel_paras_2']['freq'])

                matern_w_list_k1.append(
                    np.exp(current_params['kernel_paras_1']['log-w-matern']))
                matern_ls_list_k1.append(
                    np.exp(current_params['kernel_paras_1']['log-ls-matern']))

                matern_w_list_k2.append(
                    np.exp(current_params['kernel_paras_2']['log-w-matern']))
                matern_ls_list_k2.append(
                    np.exp(current_params['kernel_paras_2']['log-ls-matern']))

                if i > 0 and err < threshold:
                    print('get thr of relative l2 loss:  %f,  early stop at epoch %d' % (
                        threshold, i))
                    break

        log_dict = {'loss_list': loss_list,
                    'err_list': err_list, 'epoch_list': epoch_list, 'w_list_k1': w_list_k1, 'ls_list_k1': ls_list_k1,
                    'freq_list_k1': freq_list_k1, 'w_list_k2': w_list_k2, 'ls_list_k2': ls_list_k2, 'freq_list_k2': freq_list_k2,
                    'matern_w_list_k1': matern_w_list_k1, 'matern_ls_list_k1': matern_ls_list_k1,
                    'matern_w_list_k2': matern_w_list_k2, 'matern_ls_list_k2': matern_ls_list_k2}

        print('gen fig ...')
        other_paras = self.trick_paras[
            'other_paras'] + '-change_point-%.2f' % self.trick_paras[
                'change_point']
        utils.make_fig_2d_extra_GP(self, params, log_dict, other_paras)

# Poisson source, u_xx + u_yy = f
# u should be the target function


def get_source_val(u, x_pos, y_pos):
    x_mesh, y_mesh = np.meshgrid(x_pos, y_pos, indexing='ij')
    x_vec = x_mesh.reshape(-1)
    y_vec = y_mesh.reshape(-1)
    return vmap(grad(grad(u, 0), 0), (0, 0))(x_vec, y_vec) + \
        vmap(grad(grad(u, 1), 1), (0, 0))(x_vec, y_vec) + \
        u(x_vec, y_vec)*(u(x_vec, y_vec)**2-1)


def get_mesh_data(u, M1, M2):
    x_coor = np.linspace(0, 1, num=M1)
    y_coor = np.linspace(0, 1, num=M2)
    x_mesh, y_mesh = np.meshgrid(x_coor, y_coor, indexing='ij')
    u_mesh = u(x_mesh, y_mesh)
    return x_coor, y_coor, u_mesh


def get_boundary_vals(u_mesh):
    return jnp.hstack((u_mesh[0, :], u_mesh[-1, :], u_mesh[:, 0], u_mesh[:, -1]))


def test_2d(trick_paras, fix_dict=None):
    # equation
    # u = lambda x,y: jnp.sin(91.3*jnp.pi*x)*jnp.sin(95.7*jnp.pi*y)

    equation_dict = {
        'allencahn2d-sincos': lambda x, y: jnp.sin(91.3*jnp.pi*x)*jnp.sin(95.7*jnp.pi*y),
        'allencahn2d-mix-sincos': lambda x, y: (jnp.sin(x) + 0.1*jnp.sin(20*x) + jnp.cos(100*x))*(jnp.sin(y) + 0.1*jnp.sin(20*y) + jnp.cos(100*y)),
        'allencahn2d-mix-sincos-high': lambda x, y: (jnp.sin(2*jnp.pi*x) + 0.1*jnp.sin(2*jnp.pi*20*x) + jnp.cos(jnp.pi*100*x))*(jnp.sin(2*jnp.pi*y) + 0.1*jnp.sin(2*jnp.pi*20*y) + jnp.cos(jnp.pi*100*y)),
        'allencahn2d-mix-sincos-simple1': lambda x, y: (jnp.sin(2*jnp.pi*20*x) + jnp.cos(jnp.pi*100*x))*(jnp.sin(25*jnp.pi*y) + jnp.cos(jnp.pi*100*y)),
        'allencahn2d-mix-sincos-simple2': lambda x, y: (0.1*jnp.sin(20*x) + jnp.cos(100*x))*(0.1*jnp.sin(20*y) + jnp.cos(100*y)),
        'allencahn2d-mix-sincos-simple3': lambda x, y: (jnp.sin(x) + jnp.cos(100*x))*(jnp.sin(y) + jnp.cos(100*y)),
        'allencahn2d-mix-sincos-simple4': lambda x, y: (jnp.sin(x) + 0.1*jnp.sin(20*x))*(jnp.sin(y) + 0.1*jnp.sin(20*y))
    }

    u = equation_dict[trick_paras['equation']]
    # test points
    M = 300
    x_pos_test, y_pos_test, u_test_mh = get_mesh_data(u, M, M)
    # collocation points in each dim
    N = 200
    x_pos_tr, y_pos_tr, u_mh = get_mesh_data(u, N, N)
    # take the boundary values, 1d array
    bvals = get_boundary_vals(u_mh)
    # take the source values
    src_vals = get_source_val(u, x_pos_tr, y_pos_tr)
    src_vals = src_vals.reshape((x_pos_tr.size, y_pos_tr.size))

    X_test = (x_pos_test, y_pos_test)
    u_test = u_test_mh
    X_col = (x_pos_tr, y_pos_tr)

    model_PIGP = GPRLatent(bvals, X_col, src_vals,  1e-6,
                           X_test, u_test, trick_paras, fix_dict)
    np.random.seed(123)
    random.seed(123)
    # nepoch = 250000
    nepoch = trick_paras['nepoch']
    model_PIGP.train(nepoch)


if __name__ == '__main__':

    fix_dict_list = [

        {'log-w': 0, 'freq': 0, 'log-ls': 0},
        # {'log-w':0, 'freq':1, 'log-ls':0},

    ]

    trick_list = [


        {'equation': 'allencahn2d-mix-sincos',
         'init_u_trick': init_func.zeros,
            'num_u_trick': 1,
            'Q': 30,
            'lr': 1e-2,
            'llk_weight': 1000.0,
            'kernel': kernels_new.Matern52_Cos_1d,
            'kernel_extra': kernels_new.Matern52_1d,
            'nepoch': 30000,
            'freq_scale': 50,
            'logdet': True,
         'other_paras': '-extra-GP-adamw-single-matern',
            'change_point': 0.2,
         },
    ]

    for trick_paras in trick_list:
        for fix_dict in fix_dict_list:
            test_2d(trick_paras, fix_dict)
