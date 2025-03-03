'''re-orginize the kernels.py, make them more readable and easier to use'''

import jax.numpy as jnp
from jax import grad, jit
from functools import partial
from jax.config import config
import jax
import math

config.update("jax_enable_x64", True)
'''base kernel class for 1d'''


class Kernel_1d(object):

    def __init__(self, fix_dict=None, fix_paras=None):
        '''used for analyze the effect of frezzing some kernel parameters, will not be used in the main code'''

        self.fix_dict = fix_dict
        self.fix_paras = fix_paras
        # self.sparse_prior = sparse_prior

    def kappa(self, x1, y1, paras):
        '''empty kernel, rasie error'''
        raise NotImplementedError

    @partial(jit, static_argnums=(0, ))
    def D_x1_kappa(self, x1, y1, paras):  # cov(f'(x1), f(y1))
        val = grad(self.kappa, 0)(x1, y1, paras)
        return val

    @partial(jit, static_argnums=(0, ))
    def DD_x1_kappa(self, x1, y1, paras):  # cov(f''(x1), f(y1))
        val = grad(grad(self.kappa, 0), 0)(x1, y1, paras)
        return val

    @partial(jit, static_argnums=(0, ))
    def D_y1_kappa(self, x1, y1, paras):  # cov(f(x1), f'(y1))
        val = grad(self.kappa, 1)(x1, y1, paras)
        return val

    @partial(jit, static_argnums=(0, ))
    def DD_y1_kappa(self, x1, y1, paras):  # cov(f(x1), f''(y1))
        val = grad(grad(self.kappa, 1), 1)(x1, y1, paras)
        return val

    @partial(jit, static_argnums=(0, ))
    def D_x1_D_y1_kappa(self, x1, y1, paras):  # cov(f'(x1),f'(y1))
        val = grad(grad(self.kappa, 0), 1)(x1, y1, paras)
        return val

    @partial(jit, static_argnums=(0, ))
    def DD_x1_DD_y1_kappa(self, x1, y1, paras):  # cov(f''(x1), f''(y1))
        val = grad(grad(grad(grad(self.kappa, 0), 0), 1), 1)(x1, y1, paras)
        return val

    @partial(jit, static_argnums=(0, ))
    def D_x1_DD_y1_kappa(self, x1, y1, paras):  # cov(f'(x1), f''(y1))
        val = grad(grad(grad(self.kappa, 0), 1), 1)(x1, y1, paras)
        return val

    @partial(jit, static_argnums=(0, ))
    def frezze_paras(self, paras):

        if self.fix_dict is not None and self.fix_paras is not None:

            log_w = self.fix_dict['log-w']*self.fix_paras['log-w'] + \
                (1-self.fix_dict['log-w'])*paras['log-w']
            log_ls = self.fix_dict['log-ls']*self.fix_paras['log-ls'] + \
                (1-self.fix_dict['log-ls'])*paras['log-ls']
            freq = self.fix_dict['freq']*self.fix_paras['freq'] + \
                (1-self.fix_dict['freq'])*paras['freq']

        else:
            log_w = paras['log-w']
            log_ls = paras['log-ls']
            freq = paras['freq']

        return log_w, log_ls, freq

    @partial(jit, static_argnums=(0, ))
    def make_sparse_weight(self, paras):
        key, sub_key1 = jax.random.split(self.key)
        key, sub_key2 = jax.random.split(self.key)
        u_v = paras['u_v']  # gloabl variance - mean
        ln_s_v = paras['ln_s_v']  # gloabl variance - var
        M_mu = paras['M_mu']  # (local variance, Gaussian noise) 2Q*1
        M_U = paras['M_U']  # 2Q x 2Q
        L = jnp.tril(M_U)
        s_M = M_mu + \
            jnp.matmul(L, jax.random.normal(
                sub_key1, shape=(self.num * 2, 1)))  # 2Q*1
        s_tau = jnp.exp(s_M[self.num:, 0]).reshape(1, -1)
        s_w = s_M[:self.num, ].reshape(1, -1)
        s_v = jnp.exp(u_v + jax.random.normal(sub_key2, shape=(1, )) *
                      jnp.exp(ln_s_v * 0.5))
        weights = (s_tau * s_v * s_w**2).reshape(-1)

        # weights = (s_tau**0.5  * s_v**0.5 * s_w).reshape(-1)
        # weights = weights / jnp.sum(weights)

        log_ls = paras['log-ls']
        freq = paras['freq']

        return weights, log_ls, freq

    @partial(jit, static_argnums=(0, ))
    def make_sparse_weight_local_only(self, paras):
        key, sub_key1 = jax.random.split(self.key)
        key, sub_key2 = jax.random.split(self.key)
        # u_v = paras['u_v']  # gloabl variance - mean
        # ln_s_v = paras['ln_s_v']  # gloabl variance - var
        M_mu = paras['M_mu']  # (local variance, Gaussian noise) 2Q*1
        M_U = paras['M_U']  # 2Q x 2Q
        L = jnp.tril(M_U)
        s_M = M_mu + \
            jnp.matmul(L, jax.random.normal(
                sub_key1, shape=(self.num * 2, 1)))  # 2Q*1
        s_tau = jnp.exp(s_M[self.num:, 0]).reshape(1, -1)
        s_w = s_M[:self.num, ].reshape(1, -1)
        # s_v = jnp.exp(u_v + jax.random.normal(sub_key2,
        #               shape=(1, )) * jnp.exp(ln_s_v * 0.5))
        # weights = ( s_v * s_w**2).reshape(-1)

        # weights = (s_tau**0.5 * s_w).reshape(-1)
        # weights = weights / jnp.sum(weights)

        weights = (s_tau**0.5 * s_w).reshape(-1)
        weights = jnp.exp(weights) / jnp.sum(jnp.exp(weights))

        log_ls = paras['log-ls']
        freq = paras['freq']

        return weights, log_ls, freq

    @partial(jit, static_argnums=(0, ))
    def make_block_sparse_weight(self, paras):
        key, sub_key1 = jax.random.split(self.key)
        key, sub_key2 = jax.random.split(self.key)
        u_v = paras['u_v']  # gloabl variance - mean
        ln_s_v = paras['ln_s_v']  # gloabl variance - var
        M_mu = paras['M_mu']  # (local variance, Gaussian noise) Q x 2
        M_U = paras['M_U']  # Q x 2 x2
        L = jnp.tril(M_U)  # Q x 2 x2

        s_M = M_mu + \
            jnp.matmul(L, jax.random.normal(
                sub_key1, shape=(self.num, 2, 1))).squeeze()  # Q x 2

        s_tau = jnp.exp(s_M[:, 0]).reshape(1, -1)

        s_w = s_M[:, 1].reshape(1, -1)

        s_v = jnp.exp(u_v + jax.random.normal(sub_key2, shape=(1, )) *
                      jnp.exp(ln_s_v * 0.5))

        # weights = (s_tau * s_v * s_w**2).reshape(-1)

        weights = (s_tau**0.5 * s_w).reshape(-1)
        weights = jnp.exp(weights) / jnp.sum(jnp.exp(weights))

        # weights = (s_tau**0.5 * s_v**0.5 * s_w).reshape(-1)
        # weights = (s_tau * s_w**2).reshape(-1)
        # weights = weights / (jnp.sum(weights))

        log_ls = paras['log-ls']
        freq = paras['freq']

        return weights, log_ls, freq

    def update_key(self, key):
        self.key = key

    def update_fix_set(self, fix_dict, fix_paras):
        self.fix_dict = fix_dict
        self.fix_paras = fix_paras


class Block_Sparse_Matern52_Cos_1d(Kernel_1d):
    ''' Block Sparse HS_prior +  variant Specture Mixsure kernal:
      weight x SE x cosine kernel'''

    def __init__(self, fix_dict=None, fix_paras=None, Q=30):
        super().__init__(fix_dict, fix_paras)
        self.num = Q

    @partial(jit, static_argnums=(0, ))
    def kappa(self, x1, y1, paras):

        weights, log_ls, freq = self.make_block_sparse_weight(paras)

        d = jnp.abs(x1 - y1)

        matern = (1 + jnp.sqrt(5) * d * jnp.exp(log_ls) +
                  5 / 3 * d**2 * jnp.exp(log_ls)**2) * jnp.exp(
                      -jnp.sqrt(5) * d * jnp.exp(log_ls))

        cosine = jnp.cos(2 * jnp.pi * d * freq)

        return (weights * matern * cosine).sum()


class Block_Sparse_SE_Cos_1d(Kernel_1d):
    ''' Block Sparse HS_prior +  variant Specture Mixsure kernal:
      weight x Matern52 x cosine kernel'''

    def __init__(self, fix_dict=None, fix_paras=None, Q=30):
        super().__init__(fix_dict, fix_paras)
        self.num = Q

    @partial(jit, static_argnums=(0, ))
    def kappa(self, x1, y1, paras):

        weights, log_ls, freq = self.make_block_sparse_weight(paras)

        d = jnp.abs(x1 - y1)

        # matern = (1 + jnp.sqrt(5)*d*jnp.exp(log_ls) + 5/3*d**2 *
        #   jnp.exp(log_ls)**2)*jnp.exp(-jnp.sqrt(5)*d*jnp.exp(log_ls))

        SE = jnp.exp(-d**2 * jnp.exp(log_ls))

        cosine = jnp.cos(2 * jnp.pi * d * freq)

        return (weights * SE * cosine).sum()


class Sparse_Matern52_Cos_1d(Kernel_1d):
    ''' Sparse HS_prior +  variant Specture Mixsure kernal:
      weight x Matern52 x cosine kernel'''

    def __init__(self, fix_dict=None, fix_paras=None, Q=30):
        super().__init__(fix_dict, fix_paras)
        self.num = Q

    @partial(jit, static_argnums=(0, ))
    def kappa(self, x1, y1, paras):

        weights, log_ls, freq = self.make_sparse_weight(paras)

        # softmax weights
        # weights = jnp.exp(weights)/jnp.sum(jnp.exp(weights))

        # rescale weights
        # weights = weights / jnp.sum(weights)

        d = jnp.abs(x1 - y1)

        matern = (1 + jnp.sqrt(5) * d * jnp.exp(log_ls) +
                  5 / 3 * d**2 * jnp.exp(log_ls)**2) * jnp.exp(
                      -jnp.sqrt(5) * d * jnp.exp(log_ls))

        cosine = jnp.cos(2 * jnp.pi * d * freq)

        return (weights * matern * cosine).sum()


class Local_Sparse_Matern52_Cos_1d(Kernel_1d):
    ''' Sparse HS_prior +  variant Specture Mixsure kernal:
      weight x Matern52 x cosine kernel'''

    def __init__(self, fix_dict=None, fix_paras=None, Q=30):
        super().__init__(fix_dict, fix_paras)
        self.num = Q

    @partial(jit, static_argnums=(0, ))
    def kappa(self, x1, y1, paras):

        weights, log_ls, freq = self.make_sparse_weight_local_only(paras)

        # softmax weights
        # weights = jnp.exp(weights)/jnp.sum(jnp.exp(weights))

        # rescale weights
        # weights = weights / jnp.sum(weights)

        d = jnp.abs(x1 - y1)

        matern = (1 + jnp.sqrt(5) * d * jnp.exp(log_ls) +
                  5 / 3 * d**2 * jnp.exp(log_ls)**2) * jnp.exp(
                      -jnp.sqrt(5) * d * jnp.exp(log_ls))

        cosine = jnp.cos(2 * jnp.pi * d * freq)

        return (weights * matern * cosine).sum()


class Sparse_SE_Cos_1d(Kernel_1d):
    ''' Spaese HS_prior +  variant Specture Mixsure kernal:
      weight x SE x cosine kernel'''

    def __init__(self, fix_dict=None, fix_paras=None, Q=30):
        super().__init__(fix_dict, fix_paras)
        self.num = Q

    @partial(jit, static_argnums=(0, ))
    def kappa(self, x1, y1, paras):

        weights, log_ls, freq = self.make_sparse_weight(paras)

        # softmax weights
        # weights = jnp.exp(weights)/jnp.sum(jnp.exp(weights))

        # rescale weights
        # weights = weights / jnp.sum(weights)

        d = jnp.abs(x1 - y1)

        SE = jnp.exp(-d**2 * jnp.exp(log_ls))

        cosine = jnp.cos(2 * jnp.pi * d * freq)

        return (weights * SE * cosine).sum()


class SE_Cos_1d(Kernel_1d):
    '''standard Specture Mixsure kernal:
      weight x SE kernel x cosine kernel'''

    def __init__(self, fix_dict=None, fix_paras=None):
        super().__init__(fix_dict, fix_paras)

    @partial(jit, static_argnums=(0, ))
    def kappa(self, x1, y1, paras):

        log_w, log_ls, freq = self.frezze_paras(paras)

        return (jnp.exp(log_w) * jnp.exp(-(x1 - y1)**2 * jnp.exp(log_ls)) *
                jnp.cos(2 * jnp.pi * (x1 - y1) * freq)).sum()


class Matern52_Cos_1d(Kernel_1d):
    '''variant Specture Mixsure kernal:
      weight x Matern52 x cosine kernel'''

    def __init__(self, fix_dict=None, fix_paras=None):
        super().__init__(fix_dict, fix_paras)

    @partial(jit, static_argnums=(0, ))
    def kappa(self, x1, y1, paras):

        log_w, log_ls, freq = self.frezze_paras(paras)

        d = jnp.abs(x1 - y1)

        matern = (1 + jnp.sqrt(5) * d * jnp.exp(log_ls) +
                  5 / 3 * d**2 * jnp.exp(log_ls)**2) * jnp.exp(
                      -jnp.sqrt(5) * d * jnp.exp(log_ls))

        cosine = jnp.cos(2 * jnp.pi * d * freq)

        weights = jnp.exp(log_w)

        # rescale weights
        # weights = weights / jnp.sum(weights)

        # softmax weights
        # weights = jnp.exp(weights)/jnp.sum(jnp.exp(weights))

        return (weights * matern * cosine).sum()


class Matern52_Cos_add_Matern_1d(Kernel_1d):
    '''variant Specture Mixsure kernal:
      weight x Matern52 x cosine kernel + seperate Matern52 kernel'''

    def __init__(self, fix_dict=None, fix_paras=None):
        super().__init__(fix_dict, fix_paras)

    @partial(jit, static_argnums=(0, ))
    def frezze_paras_add_Matern(self, paras):

        if self.fix_dict is not None and self.fix_paras is not None:

            log_w = self.fix_dict['log-w']*self.fix_paras['log-w'] + \
                (1-self.fix_dict['log-w'])*paras['log-w']
            log_ls = self.fix_dict['log-ls']*self.fix_paras['log-ls'] + \
                (1-self.fix_dict['log-ls'])*paras['log-ls']
            freq = self.fix_dict['freq']*self.fix_paras['freq'] + \
                (1-self.fix_dict['freq'])*paras['freq']

            log_w_matern = self.fix_dict['log-w-matern']*self.fix_paras['log-w-matern'] + \
                (1-self.fix_dict['log-w-matern'])*paras['log-w-matern']
            log_ls_matern = self.fix_dict['log-ls-matern']*self.fix_paras['log-ls-matern'] + \
                (1-self.fix_dict['log-ls-matern'])*paras['log-ls-matern']

        else:
            log_w = paras['log-w']
            log_ls = paras['log-ls']
            freq = paras['freq']
            log_w_matern = paras['log-w-matern']
            log_ls_matern = paras['log-ls-matern']

        return log_w, log_ls, freq, log_w_matern, log_ls_matern

    @partial(jit, static_argnums=(0, ))
    def kappa(self, x1, y1, paras):

        log_w, log_ls, freq, log_w_matern, log_ls_matern = self.frezze_paras_add_Matern(
            paras)

        d = jnp.abs(x1 - y1)

        matern_coef = (1 + jnp.sqrt(5) * d * jnp.exp(log_ls) +
                       5 / 3 * d**2 * jnp.exp(log_ls)**2) * jnp.exp(
                           -jnp.sqrt(5) * d * jnp.exp(log_ls))

        cosine = jnp.cos(2 * jnp.pi * d * freq)

        matern_single = (1 + jnp.sqrt(5) * d * jnp.exp(log_ls_matern) +
                         5 / 3 * d**2 * jnp.exp(log_ls_matern)**2) * jnp.exp(
                             -jnp.sqrt(5) * d * jnp.exp(log_ls_matern))

        # add with sepearate matern kernel
        return (jnp.exp(log_w) * cosine * matern_coef).sum() + (
            jnp.exp(log_w_matern) * matern_single).sum()


class Matern52_1d(Kernel_1d):

    def __init__(self, fix_dict=None, fix_paras=None):
        super().__init__(fix_dict, fix_paras)

    @partial(jit, static_argnums=(0, ))
    def kappa(self, x1, y1, paras):

        log_w_matern = paras['log-w-matern']
        # log_w_matern = 1.0

        log_ls_matern = paras['log-ls-matern']
        d = jnp.abs(x1 - y1)

        matern_single = (1 + jnp.sqrt(5) * d * jnp.exp(log_ls_matern) +
                         5 / 3 * d**2 * jnp.exp(log_ls_matern)**2) * jnp.exp(
                             -jnp.sqrt(5) * d * jnp.exp(log_ls_matern))

        return (jnp.exp(log_w_matern) * matern_single).sum()
