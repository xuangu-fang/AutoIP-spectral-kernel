import jax.numpy as jnp
from jax import grad, jit
from functools import partial
from jax.config import config
import math

config.update("jax_enable_x64", True)





class Matern52_add_poly_1d(object):

    def __init__(self,fix_dict, fix_paras):

        self.fix_dict = fix_dict
        self.fix_paras = fix_paras

    @partial(jit, static_argnums=(0, ))
    def kappa(self, x1, y1, paras):

        log_w = self.fix_dict['log-w']*self.fix_paras['log-w'] + (1-self.fix_dict['log-w'])*paras['log-w']
        log_ls = self.fix_dict['log-ls']*self.fix_paras['log-ls'] + (1-self.fix_dict['log-ls'])*paras['log-ls']
        freq = self.fix_dict['freq']*self.fix_paras['freq'] + (1-self.fix_dict['freq'])*paras['freq']

        log_w_matern = paras['log-w-matern']
        log_ls_matern = paras['log-ls-matern']
        bias = paras['bias-poly']


        # return (jnp.exp(paras['log-w'])*jnp.exp(-(x1-y1)**2*jnp.exp(paras['log-ls']))*jnp.cos(2*jnp.pi*(x1-y1)*paras['freq'])).sum()

        d = jnp.abs(x1-y1)
        matern_coef = (1 + jnp.sqrt(5)*d*jnp.exp(log_ls) + 5/3*d**2*jnp.exp(log_ls)**2)*jnp.exp(-jnp.sqrt(5)*d*jnp.exp(log_ls))
        cosine = jnp.cos(2*jnp.pi*d*freq)


         # add with sepearate polynomial kernel

        #  type 1
        return (jnp.exp(log_w)*cosine*matern_coef).sum() + (jnp.exp(log_w_matern)*(x1*y1-jnp.exp(log_ls_matern))**2).sum()


        #  type 2
        # return (jnp.exp(log_w)*cosine*matern_coef).sum() + (jnp.exp(log_w_matern)*(x1*y1*jnp.exp(log_ls_matern)-bias)**2).sum()


    @partial(jit, static_argnums=(0, ))
    def D_x1_kappa(self, x1, y1, paras): #cov(f'(x1), f(y1))
        val = grad(self.kappa, 0)(x1, y1, paras)
        return val

    @partial(jit, static_argnums=(0, ))
    def DD_x1_kappa(self, x1, y1, paras): #cov(f''(x1), f(y1))
        val = grad(grad(self.kappa, 0), 0)(x1, y1, paras)
        return val

    @partial(jit, static_argnums=(0, ))
    def D_y1_kappa(self, x1, y1, paras): #cov(f(x1), f'(y1))
        val = grad(self.kappa, 1)(x1, y1, paras)
        return val

    @partial(jit, static_argnums=(0, ))
    def DD_y1_kappa(self, x1, y1, paras): #cov(f(x1), f''(y1))
        val = grad(grad(self.kappa, 1), 1)(x1, y1, paras)
        return val

    @partial(jit, static_argnums=(0, ))
    def D_x1_D_y1_kappa(self, x1, y1, paras): #cov(f'(x1),f'(y1))
        val = grad(grad(self.kappa, 0), 1)(x1, y1, paras)
        return val
    @partial(jit, static_argnums=(0, ))
    def DD_x1_DD_y1_kappa(self, x1, y1, paras): #cov(f''(x1), f''(y1))
        val = grad(grad(grad(grad(self.kappa, 0), 0), 1),1)(x1, y1, paras)
        return val

    @partial(jit, static_argnums=(0, ))
    def D_x1_DD_y1_kappa(self, x1, y1, paras): #cov(f'(x1), f''(y1))
        val = grad(grad(grad(self.kappa, 0), 1), 1)(x1, y1, paras)
        return val




class Matern52_add_Cos_1d(object):
    """Matern 5/2 kernel times Cosine kernel + Matern, allow fix some parameters"""
    """set the sigma^2 in Matern kernel as constant 1, lengthscale as 1/rho"""

    def __init__(self,fix_dict, fix_paras):

        self.fix_dict = fix_dict
        self.fix_paras = fix_paras

    @partial(jit, static_argnums=(0, ))
    def kappa(self, x1, y1, paras):

        log_w = self.fix_dict['log-w']*self.fix_paras['log-w'] + (1-self.fix_dict['log-w'])*paras['log-w']
        log_ls = self.fix_dict['log-ls']*self.fix_paras['log-ls'] + (1-self.fix_dict['log-ls'])*paras['log-ls']
        freq = self.fix_dict['freq']*self.fix_paras['freq'] + (1-self.fix_dict['freq'])*paras['freq']

        log_w_matern = paras['log-w-matern']
        log_ls_matern = paras['log-ls-matern']
        bias = paras['bias-poly']


        # return (jnp.exp(paras['log-w'])*jnp.exp(-(x1-y1)**2*jnp.exp(paras['log-ls']))*jnp.cos(2*jnp.pi*(x1-y1)*paras['freq'])).sum()

        d = jnp.abs(x1-y1)
        matern_coef = (1 + jnp.sqrt(5)*d*jnp.exp(log_ls) + 5/3*d**2*jnp.exp(log_ls)**2)*jnp.exp(-jnp.sqrt(5)*d*jnp.exp(log_ls))
        cosine = jnp.cos(2*jnp.pi*d*freq)

        matern_single = (1 + jnp.sqrt(5)*d*jnp.exp(log_ls_matern) + 5/3*d**2*jnp.exp(log_ls_matern)**2)*jnp.exp(-jnp.sqrt(5)*d*jnp.exp(log_ls_matern))

        # add with sepearate matern kernel
        return (jnp.exp(log_w)*cosine*matern_coef).sum() + (jnp.exp(log_w_matern)*matern_single).sum()


    @partial(jit, static_argnums=(0, ))
    def D_x1_kappa(self, x1, y1, paras): #cov(f'(x1), f(y1))
        val = grad(self.kappa, 0)(x1, y1, paras)
        return val

    @partial(jit, static_argnums=(0, ))
    def DD_x1_kappa(self, x1, y1, paras): #cov(f''(x1), f(y1))
        val = grad(grad(self.kappa, 0), 0)(x1, y1, paras)
        return val

    @partial(jit, static_argnums=(0, ))
    def D_y1_kappa(self, x1, y1, paras): #cov(f(x1), f'(y1))
        val = grad(self.kappa, 1)(x1, y1, paras)
        return val

    @partial(jit, static_argnums=(0, ))
    def DD_y1_kappa(self, x1, y1, paras): #cov(f(x1), f''(y1))
        val = grad(grad(self.kappa, 1), 1)(x1, y1, paras)
        return val

    @partial(jit, static_argnums=(0, ))
    def D_x1_D_y1_kappa(self, x1, y1, paras): #cov(f'(x1),f'(y1))
        val = grad(grad(self.kappa, 0), 1)(x1, y1, paras)
        return val
    @partial(jit, static_argnums=(0, ))
    def DD_x1_DD_y1_kappa(self, x1, y1, paras): #cov(f''(x1), f''(y1))
        val = grad(grad(grad(grad(self.kappa, 0), 0), 1),1)(x1, y1, paras)
        return val

    @partial(jit, static_argnums=(0, ))
    def D_x1_DD_y1_kappa(self, x1, y1, paras): #cov(f'(x1), f''(y1))
        val = grad(grad(grad(self.kappa, 0), 1), 1)(x1, y1, paras)
        return val



class Matern52_Cos_1d(object):
    """Matern 5/2 kernel times Cosine kernel, allow fix some parameters"""
    """set the sigma^2 in Matern kernel as constant 1, lengthscale as 1/rho"""

    def __init__(self,fix_dict, fix_paras):

        self.fix_dict = fix_dict
        self.fix_paras = fix_paras

    @partial(jit, static_argnums=(0, ))
    def kappa(self, x1, y1, paras):

        log_w = self.fix_dict['log-w']*self.fix_paras['log-w'] + (1-self.fix_dict['log-w'])*paras['log-w']
        log_ls = self.fix_dict['log-ls']*self.fix_paras['log-ls'] + (1-self.fix_dict['log-ls'])*paras['log-ls']
        freq = self.fix_dict['freq']*self.fix_paras['freq'] + (1-self.fix_dict['freq'])*paras['freq']

        # return (jnp.exp(paras['log-w'])*jnp.exp(-(x1-y1)**2*jnp.exp(paras['log-ls']))*jnp.cos(2*jnp.pi*(x1-y1)*paras['freq'])).sum()

        d = jnp.abs(x1-y1)
        matern = (1 + jnp.sqrt(5)*d*jnp.exp(log_ls) + 5/3*d**2*jnp.exp(log_ls)**2)*jnp.exp(-jnp.sqrt(5)*d*jnp.exp(log_ls))
        cosine = jnp.cos(2*jnp.pi*d*freq)

        return (jnp.exp(log_w)*matern*cosine).sum()

    @partial(jit, static_argnums=(0, ))
    def D_x1_kappa(self, x1, y1, paras): #cov(f'(x1), f(y1))
        val = grad(self.kappa, 0)(x1, y1, paras)
        return val

    @partial(jit, static_argnums=(0, ))
    def DD_x1_kappa(self, x1, y1, paras): #cov(f''(x1), f(y1))
        val = grad(grad(self.kappa, 0), 0)(x1, y1, paras)
        return val

    @partial(jit, static_argnums=(0, ))
    def D_y1_kappa(self, x1, y1, paras): #cov(f(x1), f'(y1))
        val = grad(self.kappa, 1)(x1, y1, paras)
        return val

    @partial(jit, static_argnums=(0, ))
    def DD_y1_kappa(self, x1, y1, paras): #cov(f(x1), f''(y1))
        val = grad(grad(self.kappa, 1), 1)(x1, y1, paras)
        return val

    @partial(jit, static_argnums=(0, ))
    def D_x1_D_y1_kappa(self, x1, y1, paras): #cov(f'(x1),f'(y1))
        val = grad(grad(self.kappa, 0), 1)(x1, y1, paras)
        return val
    @partial(jit, static_argnums=(0, ))
    def DD_x1_DD_y1_kappa(self, x1, y1, paras): #cov(f''(x1), f''(y1))
        val = grad(grad(grad(grad(self.kappa, 0), 0), 1),1)(x1, y1, paras)
        return val

    @partial(jit, static_argnums=(0, ))
    def D_x1_DD_y1_kappa(self, x1, y1, paras): #cov(f'(x1), f''(y1))
        val = grad(grad(grad(self.kappa, 0), 1), 1)(x1, y1, paras)
        return val




class SM_kernel_u_1d_fix(object):

    def __init__(self,fix_dict, fix_paras):

        self.fix_dict = fix_dict
        self.fix_paras = fix_paras

    @partial(jit, static_argnums=(0, ))
    def kappa(self, x1, y1, paras):

        log_w = self.fix_dict['log-w']*self.fix_paras['log-w'] + (1-self.fix_dict['log-w'])*paras['log-w']
        log_ls = self.fix_dict['log-ls']*self.fix_paras['log-ls'] + (1-self.fix_dict['log-ls'])*paras['log-ls']
        freq = self.fix_dict['freq']*self.fix_paras['freq'] + (1-self.fix_dict['freq'])*paras['freq']

        # return (jnp.exp(paras['log-w'])*jnp.exp(-(x1-y1)**2*jnp.exp(paras['log-ls']))*jnp.cos(2*jnp.pi*(x1-y1)*paras['freq'])).sum()
        return (jnp.exp(log_w)*jnp.exp(-(x1-y1)**2*jnp.exp(log_ls))*jnp.cos(2*jnp.pi*(x1-y1)*freq)).sum()

    @partial(jit, static_argnums=(0, ))
    def D_x1_kappa(self, x1, y1, paras): #cov(f'(x1), f(y1))
        val = grad(self.kappa, 0)(x1, y1, paras)
        return val

    @partial(jit, static_argnums=(0, ))
    def DD_x1_kappa(self, x1, y1, paras): #cov(f''(x1), f(y1))
        val = grad(grad(self.kappa, 0), 0)(x1, y1, paras)
        return val

    @partial(jit, static_argnums=(0, ))
    def D_y1_kappa(self, x1, y1, paras): #cov(f(x1), f'(y1))
        val = grad(self.kappa, 1)(x1, y1, paras)
        return val

    @partial(jit, static_argnums=(0, ))
    def DD_y1_kappa(self, x1, y1, paras): #cov(f(x1), f''(y1))
        val = grad(grad(self.kappa, 1), 1)(x1, y1, paras)
        return val

    @partial(jit, static_argnums=(0, ))
    def D_x1_D_y1_kappa(self, x1, y1, paras): #cov(f'(x1),f'(y1))
        val = grad(grad(self.kappa, 0), 1)(x1, y1, paras)
        return val
    @partial(jit, static_argnums=(0, ))
    def DD_x1_DD_y1_kappa(self, x1, y1, paras): #cov(f''(x1), f''(y1))
        val = grad(grad(grad(grad(self.kappa, 0), 0), 1),1)(x1, y1, paras)
        return val

    @partial(jit, static_argnums=(0, ))
    def D_x1_DD_y1_kappa(self, x1, y1, paras): #cov(f'(x1), f''(y1))
        val = grad(grad(grad(self.kappa, 0), 1), 1)(x1, y1, paras)
        return val


class SM_kernel_u_1d(object):

    def __init__(self):
        pass

    @partial(jit, static_argnums=(0, ))
    def kappa(self, x1, y1, paras):
        return (jnp.exp(paras['log-w'])*jnp.exp(-(x1-y1)**2*jnp.exp(paras['log-ls']))*jnp.cos(2*jnp.pi*(x1-y1)*paras['freq'])).sum()

    @partial(jit, static_argnums=(0, ))
    def D_x1_kappa(self, x1, y1, paras): #cov(f'(x1), f(y1))
        val = grad(self.kappa, 0)(x1, y1, paras)
        return val

    @partial(jit, static_argnums=(0, ))
    def DD_x1_kappa(self, x1, y1, paras): #cov(f''(x1), f(y1))
        val = grad(grad(self.kappa, 0), 0)(x1, y1, paras)
        return val

    @partial(jit, static_argnums=(0, ))
    def D_y1_kappa(self, x1, y1, paras): #cov(f(x1), f'(y1))
        val = grad(self.kappa, 1)(x1, y1, paras)
        return val

    @partial(jit, static_argnums=(0, ))
    def DD_y1_kappa(self, x1, y1, paras): #cov(f(x1), f''(y1))
        val = grad(grad(self.kappa, 1), 1)(x1, y1, paras)
        return val

    @partial(jit, static_argnums=(0, ))
    def D_x1_D_y1_kappa(self, x1, y1, paras): #cov(f'(x1),f'(y1))
        val = grad(grad(self.kappa, 0), 1)(x1, y1, paras)
        return val
    @partial(jit, static_argnums=(0, ))
    def DD_x1_DD_y1_kappa(self, x1, y1, paras): #cov(f''(x1), f''(y1))
        val = grad(grad(grad(grad(self.kappa, 0), 0), 1),1)(x1, y1, paras)
        return val

    @partial(jit, static_argnums=(0, ))
    def D_x1_DD_y1_kappa(self, x1, y1, paras): #cov(f'(x1), f''(y1))
        val = grad(grad(grad(self.kappa, 0), 1), 1)(x1, y1, paras)
        return val



class Periodic_kernel_u_1d(object):

    def __init__(self):
        pass

    @partial(jit, static_argnums=(0, ))
    def kappa(self, x1, y1, paras):
        # return  (jnp.exp(paras['log-w'])*jnp.exp(-(jnp.sin(jnp.pi*(x1-y1)*paras['freq'])**2)*jnp.exp(paras['log-ls']))).sum()
        return (jnp.exp(paras['log-w'])*jnp.exp(-(x1-y1)**2*jnp.exp(paras['log-ls']))*jnp.cos(jnp.pi*(x1-y1)*paras['freq'])).sum()
        
    

    @partial(jit, static_argnums=(0, ))
    def D_x1_kappa(self, x1, y1, paras): #cov(f'(x1), f(y1))
        val = grad(self.kappa, 0)(x1, y1, paras)
        return val

    @partial(jit, static_argnums=(0, ))
    def DD_x1_kappa(self, x1, y1, paras): #cov(f''(x1), f(y1))
        val = grad(grad(self.kappa, 0), 0)(x1, y1, paras)
        return val

    @partial(jit, static_argnums=(0, ))
    def D_y1_kappa(self, x1, y1, paras): #cov(f(x1), f'(y1))
        val = grad(self.kappa, 1)(x1, y1, paras)
        return val

    @partial(jit, static_argnums=(0, ))
    def DD_y1_kappa(self, x1, y1, paras): #cov(f(x1), f''(y1))
        val = grad(grad(self.kappa, 1), 1)(x1, y1, paras)
        return val

    @partial(jit, static_argnums=(0, ))
    def D_x1_D_y1_kappa(self, x1, y1, paras): #cov(f'(x1),f'(y1))
        val = grad(grad(self.kappa, 0), 1)(x1, y1, paras)
        return val
    @partial(jit, static_argnums=(0, ))
    def DD_x1_DD_y1_kappa(self, x1, y1, paras): #cov(f''(x1), f''(y1))
        val = grad(grad(grad(grad(self.kappa, 0), 0), 1),1)(x1, y1, paras)
        return val

    @partial(jit, static_argnums=(0, ))
    def D_x1_DD_y1_kappa(self, x1, y1, paras): #cov(f'(x1), f''(y1))
        val = grad(grad(grad(self.kappa, 0), 1), 1)(x1, y1, paras)
        return val



class RBF_kernel_u_1d(object):

    def __init__(self):
        pass

    @partial(jit, static_argnums=(0, ))
    def kappa(self, x1, y1, s1):
        return (jnp.exp(-1 / 2 * ((x1 - y1)**2 / s1**2))).sum()

    @partial(jit, static_argnums=(0, ))
    def D_x1_kappa(self, x1, y1, s1):
        val = grad(self.kappa, 0)(x1, y1, s1)
        return val

    @partial(jit, static_argnums=(0, ))
    def DD_x1_kappa(self, x1, y1, s1):
        val = grad(grad(self.kappa, 0), 0)(x1, y1, s1)
        return val

    @partial(jit, static_argnums=(0, ))
    def D_y1_kappa(self, x1, y1, s1):
        val = grad(self.kappa, 1)(x1, y1, s1)
        return val

    @partial(jit, static_argnums=(0, ))
    def DD_y1_kappa(self, x1, y1, s1):
        val = grad(grad(self.kappa, 1), 1)(x1, y1, s1)
        return val

    @partial(jit, static_argnums=(0, ))
    def D_x1_D_y1_kappa(self, x1, y1, s1):
        val = grad(grad(self.kappa, 0), 1)(x1, y1, s1)
        return val
    @partial(jit, static_argnums=(0, ))
    def DD_x1_DD_y1_kappa(self, x1, y1, s1):
        val = grad(grad(grad(grad(self.kappa, 0), 0), 1),1)(x1, y1, s1)
        return val

    @partial(jit, static_argnums=(0, ))
    def D_x1_DD_y1_kappa(self, x1, y1, s1):
        val = grad(grad(grad(self.kappa, 0), 1), 1)(x1, y1, s1)
        return val


class RBF_kernel_u(object):

    def __init__(self):
        pass

    @partial(jit, static_argnums=(0, ))
    def kappa(self, x1, x2, y1, y2, s1, s2):
        return jnp.exp(-1 / 2 * ((x1 - y1)**2 / s1**2 + (x2 - y2)**2 / s2**2))

    @partial(jit, static_argnums=(0, ))
    def D_x1_kappa(self, x1, x2, y1, y2, s1, s2):
        val = grad(self.kappa, 0)(x1, x2, y1, y2, s1, s2)
        return val

    @partial(jit, static_argnums=(0, ))
    def D_x2_kappa(self, x1, x2, y1, y2, s1, s2):
        val = grad(self.kappa, 1)(x1, x2, y1, y2, s1, s2)
        return val

    @partial(jit, static_argnums=(0, ))
    def DD_x1_kappa(self, x1, x2, y1, y2, s1, s2):
        val = grad(grad(self.kappa, 0), 0)(x1, x2, y1, y2, s1, s2)
        return val


    @partial(jit, static_argnums=(0, ))
    def DD_y1_kappa(self, x1, x2, y1, y2, s1, s2):
        val = grad(grad(self.kappa, 2), 2)(x1, x2, y1, y2, s1, s2)
        return val

    @partial(jit, static_argnums=(0, ))
    def DD_x2_kappa(self, x1, x2, y1, y2, s1, s2):
        val = grad(grad(self.kappa, 1), 1)(x1, x2, y1, y2, s1, s2)
        return val

    @partial(jit, static_argnums=(0, ))
    def D_y1_kappa(self, x1, x2, y1, y2, s1, s2):
        val = grad(self.kappa, 2)(x1, x2, y1, y2, s1, s2)
        return val

    @partial(jit, static_argnums=(0, ))
    def D_y2_kappa(self, x1, x2, y1, y2, s1, s2):
        val = grad(self.kappa, 3)(x1, x2, y1, y2, s1, s2)
        return val

    @partial(jit, static_argnums=(0, ))
    def D_x1_D_y1_kappa(self, x1, x2, y1, y2, s1, s2):
        val = grad(grad(self.kappa, 0), 2)(x1, x2, y1, y2, s1, s2)
        return val

    @partial(jit, static_argnums=(0, ))
    def D_x1_D_y2_kappa(self, x1, x2, y1, y2, s1, s2):
        val = grad(grad(self.kappa, 0), 3)(x1, x2, y1, y2, s1, s2)
        return val

    @partial(jit, static_argnums=(0, ))
    def D_x2_D_y1_kappa(self, x1, x2, y1, y2, s1, s2):
        val = grad(grad(self.kappa, 1), 2)(x1, x2, y1, y2, s1, s2)
        return val

    @partial(jit, static_argnums=(0, ))
    def D_x2_D_y2_kappa(self, x1, x2, y1, y2, s1, s2):
        val = grad(grad(self.kappa, 1), 3)(x1, x2, y1, y2, s1, s2)
        return val

    @partial(jit, static_argnums=(0, ))
    def D_x1_DD_y2_kappa(self, x1, x2, y1, y2, s1, s2):
        val = grad(grad(grad(self.kappa, 0), 3), 3)(x1, x2, y1, y2, s1, s2)
        return val

    @partial(jit, static_argnums=(0, ))
    def D_x2_DD_y2_kappa(self, x1, x2, y1, y2, s1, s2):
        val = grad(grad(grad(self.kappa, 1), 3), 3)(x1, x2, y1, y2, s1, s2)
        return val

    @partial(jit, static_argnums=(0, ))
    def DD_x2_DD_y2_kappa(self, x1, x2, y1, y2, s1, s2):
        val = grad(grad(grad(grad(self.kappa, 1), 1), 3), 3)(x1, x2, y1, y2, s1, s2)
        return val

    @partial(jit, static_argnums=(0, ))
    def DD_x1_DD_y1_kappa(self, x1, x2, y1, y2, s1, s2):
        val = grad(grad(grad(grad(self.kappa, 0), 0), 2), 2)(x1, x2, y1, y2, s1, s2)
        return val

    @partial(jit, static_argnums=(0, ))
    def DD_x1_D_y2_kappa(self, x1, x2, y1, y2, s1, s2):
        val = grad(grad(grad(self.kappa, 0), 0), 3)(x1, x2, y1, y2, s1, s2)
        return val

    @partial(jit, static_argnums=(0, ))
    def DD_x1_D_y1_kappa(self, x1, x2, y1, y2, s1, s2):
        val = grad(grad(grad(self.kappa, 0), 0), 2)(x1, x2, y1, y2, s1, s2)
        return val

    @partial(jit, static_argnums=(0, ))
    def DD_x2_D_y1_kappa(self, x1, x2, y1, y2, s1, s2):
        val = grad(grad(grad(self.kappa, 1), 1), 2)(x1, x2, y1, y2, s1, s2)
        return val

    @partial(jit, static_argnums=(0, ))
    def DD_x2_D_y2_kappa(self, x1, x2, y1, y2, s1, s2):
        val = grad(grad(grad(self.kappa, 1), 1), 3)(x1, x2, y1, y2, s1, s2)
        return val

    @partial(jit, static_argnums=(0, ))
    def DD_x1_DD_y2_kappa(self, x1, x2, y1, y2, s1, s2):
        val = grad(grad(grad(grad(self.kappa, 0), 0), 3), 3)(x1, x2, y1, y2, s1, s2)
        return val
