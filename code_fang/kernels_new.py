'''re-orginize the kernels.py, make them more readable and easier to use'''

import jax.numpy as jnp
from jax import grad, jit
from functools import partial
from jax.config import config
import math

config.update("jax_enable_x64", True)

'''base kernel class for 1d'''

class Kernel_1d(object):

    def __init__(self,fix_dict=None, fix_paras=None):

        '''used for analyze the effect of frezzing some kernel parameters, will not be used in the main code'''

        self.fix_dict = fix_dict
        self.fix_paras = fix_paras
    
    def kappa(self, x1, y1, paras):
        '''empty kernel, rasie error'''
        raise NotImplementedError
    

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
    
    @partial(jit, static_argnums=(0, ))
    def frezze_paras(self,paras):
            
            if self.fix_dict is not None:

                log_w = self.fix_dict['log-w']*self.fix_paras['log-w'] + (1-self.fix_dict['log-w'])*paras['log-w']
                log_ls = self.fix_dict['log-ls']*self.fix_paras['log-ls'] + (1-self.fix_dict['log-ls'])*paras['log-ls']
                freq = self.fix_dict['freq']*self.fix_paras['freq'] + (1-self.fix_dict['freq'])*paras['freq']
            
            else:
                log_w = paras['log-w']
                log_ls = paras['log-ls']
                freq = paras['freq']

            return log_w, log_ls, freq

class SE_Cos_1d(Kernel_1d):
    '''standard Specture Mixsure kernal: 
      weight x SE kernel x cosine kernel'''

    def __init__(self, fix_dict=None, fix_paras=None):
        super().__init__(fix_dict, fix_paras)

    @partial(jit, static_argnums=(0, ))
    def kappa(self, x1, y1, paras):

        log_w, log_ls, freq = self.frezze_paras(paras)

        return (jnp.exp(log_w)*jnp.exp(-(x1-y1)**2*jnp.exp(log_ls))*jnp.cos(2*jnp.pi*(x1-y1)*freq)).sum()

        
class Matern52_Cos_1d(Kernel_1d):

    '''variant Specture Mixsure kernal: 
      weight x Matern52 x cosine kernel'''

    def __init__(self, fix_dict=None, fix_paras=None):
        super().__init__(fix_dict, fix_paras)

    @partial(jit, static_argnums=(0, ))
    def kappa(self, x1, y1, paras):

        log_w, log_ls, freq = self.frezze_paras(paras)

        d = jnp.abs(x1-y1)

        matern = (1 + jnp.sqrt(5)*d*jnp.exp(log_ls) + 5/3*d**2*jnp.exp(log_ls)**2)*jnp.exp(-jnp.sqrt(5)*d*jnp.exp(log_ls))

        cosine = jnp.cos(2*jnp.pi*d*freq)

        return (jnp.exp(log_w)*matern*cosine).sum()
    
class Matern52_Cos_add_Matern_1d(Kernel_1d):

    '''variant Specture Mixsure kernal: 
      weight x Matern52 x cosine kernel + seperate Matern52 kernel'''

    def __init__(self, fix_dict=None, fix_paras=None):
        super().__init__(fix_dict, fix_paras)

    @partial(jit, static_argnums=(0, ))
    def kappa(self, x1, y1, paras):

        log_w, log_ls, freq = self.frezze_paras(paras)

        log_w_matern = paras['log-w-matern']
        log_ls_matern = paras['log-ls-matern']

        d = jnp.abs(x1-y1)

        matern_coef = (1 + jnp.sqrt(5)*d*jnp.exp(log_ls) + 5/3*d**2*jnp.exp(log_ls)**2)*jnp.exp(-jnp.sqrt(5)*d*jnp.exp(log_ls))

        cosine = jnp.cos(2*jnp.pi*d*freq)

        matern_single = (1 + jnp.sqrt(5)*d*jnp.exp(log_ls_matern) + 5/3*d**2*jnp.exp(log_ls_matern)**2)*jnp.exp(-jnp.sqrt(5)*d*jnp.exp(log_ls_matern))

        # add with sepearate matern kernel
        return (jnp.exp(log_w)*cosine*matern_coef).sum() + (jnp.exp(log_w_matern)*matern_single).sum()






