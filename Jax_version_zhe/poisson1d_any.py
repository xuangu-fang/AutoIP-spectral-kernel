import numpy as np
from kernels import *
import random
import matplotlib.pyplot as plt
import optax
import jax
import jax.numpy as jnp
from kernel_matrix import Kernel_matrix
from jax import vmap
import pandas as pd

np.random.seed(0)
random.seed(0)


class GPRLatent:

    #equation: u_{xx}  = f(x)
    #Xind: the indices of X_col that corresponds to training points, i.e., boundary points
    #y: training outputs 
    #Xcol: collocation points 
    def __init__(self, Xind, y, X_col, src_col, jitter, X_test, Y_test):
        self.Xind = Xind
        self.y = y
        self.X_col = X_col
        self.src_col = src_col
        self.jitter = jitter
        #X is the 1st and the last point in X_col
        self.X_con = X_col
        self.N = self.Xind.shape[0]
        #self.N_col = self.X_col.shape[0]
        self.N_con = self.X_con.shape[0]
        self.cov_func = SM_kernel_u_1d()
        self.optimizer = optax.adam(0.01)
        self.kernel_matrix = Kernel_matrix(self.jitter, self.cov_func, 'NONE')
        self.Xte = X_test
        self.yte = Y_test 

    def KL_term(self, mu, L, K):
        Ltril = jnp.tril(L)
        hh_expt = jnp.matmul(Ltril, Ltril.T) + jnp.matmul(mu, mu.T)
        #hh_expt = jnp.matmul(Ltril, Ltril.T)
        kl = (0.5 * jnp.trace(jnp.linalg.solve(K, hh_expt)) + 0.5 * jnp.linalg.slogdet(K)[1] - 0.5 * jnp.sum(jnp.log(jnp.square(jnp.diag(Ltril)))))
        return kl

    @partial(jit, static_argnums=(0,))
    def loss(self, params, key):
        u = params['u'] #function values at the collocation points
        log_v = params['log_v'] #inverse variance for eq ll 
        log_tau = params['log_tau'] #inverse variance for boundary ll
        kernel_paras = params['kernel_paras']
        x_p = jnp.tile(self.X_con.flatten(), (self.N_con, 1)).T
        X1_p = x_p.flatten()
        X2_p = jnp.transpose(x_p).flatten()
        #only the cov matrix of func vals
        K = self.kernel_matrix.get_kernel_matrx(X1_p, X2_p, kernel_paras) 
        Kinv_u = jnp.linalg.solve(K, u)
        log_prior = -0.5*jnp.linalg.slogdet(K)[1] - 0.5*jnp.sum(u*Kinv_u)
        #boundary
        log_boundary_ll = 0.5 * self.N * log_tau - 0.5 * jnp.exp(log_tau) * jnp.sum(jnp.square(u[self.Xind].reshape(-1) - self.y.reshape(-1)))
        #equation
        K_dxx1 = vmap(self.cov_func.DD_x1_kappa, (0, 0, None))(X1_p, X2_p, kernel_paras).reshape(self.N_con, self.N_con)
        u_xx = jnp.matmul(K_dxx1, Kinv_u)
        eq_ll = 0.5 * self.N_con * log_v - 0.5 * jnp.exp(log_v) * jnp.sum(jnp.square( u_xx.flatten() - self.src_col.flatten() ))
        log_joint = log_prior + log_boundary_ll + eq_ll
        return -log_joint

    @partial(jit, static_argnums=(0,))
    def step(self, params, opt_state, key):
        loss = self.loss(params, key)
        loss, d_params = jax.value_and_grad(self.loss)(params, key)
        updates, opt_state = self.optimizer.update(d_params, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @partial(jit, static_argnums=(0,))
    def preds(self, params, Xte):
        ker_paras = params['kernel_paras']
        u = params['u']
        x_p = jnp.tile(self.X_con.flatten(), (self.N_con, 1)).T
        X1_p = x_p.flatten()
        X2_p = jnp.transpose(x_p).flatten()
        K = self.kernel_matrix.get_kernel_matrx(X1_p, X2_p, ker_paras)
        Kinv_u = jnp.linalg.solve(K, u)

        N_te = Xte.shape[0]
        x_p11 = jnp.tile(Xte.flatten(), (self.N_con, 1)).T
        x_p22 = jnp.tile(self.X_con.flatten(), (N_te, 1)).T
        X1_p2 = x_p11.flatten()
        X2_p2 = jnp.transpose(x_p22).flatten()
        Kmn = vmap(self.cov_func.kappa, (0, 0, None))(X1_p2.flatten(), X2_p2.flatten(), ker_paras).reshape(N_te, self.N_con)
        preds = jnp.matmul(Kmn, Kinv_u)
        return preds

    def train(self, nepoch):
        key = jax.random.PRNGKey(0)
        Q = 50 
        #Q = 100 
        #Q = 20 
        params = {
            "log_tau": 0.0, #inv var for data ll
            "log_v": 0.0, #inv var for eq likelihood
            #"kernel_paras": {'log-w': np.zeros(Q), 'log-ls': np.zeros(Q), 'freq': np.linspace(0, 1, Q)*100},
            "kernel_paras": {'log-w': np.log(1/Q)*np.ones(Q), 'log-ls': np.zeros(Q), 'freq': np.linspace(0, 1, Q)*100},
            "u": np.zeros((self.N_con, 1)), #u value on the collocation points
        }
        opt_state = self.optimizer.init(params)

        min_err = 2.0
        print("here")
        for i in range(nepoch):
            key, sub_key = jax.random.split(key)
            params, opt_state, loss = self.step(params, opt_state, sub_key)
            #if i % 10 == 0 or i >= 2000:
            if True:
                preds = self.preds(params, self.Xte)
                err = jnp.linalg.norm(preds.reshape(-1) - self.yte.reshape(-1))/jnp.linalg.norm(self.yte.reshape(-1))
                if True or err < min_err:
                    if err < min_err:
                        min_err = err
                    print('loss = %g'%loss)
                    print('tau = %g, v = %g'%(jnp.exp(params['log_tau']), jnp.exp(params['log_v'])))
                    print('freq') 
                    print(params['kernel_paras']['freq'])
                    print('ls')
                    print(jnp.exp(params['kernel_paras']['log-ls']))
                    print('weights')
                    print(jnp.exp(params['kernel_paras']['log-w']))
                    print("It ", i, "Relative L2 error", err)
        
        print('gen fig ...')
        preds = self.preds(params, self.Xte)
        plt.figure()
        Xtr = self.X_col[self.Xind]
        plt.plot(self.Xte.flatten(), self.yte.flatten(), 'k-', label='Truth')
        plt.plot(self.Xte.flatten(), preds.flatten(), 'r-', label='Pred')
        plt.scatter(Xtr.flatten(), self.y.flatten(), c='g', label='Train')
        plt.legend(loc=2)
        plt.savefig("poisson1d-%d.png"%nepoch)
        plt.clf()


#Poisson source, u_xx = f
#u should be the target function
def get_source_val(u, x_vec):
    return vmap(grad(grad(u, 0),0), (0))(x_vec)


def test_multi_scale():
    #equation
    u = lambda x: jnp.sin(5*jnp.pi*x) + jnp.sin(23.7*jnp.pi*x) + jnp.cos(92.3*jnp.pi*x)
    #test points
    M = 400
    X_test = np.linspace(0, 1, num=M).reshape(-1, 1)
    Y_test = u(X_test)
    #collocation points
    N_col = 200 
    X_col = np.linspace(0, 1, num=N_col).reshape(-1, 1)
    Xind = np.array([0, X_col.shape[0]-1 ])
    y = jnp.array([u(X_col[Xind[0]]), u(X_col[Xind[1]])]).reshape(-1)
    src_vals = get_source_val(u, X_col.reshape(-1))

    model_PIGP = GPRLatent(Xind, y, X_col, src_vals,  1e-6, X_test, Y_test)
    np.random.seed(123)
    random.seed(123)
    #nepoch = 250000
    nepoch = 500000
    model_PIGP.train(nepoch)


if __name__ == '__main__':
    test_multi_scale()
