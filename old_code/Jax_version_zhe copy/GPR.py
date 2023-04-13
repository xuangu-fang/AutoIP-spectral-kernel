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


class GPR:

    def __init__(self, Xtr, ytr, jitter, X_test, Y_test):
        self.Xtr = Xtr 
        self.ytr = ytr.flatten()
        self.jitter = jitter
        self.Xte = X_test
        self.yte = Y_test
        self.N = self.Xtr.shape[0]
        self.cov_func = SM_kernel_u_1d()
        self.optimizer = optax.adam(0.01)
        self.kernel_matrix = Kernel_matrix(self.jitter, self.cov_func, "NONE")

    @partial(jit, static_argnums=(0,))
    def loss(self, params, key):
        log_tau = params['log_tau']
        kernel_paras = params['kernel_paras']
        x_p = jnp.tile(self.Xtr.flatten(), (self.N, 1)).T
        X1_p = x_p.flatten()
        X2_p = jnp.transpose(x_p).flatten()
        Kmat = self.kernel_matrix.get_kernel_matrx(X1_p, X2_p, kernel_paras)
        S = Kmat  + 1.0/jnp.exp(log_tau)*jnp.eye(self.N)
        ll = -0.5*jnp.linalg.slogdet(S)[1] - 0.5*jnp.sum(self.ytr*jnp.linalg.solve(S, self.ytr))
        return -ll


    @partial(jit, static_argnums=(0,))
    def step(self, params, opt_state, key):
        loss, d_params = jax.value_and_grad(self.loss)(params, key)
        updates, opt_state = self.optimizer.update(d_params, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @partial(jit, static_argnums=(0,))
    def preds(self, params, Xte):
        ker_paras = params['kernel_paras']
        log_tau = params['log_tau']
        x_p = jnp.tile(self.Xtr.flatten(), (self.N, 1)).T
        X1_p = x_p.flatten()
        X2_p = jnp.transpose(x_p).flatten()
        S = self.kernel_matrix.get_kernel_matrx(X1_p, X2_p, ker_paras) + 1.0/jnp.exp(log_tau)*jnp.eye(self.N)
        N_te = Xte.shape[0]
        x_p11 = jnp.tile(Xte.flatten(), (self.N, 1)).T
        x_p22 = jnp.tile(self.Xtr.flatten(), (N_te, 1)).T
        X1_p2 = x_p11.flatten()
        X2_p2 = jnp.transpose(x_p22).flatten()
        Kmn = vmap(self.cov_func.kappa, (0, 0, None))(X1_p2.flatten(), X2_p2.flatten(), ker_paras).reshape(N_te, self.N)
        preds = jnp.matmul(Kmn, jnp.linalg.solve(S, self.ytr))
        S_cho = jnp.linalg.cholesky(S)
        S12Knm = jnp.linalg.solve(S_cho, Kmn.T) #Ntr by Nte
        pred_var = 1 + jnp.sum(jnp.exp(ker_paras['log-w'])) - jnp.sum(jnp.square(S12Knm), 0)
        return preds, pred_var

    def train(self, nepoch):
        key = jax.random.PRNGKey(0)
        Q = 50#30 
        params = {
            "log_tau": np.array([0.0]),
            #"kernel_paras": {'log-w': np.ones(Q), 'log-ls': np.log(1)*np.ones(Q), 'freq': np.random.rand(Q)*100},
            "kernel_paras": {'log-w': np.ones(Q), 'log-ls': np.log(1)*np.ones(Q), 'freq': np.linspace(0, 1, Q)*100},
        }
        opt_state = self.optimizer.init(params)

        min_err = 2.0
        print("here")
        for i in range(nepoch):
            key, sub_key = jax.random.split(key)
            params, opt_state, loss = self.step(params, opt_state, sub_key)
            if True:
                preds, pred_stds = self.preds(params, self.Xte)
                err = jnp.sqrt(jnp.mean(jnp.square(preds.reshape((-1, 1)) - self.yte.reshape((-1, 1)))))
                if True or err < min_err:
                    if err < min_err:
                        min_err = err
                    print('loss = %g'%loss)
                    print('tau = %g'%(jnp.exp(params['log_tau'])))
                    print('freq')
                    print(params['kernel_paras']['freq'])
                    print('ls')
                    print(jnp.exp(params['kernel_paras']['log-ls']))
                    print('weights')
                    print(jnp.exp(params['kernel_paras']['log-w']))
                    print("It ", i, "Found min RMSE ", err)
                    '''
                    plt.figure()
                    plt.plot(self.X_test, self.Y_test)
                    plt.plot(self.X_test, preds)
                    plt.plot(self.X_test, preds.reshape(-1) - pred_stds.reshape(-1), linewidth=.3)
                    plt.plot(self.X_test, preds.reshape(-1) + pred_stds.reshape(-1), linewidth=.3)
                    plt.legend(["True solution", "PIGP"], prop={"size": 15})
                    plt.savefig("damping")
                    plt.clf()
                    '''
        print('gen fig ...')
        preds, pred_stds = self.preds(params, self.Xte)
        plt.figure()
        plt.scatter(self.Xtr.flatten(), self.ytr.flatten(), c='g', label='Train')
        plt.plot(self.Xte.flatten(), self.yte.flatten(), 'k-', label='Truth')
        plt.plot(self.Xte.flatten(), preds.flatten(), 'r-', label='Pred')
        plt.plot(self.Xte.flatten(), preds.reshape(-1) - pred_stds.reshape(-1), 'b:', linewidth=.3)
        plt.plot(self.Xte.flatten(), preds.reshape(-1) + pred_stds.reshape(-1), 'b:', linewidth=.3)
        plt.legend()
        plt.savefig("gpr-%d.png"%nepoch)
        plt.clf()


def test():
    Xtr = np.random.rand(200, 1)*0.5
    K = 76
    ytr = np.sin(K*np.pi*Xtr)
    ytr = ytr + np.random.normal(scale=0.1, size=ytr.shape)
    Xte = np.linspace(0, 1, 500).reshape((-1, 1))
    yte = np.sin(K*np.pi*Xte)

    gp = GPR(Xtr, ytr, 1e-6, Xte, yte)
    gp.train(5000)
    #gp.train(20000)

def test_multi():
    Xtr = np.random.rand(200, 1)*0.5
    K1 = 10
    K2 = 76
    ytr = np.sin(K1*np.pi*Xtr) + np.cos(K2*np.pi*Xtr)
    ytr = ytr + np.random.normal(scale=0.1, size=ytr.shape)
    Xte = np.linspace(0, 1, 500).reshape((-1, 1))
    yte = np.sin(K1*np.pi*Xte) + np.cos(K2*np.pi*Xte)

    gp = GPR(Xtr, ytr, 1e-6, Xte, yte)
    gp.train(5000)


def test_multi3():
    Xtr = np.random.rand(200, 1)*0.5
    K1 = 10
    K2 = 76
    K3 = 95.33
    ytr = np.sin(K1*np.pi*Xtr) + np.cos(K2*np.pi*Xtr) + np.sin(K3*np.pi*Xtr)
    ytr = ytr + np.random.normal(scale=0.1, size=ytr.shape)
    Xte = np.linspace(0, 1, 500).reshape((-1, 1))
    yte = np.sin(K1*np.pi*Xte) + np.cos(K2*np.pi*Xte) + np.sin(K3*np.pi*Xte)

    gp = GPR(Xtr, ytr, 1e-6, Xte, yte)
    gp.train(5000)




if __name__ == '__main__':
    #test()
    #test_multi()
    test_multi3()
