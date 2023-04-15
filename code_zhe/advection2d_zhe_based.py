import numpy as np
from kernels import *
import random
import matplotlib.pyplot as plt
import pickle
import optax
import jax
import jax.numpy as jnp
from kernel_matrix import Kernel_matrix
from jax import vmap
import pandas as pd

np.random.seed(0)
random.seed(0)


class GPRLatent:

    #equation: u_{xx} + u_{yy}  = f(x,y)
    #bvals: 1d array, boundary values
    #X_col = (x_pos, y_pos), x_pos: 1d array, y_pos: 1d array
    #src_vals: source values at the collocation mesh, N1 x N2
    #X_test = (x_test_pos, y_test_pos): x_test_pos: 1d array, y_test_pos: 1d array
    #u_test:  M1 x M2
    def __init__(self, bvals, X_col, src_vals, jitter, X_test, u_test, beta):
        self.beta = beta
        self.bvals = bvals #Nb dim
        self.X_col = X_col
        self.jitter = jitter
        self.Nb = bvals.size #number of boundary points
        self.N1 = X_col[0].size
        self.N2 = X_col[1].size
        self.Nc = self.N1*self.N2
        self.src_vals = src_vals #N1 by N2
        self.cov_func = SM_kernel_u_1d()

        self.optimizer = optax.adam(0.01)
        self.KM_calc = Kernel_matrix(self.jitter, self.cov_func, 'NONE')
        self.Xte = X_test
        self.ute = u_test

        self.x_pos_tr_mesh, self.x_pos_tr_mesh_T = np.meshgrid(self.X_col[0], self.X_col[0], indexing='ij')
        self.y_pos_tr_mesh, self.y_pos_tr_mesh_T = np.meshgrid(self.X_col[1], self.X_col[1], indexing='ij')


    @partial(jit, static_argnums=(0,))
    def loss(self, params, key):
        U = params['U'] #function values at the collocation points, N1 X N2
        log_v = params['log_v'] #inverse variance for eq ll 
        log_tau = params['log_tau'] #inverse variance for boundary ll
        kernel_paras_x = params['kernel_paras_1'] #ker params for 1st dimension
        kernel_paras_y = params['kernel_paras_2'] #ker params for 2nd dimension

        K1 = self.KM_calc.get_kernel_matrix(self.x_pos_tr_mesh, self.x_pos_tr_mesh_T, kernel_paras_x) #N1 x N1
        K2 = self.KM_calc.get_kernel_matrix(self.y_pos_tr_mesh, self.y_pos_tr_mesh_T, kernel_paras_y) #N2 x N2
        K1inv_U = jnp.linalg.solve(K1, U) #N1 x N2
        K2inv_Ut = jnp.linalg.solve(K2, U.T) #N2 x N1
        log_prior = -0.5*self.N2*jnp.linalg.slogdet(K1)[1] - 0.5*self.N1*jnp.linalg.slogdet(K2)[1] - 0.5*jnp.sum(K1inv_U*K2inv_Ut.T)

        #boundary
        u_b = jnp.hstack((U[0,:], U[-1,:], U[:,0], U[:,-1]))
        log_boundary_ll = 0.5 * self.Nb * log_tau - 0.5 * jnp.exp(log_tau) * jnp.sum(jnp.square(u_b.reshape(-1) - self.bvals.reshape(-1)))


        #equation
        # K_dxx1 = vmap(self.cov_func.DD_x1_kappa, (0, 0, None))(self.x_pos_tr_mesh.reshape(-1), self.x_pos_tr_mesh_T.reshape(-1), kernel_paras_x).reshape(self.N1, self.N1)
        # U_xx = jnp.matmul(K_dxx1, K1inv_U)
        # K_dyy1 = vmap(self.cov_func.DD_x1_kappa, (0, 0, None))(self.y_pos_tr_mesh.reshape(-1), self.y_pos_tr_mesh_T.reshape(-1), kernel_paras_y).reshape(self.N2, self.N2)
        # U_yy = jnp.matmul(K_dyy1, K2inv_Ut).T

        K_dx1 = vmap(self.cov_func.D_x1_kappa, (0, 0, None))(self.x_pos_tr_mesh.reshape(-1), self.x_pos_tr_mesh_T.reshape(-1), kernel_paras_x).reshape(self.N1, self.N1)

        U_x = jnp.matmul(K_dx1, K1inv_U)

        K_dy1 = vmap(self.cov_func.D_x1_kappa, (0, 0, None))(self.y_pos_tr_mesh.reshape(-1), self.y_pos_tr_mesh_T.reshape(-1), kernel_paras_y).reshape(self.N2, self.N2)

        U_y = jnp.matmul(K_dy1, K2inv_Ut).T


        # eq_ll = 0.5 * self.Nc * log_v - 0.5 * jnp.exp(log_v) * jnp.sum(jnp.square( U_xx + U_yy - self.src_vals ))
        eq_ll = 0.5 * self.Nc * log_v - 0.5 * jnp.exp(log_v) * jnp.sum(jnp.square( self.beta * U_x  + U_y - self.src_vals ))


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
    def pred(self, params):
        ker_paras_x = params['kernel_paras_1']
        ker_paras_y = params['kernel_paras_2']
        U = params['U']
        K1 = self.KM_calc.get_kernel_matrix(self.x_pos_tr_mesh, self.x_pos_tr_mesh_T, ker_paras_x)
        K1inv_U = jnp.linalg.solve(K1, U) #N1 x N2

        x_te_cross_mh, x_tr_cross_mh = np.meshgrid(self.Xte[0], self.X_col[0], indexing='ij')
        Kmn = vmap(self.cov_func.kappa, (0, 0, None))(x_te_cross_mh.reshape(-1), x_tr_cross_mh.reshape(-1), ker_paras_x).reshape(self.Xte[0].size, self.N1)

        M1 = jnp.matmul(Kmn, K1inv_U)

        K2 = self.KM_calc.get_kernel_matrix(self.y_pos_tr_mesh, self.y_pos_tr_mesh_T, ker_paras_y)
        M2 = jnp.linalg.solve(K2, M1.T)

        y_te_cross_mh, y_tr_cross_mh = np.meshgrid(self.Xte[1], self.X_col[1], indexing='ij')
        Kmn2 = vmap(self.cov_func.kappa, (0, 0, None))(y_te_cross_mh.reshape(-1), y_tr_cross_mh.reshape(-1), ker_paras_y).reshape(self.Xte[1].size, self.N2)
        U_pred = jnp.matmul(Kmn2, M2).T
        return U_pred

    def train(self, nepoch):
        key = jax.random.PRNGKey(0)
        Q = 30
        #Q = 100 
        #Q = 20 
        params = {
            "log_tau": 0.0, #inv var for data ll
            "log_v": 0.0, #inv var for eq likelihood
            #"kernel_paras": {'log-w': np.zeros(Q), 'log-ls': np.zeros(Q), 'freq': np.linspace(0, 1, Q)*100},
            "kernel_paras_1": {'log-w': np.log(1/Q)*np.ones(Q), 'log-ls': np.zeros(Q), 'freq': np.linspace(0, 1, Q)*100},
            "kernel_paras_2": {'log-w': np.log(1/Q)*np.ones(Q), 'log-ls': np.zeros(Q), 'freq': np.linspace(0, 1, Q)*100},
            "U": np.zeros((self.N1, self.N2)), #u value on the collocation points
        }
        opt_state = self.optimizer.init(params)

        min_err = 2.0
        min_err_epoch = -1
        print("here")
        for i in range(nepoch):
            key, sub_key = jax.random.split(key)
            params, opt_state, loss = self.step(params, opt_state, sub_key)
            #if i % 10 == 0 or i >= 2000:
            if True:
                preds = self.pred(params)
                
                err = jnp.linalg.norm(preds.reshape(-1) - self.ute.reshape(-1))/jnp.linalg.norm(self.ute.reshape(-1))
                if True or err < min_err:
                    if err < min_err:
                        min_err = err
                        min_err_epoch = i
                        
                    print('loss = %g'%loss)
                    print('tau = %g, v = %g'%(jnp.exp(params['log_tau']), jnp.exp(params['log_v'])))
                    print('freq-x') 
                    print(params['kernel_paras_1']['freq'])
                    print('ls-x')
                    print(jnp.exp(params['kernel_paras_1']['log-ls']))
                    print('weights-x')
                    print(jnp.exp(params['kernel_paras_1']['log-w']))
                    print('freq-y') 
                    print(params['kernel_paras_2']['freq'])
                    print('ls-y')
                    print(jnp.exp(params['kernel_paras_2']['log-ls']))
                    print('weights-y')
                    print(jnp.exp(params['kernel_paras_2']['log-w']))

                    print("It ", i, "Relative L2 error", err)
                    print('min err = %g'%min_err)
        
        with open('paras.pickle', 'wb') as f:
            pickle.dump(params, f)
            print('save current params')

        '''
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
        '''


#Poisson source, u_xx + u_yy = f
#u should be the target function
def get_source_val(u, x_pos, y_pos,beta=30):
    x_mesh, y_mesh = np.meshgrid(x_pos, y_pos, indexing='ij')
    x_vec = x_mesh.reshape(-1)
    y_vec = y_mesh.reshape(-1)
    # return vmap(grad(grad(u, 0),0), (0, 0))(x_vec, y_vec) + \
    #         vmap(grad(grad(u, 1),1), (0, 0))(x_vec, y_vec)

    return beta*vmap(grad(u, 0), (0, 0))(x_vec, y_vec) + vmap(grad(u, 1), (0, 0))(x_vec, y_vec)

def get_mesh_data(u, M1, M2):
    x_coor = np.linspace(0, 1, num=M1)
    y_coor = np.linspace(0, 1, num=M2)
    x_mesh, y_mesh = np.meshgrid(x_coor, y_coor, indexing='ij')
    u_mesh = u(x_mesh, y_mesh)
    return x_coor, y_coor, u_mesh


def get_boundary_vals(u_mesh):
    return jnp.hstack((u_mesh[0,:], u_mesh[-1,:], u_mesh[:,0], u_mesh[:,-1]))

def test_2d():
    beta = 30
    #equation
    #u = lambda x,y: jnp.sin(91.3*jnp.pi*x)*jnp.sin(95.7*jnp.pi*y)
    u = lambda x,y: jnp.sin(x-beta*y)
    #test points
    M = 300
    x_pos_test, y_pos_test, u_test_mh = get_mesh_data(u, M, M)
    #collocation points in each dim
    N = 200
    x_pos_tr, y_pos_tr, u_mh = get_mesh_data(u, N, N)
    #take the boundary values, 1d array
    bvals = get_boundary_vals(u_mh) 
    #take the source values
    src_vals = get_source_val(u, x_pos_tr, y_pos_tr)
    src_vals = src_vals.reshape((x_pos_tr.size, y_pos_tr.size))

    X_test = (x_pos_test, y_pos_test)
    u_test = u_test_mh
    X_col = (x_pos_tr, y_pos_tr)

    
    model_PIGP = GPRLatent(bvals, X_col, src_vals,  1e-6, X_test, u_test,beta)
    np.random.seed(123)
    random.seed(123)
    #nepoch = 250000
    nepoch = 50000
    model_PIGP.train(nepoch)


if __name__ == '__main__':
    test_2d()
