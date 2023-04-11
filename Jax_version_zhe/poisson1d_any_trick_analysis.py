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
from jaxopt import LBFGS 
import time
import tqdm
import os

np.random.seed(0)
random.seed(0)


class GPRLatent:

    #equation: u_{xx}  = f(x)
    #Xind: the indices of X_col that corresponds to training points, i.e., boundary points
    #y: training outputs 
    #Xcol: collocation points 
    def __init__(self, Xind, y, X_col, src_col, jitter, X_test, Y_test,trick_paras =None):
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
        # self.cov_func = SM_kernel_u_1d()
        self.cov_func = Periodic_kernel_u_1d()

        lr = trick_paras['lr'] if trick_paras is not None else 0.01

        self.optimizer = optax.adam(lr)
        # self.optimizer = optax.adamaxw(0.01)

        self.trick_paras = trick_paras

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


        u = u.sum(axis=1).reshape(-1,1) #sum over trick

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

        u = u.sum(axis=1).reshape(-1,1) #sum over trick

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
        Q = self.trick_paras['Q'] if self.trick_paras is not None else 10
        #Q = 100 
        #Q = 20 

        num_u_trick = self.trick_paras['num_u_trick'] if self.trick_paras is not None else 1

        if self.trick_paras['init_u_trick'].__name__ != 'zeros':

            params = {
                "log_tau": 0.0, #inv var for data ll
                "log_v": 0.0, #inv var for eq likelihood
                #"kernel_paras": {'log-w': np.zeros(Q), 'log-ls': np.zeros(Q), 'freq': np.linspace(0, 1, Q)*100},
                "kernel_paras": {'log-w': np.log(1/Q)*np.ones(Q), 'log-ls': np.zeros(Q), 'freq': np.linspace(0, 1, Q)*100},
                # "u": np.zeros((self.N_con, 1)), #u value on the collocation points
                "u": self.trick_paras['init_u_trick'](self.N_con, num_u_trick), #u value on the collocation points

            }            

        else:
            params = {
                "log_tau": 0.0, #inv var for data ll
                "log_v": 0.0, #inv var for eq likelihood
                #"kernel_paras": {'log-w': np.zeros(Q), 'log-ls': np.zeros(Q), 'freq': np.linspace(0, 1, Q)*100},
                "kernel_paras": {'log-w': np.log(1/Q)*np.ones(Q), 'log-ls': np.zeros(Q), 'freq': np.linspace(0, 1, Q)*100},
                # "u": np.zeros((self.N_con, 1)), #u value on the collocation points
                "u": np.zeros((self.N_con, 1)), #u value on the collocation points

            }
        opt_state = self.optimizer.init(params)
        # self.run_lbfgs(params)

        loss_list = []
        err_list = []
        w_list = []
        freq_list = []
        ls_list = []
        epoch_list = []

        min_err = 2.0
        print("here")
        for i in tqdm.tqdm(range(nepoch)):
            key, sub_key = jax.random.split(key)
            params, opt_state, loss = self.step(params, opt_state, sub_key)
            #if i % 10 == 0 or i >= 2000:


            # evluating the error with frequency epoch/20, store the loss and error in a list

            if i % (nepoch/20) == 0:

                preds = self.preds(params, self.Xte)
                err = jnp.linalg.norm(preds.reshape(-1) - self.yte.reshape(-1))/jnp.linalg.norm(self.yte.reshape(-1))
                if True or err < min_err:
                    if err < min_err:
                        min_err = err
                    print('loss = %g'%loss)
                    print("It ", i, '  loss = %g '%loss ," Relative L2 error", err)

                loss_list.append(np.log(loss))
                err_list.append(err)
                w_list.append(np.exp(params['kernel_paras']['log-w']))
                freq_list.append(params['kernel_paras']['freq'])
                ls_list.append(np.exp(params['kernel_paras']['log-ls']))
                epoch_list.append(i)

        print('gen fig ...')
        ''' plot a figure with 6 subplots. 1 for the truth- prediction, 2 for the loss curve, 3 for the error curve, 4,5,6 for scatter of the weights, freq, and ls'''
        
        plt.figure(figsize=(20, 10))


        # first subplot
        plt.subplot(2, 3, 1)
        preds = self.preds(params, self.Xte)
        Xtr = self.X_col[self.Xind]

        preds = self.preds(params, self.Xte)
        Xtr = self.X_col[self.Xind]

        plt.plot(self.Xte.flatten(), self.yte.flatten(), 'k-', label='Truth')
        plt.plot(self.Xte.flatten(), preds.flatten(), 'r-', label='Pred')
        plt.scatter(Xtr.flatten(), self.y.flatten(), c='g', label='Train')
        plt.legend(loc=2)
        plt.title('pred-truth:loss = %g, err = %g'%(loss, err))

        # second subplot: loss curve, x-axis is the epoch, y-axis is the log-loss
        plt.subplot(2, 3, 2)
        plt.plot(epoch_list, loss_list)
        plt.title(' log-loss curve')

        # third subplot: error curve
        plt.subplot(2, 3, 3)
        plt.plot(epoch_list,err_list)
        plt.title('error curve')

        # fourth subplot: scatter of the weights at each test point, which store on the list w_list, the x-axies is epoch, y-axis is the weights, make the marker size smaller to make the plot clearer

        plt.subplot(2, 3, 4)
        for i in range(Q):
            plt.scatter(epoch_list, [w[i] for w in w_list], s=10)
        plt.title('weights scatter')

        # fifth subplot: scatter of the freq at each test point, which store on the list freq_list, the x-axies is epoch, y-axis is the freq
        plt.subplot(2, 3, 5)
        for i in range(Q):
            plt.scatter(epoch_list, [f[i] for f in freq_list], s=10)
        plt.title('freq scatter')

        # sixth subplot: scatter of the ls at each test point, which store on the list ls_list, the x-axies is epoch, y-axis is the ls
        plt.subplot(2, 3, 6)
        for i in range(Q):
            plt.scatter(epoch_list, [l[i] for l in ls_list], s=10)
        plt.title('ls scatter')

        # make the whole figure title to be the name of the trick
        plt.suptitle( self.trick_paras['init_u_trick'].__name__ + '-nU-%d-Q-%d-epoch-%d-lr-%.4f'%(num_u_trick,Q,nepoch,self.trick_paras['lr']))


        prefix = 'trick_analysis/poisson1d-any/'

        # build the folder if not exist
        if not os.path.exists(prefix):
            os.makedirs(prefix)

        fig_name = self.trick_paras['init_u_trick'].__name__ + '-nu-%d-Q-%d-epoch-%d-lr-%.4f.png'%(num_u_trick,Q,nepoch,self.trick_paras['lr'])
        print ('save fig to ', prefix+fig_name)

        plt.savefig(prefix+fig_name)
            




        # plt.figure()
        
        # plt.plot(self.Xte.flatten(), self.yte.flatten(), 'k-', label='Truth')
        # plt.plot(self.Xte.flatten(), preds.flatten(), 'r-', label='Pred')
        # plt.scatter(Xtr.flatten(), self.y.flatten(), c='g', label='Train')
        # plt.legend(loc=2)
        # plt.savefig("poisson1d-any-%d.png"%nepoch)
        # plt.clf()

        # # plot the loss and error curve
        # plt.figure()
        # plt.plot(loss_list)
        # plt.savefig("poisson1d-any-%d-loss.png"%nepoch)


#Poisson source, u_xx = f
#u should be the target function
def get_source_val(u, x_vec):
    return vmap(grad(grad(u, 0),0), (0))(x_vec)


def test_multi_scale(trick_paras):
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



    
    # trick_paras = {'init_u_trick': np.zeros, 'num_u_trick': 10, 'Q': 20}


    model_PIGP = GPRLatent(Xind, y, X_col, src_vals,  1e-6, X_test, Y_test,trick_paras)
    np.random.seed(123)
    random.seed(123)
    #nepoch = 250000
    nepoch = 100000

    model_PIGP.train(nepoch)
    # model_PIGP.train_lbfgs(nepoch)
    # model_PIGP.train_hybrid(1000)




if __name__ == '__main__':
    # trick_paras = 

    trick_list = [{'init_u_trick': np.random.rand, 'num_u_trick': 25, 'Q': 20, 'lr': 1e-2},
                   {'init_u_trick': np.random.randn, 'num_u_trick': 25, 'Q': 20, 'lr': 1e-2},
                   {'init_u_trick': np.zeros, 'num_u_trick': 20, 'Q': 25, 'lr': 1e-2},]

    for trick_paras in trick_list:
        test_multi_scale(trick_paras)
