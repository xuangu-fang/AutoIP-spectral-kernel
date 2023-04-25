import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import os
import pickle


def identity(x):
    return x


def square_norm(x):
    return x**2/jnp.sum(x**2)


def soft_max(x):
    return jnp.exp(x)/jnp.sum(jnp.exp(x))


def save_paras(model, params, log_dict, other_paras=''):

    Q = model.trick_paras['Q']
    nepoch = model.trick_paras['nepoch']
    prefix = 'result_analysis/' + model.trick_paras['equation'] + '/kernel_' + \
        model.cov_func.__class__.__name__ + \
        '/epoch_'+str(nepoch)+'/Q'+str(Q)+'/'

    # build the folder if not exist
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    fig_name = 'llk_w-%.1f-' % (model.llk_weight) + model.trick_paras['init_u_trick'].__name__ + '-Q-%d-epoch-%d-lr-%.4f-freqscale=%d' % (
        Q, nepoch, model.trick_paras['lr'], model.trick_paras['freq_scale']) + other_paras

    # save the params and log_dict as pickle
    with open(prefix + fig_name + '.pickle', 'wb') as f:
        pickle.dump([params, log_dict], f)


def make_fig_1d(model, params, log_dict, other_paras=''):

    # plot a figure with 6 subplots. 1 for the truth- prediction, 2 for the loss curve, 3 for the error curve, 4,5,6 for scatter of the weights, freq, and ls

    loss_list = log_dict['loss_list']
    err_list = log_dict['err_list']
    epoch_list = log_dict['epoch_list']
    w_list = log_dict['w_list']
    freq_list = log_dict['freq_list']
    ls_list = log_dict['ls_list']

    Q = model.trick_paras['Q']
    nepoch = model.trick_paras['nepoch']
    num_u_trick = model.trick_paras['num_u_trick']

    plt.figure(figsize=(20, 10))

    # first subplot
    plt.subplot(2, 3, 1)
    preds, _ = model.preds(params, model.Xte)
    Xtr = model.X_col[model.Xind]

    plt.plot(model.Xte.flatten(), model.yte.flatten(), 'k-', label='Truth')
    plt.plot(model.Xte.flatten(), preds.flatten(), 'r-', label='Pred')
    plt.scatter(Xtr.flatten(), model.y.flatten(), c='g', label='Train')
    plt.legend(loc=2)
    plt.title('pred-truth:loss = %g, err = %g' % (loss_list[-1], err_list[-1]))

    # second subplot: loss curve, x-axis is the epoch, y-axis is the log-loss
    plt.subplot(2, 3, 2)
    plt.plot(epoch_list, loss_list)
    plt.title('loss curve')

    # third subplot: error curve
    plt.subplot(2, 3, 3)
    plt.plot(epoch_list, err_list)
    plt.title('error curve')

    # fourth subplot: scatter of the weights at each test point, which store on the list w_list, the x-axies is epoch, y-axis is the weights, make the marker size smaller to make the plot clearer

    plt.subplot(2, 3, 4)
    for i in range(Q):
        plt.scatter(epoch_list, [w[i] for w in w_list], s=10)

        # if the weight is significant, label each scatter plot with the corresponding id and freq
        weight = [w[i] for w in w_list][-1]
        if weight > 1e-2:
            plt.text(epoch_list[-1], weight, '%s-th_freq-%.1f' %
                     (str(i), freq_list[-1][i]))

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

        # if the ls is significant, label each scatter plot with the corresponding id and freq
        ls = [l[i] for l in ls_list][-1]
        if ls > 1e-2:
            plt.text(epoch_list[-1], ls, '%s-th_freq-%.1f' %
                     (str(i), freq_list[-1][i]))
    plt.title('ls scatter')

    fix_prefix_dict = {1: '_fix_', 0: '_nonfix_'}
    fix_prefix = 'w'+fix_prefix_dict[model.fix_dict['log-w']]+'ls' + \
        fix_prefix_dict[model.fix_dict['log-ls']] + \
        'freq'+fix_prefix_dict[model.fix_dict['freq']]

    # make the whole figure title to be the name of the trick
    plt.suptitle(fix_prefix + '\n'+model.trick_paras['init_u_trick'].__name__ +
                 '-nU-%d-Q-%d-epoch-%d-lr-%.4f-freqscale-%d' % (num_u_trick, Q, nepoch, model.trick_paras['lr'], model.trick_paras['freq_scale']), )

    prefix = 'result_analysis/' + model.trick_paras['equation'] + '/kernel_' + \
        model.cov_func.__class__.__name__ + \
        '/epoch_'+str(nepoch)+'/Q'+str(Q)+'/'

    # build the folder if not exist
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    fig_name = 'llk_w-%.1f-' % (model.llk_weight) + model.trick_paras['init_u_trick'].__name__ + '-nU-%d-Q-%d-epoch-%d-lr-%.4f-freqscale=%d-logdet-%d' % (
        num_u_trick, Q, nepoch, model.trick_paras['lr'], model.trick_paras['freq_scale'], model.trick_paras['logdet']) + other_paras

    print('save fig to ', prefix+fig_name+'.png')

    plt.savefig(prefix+fig_name + '.png')


def make_fig_2d(model, params, log_dict,  other_paras=''):
    fontsize = 36
    # plot a figure with 9 subplots. 1 for the truth- prediction, 2 for the loss curve, 3 for the error curve, {4,5,6} for scatter of the weights, freq, and ls for x1, {7,8,9} for scatter of the weights, freq, and ls for x2

    loss_list = log_dict['loss_list']
    err_list = log_dict['err_list']
    epoch_list = log_dict['epoch_list']
    w_list_k1 = log_dict['w_list_k1']
    w_list_k2 = log_dict['w_list_k2']
    freq_list_k1 = log_dict['freq_list_k1']
    freq_list_k2 = log_dict['freq_list_k2']
    ls_list_k1 = log_dict['ls_list_k1']
    ls_list_k2 = log_dict['ls_list_k2']

    Q = model.trick_paras['Q']
    nepoch = model.trick_paras['nepoch']
    num_u_trick = model.trick_paras['num_u_trick']

    plt.figure(figsize=(20, 20))

    # first subplot- pred of 2d possion
    plt.subplot(3, 3, 1)
    preds, _ = model.preds(params)

    plt.imshow(preds, cmap="hot")
    plt.title('pred-2d:loss = %g, err = %g' %
              (loss_list[-1], err_list[-1]), fontsize=fontsize*0.5)

    # second subplot: ground truth of 2d possion
    plt.subplot(3, 3, 2)
    plt.imshow(model.ute, cmap="hot")
    plt.title('ground-truth-2d', fontsize=fontsize*0.5)

    # third subplot: error curve
    plt.subplot(3, 3, 3)
    plt.plot(epoch_list, err_list)
    plt.title('error curve', fontsize=fontsize*0.5)

    # fourth subplot: scatter of the weights at each test point, which store on the list w_list, the x-axies is epoch, y-axis is the weights, make the marker size smaller to make the plot clearer

    plt.subplot(3, 3, 4)
    # for i in range(Q):
    #     plt.scatter(epoch_list, [w[i] for w in w_list_k1], s=10)
    for i in range(Q):
        plt.scatter(epoch_list, [w[i] for w in w_list_k1], s=10)

        # if the weight is significant, label each scatter plot with the corresponding id and freq
        weight = [w[i] for w in w_list_k1][-1]
        if weight > 1e-2:
            plt.text(epoch_list[-1], weight, '%s-th_freq-%.1f' %
                     (str(i), freq_list_k1[-1][i]))
    plt.title('weights scatter-k1', fontsize=fontsize*0.5)

    # fifth subplot: scatter of the freq at each test point, which store on the list freq_list, the x-axies is epoch, y-axis is the freq
    plt.subplot(3, 3, 5)
    for i in range(Q):
        plt.scatter(epoch_list, [f[i] for f in freq_list_k1], s=10)
    plt.title('freq scatter-k1', fontsize=fontsize*0.5)

    # sixth subplot: scatter of the ls at each test point, which store on the list ls_list, the x-axies is epoch, y-axis is the ls
    plt.subplot(3, 3, 6)

    for i in range(Q):
        plt.scatter(epoch_list, [l[i] for l in ls_list_k1], s=10)

        # if the ls is significant, label each scatter plot with the corresponding id and freq
        ls = [l[i] for l in ls_list_k1][-1]
        if ls > 1e-2:
            plt.text(epoch_list[-1], ls, '%s-th_freq-%.1f' %
                     (str(i), freq_list_k1[-1][i]))

    plt.title('ls scatter-k1', fontsize=fontsize*0.5)

    plt.subplot(3, 3, 7)
    for i in range(Q):
        plt.scatter(epoch_list, [w[i] for w in w_list_k2], s=10)

        # if the weight is significant, label each scatter plot with the corresponding id and freq
        weight = [w[i] for w in w_list_k2][-1]
        if weight > 1e-2:
            plt.text(epoch_list[-1], weight, '%s-th_freq-%.1f' %
                     (str(i), freq_list_k2[-1][i]))

    plt.title('weights scatter-k2', fontsize=fontsize*0.5)

    plt.subplot(3, 3, 8)
    for i in range(Q):
        plt.scatter(epoch_list, [f[i] for f in freq_list_k2], s=10)
    plt.title('freq scatter-k2', fontsize=fontsize*0.5)

    plt.subplot(3, 3, 9)
    for i in range(Q):
        plt.scatter(epoch_list, [l[i] for l in ls_list_k2], s=10)

        # if the ls is significant, label each scatter plot with the corresponding id and freq
        ls = [l[i] for l in ls_list_k2][-1]
        if ls > 1e-2:
            plt.text(epoch_list[-1], ls, '%s-th_freq-%.1f' %
                     (str(i), freq_list_k2[-1][i]))

    plt.title('ls scatter-k2', fontsize=fontsize*0.5)

    fix_prefix_dict = {1: '_fix_', 0: '_nonfix_'}
    fix_prefix = 'w'+fix_prefix_dict[model.fix_dict['log-w']]+'ls' + \
        fix_prefix_dict[model.fix_dict['log-ls']] + \
        'freq'+fix_prefix_dict[model.fix_dict['freq']]

    # make the whole figure title to be the name of the trick
    plt.suptitle(fix_prefix + '\n'+model.trick_paras['init_u_trick'].__name__ + '-nU-%d-Q-%d-epoch-%d-lr-%.4f' % (
        num_u_trick, Q, nepoch, model.trick_paras['lr']), fontsize=fontsize)

    prefix = 'result_analysis/' + model.trick_paras['equation'] + \
        '/kernel_'+model.cov_func.__class__.__name__ + \
        '/epoch_'+str(nepoch)+'/Q'+str(Q)+'/'

    # build the folder if not exist
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    fig_name = fix_prefix + 'llk_w-%.1f-' % (model.llk_weight) + model.trick_paras['init_u_trick'].__name__ + '-nu-%d-Q-%d-epoch-%d-lr-%.4f-freqscale=%d-logdet-%d' % (
        num_u_trick, Q, nepoch, model.trick_paras['lr'], model.trick_paras['freq_scale'], model.trick_paras['logdet'])+other_paras
    print('save fig to ', prefix+fig_name)

    plt.savefig(prefix+fig_name+'.png')
