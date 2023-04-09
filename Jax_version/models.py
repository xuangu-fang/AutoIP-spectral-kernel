import numpy as np
import torch
import gpytorch as gp
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import (
    SpectralMixtureKernel,
    ScaleKernel,
    RBFKernel,
    AdditiveKernel,
    PolynomialKernel,
)
from torch.distributions.categorical import Categorical
from torch.distributions.independent import Independent
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.normal import Normal
from torch.optim import AdamW

from matplotlib import pyplot as plt

class GP(gp.models.ExactGP):
    def __init__(self, kernel, train_x, train_y):
        super(GP, self).__init__(train_x, train_y, GaussianLikelihood())
        self.means = gp.means.ConstantMean()
        self.cov = kernel
    
    def forward(self, x):
        return MultivariateNormal(self.means(x), self.cov(x))
    
    def predict(self, x):
        self.eval()
        self.likelihood.eval()
        with torch.no_grad(), gp.settings.fast_pred_var(): # https://arxiv.org/pdf/1803.06058.pdf
            pred = self.likelihood(self(x))
            lower, upper = pred.confidence_region()
        return pred.mean, lower, upper
    
    def spectral_density(self, smk) -> MixtureSameFamily:
        """Returns the Mixture of Gaussians thet model the spectral density
        of the provided spectral mixture kernel."""
        mus = smk.mixture_means.detach().reshape(-1, 1)
        sigmas = smk.mixture_scales.detach().reshape(-1, 1)
        mix = Categorical(smk.mixture_weights.detach())
        comp = Independent(Normal(mus, sigmas), 1)
        return MixtureSameFamily(mix, comp)

class SMKernelGP(GP):
    def __init__(self, train_x, train_y, num_mixtures=10):
        kernel = SpectralMixtureKernel(num_mixtures)
        kernel.initialize_from_data(train_x, train_y)

        super(SMKernelGP, self).__init__(kernel, train_x, train_y)
        self.mean = gp.means.ConstantMean()
        self.cov = kernel

    def spectral_density(self):
        return super().spectral_density(self.cov)

class RBFKernelGP(GP):
    def __init__(self, train_x, train_y):
        kernel = ScaleKernel(RBFKernel(train_x.shape[-1]))

        super(RBFKernelGP, self).__init__(kernel, train_x, train_y)
        self.mean = gp.means.ConstantMean()
        self.cov = kernel

def train(model, epoch, lr, train_x, train_y):
    model.train()
    model.likelihood.train()
    mll = gp.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    optimizer = AdamW(model.parameters(), lr)

    for i in range(epoch):
        optimizer.zero_grad()
        
        output = model(train_x)
        loss = -mll(output, train_y).sum()

        loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0:
            print(i, ':', loss.item())




if __name__ == '__main__':
    np_train_x = np.random.rand(200) * 0.5
    np_train_y = np.sin(100 * np.pi * np_train_x)
    np_train_y = np_train_y + np.random.normal(scale=0.1, size=np_train_y.shape)

    np_test_x = np.linspace(0, 1, 500)
    np_test_y = np.sin(100 * np.pi * np_test_x)

    train_x = torch.tensor(np_train_x)
    train_y = torch.tensor(np_train_y)

    test_x = torch.tensor(np_test_x)
    test_y = torch.tensor(np_test_y)

    smk = SMKernelGP(train_x, train_y, 100)
    train(smk, 10000, 1e-3, train_x, train_y)
    smk_mean, smk_lower, smk_upper = smk.predict(test_x)

    rbf = RBFKernelGP(train_x, train_y)
    train(rbf, 10000, 1e-3, train_x, train_y)
    rbf_mean, rbf_lower, rbf_upper = rbf.predict(test_x)

    plt.figure()

    plt.plot(np_test_x, rbf_mean.numpy(), 'g-', label='RBF')
    plt.fill_between(np_test_x, rbf_lower.numpy(), rbf_upper.numpy(), alpha=0.4, color='g')


    plt.plot(np_test_x, smk_mean.numpy(), 'r-', label='SM')
    plt.fill_between(np_test_x, smk_lower.numpy(), smk_upper.numpy(), alpha=0.4, color='r')

    plt.plot(np_test_x, np_test_y, 'k-', label='Truth')
    plt.scatter(np_train_x, np_train_y, marker='o', c='b')

    plt.legend()

    plt.savefig('figure.png')