"""
The MIT License (MIT)
Copyright (c) 2015 Evan Archer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
import helpers as h
import utils as u
import math
import numpy as np


class ProductOfGaussians(nn.Module):
    """Product of gaussians used as recognition model.

    Arguments:
        dim_x (int): dimensionality of observation space
        dim_x (int): dimensionality of latent space
        dim_hidden (int): number of neurons of hidden layer
        rec_dict (dict):
            * A (torch.tensor): shape (dim_z, dim_z)
            * QinvChol (torch.tensor): shape (dim_z, dim_z)
            * Q0invChol (torch.tensor): shape (dim_z, dim_z)
    """

    def __init__(self, dim_x, dim_z, dim_hidden, X_true, batch_size, rec_dict=None):
        super(ProductOfGaussians, self).__init__()

        # TODO: Initialize the weights of the NN layers to have 0 mean wrt training data

        """ 
        - the weight matrices are stored as their own transpose, i.e. w_in has shape (dim_hidden, dim_x)
        - w_in has shape (dim_hidden, dim_x)
        - w_in_out has shape (batch_size, dim_hidden)

        NOTE: Even though the initialization of the hidden layers doesnt seem to make sense, 
        the results obtained with this set-up are very good.
        (results in folder 'lorentz_relu_meanCenteredInit_deeperRecModel')

        However, maybe the initialization of the hidden layers like this doesnt even have that big
        of an impact and only the initialization of the input layer is of importance."""

        self.initInstanceVariables(batch_size, dim_hidden, dim_x, dim_z, rec_dict)
        self.init_encoder(X_true)

        if rec_dict is not None:
            self.load_state_dict(rec_dict, strict=False)

    def initInstanceVariables(self, batch_size, dim_hidden, dim_x, dim_z, rec_dict):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_hidden = dim_hidden
        self.rec_dict = rec_dict
        self.batch_size = batch_size
        self.A = nn.Parameter(0.9 * torch.rand(dim_z, dim_z), requires_grad=True)
        self.QinvChol = nn.Parameter(torch.rand(dim_z, dim_z), requires_grad=True)
        self.Q0invChol = nn.Parameter(torch.rand(dim_z, dim_z), requires_grad=True)

        self.useRecognitionModelClipping = False

        #for testing only
        self.last_AA = None
        self.last_BB = None
        self.last_mean = None
        self.last_cov = None

    def init_encoder(self, x_true):

        # TODO: does it make difference if we initiate the weights after first layer?

        """Encoder for mean & covariance of numerical data"""
        self.fc_mean_in = nn.Linear(self.dim_x, self.dim_hidden)
        u.init_weights_and_bias(x_true, self.fc_mean_in, firstLayer=True)
        self.fc_mean_h1 = nn.Linear(self.dim_hidden, self.dim_hidden)
        self.fc_mean_h3 = nn.Linear(self.dim_hidden, self.dim_hidden)
        self.fc_mean_out = nn.Linear(self.dim_hidden, self.dim_z)

        self.fc_cov_in = nn.Linear(self.dim_x, self.dim_hidden)
        u.init_weights_and_bias(x_true, self.fc_cov_in, firstLayer=True)
        self.fc_cov_h1 = nn.Linear(self.dim_hidden, self.dim_hidden)
        self.fc_cov_h3 = nn.Linear(self.dim_hidden, self.dim_hidden)
        self.fc_cov_out = nn.Linear(self.dim_hidden, self.dim_z * self.dim_z)

    def activateRecognitionModelClipping(self):
        if self.useRecognitionModelClipping is True:
            print("Userwarning: activating RecognitionModelClipping allthough it is True allready")
        self.useRecognitionModelClipping = True

    def deactivateRecognitionModelClipping(self):
        if self.useRecognitionModelClipping is False:
            print("Userwarning: deactivating RecognitionModelClipping allthough it is False allready")
        self.useRecognitionModelClipping = False

    def encode_mean(self, x):
        x = x.view(-1, self.dim_x)
        x_mean = F.relu(self.fc_mean_in(x))
        x_mean = F.relu(self.fc_mean_h1(x_mean))
        x_mean = F.relu(self.fc_mean_h3(x_mean))
        x_mean = self.fc_mean_out(x_mean)
        return x_mean

    def encode_cov(self, x):
        x = x.view(-1, self.dim_x)
        x_cov = F.relu(self.fc_cov_in(x))
        x_cov = F.relu(self.fc_cov_h1(x_cov))
        x_cov = F.relu(self.fc_cov_h3(x_cov))
        x_cov = self.fc_cov_out(x_cov)
        return x_cov

    def forward(self, x):
        """x = numerical data"""
        # cov is actually not the covariance matrix but instead a part of the Matrix used in the Kalman filter to
        # calculate the cholesky decomposition and hence the correct covariance matrix

        # shape (BATCH_SIZE, dim_z * dim_z)
        mean = self.encode_mean(x)
        cov = self.encode_cov(x)

        self.AA, BB, lambdaMu = h.calculate_cholesky_factors(mean, cov, self.dim_z, self.QinvChol,
                                                             self.Q0invChol, self.A, self.batch_size)

        # compute cholesky decomposition
        # the_chol[0] has shape (BATCH_SIZE, dim_z, dim_z)
        # the_chol[1] has shape ((BATCH_SIZE-1), dim_z, dim_z)

        self.the_chol = h.blk_tridag_chol(self.AA, BB)
        ib = h.blk_chol_inv(self.the_chol[0], self.the_chol[1], lambdaMu, lower=True,
                                transpose=False)
        self.mu_z = h.blk_chol_inv(self.the_chol[0], self.the_chol[1], ib, lower=False,
                                   transpose=True)
        self.ln_determinant = -2 * torch.log(torch.diagonal(self.the_chol[0], dim1=-2, dim2=-1)).sum()

    def getSample(self, noise=True):
        """
        Reparameterization to get samples of z.
        """
        normSamps = torch.randn(self.batch_size, self.dim_z)
        R = h.blk_chol_inv(self.the_chol[0], self.the_chol[1], normSamps, lower=False,
                           transpose=True)

        if noise:
            return self.mu_z + R
        else:
            return self.mu_z

    def evalEntropy(self, alpha=0.5):
        """
        Differential entropy of a gaussian distributed random variable can be calculated via determinant of the
        covariance matrix.
        """
        a = 2 * (1 - alpha)

        entropy = self.calcEntropy(self.batch_size, a, self.ln_determinant)

        return entropy

    def calcEntropy(self, T, a, ln_determinant):
        return a * (ln_determinant / 2 + self.dim_z * T / 2.0 * (1 + torch.log(torch.tensor(2 * math.pi))))

    def getHessian(self):
        choleskyFactor = h.construct_bidiagonal(self.the_chol[0], self.the_chol[1])
        hessian = choleskyFactor #@ choleskyFactor.t()
        return hessian

    def getFullCholeskyFactor(self, x):
        oldBatchSize = self.batch_size
        self.batch_size = x.shape[0]
        self.forward(x)
        self.batch_size = oldBatchSize
        return self.the_chol[0], self.the_chol[1]

    def setBatchSize(self, newBatchSize):
        self.batch_size = newBatchSize

class ProductOfGaussiansMultimodal(ProductOfGaussians):
    """Product of gaussians used as recognition model.

    Arguments:
        dim_x (int): dimensionality of observation space
        dim_x (int): dimensionality of latent space
        dim_hidden (int): number of neurons of hidden layer
        rec_dict (dict):
            * A (torch.tensor): shape (dim_z, dim_z)
            * QinvChol (torch.tensor): shape (dim_z, dim_z)
            * Q0invChol (torch.tensor): shape (dim_z, dim_z)
    """

    def __init__(self, dim_x, dim_c, dim_z, dim_hidden, X_true, batch_size, rec_dict=None):
        super(ProductOfGaussians, self).__init__()
        # TODO: Initialize the weights of the NN layers to have 0 mean wrt training data
        """
        - the weight matrices are stored as their own transpose, i.e. w_in has shape (dim_hidden, dim_x)
        - w_in has shape (dim_hidden, dim_x)
        - w_in_out has shape (batch_size, dim_hidden)

        NOTE: Even though the initialization of the hidden layers doesnt seem to make sense,
        the results obtained with this set-up are very good.
        (results in folder 'lorentz_relu_meanCenteredInit_deeperRecModel')

        However, maybe the initialization of the hidden layers like this doesnt even have that big
        of an impact and only the initialization of the input layer is of importance."""

        self.initInstanceVariables(batch_size, dim_hidden, dim_x, dim_c, dim_z, rec_dict)
        self.init_encoder(X_true)

        if rec_dict is not None:
            self.load_state_dict(rec_dict, strict=False)

    def initInstanceVariables(self, batch_size, dim_hidden, dim_x, dim_c, dim_z, rec_dict):
        super().initInstanceVariables(batch_size, dim_hidden, dim_x, dim_z, rec_dict)
        self.dim_c = dim_c

    def init_encoder(self, x_true):
        # TODO: does it make difference if we initiate the weights after first layer?

        """Encoder for mean & covariance of numerical data"""
        self.fc_mean_combined_numerical_in = nn.Linear(self.dim_x, self.dim_hidden)
        u.init_weights_and_bias(x_true, self.fc_mean_combined_numerical_in, firstLayer=True)
        self.fc_mean_combined_numerical_h1 = nn.Linear(self.dim_hidden, self.dim_hidden)
        self.fc_mean_combined_numerical_h3 = nn.Linear(self.dim_hidden, self.dim_hidden)

        self.fc_cov_combined_numerical_in = nn.Linear(self.dim_x, self.dim_hidden)
        u.init_weights_and_bias(x_true, self.fc_cov_combined_numerical_in, firstLayer=True)
        self.fc_cov_combined_numerical_h1 = nn.Linear(self.dim_hidden, self.dim_hidden)
        self.fc_cov_combined_numerical_h3 = nn.Linear(self.dim_hidden, self.dim_hidden)

        """Encoder for mean & covariance of categorical data"""
        self.fc_mean_combined_categorical_in = nn.Linear(self.dim_c, self.dim_hidden)
        nn.init.orthogonal_(self.fc_mean_combined_categorical_in.weight)
        self.fc_mean_combined_categorical_h1 = nn.Linear(self.dim_hidden, self.dim_hidden)

        self.fc_cov_combined_categorical_in = nn.Linear(self.dim_c, self.dim_hidden)
        self.fc_cov_combined_categorical_h1 = nn.Linear(self.dim_hidden, self.dim_hidden)

        """Concat layer to combine the (numerical/categorical) encoders"""
        self.fc_mean_combined_concat = nn.Linear(self.dim_hidden + self.dim_hidden, self.dim_hidden)
        self.fc_mean_combined_h1 = nn.Linear(self.dim_hidden, self.dim_hidden)
        self.fc_mean_combined = nn.Linear(self.dim_hidden, self.dim_z)

        self.fc_cov_combined_concat = nn.Linear(self.dim_hidden + self.dim_hidden, self.dim_hidden)
        self.fc_cov_combined_h1 = nn.Linear(self.dim_hidden, self.dim_hidden)
        self.fc_cov_combined = nn.Linear(self.dim_hidden, self.dim_z * self.dim_z)

    def encode_mean(self, x, c):
        x = x.view(-1, self.dim_x)

        x_mean = F.relu(self.fc_mean_combined_numerical_in(x))
        x_mean = F.relu(self.fc_mean_combined_numerical_h1(x_mean))
        x_mean = F.relu(self.fc_mean_combined_numerical_h3(x_mean))

        c_mean = F.relu(self.fc_mean_combined_categorical_in(c))
        c_mean = F.relu(self.fc_mean_combined_categorical_h1(c_mean))

        mean = F.relu(self.fc_mean_combined_concat(torch.cat([x_mean, c_mean], dim=1)))
        mean = F.relu(self.fc_mean_combined_h1(mean))

        if self.useRecognitionModelClipping:
            mean = 5*torch.tanh(self.fc_mean_combined(mean))
        else:
            mean = self.fc_mean_combined(mean)
        return mean

    def encode_cov(self, x, c):
        x = x.view(-1, self.dim_x)

        x_cov = F.relu(self.fc_cov_combined_numerical_in(x))
        x_cov = F.relu(self.fc_cov_combined_numerical_h1(x_cov))
        x_cov = F.relu(self.fc_cov_combined_numerical_h3(x_cov))

        c_cov = F.relu(self.fc_cov_combined_categorical_in(c))
        c_cov = F.relu(self.fc_cov_combined_categorical_h1(c_cov))

        cov = F.relu(self.fc_cov_combined_concat(torch.cat([x_cov, c_cov], dim=1)))
        cov = F.relu(self.fc_cov_combined_h1(cov))

        if self.useRecognitionModelClipping:
            cov = 5*torch.tanh(self.fc_cov_combined(cov))
        else:
            cov = self.fc_cov_combined(cov)
        return cov

    def forward(self, x, c):
        """x = numerical data, c = categorical data"""

        """ cov is actually not the covariance matrix but instead a part of the Matrix used in the Kalman filter to
        calculate the cholesky decomposition and hence the correct covariance matrix

        the_chol[0] has shape (BATCH_SIZE, dim_z, dim_z)
        the_chol[1] has shape ((BATCH_SIZE-1), dim_z, dim_z)"""

        mean = self.encode_mean(x, c)
        cov = self.encode_cov(x, c)


        # compute cholesky decomposition

        self.AA, BB, lambdaMu = h.calculate_cholesky_factors(mean, cov, self.dim_z, self.QinvChol,
                                                             self.Q0invChol, self.A, self.batch_size)
        try:
            self.the_chol = h.blk_tridag_chol(self.AA, BB)
        except:
            print("Something went wrong in rec_model.forward(). Printing relevant quantities from this and the previous step")
            print("Previous step: ")
            print(self.last_AA)
            print(self.last_BB)
            print(self.last_mean)
            print(self.last_cov)
            print("--------------------------------------------")
            print("This step: ")
            print(self.AA)
            print(BB)
            print(mean)
            print(cov)
        ib = h.blk_chol_inv(self.the_chol[0], self.the_chol[1], lambdaMu, lower=True,
                                transpose=False)
        self.mu_z = h.blk_chol_inv(self.the_chol[0], self.the_chol[1], ib, lower=False,
                                   transpose=True)
        #self.printCholeskyFactorAndHessian()
        self.ln_determinant = -2 * torch.log(torch.diagonal(self.the_chol[0], dim1=-2, dim2=-1)).sum()

        self.last_AA = self.AA
        self.last_BB = BB
        self.last_mean = mean
        self.last_cov = cov

    def getFullCholeskyFactor(self, x, c):
        oldBatchSize = self.batch_size
        self.batch_size = x.shape[0]
        self.forward(x, c)
        self.batch_size = oldBatchSize
        return self.the_chol[0], self.the_chol[1]

    def printCholeskyFactorAndHessian(self):
        realR = h.construct_bidiagonal(self.the_chol[0], self.the_chol[1])
        print(self.the_chol[0])
        print(self.the_chol[1])
        print("--------------------------------------------------------")
        print(realR.shape)
        # for ind in range(0, len(realR)):
        np.set_printoptions(linewidth=200)
        print(realR.data.numpy())
        print("--------------------------------------------------------")
        print(realR.t().data.numpy())
        print("--------------------------------------------------------")
        realRtimesRealRT = realR @ realR.t()
        print(realRtimesRealRT.data.numpy())
        print("--------------------------------------------------------")
        print(realRtimesRealRT.inverse().data.numpy())
        print("--------------------------------------------------------")
        print("--------------------------------------------------------")


class ProductOfGaussiansHRF(ProductOfGaussians):
    """Product of gaussians used as recognition model.

    Arguments:
        dim_x (int): dimensionality of observation space
        dim_x (int): dimensionality of latent space
        dim_hidden (int): number of neurons of hidden layer
        rec_dict (dict):
            * A (torch.tensor): shape (dim_z, dim_z)
            * QinvChol (torch.tensor): shape (dim_z, dim_z)
            * Q0invChol (torch.tensor): shape (dim_z, dim_z)
    """

    def __init__(self, dim_x, dim_z, dim_hidden, X_true, tau, batch_size, rec_dict=None):
        super(ProductOfGaussians, self).__init__()

        # TODO: Initialize the weights of the NN layers to have 0 mean wrt training data
        """ 
        - the weight matrices are stored as their own transpose, i.e. w_in has shape (dim_hidden, dim_x)
        - w_in has shape (dim_hidden, dim_x)
        - w_in_out has shape (batch_size, dim_hidden)

        NOTE: X_true in this recognition model has shape (dim_x, (tau+1)), to contain all the needed timesteps in 
        regard to the HRF """

        self.initInstanceVariables(batch_size, dim_hidden, dim_x, dim_z, tau, rec_dict)
        self.init_encoder(X_true)

        if rec_dict is not None:
            self.load_state_dict(rec_dict, strict=False)

    def initInstanceVariables(self, batch_size, dim_hidden, dim_x, dim_z, tau, rec_dict):
        super().initInstanceVariables(batch_size, dim_hidden, dim_x, dim_z, rec_dict)
        self.dim_x = dim_x * (tau + 1)


class ProductOfGaussiansMultimodalHRF(ProductOfGaussiansMultimodal):
    """Product of gaussians used as recognition model.

    Arguments:
        dim_x (int): dimensionality of observation space
        dim_x (int): dimensionality of latent space
        dim_hidden (int): number of neurons of hidden layer
        rec_dict (dict):
            * A (torch.tensor): shape (dim_z, dim_z)
            * QinvChol (torch.tensor): shape (dim_z, dim_z)
            * Q0invChol (torch.tensor): shape (dim_z, dim_z)
    """

    def __init__(self, dim_x, dim_c, dim_z, dim_hidden, X_true, tau, batch_size, rec_dict=None):
        super(ProductOfGaussians, self).__init__()

        # TODO: Initialize the weights of the NN layers to have 0 mean wrt training data
        """ 
        - the weight matrices are stored as their own transpose, i.e. w_in has shape (dim_hidden, dim_x)
        - w_in has shape (dim_hidden, dim_x)
        - w_in_out has shape (batch_size, dim_hidden)

        NOTE: X_true in this recognition model has shape (dim_x, (tau+1)), to contain all the needed timesteps in 
        regard to the HRF """

        self.initInstanceVariables(batch_size, dim_hidden, dim_x, dim_c, dim_z, tau, rec_dict)
        self.init_encoder(X_true)

        if rec_dict is not None:
            self.load_state_dict(rec_dict, strict=False)

    def initInstanceVariables(self, batch_size, dim_hidden, dim_x, dim_c, dim_z, tau, rec_dict):
        super().initInstanceVariables(batch_size, dim_hidden, dim_x, dim_c, dim_z, rec_dict)
        self.dim_x = dim_x * (tau + 1)


'''Recognition Models as used by Leo. Until now I did not test the performance of these models. Most likely need to implement line 
regularization and anealing first. '''


class DiagonalCovariance(nn.Module):
    def __init__(self, dim_x, dim_z):
        super(DiagonalCovariance, self).__init__()
        self.d_x = dim_x
        self.d_z = dim_z
        self.w_filter = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.mean = nn.Linear(self.d_x, self.d_z, bias=False)
        self.logvar = nn.Linear(self.d_x, self.d_z, bias=True)
        # self.B_inv = tc.pinverse(nn.Parameter(tc.randn(self.d_z, self.d_x), requires_grad=False))
        # self.mean = nn.Sequential(nn.Linear(self.d_x, self.d_z), nn.LeakyReLU(), nn.Linear(self.d_z, self.d_z))

    def filter(self, x):
        xm1 = torch.cat((x[0:1, :], x), dim=0)[:-1]
        xp1 = torch.cat((x, x[-1, :].unsqueeze(0)), dim=0)[1:]
        return self.w_filter * xm1 + (1 - 2 * self.w_filter) * x + self.w_filter * xp1

    def forward(self, x):
        x = x.view(-1, self.d_x)
        x = self.filter(x)
        mu = self.mean(x)
        log_stddev = self.logvar(x)
        # mu = tc.einsum('xz,bx->bz', (self.B_inv, x))        T = x.shape[0]
        sample = mu + torch.exp(log_stddev) * torch.randn(self.T, self.d_z, device=x.device)
        entropy = torch.sum(log_stddev) / self.T
        return sample, entropy


class DiagonalCovarianceHRF(nn.Module):
    def __init__(self, dim_x, dim_z, tau):
        super(DiagonalCovarianceHRF, self).__init__()
        self.d_x = dim_x * (tau + 1)
        self.d_z = dim_z
        self.w_filter = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.mean = nn.Linear(self.d_x, self.d_z, bias=False)
        self.logvar = nn.Linear(self.d_x, self.d_z, bias=True)
        # self.B_inv = tc.pinverse(nn.Parameter(tc.randn(self.d_z, self.d_x), requires_grad=False))
        # self.mean = nn.Sequential(nn.Linear(self.d_x, self.d_z), nn.LeakyReLU(), nn.Linear(self.d_z, self.d_z))

    def filter(self, x):
        xm1 = torch.cat((x[0:1, :], x), dim=0)[:-1]
        xp1 = torch.cat((x, x[-1, :].unsqueeze(0)), dim=0)[1:]
        return self.w_filter * xm1 + (1 - 2 * self.w_filter) * x + self.w_filter * xp1

    def forward(self, x):
        x = x.view(-1, self.d_x)
        # x = self.filter(x)
        mu = self.mean(x)
        log_stddev = self.logvar(x)
        # mu = tc.einsum('xz,bx->bz', (self.B_inv, x))
        T = x.shape[0]
        sample = mu + torch.exp(log_stddev) * torch.randn(T, self.d_z, device=x.device)
        entropy = torch.sum(log_stddev) / T
        return sample, entropy
