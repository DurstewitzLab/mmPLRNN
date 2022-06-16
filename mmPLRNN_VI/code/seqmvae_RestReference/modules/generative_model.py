import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import utils
import hrf_convolution


class PLRNN(nn.Module):
    """Implements the observation and transition equations of the PLRNN in the log_likelihood method.

    Arguments:
        dim_x (int): dimension of observation space
        dim_c (int): dimension of categorical observation space
        dim_z (int): dimension of latent space
        args (dict): dictionary that must specify the following parameters:
            * stabilize (bool): whether or not to make use of the stability condition of the system
            * init_distr (str): initialize the parameters with a uniform or standard normal distribution
        gen_dict (dict, optional): dictionary that can specify any one out of the following parameters (all torch.tensors):
            * A : (dim_z) diagonal of auto-regressive weights matrix
            * W : (dim_z, dim_z) off-diagonal matrix of connection weights
            * h : (dim_z) bias term
            * R_x : (dim_x) diagonal of square root of covariance matrix of observations xt
            * R_z0 : (dim_z) diagonal of square root of covariance matrix of initial latent state z0
            * R_z : (dim_z) diagonal of square root of covariance matrix of latent states zt
            * mu0 : (dim_z) mean of the initial latent state z0
            * B : (dim_x, dim_z) matrix of regression weights
        nonlinearity (torch.nn.functional, optional): which nonlinearity to be used for the observation model

    """

    def __init__(self, dim_x, dim_c, dim_z, tau, args, gen_dict=None, nonlinearity=None):
        super(PLRNN, self).__init__()

        # NOTE: Put the threshold parameter h into the ReLU function (in the calculation of the log likelihood)

        self.dim_x = dim_x
        #self.dim_c = dim_c
        self.dim_c = dim_c - 1
        self.dim_z = dim_z
        # Todo: think about how these dimensions should be given to the PLRNN. Probably needs to be given to it
        #  directly. It would also make sense to determine all these dimensions with the data that we put in.
        self.dim_ex = args['dim_ex']
        self.dim_reg = args['dim_reg']

        self.tau = tau
        self.use_hrf = args['use_hrf']
        self.useExplicitHrf = args['useExplicitHrf']
        if self.use_hrf and not self.useExplicitHrf:
            self.dim_hidden = 20
            self.init_hrf_NN(self.dim_z * (self.tau + 1))
        if self.useExplicitHrf:
            self.repetitionTime = args['repetitionTime']
            self.hrf = hrf_convolution.haemodynamicResponseFunction(self.repetitionTime)
            if self.tau == 0:
                assert len(
                    self.hrf) == 1, "If tau = 0, the hrf vector should be exactly one element long, got {}".format(
                    len(self.hrf))

        #self.hrf_times_z = 0

        if args['init_distr'] == 'uniform':
            self.init_distr = torch.rand
        elif args['init_distr'] == 'normal':
            self.init_distr = torch.randn
        else:
            raise NameError('use \'uniform\' or \'normal\' ')

        self.nonlinearity = nonlinearity

        factor = 5
        upperBound = 1*factor
        lowerBound = -1*factor

        self.A = nn.Parameter(self.init_distr(self.dim_z), requires_grad=True)
        #self.A = nn.Parameter(torch.ones(self.dim_z) - torch.DoubleTensor(self.dim_z).uniform_(lowerBound, upperBound), requires_grad=True)
        
        self.W = nn.Parameter(self.init_distr(self.dim_z, self.dim_z), requires_grad=True)
        #self.W = nn.Parameter(torch.zeros(self.dim_z, self.dim_z) + torch.DoubleTensor(self.dim_z, self.dim_z).uniform_(lowerBound, upperBound), requires_grad=True)
        self.W = nn.Parameter(self.W * (1 - torch.eye(self.dim_z, self.dim_z)), requires_grad=True)

        # Matrix for external Inputs
        self.C = nn.Parameter(self.init_distr(self.dim_z, self.dim_ex), requires_grad=True)
        # Matrix for movement regressors
        self.J = nn.Parameter(self.init_distr(self.dim_x, self.dim_reg), requires_grad=True)

        if args['stabilize']:
            print("Stabilizing eigenvalues of A+W")
            self.stabilize()

        self.h = nn.Parameter(torch.zeros(self.dim_z), requires_grad=True)

        # R_x is the 'square root' of the diagonal elements of the diagonal covariance matrix of
        # observations xt and therefore stores the standard deviations which must not be negative.
        self.R_x = nn.Parameter(torch.rand(self.dim_x) / 10, requires_grad=True)
        self.R_z = nn.Parameter(torch.rand(self.dim_z), requires_grad=True)
        self.R_z0 = nn.Parameter(torch.rand(self.dim_z), requires_grad=True)

        self.mu0 = nn.Parameter(self.init_distr(self.dim_z), requires_grad=True)
        self.B = nn.Parameter(self.init_distr(self.dim_x, self.dim_z), requires_grad=True)
        # weights for categorical input
        self.beta = nn.Parameter(torch.rand(self.dim_c, self.dim_z), requires_grad=True)

        # override the parameters with the given gen_dict
        if gen_dict is not None:
            self.load_state_dict(gen_dict, strict=False)
       

    def stabilize(self):
        """Divide both W and A by the maximum eigenvector of W+A."""
        eigs = torch.eig(torch.diag(self.A) + self.W)[0]
        eigs_abs = torch.norm(eigs, p=2, dim=1)
        max_eig = torch.max(eigs_abs).item()
        self.W = nn.Parameter((self.W / max_eig) * 0.9999, requires_grad=True)
        self.A = nn.Parameter((self.A / max_eig) * 0.9999, requires_grad=True)

    def init_hrf_NN(self, dim_ztau):
        self.fc_hrf_in = nn.Linear(dim_ztau, self.dim_hidden * dim_ztau)
        self.fc_hrf_out = nn.Linear(self.dim_hidden * dim_ztau, self.dim_z)

    def encode_hrf(self, z, dim_ztau):
        # TODO: does in this case view work as expected?
        z = z.view(-1, dim_ztau)
        z = F.relu(self.fc_hrf_in(z))
        z = self.fc_hrf_out(z).detach().clone()
        return z

    def forward(self, z_tau_to_t, dim_ztau):
        self.hrf_times_z = self.encode_hrf(z_tau_to_t, dim_ztau)

    def get_hrfTimesZ(self):
        return self.hrf_times_z

    def getAWandH(self):
        return self.A, self.W, self.h

    def log_likelihood(self, xTrue, zSample, externalInputs=None, movementRegressors=None):
        """Calculate the log_likelihood of the generated timeseries.

        Arguments:
            xTrue (torch.tensor): (time_dim, dim_x) matrix of training set observations
            C_true (torch.tensor): (time_dim, dim_c)
            zSample (torch.tensor): (time_dim, dim_z) matrix of z samples from the approximate posterior
        Note: Here and in the following, 'time_dim' denotes the length of the input timeseries,
        which can either be the full timeseries or only a snippet (e.g. a batch)
        Returns:
            torch.tensor: Tensor of shape (1) containing the log_likelihood
        """
        time_dim = xTrue.shape[0]
        if self.use_hrf:
            batch_Z_tau_to_t_3d = utils.reshapeZSamplesForHRF(zSample,
                                                              time_dim,
                                                              self.dim_z, self.tau)
            Z_sample_convolved = self.calcHRFtimesZ(batch_Z_tau_to_t_3d)
            xTrue = xTrue[self.tau:]
            zSample = zSample[self.tau:]

        # mean of z0, shape=(dim_z)
        mu_0 = self.mu0
        if externalInputs is not None:
            mu_0 = mu_0 + externalInputs[0] @ self.C.t()

        # Use unsequeeze(0) to reshape from (dim_z) to (1, dim_z), where the zeroth dimension is the time axis.
        res_z0 = (zSample[0] - mu_0).unsqueeze(0)

        # Z_t contains all z_t for t=2,..., time_dim; Z_t.size()=(time_dim-1, dim_z)
        Z_t = zSample[1:, :]

        # Z_t1 contains the (z_t-1) values, i.e., the z_t for which t = 1,..., time_dim-1
        # Z_t1.size()=(time_dim-1, dim_z)
        Z_t1 = zSample[:-1, :]

        # A is a diagonal matrix and therefore the transposing is unnecessary but it is still used to
        # stay consistent. mu_z.size()=(time_dim-1, dim_z)

        mu_z = Z_t1 @ torch.diag(self.A).t() + F.relu(Z_t1) @ self.W.t()
        if externalInputs is not None:
            mu_z += externalInputs[1:] @ self.C.t()

        # res_z.size()=(time_dim-1, dim_z)
        res_z = Z_t - mu_z - self.h

        # mu_xt.size()=(time_dim, dim_x)
        if self.nonlinearity is None:
            mu_x = zSample @ self.B.t()

        else:
            mu_x = self.nonlinearity(zSample) @ self.B.t()
        if self.use_hrf:
            mu_x = Z_sample_convolved @ self.B.t()
        if movementRegressors is not None:
            mu_x += movementRegressors @ self.J.t()

        # res_x.size()=(time_dim, dim_x)
        res_x = xTrue - mu_x

        def exponent(res, cov_sqrt):

            """Calculate the expression in the exponent of a gaussian.
            Returns:
                torch.tensor: Tensor of size (time, time), where time is time_dim for x,
                              (time_dim-1) for z and 1 for z0.
            """
            assert (len(res.size()) == 2)
            assert (len(cov_sqrt.size()) == 1)  # for R_x
            return -0.5 * (res @ torch.inverse(torch.diag(cov_sqrt ** 2)) @ res.t())

        def log_det(cov_sqrt):

            """Calculates the constant terms of the log likelihood of a multivariate gaussian.
            Returns:
                torch.tensor: Tensor of shape (1)
            """

            assert (len(cov_sqrt.size()) == 1)
            dim = cov_sqrt.size()[0]
            return -0.5 * dim * torch.log(torch.tensor([2 * math.pi])) - 0.5 * torch.log(
                torch.det(torch.diag(cov_sqrt ** 2)))

        return log_det(self.R_z0) + torch.trace(exponent(res_z0, self.R_z0)) \
               + (time_dim - 1) * log_det(self.R_z) + torch.trace(exponent(res_z, self.R_z)) \
               + time_dim * log_det(self.R_x) + torch.trace(exponent(res_x, self.R_x))

    def log_likelihood_multimodal(self, xTrue, cTrue, zSample, externalInputs=None, movementRegressors=None):
        """Calculate the log_likelihood of the generated timeseries.
        Arguments:
            xTrue (torch.tensor): (time_dim, dim_x) matrix of training set observations
            cTrue (torch.tensor): (time_dim, dim_c)
            zSample (torch.tensor): (time_dim, dim_z) matrix of z samples from the approximate posterior
        Note: Here and in the following, 'time_dim' denotes the length of the input timeseries,
        which can either be the full timeseries or only a snippet (e.g. a batch)
        Returns:
            torch.tensor: Tensor of shape (1) containing the log_likelihood
        """
        log_likelihood = self.log_likelihood(xTrue, zSample, externalInputs, movementRegressors)

        if self.use_hrf:
            zSample = zSample[self.tau:]

        return log_likelihood + self.categorical_log_likelihood(cTrue, zSample[:-1, :])

    def categorical_log_likelihood(self, categorical_data, latent_data):
        result = 0
        if self.use_hrf:
            categorical_data = categorical_data[self.tau:]
        for ind in range(0, len(latent_data)):
            normalizationTerm = 1
            for idx in range(0, self.dim_c):
                normalizationTerm += torch.exp(self.beta[idx] @ latent_data[ind])
            for idx in range(0, self.dim_c + 1):
            #for idx in range(0, self.dim_c):
                #if categorical_data[ind][idx] == 1:
                if categorical_data[ind][idx] == 1 and idx < self.dim_c:
                    result += self.beta[idx] @ latent_data[ind] - torch.log(normalizationTerm)
                if categorical_data[ind][idx] == 1 and idx == self.dim_c:
                    result -= torch.log(normalizationTerm)
        return result

    def calcHRFtimesZ(self, Z_tau_to_t_3d):
        hrf_times_z = 0
        for ind in range(0, self.tau + 1):
            hrf_times_z += self.hrf[self.tau - ind] * Z_tau_to_t_3d[ind]
        return hrf_times_z

