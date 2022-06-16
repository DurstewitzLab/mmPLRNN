import torch
import torch.nn as nn
import torch.nn.functional as F
from mpl_toolkits.mplot3d import axes3d
from matplotlib import pyplot as plt
import utils
import numpy as np
from sklearn import metrics
import hrf_convolution
import random


class DataGenerator(nn.Module):
    """Generate a timeseries of latent variables Z and observations X according to the PLRNN framework.

    Arguments:
        dim_x (int): dimension of observation space
        dim_z (int): dimension of latent space
        gen_dict (dict): dictionary, that can specify any one out of the following parameters:
            * A : (dim_z) diagonal of auto-regressive weights matrix
            * W : (dim_z, dim_z) off-diagonal matrix of connection weights
            * h : (dim_z) bias term
            * R_x : (dim_x) diagonal of square root of covariance matrix of observations xt
            * R_z0 : (dim_z) diagonal of square root of covariance matrix of initial latent state z0
            * R_z : (dim_z) diagonal of square root of covariance matrix of latent states zt
            * mu0 : (dim_z) mean of the initial latent state z0
            * B : (dim_x, dim_z) matrix of regression weights
        init_distr (str): initialize the parameters with a uniform or standard normal distribution
        stabilize (bool): whether or not to make use of the stability condition for the system. It is
                          highly recommended to use stabilize=True, otherwise the covariance matrix
                          of the recognition model will usually not be positive semidefinite and
                          therefore the cholesky decomposition will not work.
        nonlinearity (torch.nn.functional): which nonlinearity to be used for the observation model
        B_ortho (bool): Whether or not the observation matrix B should be initialized as a (semi)
                        orthogonal matrix which is recommended as it leads to observations that are
                        not correlated that strongly

    """

    def __init__(self, dim_x, dim_z, args_dict, dim_c=None, gen_dict=None, init_distr='normal', stabilize=True,
                 nonlinearity=None, B_ortho=True, externalInputs=None, movementRegressors=None, zTrue=None):

        super(DataGenerator, self).__init__()
        self.dim_x = dim_x
        self.dim_z = dim_z
        #self.dim_c = dim_c
        self.dim_c = dim_c - 1

        self.dim_ex = args_dict['dim_ex']
        self.dim_reg = args_dict['dim_reg']

        self.tau = args_dict['tau']
        self.dim_hidden = 20
        self.use_hrf = args_dict['use_hrf']
        self.useExplicitHrf = args_dict['useExplicitHrf']
        self.repetitionTime = args_dict['repetitionTime']
        self.hrf = hrf_convolution.haemodynamicResponseFunction(self.repetitionTime)
        self.shuffleInitialConditions = False

        if init_distr == 'uniform':
            self.init_distr = torch.rand
        elif init_distr == 'normal':
            self.init_distr = torch.randn
        else:
            raise NameError('use \'uniform\' or \'normal\' ')

        self.stabilize_ = stabilize
        self.nonlinearity = nonlinearity

        self.A = nn.Parameter(self.init_distr(dim_z), requires_grad=False)
        self.W = nn.Parameter(self.init_distr(dim_z, dim_z), requires_grad=False)
        self.W = nn.Parameter(self.W * (1 - torch.eye(dim_z, dim_z)), requires_grad=False)

        if self.stabilize_:
            self.stabilize()

        self.h = nn.Parameter(torch.zeros(dim_z, requires_grad=False))

        # R_x is the 'square root' of the diagonal elements of the diagonal covariance matrix of
        # observations xt and therefore stores the standard deviations which must not be negative.
        self.R_x = nn.Parameter(torch.rand(self.dim_x) / 10, requires_grad=False)
        self.R_z = nn.Parameter(torch.rand(self.dim_z), requires_grad=False)
        self.R_z0 = nn.Parameter(torch.rand(self.dim_z), requires_grad=False)

        self.mu0 = nn.Parameter(self.init_distr(self.dim_z), requires_grad=False)
        self.B = nn.Parameter(self.init_distr(self.dim_x, dim_z), requires_grad=False)
        if B_ortho:
            torch.nn.init.orthogonal_(self.B)

        # weights for categorical input
        self.beta = nn.Parameter(torch.rand(self.dim_c, self.dim_z), requires_grad=False)
        self.hrf_times_z = 0
        self.init_hrf_NN(self.dim_z * (self.tau + 1))

        # Matrix for external Inputs
        self.C = nn.Parameter(self.init_distr(self.dim_z, self.dim_ex), requires_grad=False)
        # Matrix for movement regressors
        self.J = nn.Parameter(self.init_distr(self.dim_x, self.dim_reg), requires_grad=False)


        # override the parameters with the given gen_dict
        if gen_dict is not None:
            self.load_state_dict(gen_dict, strict=False)
            if self.shuffleInitialConditions:
                adaptionRate = 0.001
                self.h.requires_grad = False
                self.A -= adaptionRate * torch.rand(self.dim_z)
                self.W += adaptionRate * torch.randn(self.dim_z, dim_z)
                self.h += adaptionRate * torch.rand(self.dim_z)
                self.B -= adaptionRate * torch.rand(self.dim_x, self.dim_z)
                self.mu0 -= adaptionRate * torch.rand(self.dim_z)

        self.externalInputs = externalInputs
        self.movementRegressors = movementRegressors
        self.zTrue = zTrue

        #print(self.A)

    def init_hrf_NN(self, dim_ztau):
        self.fc_hrf_in = nn.Linear(dim_ztau, self.dim_hidden * dim_ztau)
        self.fc_hrf_out = nn.Linear(self.dim_hidden * dim_ztau, self.dim_z)

    def encode_hrf(self, z):
        dim_ztau = self.dim_z * (self.tau + 1)
        z = z.view(-1, dim_ztau)
        z = F.relu(self.fc_hrf_in(z))
        z = self.fc_hrf_out(z).detach().clone()
        return z

    def forward(self, z_tau_to_t):
        self.hrf_times_z = self.encode_hrf(z_tau_to_t)

    def stabilize(self):
        """Devide both W and A by the maximum eigenvector of W+A."""
        eigs = torch.eig(torch.diag(self.A) + self.W)[0]
        # eigenvalues can be complex therefore use 2-norm to calculate absolute value
        eigs_abs = torch.norm(eigs, p=2, dim=1)
        max_eig = torch.max(eigs_abs).item()
        self.W = nn.Parameter((self.W / max_eig) * 0.9999, requires_grad=False)
        self.A = nn.Parameter((self.A / max_eig) * 0.9999, requires_grad=False)

    def generate_timeseries(self, T, noise=True, z0=None, nstepIndex=None, perturbations=None):
        """Generates a time series of length T of observations xt and latent states zt.

        Arguments:
            T (int): length of the timeseries to be sampled
            noise (bool, optional): Use noise for sampling when True

        Returns:
            X (torch.tensor): (T, dim_x) matrix of observations xt
            Z (torch.tensor): (T, dim_z) matrix of latent states zt
        """

        # initialize the matrices containing all observations and latent states.
        Z = torch.zeros(T, self.dim_z, requires_grad=False)
        X = torch.zeros(T, self.dim_x, requires_grad=False)

        if noise:
            # draw the noise of zt from a standard normal
            epsilon = torch.randn(T, self.dim_z)

            # draw the noise of xt from the standard normal
            eta = torch.randn(T, self.dim_x)

        # generate initial latent state z0
        if z0 is not None:
            z0 = z0 + self.R_z0 * epsilon[0] if noise else z0
        else:
            z0 = self.mu0 + self.R_z0 * epsilon[0] if noise else self.mu0
        if self.externalInputs is not None:
            if nstepIndex is None:
                z0 = z0 + self.C @ self.externalInputs[0]
            else:
                z0 = z0 + self.C @ self.externalInputs[nstepIndex]
        if perturbations is not None:
            z0 = z0 + perturbations

        Z[0] = z0
        zt = z0  # torch always copies tensors
        #print(T, len(self.externalInputs))
        for t in range(1, T):
            # generate the subsequent latent states zt
            # in the transition equation there must always be a relu
            mu_zt = self.A * zt + self.W @ F.relu(zt) + self.h
            zt = mu_zt + self.R_z * epsilon[t] if noise else mu_zt
            if self.externalInputs is not None:
                if nstepIndex is None:
                    zt += self.C @ self.externalInputs[t]
                else:
                    zt += self.C @ self.externalInputs[nstepIndex + t]
                if perturbations is not None:
                    z0 = z0 + perturbations
            Z[t] = zt

        # generate the corresponding initial observation x0, therefore we also need z_tau_to_t in case of hrf.
        mu_xt = self.B @ self.nonlinearity(z0) if self.nonlinearity is not None else self.B @ z0
        if self.movementRegressors is not None:
            if nstepIndex is None:
                mu_xt += self.J @ self.movementRegressors[0]
            else:
                mu_xt += self.J @ self.movementRegressors[nstepIndex]

        '''These two lines are for testing only'''
        # self.forward(z_tau_to_t)
        # self.plotSomeStuff(Z, self.hrf_times_z)

        if self.use_hrf:
            z_tau_to_t_3d = utils.reshapeZSamplesForHRF(Z, T, self.dim_z, self.tau)
            if self.useExplicitHrf:
                 hrf_times_z = self.calcHRFtimesZ(z_tau_to_t_3d[:, 0])
            #else:
            #    self.forward(z_tau_to_t[0])

            temp = mu_xt.detach().clone()
            mu_xt = hrf_times_z @ self.B.t()
            if self.tau is 0:
                assert torch.all(
                    hrf_times_z.eq(Z[0])), "z_tau:t is unequal z_t allthough tau = 0. This should never happen."
                assert torch.all(mu_xt.eq(temp)), "z_tau:t is unequal z_t allthough tau = 0. This should never happen."

        xt = mu_xt + self.R_x * eta[0] if noise else mu_xt
        X[0] = xt

        tStart = 1
        if self.use_hrf:
            T -= self.tau

        for t in range(tStart, T):
            # generate the corresponding observations xt
            if self.nonlinearity is not None:
                mu_xt = self.B @ self.nonlinearity(Z[t])
            else:
                mu_xt = self.B @ Z[t]
            if self.use_hrf:
                if self.useExplicitHrf:
                    hrf_times_z = self.calcHRFtimesZ(z_tau_to_t_3d[:, t])

                    temp = mu_xt.detach().clone()
                    mu_xt = hrf_times_z @ self.B.t()  # TODO: implement + self.J @ r_t
                    if self.tau is 0:
                        assert torch.all(hrf_times_z.eq(
                            Z[t])), "z_tau:t is unequal z_t allthough tau = 0. This should never happen."
                        assert torch.all(mu_xt.eq(temp)), "{}  vs {}".format(mu_xt, temp)
                #else:
                #    self.forward(z_tau_to_t[t])
                #    mu_xt = self.hrf_times_z @ self.B.t()  # TODO: implement + self.J @ r_t
            #print(self.movementRegressors)
            if self.movementRegressors is not None:
                if nstepIndex is None:
                    mu_xt += self.J @ self.movementRegressors[t]
                else:
                    mu_xt += self.J @ self.movementRegressors[nstepIndex + t]


            xt = mu_xt + self.R_x * eta[t] if noise else mu_xt
            X[t] = xt

        if self.use_hrf:
            X = X[:-self.tau]
            Z = Z[self.tau:]
        # have to detach the tensors otherwise this will lead to errors in training during backprop
        # since the graph of the datagenerator will still be attached to X and Z
        return X.detach(), Z.detach()

    def generate_timeseries_without_hrf(self, T, noise=True):
        current_hrf_usage = self.use_hrf
        self.use_hrf = False
        X, Z = self.generate_timeseries(T, noise)
        self.use_hrf = current_hrf_usage
        return X, Z

    def generate_timeseries_with_hrf(self, T, noise=True):
        current_hrf_usage = self.use_hrf
        self.use_hrf = True
        current_explicitHRF_usage = self.useExplicitHrf
        self.useExplicitHrf = True
        X, Z = self.generate_timeseries(T, noise)
        self.use_hrf = current_hrf_usage
        self.useExplicitHrf = current_explicitHRF_usage
        return X, Z

    def generate_timeseries_without_regressors(self, T, noise=True):
        current_regressors = self.movementRegressors
        self.movementRegressors = None
        X, Z = self.generate_timeseries(T, noise)
        self.movementRegressors = current_regressors
        return X, Z

    def generate_timeseries_without_externalInputs(self, T, noise=True):
        current_externalInputs = self.externalInputs
        self.externalInputs = None
        X, Z = self.generate_timeseries(T, noise)
        self.externalInputs = current_externalInputs
        return X, Z

    def generate_nstep_timeseries(self, T, nsteps, zSample, noise=False):
        z0s = zSample[::nsteps]
        X = torch.zeros(T, self.dim_x, requires_grad=False)
        for ind in range(0, len(z0s)-1):
            nstepIndex = nsteps * ind
            x_nstep, z_nstep = self.generate_timeseries(nsteps, noise, z0s[ind], nstepIndex)
            X[ind*nsteps:(ind+1)*nsteps] = x_nstep
        return X.detach()

    def generate_reset_timeseries_for_fmriExperiment_fromCV(self, T, zInferredLeftOut, zInferredOther, cTrue, fileIndex, noise=False):
        '''We want to reset the generated trajectory to the inferred trajectory on every instruction phase'''
        z0s = []
        timesteps = [0]
        timestep = 0
        z0 = None
        for ind in range(0, len(cTrue)):
            timestep += 1
            if ind > 0 and ind < fileIndex * 72:
                if cTrue[ind] == 1:
                    z0 = zInferredOther[ind]
            if ind > fileIndex * 72 and ind < (fileIndex + 1) * 72:
                if cTrue[ind] == 1:
                    z0 = zInferredLeftOut[ind - fileIndex * 72]
            if ind > (fileIndex + 1) * 72:
                if cTrue[ind] == 1:
                    z0 = zInferredOther[ind - (fileIndex + 1) * 72]
            if z0 is not None:
                z0s.append(z0)
                timesteps.append(ind)
                timestep = 0
                z0 = None

        X = torch.zeros(T, self.dim_x, requires_grad=False)
        Z = torch.zeros(T, self.dim_z, requires_grad=False)
        for ind in range(0, len(z0s)-1):
            x_nstep, z_nstep = self.generate_timeseries(timesteps[ind+1]-timesteps[ind], noise, z0s[ind])
            X[timesteps[ind]:timesteps[ind+1]] = x_nstep
            Z[timesteps[ind]:timesteps[ind+1]] = z_nstep
        return X.detach(), Z.detach()

    def generate_reset_timeseries_for_fmriExperiment_fromFullTS(self, T, zInferred, cTrue, noise=False):
        '''We want to reset the generated trajectory to the inferred trajectory on every instruction phase'''
        z0s = [None]
        timesteps = [0]

        for ind in range(1, len(cTrue)):
            #"Normal experiment"
            if cTrue[ind] == 1 and (cTrue[ind] - cTrue[ind-1]) > 0:
            #"Rest category for reference experiment"
            #if cTrue[ind] == 1 and (cTrue[ind] - cTrue[ind - 1]) < 0:
                z0 = zInferred[ind]
                z0s.append(z0)
                timesteps.append(ind)

        timesteps.append(T)
        X = torch.zeros(T, self.dim_x, requires_grad=False)
        Z = torch.zeros(T, self.dim_z, requires_grad=False)
        print(len(z0s))
        for ind in range(0, len(timesteps)-1):
            x_nstep, z_nstep = self.generate_timeseries(timesteps[ind+1]-timesteps[ind], noise, z0s[ind])
            X[timesteps[ind]:timesteps[ind+1]] = x_nstep
            Z[timesteps[ind]:timesteps[ind+1]] = z_nstep
        return X.detach(), Z.detach()

    def generate_reset_timeseries_for_fmriExperiment_fromFullTS_randomResets(self, T, zInferred, noise=False):
        '''We want to reset the generated trajectory to the inferred trajectory on every instruction phase'''
        z0s = [None]
        timesteps = [0]

        for ind in range(0, 15):
            timestep = random.randint(20, 30)
            if timesteps[ind]+timestep >= 360:
                break
            else:
                timesteps.append(timesteps[ind]+timestep)
                z0 = zInferred[timesteps[ind]+timestep]
                z0s.append(z0)

        #for ind in range(0, len(zInferred)):
        #    timestep = random.randint(15,40)
        #    ind += timestep
        #    if ind > len(zInferred):
        #        break
        #    z0 = zInferred[ind]
        #    z0s.append(z0)
        #    timesteps.append(ind)

        timesteps.append(T)
        X = torch.zeros(T, self.dim_x, requires_grad=False)
        Z = torch.zeros(T, self.dim_z, requires_grad=False)

        #print(len(timesteps), len(z0s))
        #print(timesteps)
        for ind in range(0, len(timesteps)-1):
            x_nstep, z_nstep = self.generate_timeseries(timesteps[ind+1]-timesteps[ind], noise, z0s[ind])
            X[timesteps[ind]:timesteps[ind+1]] = x_nstep
            Z[timesteps[ind]:timesteps[ind+1]] = z_nstep
        return X.detach(), Z.detach()


    # def generate_reset_timeseries_for_fmriExperiment_AttractingBehaviour(self, T, noise=False):
    #     '''We want to reset the generated trajectory to the inferred trajectory on every instruction phase'''
    #
    #     timesteps = [0, 510, 700, T]
    #     perturbations = torch.zeros(20)
    #
    #     X = torch.zeros(T, self.dim_x, requires_grad=False)
    #     Z = torch.zeros(T, self.dim_z, requires_grad=False)
    #
    #     z0 = None
    #     for ind in range(0, len(timesteps)-1):
    #         x_nstep, z_nstep = self.generate_timeseries(timesteps[ind+1]-timesteps[ind], noise, z0, None, perturbations)
    #         X[timesteps[ind]:timesteps[ind+1]] = x_nstep
    #         Z[timesteps[ind]:timesteps[ind+1]] = z_nstep
    #         z0 = z_nstep[-1]
    #         #perturbations[10] = 0.5
    #     return X.detach(), Z.detach()

    def generate_reset_timeseries_for_fmriExperiment_AttractingBehaviour(self, T, cTrue, zInferred, noise=False):
        '''We want to reset the generated trajectory to the inferred trajectory on every instruction phase'''
        z0s = [None]
        timesteps = [0]

        ind = 1
        while ind < len(cTrue):
            if cTrue[ind] == 1 and (cTrue[ind] - cTrue[ind-1]) > 0 and ind > 50:
                z0 = zInferred[ind]
                z0s.append(z0)
                timesteps.append(ind)
                ind += 200
                break
            ind += 1

        timesteps.append(T)
        #print(timesteps)
        X = torch.zeros(T, self.dim_x, requires_grad=False)
        Z = torch.zeros(T, self.dim_z, requires_grad=False)

        for ind in range(0, len(timesteps)-1):
            x_nstep, z_nstep = self.generate_timeseries(timesteps[ind+1]-timesteps[ind], noise, z0s[ind])
            X[timesteps[ind]:timesteps[ind+1]] = x_nstep
            Z[timesteps[ind]:timesteps[ind+1]] = z_nstep
        return X.detach(), Z.detach()

    def generate_reset_timeseries_for_fmriExperiment_fromFullTS_perturb(self, T, cTrue, noise=False):
        '''We want to reset the generated trajectory to the inferred trajectory on every instruction phase'''
        timesteps = [0]
        for ind in range(1, len(cTrue)):
            if cTrue[ind] == 1 and (cTrue[ind] - cTrue[ind-1]) > 0:
                timesteps.append(ind)

        timesteps.append(T)
        perturbations = 0.1 * torch.randn(self.dim_z)
        X = torch.zeros(T, self.dim_x, requires_grad=False)
        Z = torch.zeros(T, self.dim_z, requires_grad=False)

        z0 = None
        for ind in range(0, len(timesteps)-1):
            x_nstep, z_nstep = self.generate_timeseries(timesteps[ind+1]-timesteps[ind], noise, z0, None, perturbations)
            X[timesteps[ind]:timesteps[ind+1]] = x_nstep
            Z[timesteps[ind]:timesteps[ind+1]] = z_nstep
            z0 = z_nstep[-1]
        return X.detach(), Z.detach()


    def generate_nstep_timeseries_carlo(self, T, nsteps, zSample):

        z_f = torch.zeros(T, self.dim_z)
        x_f = torch.zeros(T, self.dim_x)
        z_inf = zSample

        for t in range(nsteps, T):
            t_n = t - nsteps
            zt = z_inf[t_n]

            for i in range(nsteps, 0, -1):
                zt = self.A * zt + self.W @ F.relu(zt) + self.h

            z_f[t] = zt
            x_f[t] = self.B @ self.nonlinearity(zt)

        xAhead = x_f[nsteps:]
        return xAhead.detach()

#        xTrue = true_model.obs_latent[nsteps:]
#        nd = (T - nsteps) * dim_x
#        mse_z_for = ((xAhead - xTrue) ** 2).sum() / nd

    def generate_timeseries_from_zSamples(self, T, zSample):
        X = torch.zeros(T, self.dim_x, requires_grad=False)
        Z = zSample
        for t in range(0, T):
            zt = Z[t]
            if self.externalInputs is not None:
                zt += self.C @ self.externalInputs[t]

            if self.nonlinearity is not None:
                mu_xt = self.B @ self.nonlinearity(zt)
            else:
                mu_xt = self.B @ zt
            if self.movementRegressors is not None:
                mu_xt += self.J @ self.movementRegressors[t]

            xt = mu_xt
            X[t] = xt

        return X.detach(), Z.detach()


    def plotSomeStuff(self, zTrue, zHRF):
        zTrue = zTrue.detach().clone().numpy()
        zHRF = zHRF.detach().clone().numpy()
        plt.plot(zTrue, color='b', alpha=0.8)
        plt.plot(zHRF, color='r', alpha=0.8)
        # plt.show()

    def calc_categorical_pdf(self, latent_data):
        '''The categorical distribution is a product of probabilities^the ith component of the one-hot encoded
        category vector (p_i(c_t|z_t)^c_it). Since c_it only has 0's and one 1, it's a product of 1's and p_j
        where c_jt = 1. Thus we don't actually need the category vectors to calculate the probablities but rather
        need to calculate the probabilites for each j '''

        probabilities = torch.zeros(len(latent_data), self.dim_c+1)
        betas = self.beta.detach().clone()

        betas, latent_data = self.normalize(betas, latent_data)

        for ind in range(0, len(latent_data)):
            normalizationTerm = 1
            for idx in range(0, self.dim_c):
                normalizationTerm += torch.exp(betas[idx] @ latent_data[ind])
            for idx in range(0, self.dim_c):
                temp = torch.exp(betas[idx] @ latent_data[ind]) / normalizationTerm
                probabilities[ind][idx] = temp
                if (temp < 0 or temp > 1):
                    raise ValueError(
                        "Calculated probability outside of interval [0,1], this should never happen: {}".format(temp))
                if np.isnan(temp.detach().numpy()):
                    raise ValueError("Calculated nan, this should never happen: {}/{}....{}".format(
                        torch.exp(betas[idx] @ latent_data[ind]), normalizationTerm,
                        (betas[idx] @ latent_data[ind])))
            probabilities[ind][-1] = 1 / normalizationTerm

        #probabilities = self.sample_from_probabilities(probabilities)
        probabilities = self.choose_most_probable_probabilities(probabilities)
        return probabilities

    def normalize(self, betas, latent_data):

        for ind in range(len(betas)):
            max = torch.max(torch.abs(betas[ind]))
            betas[ind] /= max
            max = torch.max(torch.abs(latent_data[ind]))
            latent_data[ind] /= max
        return betas, latent_data

    def sample_from_probabilities(self, probabilities):
        sampled_probabilities = np.zeros(len(probabilities))
        category_vector = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        for ind in range(0, len(probabilities)):
            sampled_probabilities[ind] = np.random.choice(a=category_vector.detach().numpy(),
                                                          p=probabilities.detach().numpy()[ind])
        sampled_probabilities = torch.from_numpy(sampled_probabilities).type(torch.DoubleTensor)
        return sampled_probabilities

    def choose_most_probable_probabilities(self, probabilities):
        '''This returns the expected value'''
        filtered_probabilites = self.category_oneHot_to_indices(probabilities)
        return filtered_probabilites

    def category_reconstruction_success(self, C_true, C_trained_indices):
        '''C_true contains the one-hot-encoded category vector. C_trained_indices contains the indices of the categoires.'''

        if not len(C_true) == len(C_trained_indices):
            print("Input vectors must be of same length: {} != {}".format(len(C_true), len(C_trained_indices)))
            return 0
        C_true_indices = self.category_oneHot_to_indices(C_true)
        count = 0
        for ind in range(0, len(C_true_indices)):
            if C_true_indices[ind] == C_trained_indices[ind]:
                count += 1
        return count / len(C_true_indices)

    def category_oneHot_to_indices(self, C_true):
        '''Tested. This function works fine.'''
        C_true_indices = torch.zeros(len(C_true)).type(torch.DoubleTensor)
        for ind in range(0, len(C_true)):
            max, index = torch.max(C_true[ind], 0)
            C_true_indices[ind] = index
        return C_true_indices

    def make_category_confusion_matrix(self, C_true_categories, C_trained_categories):
        confusion_matrix = metrics.confusion_matrix(C_true_categories, C_trained_categories)
        return confusion_matrix

    def calc_categories_from_data(self, data):

        data = data.t()
        thresh_x1, thresh_x2, thresh_x3 = torch.mean(data[0]), torch.mean(data[1]), torch.mean(data[2])
        data = data.t()

        category_index_list = []
        index = 0
        for ind in range(0, len(data)):

            if (data[ind][0] >= thresh_x1
                    and data[ind][1] >= thresh_x2
                    and data[ind][2] >= thresh_x3):
                index = 1
            elif (data[ind][0] >= thresh_x1
                  and data[ind][1] >= thresh_x2
                  and data[ind][2] < thresh_x3):
                index = 2
            elif (data[ind][0] >= thresh_x1
                  and data[ind][1] < thresh_x2
                  and data[ind][2] > thresh_x3):
                index = 3
            elif (data[ind][0] >= thresh_x1
                  and data[ind][1] < thresh_x2
                  and data[ind][2] < thresh_x3):
                index = 4
            elif (data[ind][0] < thresh_x1
                  and data[ind][1] >= thresh_x2
                  and data[ind][2] >= thresh_x3):
                index = 5
            elif (data[ind][0] < thresh_x1
                  and data[ind][1] >= thresh_x2
                  and data[ind][2] < thresh_x3):
                index = 6
            elif (data[ind][0] < thresh_x1
                  and data[ind][1] < thresh_x2
                  and data[ind][2] >= thresh_x3):
                index = 7
            elif (data[ind][0] < thresh_x1
                  and data[ind][1] < thresh_x2
                  and data[ind][2] < thresh_x3):
                index = 8
            category_index_list.append(index - 1)
        return category_index_list

    def calcHRFtimesZ(self, Z_tau_to_t_3d):
        hrf_times_z = 0
        for ind in range(0, self.tau + 1):
            hrf_times_z += self.hrf[self.tau - ind] * Z_tau_to_t_3d[ind]

        return hrf_times_z

if __name__ == "__main__":
    dim_z = 3
    dim_x = 5
    dim_c = 8
    T = 100

    gendict = dict([
        ('A', 0.99 * torch.ones(dim_z)),
        ('W', torch.zeros(dim_z, dim_z)),
        ('R_x', torch.randn(dim_x)),
        ('R_z', torch.ones(dim_z)),
        ('R_z0', torch.ones(dim_z)),
        ('mu0', torch.zeros(dim_z)),
        ('B', torch.rand(dim_x, dim_z)),
    ])

    gendict = None

    while (True):
        # ts = DataGenerator(dim_x, dim_z, T, gendict, 'uniform', False, torch.nn.functional.relu)
        ts = DataGenerator(dim_x, dim_z, dim_c, gendict, 'uniform', True, None)
        X, Z = ts.generate_timeseries(T, noise=False)

        print('B = ', ts.B)
        print('X = ', X)
        print('Z = ', Z)

        fig = plt.figure(figsize=(16, 9))
        # `ax` is a 3D-aware axis instance, because of the projection='3d' keyword argument to add_subplot
        bx = fig.add_subplot(1, 1, 1, projection='3d')
        bx.set_xlabel('$x_1$', size=15)
        bx.set_ylabel('$x_2$', size=15)
        bx.set_zlabel('$x_3$', size=15)
        p = bx.plot(X[:, 0].numpy(), X[:, 1].numpy(), X[:, 2].numpy(),
                    antialiased=True,
                    linewidth=0.5, label='true')
        plt.show()

    ts = DataGenerator(dim_x, dim_z, dim_c, gendict, 'uniform', False, None)
    print(ts.state_dict())
    X_noise, Z_noise = ts.generate_timeseries(T, noise=True)
    X, Z = ts.generate_timeseries(T, noise=False)
    print('B = ', ts.B)
