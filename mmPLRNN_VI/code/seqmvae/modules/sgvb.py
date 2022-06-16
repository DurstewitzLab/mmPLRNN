import torch
from torch import nn, optim
import statespacedataset
from torch.utils.data import DataLoader
import generative_model
import recognition_model
import datagenerator
import hrf_convolution
import plotting
import testing
import helpers
import utils
import numpy as np
import os
import time
from matplotlib import pyplot as plt
import psutil


class SGVB:
    """Class to train a recognition and generative model jointly via Stochastic Gradient 
    Variational Bayes.

    Arguments:
        args (dict): dictionary that has to contain the following parameters:
            * dim_x (int): dimension of the observation space
            * dim_c (int): dimension of the categorical observation space
            * dim_z (int): dimension of the latent space
            * lr (float): learning rate
            * use_cuda (bool): Train the model on the GPU if True
            * seed (int): Set a random seed for reproducibility of the results
            * batch_size (int): Size of the training batches
            * drop_last (bool): Whether or not to drop the remaining (smaller) batch if the total
                                length of the timeseries is not divisible by batch_size
            * num_worker (int): Number of subprocesses used for loading the dataset
            * N_samples (int): The number of samples drawn from the approximate posterior to
                               calculate an average log_likelihood out of these N_samples
            * sampler (str): choose the batch sampler between 'Shuffling' and 'Stochastic'
            * rec_model (str): Choose the recognition model between 'ProductOfGaussians' or 
                               'StructuredApproxPosterior'
        xTrue (torch.tensor): (T, dim_x) matrix of observations
        rec_dict (dict): Only required when using 'ProductOfGaussians' as recognition model. Must then contain
                         the initial values for the parameter matrices 'A', 'QinvChol', Q0invChol'
        gen_dict (dict, optional): See doc string of 'generative_model.py'
        nonlinearity (torch.nn.functional, optional): which nonlinearity to be used for the observation model
    """

    def __init__(self, args: dict, xTrue, cTrue=None, rec_dict=None, gen_dict=None, xTrueWithoutNoise=None,
                 xTrueUnconvolved=None, externalInput=None, movementRegressors=None, zTrue=None):

        self.initInstanceVariablesFromArgs(cTrue, xTrue, zTrue, args, xTrueWithoutNoise, xTrueUnconvolved, externalInput,
                                           movementRegressors)
        #torch.manual_seed(self.seed)

        self.rec_model = self.choose_recognition_model(self.xTrue, args, rec_dict)
        self.rec_model.apply(helpers.weights_init_uniform_rule)

        self.gen_model = generative_model.PLRNN(self.dim_x, self.dim_c, self.dim_z, self.tau, args, gen_dict,
                                                self.nonLinearity)

        self.init_cuda()
        self.model_params = list(self.rec_model.parameters()) + list(self.gen_model.parameters())
        self.optimizer = optim.Adam(self.model_params, self.lr)
        #if not self.analyseRealData:
            #plotting.plotAndSaveBothLorenz(1500, 'xNoiseVsX', self.xTrue, self.xTrueWithoutNoise, self.trial_path)

        self.init_sampler(args)
        if self.use_convolution:
            self.state_space_data = statespacedataset.StateSpaceDataset(self.xConvolved[:self.T],
                                                                        self.cTrue[:self.T])
        if self.analyseRealData:
            self.state_space_data = statespacedataset.StateSpaceDataset(self.xTrue, self.externalInput,
                                                                        self.movementRegressors)
        if self.analyseRealDataMultimodal:
            self.state_space_data = statespacedataset.StateSpaceDataset(self.xTrue, self.cTrue, self.externalInput, self.movementRegressors)
        if not self.analyseRealData and not self.analyseRealDataMultimodal:
            self.state_space_data = statespacedataset.StateSpaceDataset(self.xTrue[:self.T], self.cTrue[:self.T])

        self.dataloader = DataLoader(self.state_space_data, batch_sampler=self.sampler,
                                     **self.kwargs)

    def initInstanceVariablesFromArgs(self, cTrue, xTrue, zTrue, args, xTrueWithoutNoise=None, xTrueUnconvolved=None,
                                      externalInput=None, movementRegressors=None):
        self.args = args
        self.dim_x = args['dim_x']
        self.dim_c = args['dim_c']
        self.dim_z = args['dim_z']
        self.lr = args['lr']
        self.epochs = args['epochs']
        self.use_cuda = args['use_cuda']
        self.seed = args['seed']
        self.batchSize = args['batch_size']
        self.num_workers = args['num_workers']
        self.N_samples = args['N_samples']
        self.use_clipping = args['use_clipping']
        self.clipping_value = args['clipping_value']
        self.useRecognitionModelClipping = args['useRecognitionModelClipping']
        self.chosen_rec_model = args['rec_model']
        self.trial_path = args['trial_path']
        self.use_annealing = args['use_annealing']
        self.validate_after_epoch = args['validate_after_epoch']
        self.tau = args['tau']
        self.add_noise = args['add_noise']
        self.noise_percentage = args['noise_percentage']
        self.log_trajectory = args['log_trajectory']
        self.use_convolution = args['use_convolution']
        self.T = args['T']
        self.use_regularization = args['use_regularization']
        self.nonLinearity = args['nonLinearity']
        self.analyseRealData = args['analyseRealData']
        self.analyseRealDataMultimodal = args['analyseRealDataMultimodal']

        self.use_hrf = args['use_hrf']
        self.useExplicitHrf = args['useExplicitHrf']
        if self.use_hrf and not self.useExplicitHrf:
            self.dim_hidden = 20
            self.init_hrf_NN(self.dim_z * (self.tau + 1))
        self.repetitionTime = args['repetitionTime']
        self.hrf = hrf_convolution.haemodynamicResponseFunction(self.repetitionTime)
        if self.tau == 0:
            assert len(self.hrf) == 1, "If tau = 0, the hrf vector should be exactly one element long, got {}".format(
                len(self.hrf))

        if self.analyseRealData or self.analyseRealDataMultimodal:
            self.T = len(xTrue)

        self.xTrue = xTrue
        self.zTrue = zTrue

        if self.useExplicitHrf:
            self.xTrueUnconvolved = xTrueUnconvolved
        if xTrueWithoutNoise is not None:
            self.xTrueWithoutNoise = xTrueWithoutNoise
        elif xTrueWithoutNoise is None:
            self.xTrueWithoutNoise = xTrue.detach().clone()

        if self.add_noise:
            self.xTrue = helpers.addGaussianNoiseToData(self.xTrue, self.noise_percentage)
        if self.use_convolution:
            self.setConvolvedData()
        self.cTrue = cTrue

        self.externalInput = externalInput
        self.movementRegressors = movementRegressors
        print(externalInput, movementRegressors)
        self.epoch_val_kldx_loss = float(-1)
        self.best_loss = float(100)

        self.previousBatch_Zsamples = None

    def setConvolvedData(self):
        data = self.xTrue[:self.T].detach().clone().t()
        hrf = hrf_convolution.haemodynamicResponseFunction()
        convolvedData = torch.zeros(len(data), len(data[0]) + len(hrf))
        for ind in range(0, len(data)):
            convolvedData[ind] = hrf_convolution.convolve_data(data[ind], hrf)
        self.xConvolved = convolvedData.t()
        # plt.plot(self.xTrue.t()[0][:1000], color='b', alpha=0.6)
        # plt.plot(convolvedData[0][:1000], color='r', alpha=0.6)
        # plt.show()
        # plotting.plotBothLorentz(1500, self.xTrue, self.xConvolved)

    def init_sampler(self, args):
        if args['sampler'] == 'Shuffling':
            self.drop_last = args['drop_last']
            self.sampler = statespacedataset.ShufflingBatchSampler(self.xTrue[:self.T], self.batchSize,
                                                                   self.drop_last)
            if args['use_convolution']:
                self.sampler = statespacedataset.ShufflingBatchSampler(self.xConvolved[:self.T], self.batchSize,
                                                                       self.drop_last)

        elif args['sampler'] == 'Stochastic':
            self.sampler = statespacedataset.StochasticBatchSampler(self.xTrue[:self.T], self.batchSize)
            if args['use_convolution']:
                self.sampler = statespacedataset.StochasticBatchSampler(self.xConvolved[:self.T], self.batchSize)

        else:
            raise NameError('use \'Shuffling\' or \'Stochastic\' ')

    def choose_recognition_model(self, X_true, args, rec_dict):
        if self.chosen_rec_model == 'ProductOfGaussians':
            rec_model = recognition_model.ProductOfGaussians(self.dim_x, self.dim_z,
                                                             args['dim_hidden'], X_true, self.batchSize, rec_dict)
        elif self.chosen_rec_model == 'ProductOfGaussiansMultimodal':
            rec_model = recognition_model.ProductOfGaussiansMultimodal(self.dim_x, self.dim_c, self.dim_z,
                                                                       args['dim_hidden'], X_true, self.batchSize,
                                                                       rec_dict)
        elif self.chosen_rec_model == 'ProductOfGaussiansHRF':
            X_true_hrf = utils.reshape_data_for_hrfEncoderInit(self.tau, X_true)
            rec_model = recognition_model.ProductOfGaussiansHRF(self.dim_x, self.dim_z,
                                                                args['dim_hidden'], X_true_hrf, self.tau,
                                                                self.batchSize, rec_dict)

        elif self.chosen_rec_model == 'ProductOfGaussiansMultimodalHRF':
            X_true_hrf = utils.reshape_data_for_hrfEncoderInit(self.tau, X_true)
            rec_model = recognition_model.ProductOfGaussiansMultimodalHRF(self.dim_x, self.dim_c, self.dim_z,
                                                                          args['dim_hidden'], X_true_hrf, self.tau,
                                                                          self.batchSize,
                                                                          rec_dict)
        elif self.chosen_rec_model == 'DiagonalCovarianceHRF':
            rec_model = recognition_model.DiagonalCovarianceHRF(self.dim_x, self.dim_z, self.tau)
        else:
            raise NameError(
                'use \'ProductOfGaussians\', \'ProductOfGaussiansHRF\' \'ProductOfGaussiansMultimodal\' or \'ProductOfGaussiansMultimodalHRF\' or '
                '\'DiagonalCovarianceHRF\'')

        if self.useRecognitionModelClipping:
            rec_model.activateRecognitionModelClipping()

        return rec_model

    def init_cuda(self):
        """Set up Cuda if training on GPU is available and desired."""

        self.cuda = self.use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")
        print("Training on " + str(self.device) + "...")

        # enable usage of pin memory if using cuda (only required when using Dataloader)
        self.kwargs = {'num_workers': self.num_workers, 'pin_memory': True} if self.cuda else {}

        # If there are mode than just one GPUs available, use them:
        if self.use_cuda and torch.cuda.device_count() > 1:
            print("We train our model on ", torch.cuda.device_count(), " GPUs")
            self.rec_model = nn.DataParallel(self.rec_model)
            self.gen_model = nn.DataParallel(self.gen_model)

        # push the model onto the device (GPU or CPU)
        self.rec_model.to(self.device)
        self.gen_model.to(self.device)

    def getBestLossAndEpoch(self):
        return self.best_loss, self.epoch_val_kldx_loss

    def train_batch(self):
        """ Training method using batches of the timeseries.
        Returns: cost (list): Contains the cost for each of the 'self.epochs' epochs
        """
        process = psutil.Process(os.getpid())
        self.rec_model.train()
        self.gen_model.train()
        model_dir = self.trial_path
        cost = []

        print("Starting training...")

        for epoch in range(1, self.epochs + 1):

            train_loss = 0
            start_time = time.time()

            if not self.analyseRealData and not self.analyseRealDataMultimodal:
                for batch_idx, (batch_X, batch_C) in enumerate(self.dataloader):
                    #print(batch_idx)
                    train_loss = self.makeGradientStep(batch_X, batch_C, train_loss)
            if self.analyseRealData:
                for batch_idx, (batch_X, batch_ex, batch_reg) in enumerate(self.dataloader):
                    train_loss = self.makeGradientStepForRealData(batch_X, batch_ex, batch_reg, train_loss)
            if self.analyseRealDataMultimodal:
                for batch_idx, (batch_X, batch_C, batch_ex, batch_reg) in enumerate(self.dataloader):
                    train_loss = self.makeGradientStepForRealData(batch_X, batch_ex, batch_reg, train_loss, batch_C)

            self.validateAndSaveModels(cost, epoch, model_dir, train_loss)
            cost.append(train_loss)
            print('====> Epoch: {} Average loss: {:.4f}'.format(
                epoch, train_loss))
            print("Memory percent: {}".format(process.memory_percent()))
            end_time = time.time()
            print("t = {}".format(end_time - start_time))

        return cost

    def makeGradientStep(self, batch_X, batch_C, train_loss):
        self.optimizer.zero_grad()
        if self.use_hrf:
            #TODO: adapt this part for the mathematically correct recognition model to work (PoGMHRF).
            batch_X_reshaped = None
            if self.tau is 0:
                assert torch.all(batch_X_reshaped.eq(
                    batch_X)), "x_t:tau is unequal x_t allthough tau = 0. This should never happen."
            #print("Zero gradients:")
            loss = self.lossHRF(batch_X, batch_X_reshaped, batch_C)
            loss.backward()
            #print("Nonzero gradients:")
            #loss2 = self.lossHRF(batch_X, batch_X_reshaped, batch_C)

        else:
            #print("Zero gradients:")
            loss = self.loss(batch_X, batch_C)
            loss.backward()
            #print("Nonzero gradients:")
            #loss2 = self.loss(batch_X, batch_C)


        if self.use_clipping:
            torch.nn.utils.clip_grad_norm_(self.rec_model.parameters(), self.clipping_value)
        train_loss += loss.item()
        self.optimizer.step()
        return train_loss

    def makeGradientStepForRealData(self, batch_X, batch_ex, batch_reg, train_loss, batch_C=None):
        self.optimizer.zero_grad()
        loss = self.lossRealData(batch_X, batch_ex, batch_reg, batch_C)
        loss.backward()
        if self.use_clipping:
            torch.nn.utils.clip_grad_norm_(self.rec_model.parameters(), self.clipping_value)
        train_loss += loss.item()
        self.optimizer.step()
        return train_loss

    def loss(self, batch_X, batch_C):
        """Compute the average cost for a sample in the given batch X_data for the
        evidence lower bound (ELBO) cost function.
        Arguments:
            batch_X (torch.tensor): (time_dim, dim_x) batch of observations
        Returns:
            cost/time_dim (torch.tensor): shape () tensor containing the average cost of a sample
        """
        time_dim = batch_X.shape[0]

        if self.chosen_rec_model == 'ProductOfGaussians':
            self.rec_model.forward(batch_X)
            rec_entropy = self.rec_model.evalEntropy()
        if self.chosen_rec_model == 'ProductOfGaussiansMultimodal':
            self.rec_model.forward(batch_X, batch_C)
            rec_entropy = self.rec_model.evalEntropy()

        Z_sample = self.rec_model.getSample()

        if self.chosen_rec_model == 'ProductOfGaussians':
            gen_log_likelihood = self.gen_model.log_likelihood(batch_X, Z_sample)

        if self.chosen_rec_model == 'ProductOfGaussiansMultimodal':
            gen_log_likelihood = self.gen_model.log_likelihood_multimodal(batch_X, batch_C, Z_sample)

        gen_log_likelihood = gen_log_likelihood / self.N_samples / self.batchSize
        rec_entropy /= self.batchSize

        cost = -(rec_entropy + gen_log_likelihood)

        if self.use_regularization:
            cost += self.lossRegularization(5)

        return cost / time_dim

    def lossHRF(self, X_data, X_data_reshaped, C_data):
        """Compute the average cost for a sample in the given batch X_data for the
        evidence lower bound (ELBO) cost function.

        Arguments:
            epoch (int): number of the current epoch in the training procedure (required for logging)
            X_data (torch.tensor): (time_dim, dim_x) batch of observations

        Returns:
            cost/time_dim (torch.tensor): shape () tensor containing the average cost of a sample
        """

        time_dim = X_data.shape[0]
        Z_sample, rec_entropy = self.getZSamplesAndEntropy(C_data, X_data, X_data_reshaped)
        #print("Z_sample grad:")
        #print(Z_sample.grad_fn)

        batch_Z_tau_to_t_3d = utils.reshapeZSamplesForHRF(Z_sample,
                                                                            self.batchSize,
                                                                            self.dim_z, self.tau)
        Z_sample_convolved = self.calcHRFtimesZ(batch_Z_tau_to_t_3d)
        #print(Z_sample.shape, Z_sample_convolved.shape, batch_Z_tau_to_t_3d.shape, batch_Z_tau_to_t.shape)
        #print("Z_sample_convolved grad:")
        #print(Z_sample_convolved.grad_fn)
        #print("Z_tau_to_t_3d grad:")
        #print(batch_Z_tau_to_t_3d.grad_fn)
        #print("batch_Z_tau_to_t grad")
        #print(batch_Z_tau_to_t.grad_fn)
        #print("---------------------------------")
        #testing.test_matrix_shape_on_timedependency(batch_Z_tau_to_t, self.dim_z, self.tau)
        #print("get_Z_samples_for_HRF tested succesfully")
        #if self.tau is 0:
        #    assert torch.all(
        #        batch_Z_tau_to_t.eq(Z_sample_convolved)), "z_t:tau is unequal z_t allthough tau = 0. This should never happen."
        #    assert torch.all(
        #        batch_Z_tau_to_t_3d.eq(Z_sample_convolved)), "z_t:tau is unequal z_t allthough tau = 0. This should never happen."

        #if self.useExplicitHrf:
        #    self.gen_model.calcAndSetHRFtimesZ(batch_Z_tau_to_t_3d)
        #else:
        #    self.gen_model.forward(batch_Z_tau_to_t, self.dim_z * (self.tau + 1))

        gen_log_likelihood = self.getGenLogLikelihood(C_data, X_data, Z_sample)
        #else:
        #    gen_log_likelihood = self.getGenLogLikelihood(C_data, X_data, Z_sample)

        gen_log_likelihood = gen_log_likelihood / self.N_samples / (self.batchSize - self.tau)
        rec_entropy /= (self.batchSize - self.tau)

        cost = -(rec_entropy + gen_log_likelihood)

        if self.use_regularization:
            cost += self.lossRegularization(3)

        return cost / time_dim

    def lossRealData(self, batch_X, batch_ex, batch_reg, batch_C=None):
        """Compute the average cost for a sample in the given batch X_data for the
        evidence lower bound (ELBO) cost function.
        Arguments:
            batch_X (torch.tensor): (time_dim, dim_x) batch of observations
        Returns:
            cost/time_dim (torch.tensor): shape () tensor containing the average cost of a sample
        """
        time_dim = batch_X.shape[0]

        if self.chosen_rec_model == 'ProductOfGaussians':
            self.rec_model.forward(batch_X)
            rec_entropy = self.rec_model.evalEntropy()
        #else:
        #    raise NameError("lossRealData is only implemented for ProductOfGaussians at the moment.")
        if self.chosen_rec_model == 'ProductOfGaussiansMultimodal':
            self.rec_model.forward(batch_X, batch_C)
            rec_entropy = self.rec_model.evalEntropy()

        Z_sample = self.rec_model.getSample()

        if self.chosen_rec_model == 'ProductOfGaussians':
            gen_log_likelihood = self.gen_model.log_likelihood(batch_X, Z_sample, batch_ex, batch_reg)

        if self.chosen_rec_model == 'ProductOfGaussiansMultimodal':
            gen_log_likelihood = self.gen_model.log_likelihood_multimodal(batch_X, batch_C, Z_sample, batch_ex, batch_reg)

        gen_log_likelihood = gen_log_likelihood / self.N_samples / self.batchSize
        rec_entropy /= self.batchSize

        cost = -(rec_entropy + gen_log_likelihood)

        if self.use_regularization:
            cost += self.lossRegularization(3)

        return cost / time_dim

    def lossRegularization(self, nRegStates):
        A, W, h = self.gen_model.getAWandH()
        AW = torch.diag(A) + W
        n = nRegStates
        return utils.squared_error(AW[:n, :n] - torch.eye(n)) + utils.squared_error(h[:n])

    def getZSamplesAndEntropy(self, C_data, X_data, X_data_reshaped):
        if self.chosen_rec_model == 'ProductOfGaussiansMultimodal':
            self.rec_model.forward(X_data, C_data)
            rec_entropy = self.rec_model.evalEntropy()
            Z_sample = self.rec_model.getSample()
        elif self.chosen_rec_model == 'ProductOfGaussiansHRF':
            #TODO: make sure this needs no further adjustment, if X_data_reshaped is handeled correctly
            self.rec_model.forward(X_data_reshaped)
            rec_entropy = self.rec_model.evalEntropy()
            Z_sample = self.rec_model.getSample()
        elif self.chosen_rec_model == 'ProductOfGaussiansMultimodalHRF':
            # TODO: make sure this needs no further adjustment, if X_data_reshaped is handeled correctly
            self.rec_model.forward(X_data_reshaped, C_data)
            rec_entropy = self.rec_model.evalEntropy()
            Z_sample = self.rec_model.getSample()
        elif self.chosen_rec_model == 'DiagonalCovarianceHRF':
            Z_sample, rec_entropy = self.rec_model.forward(X_data_reshaped)
        else:
            raise NameError(
                'This loss function should only be used in combination with \'ProductOfGaussiansHRF\', '
                '\'ProductOfGaussiansMultimodalHRF\', \'ProductOfGaussiansMultimodal + -simple\' or \'DiagonalCovarianceHRF\'')
        return Z_sample, rec_entropy

    def getGenLogLikelihood(self, C_data, X_data, Z_sample):
        gen_log_likelihood = None
        if self.chosen_rec_model == 'ProductOfGaussiansMultimodal':
            gen_log_likelihood = self.gen_model.log_likelihood_multimodal(X_data, C_data, Z_sample)
        elif self.chosen_rec_model == 'ProductOfGaussiansHRF':
            gen_log_likelihood = self.gen_model.log_likelihood(X_data, Z_sample)
        elif self.chosen_rec_model == 'ProductOfGaussiansMultimodalHRF':
            gen_log_likelihood = self.gen_model.log_likelihood_multimodal(X_data, C_data, Z_sample)
        elif self.chosen_rec_model == 'DiagonalCovarianceHRF':
            gen_log_likelihood = self.gen_model.log_likelihood(X_data, Z_sample)

        if gen_log_likelihood is None:
            raise ValueError("log_likelihood = None. This shouldn't happen. Aborting.")

        return gen_log_likelihood

    def validateAndSaveModels(self, cost, epoch, model_dir, train_loss):

        if (self.validate_after_epoch < epoch) and cost:
            if train_loss < min(cost):
                torch.save(self.rec_model.state_dict(), model_dir + '/best_loss_rec_model.pt')
                torch.save(self.gen_model.state_dict(), model_dir + '/best_loss_gen_model.pt')
                self.epoch_val_kldx_loss = epoch
                self.best_loss = train_loss

        if self.log_trajectory and (epoch % 500 == 0):
            trajectory_dir = model_dir + '/trajectory_over_time'
            if not os.path.exists(trajectory_dir):
                os.mkdir(trajectory_dir)
            torch.save(self.rec_model.state_dict(), trajectory_dir + '/rec_model_{}.pt'.format(epoch))
            torch.save(self.gen_model.state_dict(), trajectory_dir + '/gen_model_{}.pt'.format(epoch))

    def calcKLx(self, T, dict=None):
        current_model = self.getDataGeneratorFromDict(dict)
        if not self.use_hrf:
            T -= self.tau
        X, Z = current_model.generate_timeseries(T, noise=False)
        print(T, len(X), len(self.xTrueWithoutNoise[:T]))
        #X = X[::3]
        #if self.useExplicitHrf:
        #    kld = spatial_kullback(X, self.xTrueUnconvolved[:T], n_bins=10)
        #else:
        kld = utils.spatial_kullback(X, self.xTrueWithoutNoise[:T], n_bins=10)
        return kld

    def calcKLxForHRF(self, T, dict=None):
        current_model = self.getDataGeneratorFromDict(dict)
        X, Z = current_model.generate_timeseries_with_hrf(T, noise=False)
        #Z = Z[::3]
        zTrue = self.zTrue[:T].clone().detach()
        #print(len(Z), len(zTrue))
        zTrue = utils.standardiseData(self.dim_z, zTrue)
        #if self.use_hrf:
        #    Z = Z[self.tau:T]
        #    zTrue = zTrue[self.tau:T]
        #    print(len(Z), len(zTrue))
        #Z = utils.standardiseData(self.dim_z, Z)
        kld = utils.spatial_kullback(Z, zTrue, n_bins=5)
        return kld

    def calcKLz(self, T, dict=None):
        current_model = self.getDataGeneratorFromDict(dict)
        muX_gen, muZ_gen = current_model.generate_timeseries(T, noise=False)
        self.rec_model.setBatchSize(T)
        if self.chosen_rec_model == 'ProductOfGaussians':
            self.rec_model.forward(self.xTrue[:T])
        if self.chosen_rec_model == 'ProductOfGaussiansMultimodal':
            self.rec_model.forward(self.xTrue[:T], self.cTrue[:T])
        muZ_inf = self.rec_model.getSample(noise=False)

        '''In Leos code, they replace the standard deviation (std) with ones. This might be due to the case, 
        that they use a very simple recognition model, and therefore the covariance matrix is not really meaningfull. 
        For the archer model, this might not be the case and maybe we can directly use the std. '''
        cov_inf = torch.ones(T, self.dim_z)
        cov_gen = torch.ones(T, self.dim_z)

        klZ_var, _ = utils.calc_kl_var(muZ_inf, cov_inf, muZ_gen, cov_gen)
        klZ_mc, _ = utils.calc_kl_mc(muZ_inf, cov_inf, muZ_gen, cov_gen)
        return float(klZ_var), float(klZ_mc)

    def getRealDataMSE(self, dict=None):
        current_model = self.getDataGeneratorFromDict(dict)
        T = len(self.xTrueWithoutNoise)
        X, Z = current_model.generate_timeseries(T, noise=False)
        loss = torch.nn.MSELoss()
        meanSquaredErrors = []
        for ind in range(0,len(X.t())):
            meanSquaredError = loss(self.xTrueWithoutNoise.t()[ind], X.t()[ind])
            meanSquaredErrors.append(float(meanSquaredError))
        return meanSquaredErrors

    def getRealDataMSE_nstep(self, dict=None):
        nsteps = 15
        current_model = self.getDataGeneratorFromDict(dict)
        T = len(self.xTrueWithoutNoise)
        self.rec_model.setBatchSize(T)
        if self.chosen_rec_model == 'ProductOfGaussians':
            self.rec_model.forward(self.xTrue)
        if self.chosen_rec_model == 'ProductOfGaussiansMultimodal':
            self.rec_model.forward(self.xTrue, self.cTrue)
        zSamples = self.rec_model.getSample(noise=False)
        X = current_model.generate_nstep_timeseries(T, nsteps, zSamples, noise=False)

        loss = torch.nn.MSELoss()
        meanSquaredErrors = []
        for ind in range(0,len(X.t())):
            meanSquaredError = loss(self.xTrueWithoutNoise.t()[ind], X.t()[ind])
            meanSquaredErrors.append(float(meanSquaredError))
        return meanSquaredErrors

    def getDataGeneratorFromDict(self, dict=None):
        if dict is None:
            dict = self.gen_model.state_dict()
        current_model = datagenerator.DataGenerator(self.dim_x, self.dim_z, self.args, self.dim_c,
                                                    dict, 'uniform', False, self.nonLinearity, True,
                                                    self.externalInput, self.movementRegressors, self.zTrue)
        return current_model

    def findTransformationForMissDimTest(self):
        # Todo: Find some routine that can solve Ax=b for A (underdetermined linear system)
        transformationMatrix = 0
        return transformationMatrix

    def calcHRFtimesZ(self, Z_tau_to_t_3d):
        hrf_times_z = 0
        for ind in range(0, self.tau + 1):
            hrf_times_z += self.hrf[self.tau - ind] * Z_tau_to_t_3d[ind]
        return hrf_times_z

