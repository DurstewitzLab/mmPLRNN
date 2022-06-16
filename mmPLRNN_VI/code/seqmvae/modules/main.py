import torch
import argparse
import datagenerator
import utils
import load_matlab_data
import os
import plotting
import copy
import sgvb
import hrf_convolution
import numpy as np
import helpers as h
import faulthandler

faulthandler.enable()


class Main():
    def __init__(self):
        super(Main, self).__init__

    def main(self):
        '''Init args and some storage related things'''
        #dir_path = os.path.dirname(os.path.realpath(__file__))
        #print(dir_path)

        args = self.initArgs()
        self.setArgs(args)

        args_dict = self.initArgsDict(args)

        print("Using the following arguments (args_dict):")
        print("--------------------------------------------------------------------")
        print(args_dict)
        print("--------------------------------------------------------------------")

        filePath, writer = self.initWriter(args, args_dict)
        self.writeParameterFile(args_dict, filePath)

        print("Storing data in " + str(filePath) + ".")

        dim_z = args.dim_z
        dim_x = args.dim_x
        dim_c = args.dim_c
        T = args.T

        gendict, recdict = self.initModelDicts(dim_c, dim_x, dim_z)

        '''First get true data that we want to reconstruct'''
        dataPath = self.getDataPath(args)
        xTrueWithoutNoise, xTrueReduced, xTrue_unconvolved = 0, 0, 0
        zTrue = None
        trueDataGendict = None
        cTrue = None
        if not args.analyseRealData and not args.analyseRealDataMultimodal:
            xTrue, cTrue, xTrue_unconvolved, xTrueReduced, xTrueWithoutNoise, zTrue, trueDataGendict = self.getDataForInference(args, args_dict,
                                                                                                        dataPath,
                                                                                                        filePath)
        if args.analyseRealData:
            xTrue, externalInputs, movementRegressors = self.getRealData(args_dict, dataPath)
        if args.analyseRealDataMultimodal:
            xTrue, externalInputs, movementRegressors, cTrue = self.getRealDataMultimodal(args_dict, dataPath)
            #externalInputs = torch.zeros(externalInputs.shape)
            #movementRegressors = torch.zeros(movementRegressors.shape)
        if not args.analyseRealData and not args.analyseRealDataMultimodal:
            externalInputs, movementRegressors = 0, 0
        '''Create a SGVB object and train it'''
        args.T = len(xTrue)
        T = len(xTrue)

        if args.analyseRealData:
            sgvb_model = sgvb.SGVB(args_dict, xTrue, None, recdict, gendict, None, None, externalInputs,
                                   movementRegressors)
        elif args.analyseRealDataMultimodal:
            sgvb_model = sgvb.SGVB(args_dict, xTrue, cTrue, recdict, gendict, None, None, externalInputs,
                                   movementRegressors)
        else:
            sgvb_model = self.initSgvb(args, args_dict, cTrue, None, recdict, xTrue, xTrueWithoutNoise,
                                       xTrue_unconvolved, zTrue)

        '''Copy init dicts so we can save them'''

        genmodel_init = copy.deepcopy(sgvb_model.gen_model.state_dict())
        recmodel_init = copy.deepcopy(sgvb_model.rec_model.state_dict())

        '''Starting the inference algorithm'''

        cost = sgvb_model.train_batch()

        '''Save stuff and do some plots'''
        if not args.analyseRealData and not args.analyseRealDataMultimodal:
            T = 100000
        self.saveDataAndPlots(T, args, args_dict, cost, filePath, gendict, genmodel_init, recdict, recmodel_init,
                              sgvb_model, xTrue, xTrueWithoutNoise, xTrue_unconvolved, cTrue, externalInputs, movementRegressors, zTrue, trueDataGendict)

        """---------------------------------------------------------------------------------------------------------"""
        """---------------------------------------------------------------------------------------------------------"""

    def initArgsDict(self, args):
        args_dict = {
            'dim_x': args.dim_x, 'dim_c': args.dim_c, 'dim_z': args.dim_z, 'dim_ex': args.dim_ex,
            'dim_reg': args.dim_reg, 'T': args.T, 'dim_hidden': args.dim_hidden,
            'lr': args.lr, 'epochs': args.epochs, 'use_cuda': args.use_cuda, 'seed': args.seed,
            'use_tb': args.use_tb, 'trial_dir': args.trial_dir, 'trial_path': args.trial_path, 'sampler': args.sampler,
            'N_samples': args.N_samples, 'num_workers': args.num_workers, 'batch_size': args.batch_size,
            'drop_last': args.drop_last, 'rec_model': args.rec_model, 'init_distr': args.init_distr,
            'stabilize': args.stabilize, 'double_precision': args.double_precision, 'use_clipping': args.use_clipping,
            'clipping_value': args.clipping_value, 'use_annealing': args.use_annealing, 'add_noise': args.add_noise,
            'noise_percentage': args.noise_percentage, 'input_data_path': args.input_data_path,
            'validate_after_epoch': args.validate_after_epoch, 'log_trajectory': args.log_trajectory, 'tau': args.tau,
            'use_convolution': args.use_convolution, 'use_hrf': args.use_hrf,
            'use_simplified_hrf_recognitionModel': args.use_simplified_hrf_recognitionModel,
            'make_limit_cycle_test': args.make_limit_cycle_test, 'makeHRFtest': args.makeHRFtest,
            'useExplicitHrf': args.useExplicitHrf, 'use_regularization': args.use_regularization,
            'makeMVAEevaluation': args.makeMVAEevaluation,
            'index': args.index, 'mvaeEvaluationTestName': args.mvaeEvaluationTestName,
            'makeMissingDimensionTest': args.makeMissingDimensionTest,
            'nonLinearity': args.nonLinearity, 'analyseRealData': args.analyseRealData, 'analyseRealDataMultimodal': args.analyseRealDataMultimodal, 'timestepsKLx': args.timestepsKLx,
            'useRecognitionModelClipping': args.useRecognitionModelClipping, 'repetitionTime': args.repetitionTime
        }
        return args_dict

    def getDataForMVAEevaluation(self, path):
        xTrueWithoutNoise, cTrueWithoutNoise, xTrue, cTrue = load_matlab_data.loadMatlabDataForPaper(path)
        xTrueWithoutNoise, cTrueWithoutNoise, xTrue, cTrue = xTrueWithoutNoise.clone().detach().type(torch.DoubleTensor), \
                                                             cTrueWithoutNoise.clone().detach().type(torch.DoubleTensor), \
                                                             xTrue.clone().detach().type(torch.DoubleTensor), \
                                                             cTrue.clone.detach().type(torch.DoubleTensor)
        return xTrue, cTrue, xTrueWithoutNoise

    def saveDataAndPlots(self, T, args, args_dict, cost, filePath, gendict, genmodel_init, recdict,
                         recmodel_init, sgvb_model, xTrue, xTrueWithoutNoise, xTrue_unconvolved, cTrue, externalInputs,
                         movementRegressors, zTrue, trueDataGendict):

        self.getUsefulValuesAndSave(cost, filePath, sgvb_model, args)
        self.save_data(xTrue, args_dict, filePath, gendict, genmodel_init, recdict, recmodel_init, sgvb_model, trueDataGendict)

        trainedModelNames, trainedModels = self.getTrainedModels(filePath, sgvb_model)

        if args.makeMissingDimensionTest:
            xTrue = xTrueWithoutNoise

        for idx, trained_model in enumerate(trainedModels):
            #pgrint(T)
            xTrained, zTrained = trained_model.generate_timeseries(T, noise=False)
            xTrainedNoisy, zTrainedNoisy = trained_model.generate_timeseries(T, noise=True)

            if args.analyseRealData or args.analyseRealDataMultimodal:
                self.makePlotsForRealData(T, filePath, trained_model, xTrue, cTrue, externalInputs, movementRegressors, sgvb_model, args)
            self.makeSomePlots(args, filePath, idx, trainedModelNames, xTrained, xTrainedNoisy, xTrue,
                               xTrue_unconvolved, zTrained, zTrue)

    def makeSomePlots(self, args, filePath, idx, trainedModelNames, xTrained, xTrainedNoisy, xTrue, xTrue_unconvolved,
                      zTrained, zTrue):
        plotting.plotAndSaveObservations(1000, trainedModelNames[idx], xTrue, xTrained, filePath)
        #plotting.plotAndSaveObservations(1000, "observationWithNoise", xTrue, xTrainedNoisy, filePath)
        plotting.plotFourierObservations(xTrue, xTrained, filePath)
        try:
            plotting.plotAndSaveObservations(1000, "latent", zTrained, zTrained, filePath)
            #plotting.plotAndSaveObservations(1000, "latent_withTrue", zTrue, zTrained, filePath)
        except:
            print("Something went wrong with plotting latent trajectories")
        if not args.mvaeEvaluationTestName == 'Yclass' and not args.analyseRealData:
            plotting.plotAndSaveBothLorenz(1500, trainedModelNames[idx], xTrue, xTrained, filePath)
        if args.useExplicitHrf:
            plotting.plotAndSaveBothLorenz(1500, trainedModelNames[idx] + '_unconvolved', xTrue_unconvolved, xTrained,
                                           filePath)

    def makePlotsForRealData(self, T, filePath, trained_model, xTrue, cTrue, externalInputs, movementRegressors, sgvb, args):
        xTrainedWithoutRegressors, zTrainedWithoutRegressors = trained_model.generate_timeseries_without_regressors(T,
                                                                                                                    noise=False)
        plotting.plotAndSaveObservations(1000, "observationWithoutRegressors", xTrue, xTrainedWithoutRegressors,
                                         filePath)
        xTrainedWithoutExternalInputs, zTrainedWithoutExternalInputs = trained_model.generate_timeseries_without_externalInputs(
            T, noise=False)
        plotting.plotAndSaveObservations(1000, "observationWithoutExternalInputs", xTrue, xTrainedWithoutExternalInputs,
                                         filePath)
        plotting.plotAndSaveObservations(1000, "externalInputs", xTrue, externalInputs,
                                         filePath)
        plotting.plotAndSaveObservations(1000, "movementRegressors", xTrue, movementRegressors,
                                         filePath)
        sgvb.rec_model.setBatchSize(T)
        if args.analyseRealData:
            sgvb.rec_model.forward(xTrue)
        elif args.analyseRealDataMultimodal:
            sgvb.rec_model.forward(xTrue, cTrue)
        zSamples = sgvb.rec_model.getSample(noise=False)
        xFromZsamples, zSamplesWithInputs = trained_model.generate_timeseries_from_zSamples(T, zSamples)
        plotting.plotAndSaveObservations(1000, "observationFromZsamples", xTrue, xFromZsamples,
                                         filePath)

    def initSgvb(self, args, args_dict, cTrue, gendict, recdict, xTrue, xTrueWithoutNoise, xTrue_unconvolved, zTrue):
        if not args.makeHRFtest:
            sgvb_model = sgvb.SGVB(args_dict, xTrue, cTrue, recdict, gendict, xTrueWithoutNoise)
        else:
            sgvb_model = sgvb.SGVB(args_dict, xTrue, cTrue, recdict, gendict, None, xTrue_unconvolved, None, None, zTrue)
        return sgvb_model

    def initSgvbForMVAEevaluation(self, args, args_dict, cTrue, gendict, recdict, xTrue,
                                  xTrueWithoutNoise):
        xTrueReduced = h.removeYdimension(xTrue)
        if args.mvaeEvaluationTestName == 'Yclass':
            print("MVAE Yclass Test run {}...".format(args.index))
            xTrueWithoutNoiseReduced = h.removeYdimension(xTrueWithoutNoise)
            sgvb_model = sgvb.SGVB(args_dict, xTrueReduced, cTrue, recdict, gendict,
                                   xTrueWithoutNoiseReduced)
        elif args.mvaeEvaluationTestName == 'NoisyLorenz':
            print("MVAE NoisyLorenz Test run {}...".format(args.index))
            sgvb_model = sgvb.SGVB(args_dict, xTrue, cTrue, recdict, gendict, xTrueWithoutNoise)
        elif args.mvaeEvaluationTestName is not None and args.mvaeEvaluationTestName != 'Yclass' and args.mvaeEvaluationTestName != 'NoisyLorenz':
            raise NameError("MVAEevaluation is only implemented for Yclass and NoisyLorenz. Got {} instead.".format(
                args.mvaeEvaluationTestName))
        return sgvb_model, xTrueReduced

    def initArgs(self):
        parser = argparse.ArgumentParser(description="Estimate PLRNN with Sequential Variational Autoencoder")
        parser.add_argument('-dx', '--dim_x', type=int, default=5, help="dimension of observation space")
        parser.add_argument('-dz', '--dim_z', type=int, default=3, help="dimension of latent space")
        parser.add_argument('-dh', '--dim_hidden', type=int, default=25, help="dimension of hidden layer")
        parser.add_argument('--T', type=int, default=2500, help="length of time series")
        parser.add_argument('-lr', type=float, default=1e-3, help="learning rate of Adam optimizer")
        parser.add_argument('-e', '--epochs', type=int, default=2, help="number of epochs")
        parser.add_argument('-c', '--use_cuda', type=bool, default=False, help="enables usage of CUDA")
        parser.add_argument('-s', '--seed', type=int, default=100, help="random seed")
        parser.add_argument('--use_tb', default=False, action='store_true', help="use tensorboard")
        parser.add_argument('-td', '--trial_dir', type=str, default='test',
                            help="specify set-up of experiments in the folder name")
        parser.add_argument('-N', '--N_samples', type=int, default=1,
                            help="number of z samples drawn to calculate likelihood")
        parser.add_argument('-nw', '--num_workers', type=int, default=1,
                            help="how many subprocesses to use for dataloading. 0 means that the data will be loaded in the main process")
        parser.add_argument('-b', '--batch_size', type=int, default=250, help="number of samples per batch")
        parser.add_argument('--sampler', type=str, default='Stochastic',
                            help="Choose randomb batch sampler between \'Shuffling\' and \'Stochastic\' ")
        parser.add_argument('-dl', '--drop_last', type=bool, default=False,
                            help="In case that len(dataset) is not divisible by the batch_size: Drop the last batch of smaller size? (only "
                                 "for \'Shuffling\' sampler)")
        parser.add_argument('-rm', '--rec_model', type=str, default='ProductOfGaussiansMultimodal',
                            help="Which recognition model to use: Choose between 'ProductOfGaussians', 'ProductOfGaussiansMultimodal', "
                                 "'ProductOfGaussiansHRF','ProductOfGaussiansMultimodalHRF' and 'DiagonalCovarianceHRF'")
        parser.add_argument('-id', '--init_distr', type=str, default='uniform',
                            help="Initialize the parameter matrices of the PLRNN either with a 'uniform' or a'normal' distribution")
        parser.add_argument('-st', '--stabilize', default=False, action='store_true',
                            help="Whether or not to make use of the stability condition when initializing the PLRNN")
        parser.add_argument('-dp', '--double_precision', default=False, action='store_true',
                            help="Use double precision for floating point numbers in torch")
        parser.add_argument('-idp', '--input_data_path', type=str, default=None,
                            help="path to the timeseries data under investigation")
        parser.add_argument('--use_annealing', dest='use_annealing', default=False, action='store_true',
                            help="Use annealing in training (weigh the loss function)")
        parser.add_argument('-an', '--add_noise', dest='add_noise', default=False, action='store_true',
                            help="Add noise to input data, mainly required to validate multimodal model")
        parser.add_argument('-np', '--noise_percentage', dest='noise_percentage', type=float, default=0.1,
                            help="percentage of gaussian noise added to input data (1 equals 100%), default is 0.1 (10%)")
        parser.add_argument('-lt', '--log_trajectory', dest='log_trajectory', default=False, action='store_true',
                            help="Save model over training (all 1000 Epochs) to log evolution of state space trajectory")
        # parser.add_argument('-uc', '--use_convolution', dest='use_convolution', default=False, action='store_true',
        #                    help="Convolve the input data with hrf distribution")

        parser.add_argument('-simple', '--use_simplified_hrf_recognitionModel',
                            dest='use_simplified_hrf_recognitionModel',
                            default=False,
                            action='store_true',
                            help="Take easier recognition model in case of hrf")
        parser.add_argument('-vanilla', '--make_limit_cycle_test', dest='make_limit_cycle_test', default=False,
                            action='store_true',
                            help="Make vanilla test with limit cycle reconstruction")
        parser.add_argument('-mht', '--makeHRFtest', dest='makeHRFtest', default=False,
                            action='store_true', help="Make vanilla test with Lorenz reconstruction")
        parser.add_argument('-eh', '--itHrf', dest='useExplicitHrf', default=False,
                            action='store_true',
                            help="For hrf implementation take mathematical calculcation instead of MLP")
        parser.add_argument('-reg', '--regularize', dest='use_regularization', default=False,
                            action='store_true',
                            help="Take regularization into loss to suggest line attractor behaviour")
        parser.add_argument('-evalMVAE', dest='makeMVAEevaluation', default=False, action='store_true')
        parser.add_argument('-index', dest='index', type=int, default=None,
                            help="index variable for running experiments")
        parser.add_argument('-evalMVAEname', dest='mvaeEvaluationTestName', type=str, default=None,
                            help="name of the running Test")
        parser.add_argument('-missDim', dest='makeMissingDimensionTest', default=False, action='store_true')
        parser.add_argument('-realData', dest='analyseRealData', default=False, action='store_true')
        parser.add_argument('-realDataMultimodal', dest='analyseRealDataMultimodal', default=False, action='store_true')
        parser.add_argument('-recClipping', dest='useRecognitionModelClipping', default=False, action='store_true')
        parser.add_argument('-rT', '--repetitionTime', type=float, default=32, help="regulates the value of tau in the hrf")

        args = parser.parse_args()
        return args

    def setArgs(self, args):
        args.dim_z = 10 
        args.dim_x = 3
        args.dim_c = 8
        if args.mvaeEvaluationTestName == 'Yclass':
            args.dim_x = 2
        if args.analyseRealData or args.analyseRealDataMultimodal:
            args.dim_x = 20
            #args.dim_x = 28
            args.dim_c = 4
            args.dim_z = 20

        """For experimental data: """
        args.dim_ex = 5
        #args.dim_ex = 2
        args.dim_reg = 6

        """This option is rather important to prevent excessive cpu usage."""
        torch.set_num_threads(1)
        args.T = 1000
        args.timestepsKLx = 100000
        args.dim_hidden = 25
        args.lr = 0.003
        args.drop_last = False
        args.use_clipping = True
        args.clipping_value = 20
        args.trial_path = None
        args.nonLinearity = torch.nn.functional.relu
        #args.nonLinearity = None
        args.validate_after_epoch = 1
        args.tau = len(hrf_convolution.haemodynamicResponseFunction(args.repetitionTime)) - 1
        print("tau = {}".format(args.tau))
        if (args.rec_model == 'ProductOfGaussiansHRF' or args.rec_model == 'ProductOfGaussiansMultimodalHRF' \
                or args.rec_model == 'DiagonalCovarianceHRF' or args.use_simplified_hrf_recognitionModel is True):
            args.use_hrf = True
            args.use_convolution = True
            if args.makeHRFtest:
                args.use_convolution = False
        else:
            args.use_hrf = False
            args.use_convolution = False

    def initWriter(self, args, args_dict):
        if args.use_tb:
            writer, file_path = utils.initWriter(args_dict)
            args_dict['trial_path'] = file_path
        else:
            file_path = utils.create_trial_directory(args_dict)
            args_dict['trial_path'] = file_path
            writer = None
        return file_path, writer

    def writeParameterFile(self, args_dict, file_path):
        f = open(file_path + "/parameter_settings.txt", "x")
        for key, value in args_dict.items():
            f.write('{} = {}\n'.format(key, value))
        f.close()

    def getDataPath(self, args):
        if args.input_data_path == None:
            # path = '/home/daniel.kramer/algorithms/data/DataLong/3Dclassification/lorenz_traj_chaos_n1_T200000_0_02_0.1.mat'
            path = '/zi-flstorage/Daniel.Kramer/code/data/lorentz_attractor_long/lorenz_traj_chaos_n1_T200000_0_01_0.1.mat'
            #path = '/zi-flstorage/Daniel.Kramer/code/data/LorenzData/NoisyLorenz/lorenz_traj_chaos_n1_T1000_0.001_01_0.1.mat'
            # path = '/home/daniel/doktorarbeit/code/DataLong/lorenz_traj_chaos_n1_T200000_0_01_0.1.mat'
            #path = '//wsl$/Ubuntu-18.04/home/daniel/code/data/lorenz_traj_chaos_n1_T200000_0_01_0.1.mat'
            #path = '/home/daniel/code/data/lorenz_traj_chaos_n1_T200000_0_01_0.1.mat'
            #path = '/home/daniel/code/data/NoisyLorenz/lorenz_traj_chaos_n1_T1000_0.001_01_0.1.mat'
            # path = '/home/daniel/code/data/Yclass/lorenz_traj_chaos_n1_T1000_0.001_01_0.001.mat'
        else:
            path = args.input_data_path
        return path

    def getDataForInference(self, args, args_dict, dataPath, filePath):

        dim_x = args_dict['dim_x']
        xTrue, cTrue = load_matlab_data.loadMatlabData(dataPath)
        xTrue_unconvolved = None
        xTrueReduced = None
        xTrueWithoutNoise = None
        zTrue = None
        trueDataGendict = None

        '''64 Bit Tensors used because of numerical instabilities in Cholesky Decomposition'''

        torch.set_default_tensor_type(torch.DoubleTensor)

        '''Some specific Tests'''

        if args.make_limit_cycle_test:
            xTrue, cTrue = self.makeLimitCycleTest(args_dict)

        if args.makeHRFtest:
            xTrue, cTrue, xTrue_unconvolved, zTrue, trueDataGendict = self.makeLorenzTestWithHRF(args_dict, filePath)
            xTrue_unconvolved = xTrue_unconvolved.clone().detach().type(torch.DoubleTensor)
            zTrue = zTrue.clone().detach().type(torch.DoubleTensor)

        if args.makeMissingDimensionTest:
            xTrue = torch.tensor(xTrue).type(torch.DoubleTensor)
            xTrue = utils.standardiseData(dim_x, xTrue)
            xTrueWithoutNoise = xTrue.detach().clone()
            xTrue = xTrue.t()
            xTrue[1] = torch.randn(len(xTrue[1]))
            xTrue = xTrue.t()
	
        xTrue, cTrue = torch.tensor(xTrue).type(torch.DoubleTensor), torch.tensor(cTrue).type(torch.DoubleTensor)
        #xTrue, cTrue = xTrue.clone().detach().type(torch.DoubleTensor), cTrue.clone().detach().type(torch.DoubleTensor)
        xTrue = utils.standardiseData(dim_x, xTrue)

        print("Using tensors of type {}".format(xTrue.type()))

        return xTrue, cTrue, xTrue_unconvolved, xTrueReduced, xTrueWithoutNoise, zTrue, trueDataGendict

    def getRealData(self, args_dict, dataPath):

        dim_x = args_dict['dim_x']
        dim_reg = args_dict['dim_reg']
        xTrue, externalInputs, movementRegressors = load_matlab_data.loadMatlabDataRealData(dataPath)

        '''64 Bit Tensors used because of numerical instabilities in Cholesky Decomposition'''

        torch.set_default_tensor_type(torch.DoubleTensor)

        xTrue, externalInputs, movementRegressors = torch.tensor(xTrue).type(torch.DoubleTensor), torch.tensor(
            externalInputs).type(torch.DoubleTensor), torch.tensor(movementRegressors).type(torch.DoubleTensor)
        xTrue = utils.standardiseData(dim_x, xTrue)
        #movementRegressors = utils.standardiseData(dim_reg, movementRegressors)


        print("Using tensors of type {}".format(xTrue.type()))

        return xTrue, externalInputs, movementRegressors

    def getRealDataMultimodal(self, args_dict, dataPath):

        dim_x = args_dict['dim_x']
        xTrue, externalInputs, movementRegressors, cIndices = load_matlab_data.loadMatlabDataRealDataMultimodal(dataPath)

        '''64 Bit Tensors used because of numerical instabilities in Cholesky Decomposition'''

        torch.set_default_tensor_type(torch.DoubleTensor)

        xTrue, externalInputs, movementRegressors = torch.tensor(xTrue).type(torch.DoubleTensor), torch.tensor(
            externalInputs).type(torch.DoubleTensor), torch.tensor(movementRegressors).type(torch.DoubleTensor)
        xTrue = utils.standardiseData(dim_x, xTrue)
        #movementRegressors = utils.standardiseData(dim_reg, movementRegressors)
        cTrue = torch.zeros(len(cIndices), args_dict['dim_c'])
        for j in range(0, len(cIndices)):
            if cIndices[j] == 0:
                continue
            else:
                cTrue[j][cIndices[j]-1] = 1

        print("Using tensors of type {}".format(xTrue.type()))
        #externalInputs = None
        return xTrue, externalInputs, movementRegressors, cTrue

    def makeLimitCycleTest(self, args_dict):
        limitCyclePath = '/home/daniel/code/data/limitCicle_gen_model.pt'
        # limitCyclePath = '/zi-flstorage/Daniel.Kramer/papers/MultimodalVAE/data/limitCicle_gen_model.pt'

        trained_mdl = torch.load(limitCyclePath)
        trained_mdl = datagenerator.DataGenerator(args_dict['dim_x'], args_dict['dim_z'], args_dict, args_dict['dim_c'],
                                                  trained_mdl, 'uniform',
                                                  False,
                                                  nonlinearity=args_dict['nonLinearity'])
        x, z = trained_mdl.generate_timeseries(args_dict['T'], noise=False)
        cIndices = trained_mdl.calc_categories_from_data(x)
        c = torch.zeros(len(cIndices), args_dict['dim_c'])
        for j in range(0, len(cIndices)):
            c[j][cIndices[j]] = 1
        return x, c

    def makeLorenzTestWithHRF(self, args_dict, filePath):
        # LorenzPath = '/home/daniel/doktorarbeit/data/lorentz_gen_model.pt'
        #LorenzPath = '/home/daniel/code/data/lorentz_gen_model.pt'
        #LorenzPath = '//wsl$/Ubuntu-18.04//home/daniel/code/data/lorentz_gen_model.pt'
        LorenzPath = '/zi-flstorage/Daniel.Kramer/papers/MultimodalVAE/data/lorentz_gen_model.pt'

        trained_mdl = torch.load(LorenzPath)
        trained_mdl = datagenerator.DataGenerator(args_dict['dim_x'], args_dict['dim_z'], args_dict, args_dict['dim_c'],
                                                  trained_mdl, 'uniform',
                                                  False,
                                                  nonlinearity=args_dict['nonLinearity'])
        x, z = trained_mdl.generate_timeseries_with_hrf(args_dict['timestepsKLx']*3, noise=False)
        x = x[::3]
        x = x[:args_dict['timestepsKLx']-args_dict['tau']]
        print(len(x))
        z = z[::3]
        cIndices = trained_mdl.calc_categories_from_data(x)
        cIndices[::3]
        c = torch.zeros(len(cIndices), args_dict['dim_c'])
        for j in range(0, len(cIndices)):
            c[j][cIndices[j]] = 1

        xUnconvolved, zUnconvolved = trained_mdl.generate_timeseries_without_hrf(args_dict['timestepsKLx']*3, noise=False)
        trueDataDict = trained_mdl.state_dict()



        x = utils.standardiseData(args_dict['dim_x'], x)
        xUnconvolved = utils.standardiseData(args_dict['dim_x'], xUnconvolved)

        plotting.plotAndSaveObservations(500, "", xUnconvolved, x, filePath, 'observations_xTrueVsxConvolved')
        plotting.plotAndSaveBothLorenz(2500, "", xUnconvolved, x, filePath, 'lorenz_xTrueVsxConvolved')
        return x, c, xUnconvolved, z, trueDataDict

    def initModelDicts(self, dim_c, dim_x, dim_z):
        gendict = dict([
            ('A', 0.99 * torch.ones(dim_z)),
            ('W', torch.zeros(dim_z, dim_z)),
            ('R_x', torch.randn(dim_x)),
            ('R_z', torch.ones(dim_z)),
            ('R_z0', torch.ones(dim_z)),
            ('mu0', torch.zeros(dim_z)),
            ('B', torch.rand(dim_x, dim_z)),
            ('beta', torch.rand(dim_c-1, dim_z)),
        ])
        recdict = dict([('A', .9 * torch.rand(dim_z, dim_z)),
                        ('QinvChol', torch.rand(dim_z, dim_z)),
                        ('Q0invChol', torch.rand(dim_z, dim_z)),
                        ])
        return gendict, recdict

    def getUsefulValuesAndSave(self, cost, file_path, sgvb, args):
        lastLoss = cost[-1]
        bestLoss, epochBestLoss = sgvb.getBestLossAndEpoch()
        #klZ = sgvb.calcKLz(1000)
        if args.analyseRealData or args.analyseRealDataMultimodal:
            meanSquaredError = sgvb.getRealDataMSE_nstep()
            kld = meanSquaredError
            kldFromBestLoss = 1
        if not args.analyseRealData and not args.analyseRealDataMultimodal:
            kld = sgvb.calcKLx(args.timestepsKLx)
            bestLossDict = torch.load(file_path + '/best_loss_gen_model.pt')
            kldFromBestLoss = sgvb.calcKLx(args.timestepsKLx, bestLossDict)
        #else:
        #    kld = 1
        #    kldFromBestLoss = 1

        klXforHRF = 0
        #sgvb.calcKLxForHRF(50000)
        try:
            f = open(file_path + "/useful_values.txt", "x")  # The 'x' means: create a file, if it exists throw error
            f.write('kldBestLoss\t{}\n'.format(kldFromBestLoss))
            f.write('epochBestLoss\t{}\n'.format(epochBestLoss))
            f.write('bestLoss\t{}\n'.format(bestLoss))
            f.write('kld\t{}\n'.format(kld))
            #f.write('klZ\t{}\n'.format(klZ[1]))
            #if args.makeHRFtest:
            f.write('klX_latent\t{}\n'.format(klXforHRF))
            f.write('lastLoss\t{}\n'.format(lastLoss))
            f.close()
        except:
            print("Something went wrong with saving useful values")

    def getTrainedModels(self, file_path, sgvb):
        modelDir = file_path
        trainedModels = []
        trainedModelNames = []

        trainedModelChriterion = 'last_loss'
        trainedMdl = sgvb.getDataGeneratorFromDict()

        trainedModels.append(trainedMdl)
        trainedModelNames.append(trainedModelChriterion)
        #try:
        #    trainedModelChriterion = 'best_loss'
        #    trainedDict = torch.load(modelDir + '/best_loss_gen_model.pt')
        #    trainedMdl = sgvb.getDataGeneratorFromDict(trainedDict)

        #    trainedModels.append(trainedMdl)
        #    trainedModelNames.append(trainedModelChriterion)
        #except:
        #    print("no best_loss found")

        return trainedModelNames, trainedModels

    def save_data(self, X, args_dict, file_path, gendict, genmodel_init, recdict, recmodel_init, sgvb, trueDataGendict):
        data_storage = file_path + '/data'
        if not os.path.exists(data_storage):
            os.mkdir(data_storage)
        torch.save(X, data_storage + '/true_X.pt')  # required since we sample with noise, i.e., not reproducible
        torch.save(sgvb.gen_model.state_dict(), data_storage + '/trained_gen_statedict.pt')
        torch.save(sgvb.rec_model.state_dict(), data_storage + '/trained_rec_statedict.pt')
        torch.save(recdict, data_storage + '/recdict.pt')
        torch.save(gendict, data_storage + '/gendict.pt')
        torch.save(args_dict, data_storage + '/args_dict.pt')
        torch.save(genmodel_init, data_storage + '/genmodel_init.pt')
        torch.save(recmodel_init, data_storage + '/recmodel_init.pt')
        if trueDataGendict is not None:
            torch.save(trueDataGendict, data_storage + '/trueData_gendict.pt')


if __name__ == "__main__":
    seqMVAE = Main()
    seqMVAE.main()
