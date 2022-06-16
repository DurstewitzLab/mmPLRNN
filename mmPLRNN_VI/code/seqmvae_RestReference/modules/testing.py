import argparse
import sgvb
import torch
import numpy as np
import datagenerator
import torch.nn.functional as F
import load_matlab_data
import hrf_convolution
import utils
from mpl_toolkits.mplot3d import axes3d
from matplotlib import pyplot as plt


def init_dicts_for_sgvb_init():
    args = initArgs()

    args.dim_z = 5
    args.dim_x = 3
    args.dim_c = 8

    """This option is rather important to prevent excessive cpu usage."""
    torch.set_num_threads(1)

    args.dim_z = 10
    args.dim_x = 3
    if args.mvaeEvaluationTestName == 'Yclass':
        args.dim_x = 2
    if args.analyseRealData:
        args.dim_x = 28
    args.dim_c = 8

    """For experimental data: """
    args.dim_ex = 2
    args.dim_reg = 6

    """This option is rather important to prevent excessive cpu usage."""
    torch.set_num_threads(1)
    args.T = 1000
    args.timestepsKLx = 1000
    args.dim_hidden = 25
    args.lr = 0.003
    args.drop_last = False
    args.use_clipping = True
    args.clipping_value = 20
    args.trial_path = None
    args.nonLinearity = None
    args.validate_after_epoch = 1
    args.tau = len(hrf_convolution.haemodynamicResponseFunction()) - 1
    args.use_simplified_hrf_recognitionModel = True
    args.useExplicitHrf = True
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

    args_dict = initArgsDict(args)
    print("Using the following options (args_dict):")
    print(args_dict)
    print("--------------------------------------------------------------------")

    dim_z = args.dim_z
    dim_x = args.dim_x
    dim_c = args.dim_c
    dim_hidden = args.dim_hidden
    T = args.T

    gendict = dict([
        ('A', 0.99 * torch.ones(dim_z)),
        ('W', torch.zeros(dim_z, dim_z)),
        ('R_x', torch.randn(dim_x)),
        ('R_z', torch.ones(dim_z)),
        ('R_z0', torch.ones(dim_z)),
        ('mu0', torch.zeros(dim_z)),
        ('B', torch.rand(dim_x, dim_z)),
        ('beta', torch.rand(dim_c, dim_z)),
    ])

    recdict = dict([('A', .9 * torch.rand(dim_z, dim_z)),
                    ('QinvChol', torch.rand(dim_z, dim_z)),
                    ('Q0invChol', torch.rand(dim_z, dim_z)),
                    ])

    return args, args_dict, gendict, recdict

def initArgs():
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
    parser.add_argument('-st', '--stabilize', default='False', action='store_true',
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
    parser.add_argument('-recClipping', dest='useRecognitionModelClipping', default=False, action='store_true')

    args = parser.parse_args()
    return args

def initArgsDict(args):
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
        'nonLinearity': args.nonLinearity, 'analyseRealData': args.analyseRealData, 'timestepsKLx': args.timestepsKLx,
        'useRecognitionModelClipping': args.useRecognitionModelClipping,
    }
    return args_dict


def test_get_X_data_for_HRF(sgvb, dim_x, batch_size, tau):
    batched_X = torch.rand(4, batch_size, dim_x)
    iterator = iter(batched_X)
    batched_X = next(iterator)

    reshaped_data = utils.getXDataForHRF(batched_X, iterator)
    """shape has to be batchsize x dim_x*(tau+1)"""
    assert reshaped_data.shape == torch.zeros(batch_size, dim_x * (tau + 1)).shape, "Shape of reshaped data not as " \
                                                                                    "expected "
    """Actual Test:"""

    test_matrix_shape_on_timedependency(reshaped_data, dim_x, tau)

    print("get_X_data_for_HRF tested succesfully")


def test_get_Z_samples_for_HRF(sgvb, dim_z, batch_size, tau):
    Z_tau_to_t, Z_tau_to_t_3d = utils.reshapeZSamplesForHRF()
    expected_shape = torch.zeros(batch_size, dim_z * (tau + 1)).shape
    assert Z_tau_to_t.shape == expected_shape, "Shape of reshaped data not as " \
                                               "expected:{} vs {}".format(Z_tau_to_t.shape, expected_shape)

    """Actual Test:"""

    test_matrix_shape_on_timedependency(Z_tau_to_t, dim_z, tau)

    print("get_Z_samples_for_HRF tested succesfully")


def test_matrix_shape_on_timedependency(matrix_tau_to_t, dim, tau):
    matrix_tau_to_t = matrix_tau_to_t.detach().clone()
    flip_this = matrix_tau_to_t.numpy()
    Z_tau_to_t = torch.from_numpy(np.flip(flip_this, axis=0).copy())
    count = 0
    for idx in range(0, len(matrix_tau_to_t)):
        if idx > 0:
            for ind in range(0, dim * (tau - 1)):
                errorMessage = 'Found missmatch in B[{}][{}] == B[{}][{}]: {} =/= {} (dim = {}, tau = {})'.format(idx, ind, idx - 1,
                                                                                                                  ind + dim,
                                                                                                                  matrix_tau_to_t[idx][ind],
                                                                                                                  matrix_tau_to_t[idx - 1][
                                                                                                                      ind + dim],
                                                                                                                  dim, tau)
                assert matrix_tau_to_t[idx][ind] == matrix_tau_to_t[idx - 1][ind + dim], errorMessage
                print(errorMessage)
                count += 1
            if idx < len(matrix_tau_to_t) - 1:
                for ind in range(0, dim):
                    errorMessage = 'Found missmatch in B[{}][{}] == B[{}][{}]: {} =/= {} (dim = {}, tau = {})'.format(
                        idx + 1,
                        ind + dim * (tau - 1),
                        idx,
                        ind + dim * tau,
                        matrix_tau_to_t[idx + 1][ind + dim * (tau - 1)],
                        matrix_tau_to_t[idx][ind + dim * tau],
                        dim, tau)
                    assert matrix_tau_to_t[idx + 1][ind + dim * (tau - 1)] == matrix_tau_to_t[idx][
                        ind + dim * tau], errorMessage
                    count += 1
    print("{}/{}".format(count, len(Z_tau_to_t) * len(Z_tau_to_t[0])))


def init_sgvb(args_dict, dim_c, dim_x, gendict, recdict):
    import sgvb
    Z_true = None
    writer = None
    dataset_X = torch.rand(1000, dim_x)
    dataset_C = torch.rand(1000, dim_c)
    sgvb = sgvb.SGVB(args_dict, dataset_X, dataset_C, recdict, gendict)
    return sgvb


def plot_and_save_both_lorentz(timesteps, model_name, true_data, trained_data, file_path):
    fig = plt.figure(figsize=(16, 9))
    # `ax` is a 3D-aware axis instance, because of the projection='3d' keyword argument to add_subplot
    dx = fig.add_subplot(1, 1, 1, projection='3d')
    dx.set_xlabel('$x_1$', size=15)
    dx.set_ylabel('$x_2$', size=15)
    dx.set_zlabel('$x_3$', size=15)
    p = dx.plot(true_data[:timesteps, 0].numpy(), true_data[:timesteps, 1].numpy(), true_data[:timesteps, 2].numpy(),
                antialiased=True,
                linewidth=0.5, label='true')
    p = dx.plot(trained_data[:timesteps, 0].numpy(), trained_data[:timesteps, 1].numpy(),
                trained_data[:timesteps, 2].numpy(), color='red', alpha=0.7,
                antialiased=True,
                linewidth=0.5, label='reconstructed')
    plt.savefig(file_path + '/true_and_reconstructed_lorentz_test_{}_timesteps_{}'.format(timesteps, model_name))
    plt.close()


def get_trained_models(dim_c, dim_x, dim_z, file_path, non_linearity):
    model_dir = file_path
    trained_models = []
    trained_model_names = []

    try:
        trained_model_chriterion = 'best_loss'
        trained_mdl = torch.load(model_dir + '/best_loss_gen_model.pt')
        trained_mdl = datagenerator.DataGenerator(dim_x, dim_z, dim_c,
                                                  trained_mdl, 'uniform',
                                                  False,
                                                  nonlinearity=non_linearity)
        trained_models.append(trained_mdl)
        trained_model_names.append(trained_model_chriterion)
    except:
        print("no best_loss found in " + str(model_dir + '/best_loss_gen_model.pt'))
    try:
        trained_model_chriterion = 'best_klx'
        trained_mdl = torch.load(model_dir + '/best_klx_gen_model.pt')
        trained_mdl = datagenerator.DataGenerator(dim_x, dim_z, dim_c,
                                                  trained_mdl, 'uniform',
                                                  False,
                                                  nonlinearity=non_linearity)
        trained_models.append(trained_mdl)
        trained_model_names.append(trained_model_chriterion)
    except:
        print("no best_klx found in " + str(model_dir + '/best_klx_gen_model.pt'))
    return trained_model_names, trained_models


def test_spatial_kullback(T, dim_c, dim_x, dim_z, non_linearity):
    dir = '/home/daniel.kramer/algorithms/seqMVAE/modules/testing/'
    file_paths = [dir + 'run_8', dir + 'run_67', dir + 'run_115', dir + 'run_148', dir + 'run_162']

    for path in file_paths:

        dict = {}
        with open(path + "/parameter_settings.txt") as f:
            for line in f:
                line = line.strip()
                (key, val) = line.split(" = ")
                dict[(key)] = (val)

        dict_useful_values = {}
        with open(path + "/useful_values.txt") as f:
            for line in f:
                line = line.strip()
                (key, val) = line.split()
                dict_useful_values[(key)] = float(val)

        kld_max = dict_useful_values['validation_kldx_max']

        true_data_path = '/zifnas/daniel.kramer/code/seqMVAE/seqmvae/' + dict['input_data_path']
        X_true, C_true = load_matlab_data.loadMatlabData(true_data_path)
        X_true, C_true = torch.tensor(X_true).type(torch.FloatTensor), torch.tensor(C_true).type(torch.FloatTensor)

        trained_model_names, trained_models = get_trained_models(dim_c, dim_x, dim_z, path, non_linearity)

        print("Evaluating: {} \n\t with input file {}".format(path, true_data_path))
        for idx, trained_model in enumerate(trained_models):
            X_trained, Z_trained = trained_model.generate_timeseries(T, noise=False)
            kld = utils.spatial_kullback(X_trained, X_true[:T], n_bins=10) / kld_max
            print("kld: {}".format(kld))
            plot_and_save_both_lorentz(T, trained_model_names[idx], X_true[:T], X_trained, path)
        print("---------------------")


def main():
    args, args_dict, gendict, recdict = init_dicts_for_sgvb_init()

    batch_size = args_dict['batch_size']
    dim_x = args_dict['dim_x']
    dim_c = args_dict['dim_c']
    dim_z = args_dict['dim_z']
    tau = args_dict['tau']
    T = args_dict['T']
    non_linearity = F.relu

    sgvb = init_sgvb(args_dict, dim_c, dim_x, gendict, recdict)

    # test_spatial_kullback(T, dim_c, dim_x, dim_z, non_linearity)
    #test_get_X_data_for_HRF(sgvb, dim_x, batch_size, tau)
    #
    batched_X = torch.rand(4, batch_size, dim_x)
    batched_C = torch.rand(batch_size, dim_c)
    iterator = iter(batched_X)
    batched_X = next(iterator)
    #
    #reshaped_batched_X = utils.getXDataForHRF(batched_X, iterator)
    reshaped_batched_X = None
    cost = sgvb.lossHRF(batched_X, reshaped_batched_X, batched_C)
    test_get_Z_samples_for_HRF(sgvb, dim_z, batch_size, tau)


if __name__ == "__main__":
    main()
