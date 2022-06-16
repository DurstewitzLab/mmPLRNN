import os
from tensorboardX import SummaryWriter
import torch as tc
from matplotlib import pyplot as plt
from torch import nn
from torch.nn.modules.module import _addindent
import torch
import numpy as np


# from torchsummary import summary as torch_summarize


def init_writer(args):
    """Automatically create a new folder for each trial and initialize the tensorboardX writer for logging.

    Arguments:
        args(dict):
            * trial_dir (str): Name for the directory for the current set of trials

    returns:
        writer (tensorboardX.SummaryWriter)
        file path (str): the file path where the writer should store the results

    """
    file_path = create_trial_directory(args)
    writer = SummaryWriter(file_path)
    return writer, file_path


def create_trial_directory(args):
    """Automatically create a new folder for each trial to store the results in.

    In the very first trial, a directory called 'experiments' is created. In that directory a subdirectory 
    is created with the name specified by trial_dir. In that subdirectory new folders are automatically
    created for each new trial subsequently being named run_0, run_1, run_2 etc.

    Arguments:
        args(dict):
            * trial_dir (str): Name for the directory for the current set of trials

    returns:
        file path (str): the file path where the writer should store the results

    """
    # check if folder 'experiments' exisits. If not, create it.
    trial_dir = './experiments'
    #trial_dir = '//wsl$/Ubuntu-18.04/home/daniel/code/seqmvae/experiments'
    if not os.path.exists(trial_dir):
        os.mkdir(trial_dir)

    trial_dir = trial_dir + '/{}'.format(args['trial_dir'])
    if not os.path.exists(trial_dir):
        print('a new directory is being created')
        os.mkdir(trial_dir)

    # get all the names of the directories contained in the directory 'experiments'
    files_and_directories = os.listdir(trial_dir)
    directories = [d for d in files_and_directories if os.path.isdir(os.path.join(trial_dir, d))]

    # if there are no directories in 'experiments/trial_dir' so far, create a dummy directory 'run_0' which
    # however will never be used
    if len(directories) == 0:
        os.mkdir(trial_dir + '/run_0')
        directories.append('run_0')

    # all the directories in 'experiments/trial_dir' will have the naming convention 'run_x', where x is an integer
    run_no = [r.split('_')[1] for r in directories]
    run_no_int = [int(r) for r in run_no]
    max_run_no = max(run_no_int)

    new_test_dir = 'run_{}'.format(str(max_run_no + 1))

    file_path = trial_dir + '/' + new_test_dir

    # we now create a new directory that has the name 'run_x+1', where x is the highest number so far.
    if not os.path.exists(file_path):
        os.mkdir(file_path)

    return file_path


# Helper function to summarize the network architecture
def summary(model, target_folder=None, filename=None, show_weights=True, show_parameters=True, show_biases=True):
    """Summarizes torch model by showing trainable parameters and weights."""

    tmpstr = model.__class__.__name__ + ' (\n'

    if 'rec' in filename:
        for key, module in model._modules.items():
            # if it contains layers let call it recursively to get params and weights

            modstr = module.__repr__()
            modstr = _addindent(modstr, 2)

            params = sum([np.prod(p.size()) for p in module.parameters()])
            weights = tuple([tuple(p.size()) for p in module.parameters()])

            # Names of the parameters in recognition model
            weight_list = []
            for name in model.named_parameters():
                #             print(name[0])
                if 'weight' or 'bias' in name[0]:
                    weight_list.append(name[0])

            # Rec model
            w = dict(zip(weight_list, weights))

            tmpstr += '  (' + key + '): ' + modstr

            tmpstr += '\n' '  Parameters={}''\n'.format(w)
            tmpstr += '  Total_parameters={}''\n'.format(params)

    if 'gen' in filename:

        for key, module in model._modules.items():
            if type(module) in [
                torch.nn.modules.container.Container,
                torch.nn.modules.container.Sequential
            ]:
                modstr = torch_summarize(module)
            else:
                modstr = module.__repr__()
            modstr = _addindent(modstr, 2)

            weights = tuple([tuple(p.size()) for p in module.parameters()])

            weight_list = []
            weights = []
            params = 0
            for name in model.named_parameters():
                weight_list.append(name[0])
                weights.append(tuple(name[1].size()))
                params += np.prod(tuple(name[1].size()))

            w = dict(zip(weight_list, weights))

            tmpstr += '  (' + key + '): ' + modstr

            tmpstr += '\n''  Parameters ={}'.format(w)
            tmpstr += '  Total_parameters={}''\n'.format(params)

            tmpstr = tmpstr + ')'

    if target_folder != None:
        with open(target_folder + filename, 'w') as f:
            f.write(tmpstr)

    return tmpstr


def calc_hist(x, n_bins, min_, max_):
    """

    Calculate a multidimensional histogram in the range of min and max

    works by aggregating values in sparse tensor,

    then exploits the fact that sparse matrix indices may contain the same coordinate multiple times,

    the matrix entry is then the sum of all values at the coordinate

    for reference: https://discuss.pytorch.org/t/histogram-function-in-pytorch/5350/9

    Outliers are discarded!

    :param x: multidimensional data: shape (N, D) with N number of entries, D number of dims

    :param n_bins: number of bins in each dimension

    :param min_: minimum value

    :param max_: maximum value to consider for histogram

    :return: histogram

    """

    D = x.shape[1]  # number of dimensions
    # get coordinates

    coord = tc.LongTensor(x.shape)

    for d in range(D):
        span = max_[d] - min_[d]

        xd = (x[:, d] - min_[d]) / span

        xd = xd * n_bins

        xd = xd.long()

        coord[:, d] = xd

    # discard outliers

    cond1 = coord > 0

    cond2 = coord < n_bins

    inlier = cond1.all(1) * cond2.all(1)

    coord = coord[inlier]

    size_ = tuple(n_bins for d in range(D))

    hist = tc.sparse.FloatTensor(coord.t(), tc.ones(coord.shape[0]), size=size_).to_dense()

    return hist


def calc_pdf(x1, x2, n_bins=10):
    """

    Calculate spatial pdf of time series x1 and x2

    :param x1: multivariate time series: shape (T, dim)

    :param x2: multivariate time series, used for choosing range of histogram

    :param n_bins: number of histogram bins

    :return: pdfs

    """

    assert len(x1)==len(x2), "Length of timeseries for the calculation of KLx are different: {} vs {}".format(len(x1),len(x2))

    # find range of histogram
    min_ = tc.min(x2, 0)[0]
    max_ = tc.max(x2, 0)[0]

    #min_ = torch.tensor([-4,-4,-4])
    #max_ = torch.tensor([4,4,4])

    # calculate histogram

    h1 = calc_hist(x1, n_bins=n_bins, min_=min_, max_=max_)

    h2 = calc_hist(x2, n_bins=n_bins, min_=min_, max_=max_)

    # convert histogram to pdf by normalizing, with laplace smoothing

    alpha = 10e-6

    dim_x = x1.shape[1]
    T = x1.shape[0]

    assert x1.shape[1] == x2.shape[1]  # number of dimensions need to be same

    p1 = (h1 + alpha) / (T + alpha * n_bins ** dim_x)

    p2 = (h2 + alpha) / (T + alpha * n_bins ** dim_x)

    return p1, p2


def kl(p1, p2):
    # calculate kullback-leibler divergences

    kl12 = (p1 * tc.log(p1 / p2)).sum()

    kl21 = (p2 * tc.log(p2 / p1)).sum()

    return kl12, kl21


def loss_kl(x1, x2, n_bins=10, symmetric=False):
    """

    Spatial KL-divergence loss function

    :param x1: time series 1

    :param x2: time series 2, reference time series

    :param n_bins: number of histogram bins

    :param symmetric: symmetrized KL-divergence

    :return: loss (skalar)

    """

    p1, p2 = calc_pdf(x1, x2, n_bins)

    kl12, kl21 = kl(p1, p2)

    if not symmetric:

        loss = kl21  # assuming p2 is ground truth

    else:

        loss = (kl12 + kl21) / 2

    return loss


def spatial_kullback(x_gen, x_true, n_bins=10):
    p1, p2 = calc_pdf(x_gen, x_true, n_bins)

    kl12, kl21 = kl(p1, p2)

    def marginalize(p, dims=(0, 1)):
        """

        Marginalize out all except the specified dims

        :param p: multidimensional pdf

        :param dims: specify dimensions to keep

        :return: marginalized pdf

        """

        if len(p.shape) > 2:
            l = list(range(len(p.shape)))

            l = [i for i in l if i not in dims]

            p = p.sum(tuple(l))

        return p

    # print('KL(GroundTruth||Generated) : {}'.format(kl21))
    return float(kl21)


def exponential_func(num_points, c):
    """Args:
    C(int) - Conrols the steepness of the curves"""

    y = []

    for x in range(num_points):
        x = x / num_points
        y.append((1 - np.exp(-c * x)) / (1 - np.exp(-c)))

    return torch.Tensor(y)


def linear_func(num_points):
    return torch.Tensor(np.array(list(range(num_points))) / num_points)


def init_weights_and_bias(regularization_weights, layer, firstLayer=False):
    nn.init.orthogonal_(layer.weight)
    if firstLayer:
        layer.weight = nn.Parameter(
            (layer.weight.t() / torch.matmul(layer.weight, regularization_weights.t()).std(dim=1)).t())
    else:
        layer.weight = nn.Parameter(
            (layer.weight.t() / torch.matmul(layer.weight, regularization_weights).std(dim=1)).t())

    if firstLayer:
        layer.bias = nn.Parameter(-(torch.matmul(layer.weight, regularization_weights.t())).mean(dim=1))
    else:
        layer.bias = nn.Parameter(-(torch.matmul(layer.weight, regularization_weights)).mean(dim=1))


def reshapeZSamplesForHRF(Z_sample, batch_size, dim_z, tau):
    Z_tau_to_t_3d_init = torch.zeros(tau + 1, batch_size-tau, dim_z)
    #Z_tau_to_t, Z_tau_to_t_3d = reshapeZSamples(Z_tau_to_t_3d_init, Z_sample, tau)
    Z_tau_to_t_3d = reshapeZSamples(Z_tau_to_t_3d_init, Z_sample, tau)
    return Z_tau_to_t_3d

def reshapeZSamples(Z_tau_to_t_3d, Z_sample, tau):
    tau = tau + 1

    Z_sampleFrom_T_ToEnd = Z_sample[tau-1:].clone()
    Z_sampleFromTauTo_T = Z_sample[:tau-1].clone()
    Z_tau_to_t_3d[-1] = Z_sampleFrom_T_ToEnd.clone()

    for idx in range(1, tau):
        Z_tau_to_t_3d[-1 - idx][idx:] = Z_sampleFrom_T_ToEnd[:-idx].clone()
        Z_tau_to_t_3d[-1 - idx][:idx] = Z_sampleFromTauTo_T[-idx:].clone()

    """Bring shape (tau+1, batch_size, dim_z) to shape (batch_size, dim_z*(tau+1)). Unfortunately, x.view(-1,
    dim_z*(tau+1)) is not working as expected for 3d arrays (or generally?) """

    #for ind in range(0, len(Z_tau_to_t_3d)):
    #    if ind == 0:
    #        Z_tau_to_t = Z_tau_to_t_3d[ind].clone()
    #    else:
    #        Z_tau_to_t = torch.cat((Z_tau_to_t, Z_tau_to_t_3d[ind]), 1)

    #numpyarray = Z_tau_to_t.detach().clone().numpy()
    #numpyarray2 = Z_tau_to_t_3d.detach().clone().numpy()
    #numpyarray3 = Z_sampleFrom_T_ToEnd.detach().clone().numpy()
    #numpyarray4 = Z_sampleFromTauTo_T.detach().clone().numpy()
    return Z_tau_to_t_3d #Z_tau_to_t,

def getXDataForHRF(batch_X, batch_size, dim_x, tau):
    #TODO: finish this routine. Should be pretty similar to reshapeZsamples
    X_tau_to_t_3d_init = torch.zeros(tau + 1, batch_size - tau, dim_x)
    X_tau_to_t, X_tau_to_t_3d = reshapeXDataForHRF(X_tau_to_t_3d_init, batch_X, tau)

    batch_X_reshaped = reshapeXDataForHRF(X_tau_to_t_3d, batch_X,  batch_size, tau)
    numpyarray = batch_X_reshaped.clone().numpy()
    return batch_X_reshaped


def reshapeXDataForHRF(X_tau_to_t_3d_init, batchX, batch_size, tau):
    #dataHRF = torch.cat((batchX, nextbatchX[:tau]))
    reshapedData = batchX.detach().clone()
    #for idx in range(0, tau):
    #    reshapedData = torch.cat((reshapedData, dataHRF[idx + 1:batch_size + idx + 1]), 1)
    return reshapedData


def reshape_data_for_hrfEncoderInit(tau, X_true):
    """Just used for weight initialization, therefore we stack the input data such that it fits the encoder
    structure """
    X_true_hrf = X_true.detach().clone()
    for idx in range(0, tau):
        X_true_hrf = tc.cat((X_true_hrf, X_true), 1)
    return X_true_hrf

def squared_error(x):
    return tc.sum(tc.pow(x, 2))

def calc_kl_var(mu_inf, cov_inf, mu_gen, cov_gen):
    """
    Variational approximation of KL divergence (eq. 20, Hershey & Olsen, 2007)
    """
    kl_posterior_posterior = kl_between_two_gaussians(mu_inf, cov_inf, mu_inf, cov_inf)
    kl_posterior_prior = kl_between_two_gaussians(mu_inf, cov_inf, mu_gen, cov_gen)

    denominator = tc.sum(tc.exp(-kl_posterior_posterior), dim=1)
    nominator = tc.sum(tc.exp(-kl_posterior_prior), dim=1)
    nominator, denominator, outlier_ratio = clean_from_outliers(nominator, denominator)
    kl_var = (tc.mean(tc.log(denominator), dim=0) - tc.mean(tc.log(nominator), dim=0))
    return kl_var, outlier_ratio


def kl_between_two_gaussians(mu0, cov0, mu1, cov1):
    """
    For every time step t in mu0 cov0, calculate the kl to all other time steps in mu1, cov1.
    """
    T = mu0.shape[0]
    n = mu0.shape[1]

    cov1inv_cov0 = tc.einsum('tn,dn->tdn', cov0, 1 / cov1)  # shape T, T, n
    trace_cov1inv_cov0 = tc.sum(cov1inv_cov0, dim=-1)  # shape T,

    diff_mu1_mu0 = mu1.reshape(1, T, n) - mu0.reshape(T, 1, n)  # subtract every possible combination
    mahalonobis = tc.sum(diff_mu1_mu0 / cov1 * diff_mu1_mu0, dim=2)

    det1 = tc.prod(cov1, dim=1)
    det0 = tc.prod(cov0, dim=1)
    logdiff_det1det0 = tc.log(det1).reshape(1, T) - tc.log(det0).reshape(T, 1)

    kl = 0.5 * (logdiff_det1det0 - n + trace_cov1inv_cov0 + mahalonobis)
    return kl


def calc_kl_mc(mu_inf, cov_inf, mu_gen, cov_gen):
    mc_n = 10
    t = tc.randint(0, mu_inf.shape[0], (mc_n,))

    std_inf = tc.sqrt(cov_inf)
    std_gen = tc.sqrt(cov_gen)

    z_sample = (mu_inf[t] + std_inf[t] * tc.randn(mu_inf[t].shape)).reshape((mc_n, 1, -1))

    prior = eval_likelihood_gmm_for_diagonal_cov(z_sample, mu_gen, std_gen)
    posterior = eval_likelihood_gmm_for_diagonal_cov(z_sample, mu_inf, std_inf)
    prior, posterior, outlier_ratio = clean_from_outliers(prior, posterior)
    kl_mc = tc.mean(tc.log(posterior) - tc.log(prior), dim=0)

    return kl_mc, outlier_ratio


def clean_from_outliers(prior, posterior):
    nonzeros = (prior != 0)
    if any(prior == 0):
        prior = prior[nonzeros]
        posterior = posterior[nonzeros]
    outlier_ratio = (1 - nonzeros.float()).mean()
    return prior, posterior, outlier_ratio


def eval_likelihood_gmm_for_diagonal_cov(z, mu, std):
    T = mu.shape[0]
    mu = mu.reshape((1, T, -1))

    vec = z - mu  # calculate difference for every time step
    precision = 1 / (std ** 2)
    precision = tc.diag_embed(precision)

    prec_vec = tc.einsum('zij,azj->azi', precision, vec)

    exponent = tc.einsum('abc,abc->ab', vec, prec_vec)
    sqrt_det_of_cov = tc.prod(std, dim=1)
    likelihood = tc.exp(-0.5 * exponent) / sqrt_det_of_cov
    return likelihood.sum(dim=1) / mu.shape[0]

def standardiseData(dim, data):
    data = data.t()
    for ind in range(0, dim):
        data[ind] = (data[ind] - torch.mean(data[ind])) / torch.std(data[ind])
    data = data.t()
    return data
