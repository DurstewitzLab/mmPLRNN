import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.font_manager
from matplotlib import rcParams
import pandas as pd
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd


rc('font', **{'family': 'sans-serif', 'sans-serif': ['Computer Modern Roman serif']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


sys.path.insert(0, "seqmvae/modules/")

import load_matlab_data
import datagenerator
import scipy

PLOTPATH = '../plots/'


'''Full TS Experiment with Rest as Reference Category'''

pwd = '../data/FullTS_restReference'


rootdir = pwd

THIS_DIR = os.path.normpath(os.path.abspath(os.path.dirname(__file__)))
DIR = os.path.normpath(os.path.join(THIS_DIR, '..', '.'))

colors = []
cmap_temp = np.linspace(0.0, 0.7, 3)
for number in cmap_temp:
    cmap = matplotlib.cm.get_cmap('viridis')
    colors.append(cmap(number))

'''Full TS Experiment'''

'''Full TS Experiment with Rest as Reference Category'''

relevantRuns = ['Sparse_PLRNN_m20pat_1_init_1.mat', 'Sparse_PLRNN_m20pat_2_init_4.mat','Sparse_PLRNN_m20pat_3_init_1.mat',
                'Sparse_PLRNN_m20pat_4_init_4.mat','Sparse_PLRNN_m20pat_6_init_3.mat',
                'Sparse_PLRNN_m20pat_7_init_4.mat','Sparse_PLRNN_m20pat_8_init_1.mat','Sparse_PLRNN_m20pat_9_init_1.mat',
                'Sparse_PLRNN_m20pat_10_init_1.mat','Sparse_PLRNN_m20pat_11_init_4.mat','Sparse_PLRNN_m20pat_12_init_1.mat',
                'Sparse_PLRNN_m20pat_13_init_1.mat','Sparse_PLRNN_m20pat_14_init_1.mat','Sparse_PLRNN_m20pat_15_init_2.mat',
                'Sparse_PLRNN_m20pat_17_init_2.mat','Sparse_PLRNN_m20pat_18_init_2.mat','Sparse_PLRNN_m20pat_19_init_1.mat',
                'Sparse_PLRNN_m20pat_21_init_5.mat',
                'Sparse_PLRNN_m20pat_22_init_4.mat', 'Sparse_PLRNN_m20pat_23_init_4.mat',
                'Sparse_PLRNN_m20pat_24_init_1.mat', 'Sparse_PLRNN_m20pat_25_init_4.mat',
                'mmPLRNN_m20pat_1_init_4.mat', 'mmPLRNN_m20pat_2_init_1.mat', 'mmPLRNN_m20pat_3_init_3.mat',
                'mmPLRNN_m20pat_4_init_4.mat', 'mmPLRNN_m20pat_6_init_5.mat', 
                'mmPLRNN_m20pat_7_init_3.mat', 'mmPLRNN_m20pat_8_init_4.mat', 'mmPLRNN_m20pat_9_init_1.mat',
                'mmPLRNN_m20pat_10_init_2.mat', 'mmPLRNN_m20pat_11_init_3.mat', 'mmPLRNN_m20pat_12_init_5.mat',
                'mmPLRNN_m20pat_13_init_3.mat', 'mmPLRNN_m20pat_14_init_1.mat', 'mmPLRNN_m20pat_15_init_1.mat',
                'mmPLRNN_m20pat_17_init_3.mat', 'mmPLRNN_m20pat_18_init_3.mat', 'mmPLRNN_m20pat_19_init_5.mat',
                'mmPLRNN_m20pat_21_init_2.mat', 'mmPLRNN_m20pat_22_init_1.mat', 'mmPLRNN_m20pat_23_init_2.mat',
                'mmPLRNN_m20pat_24_init_3.mat', 'mmPLRNN_m20pat_25_init_3.mat']


def main():
    print("Getting files from " + pwd)
    nStepErrorsMultimodal, stdsMultimodal, klxListMultimodal = getNstepErrorsFromTrajectories('multimodal')
    nStepErrorsUnimodal, stdsUnimodal, klxListUnimodal = getNstepErrorsFromTrajectories('unimodal')

    print(klxListMultimodal)
    print("----------------")
    print(klxListUnimodal)

    print(len([i for i in klxListUnimodal if i <= 1.]), len([i for i in klxListMultimodal if i <= 1.]))
    print(torch.mean(torch.tensor(klxListUnimodal)), torch.std(torch.tensor(klxListUnimodal))/len(torch.tensor(klxListUnimodal)),
          torch.mean(torch.tensor(klxListMultimodal)), torch.std(torch.tensor(klxListMultimodal))/len(torch.tensor(klxListMultimodal)))


    print("Wilcoxon Test D_kl:")
    wilcoxon = scipy.stats.wilcoxon(klxListUnimodal, klxListMultimodal)
    print(wilcoxon)

    print("Ttest D_kl:")

    res = scipy.stats.ttest_ind(klxListUnimodal, klxListMultimodal)
    print(res)

    nStepErrorsMultimodal, stdsMultimodal = torch.tensor(nStepErrorsMultimodal), torch.tensor(stdsMultimodal)
    nStepErrorsUnimodal, stdsUnimodal = torch.tensor(nStepErrorsUnimodal), torch.tensor(stdsUnimodal)

    meanMultimodal = []
    stdMultimodal = []
    meanUnimodal = []
    stdUnimodal = []

    #nsteps = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    nsteps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    print("-------------------------")

    nstepStatisticSteps = [1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20]

    for ind in range(0,len(nsteps)):
        meanMultimodal.append(torch.mean(nStepErrorsMultimodal.t()[ind]))
        stdMultimodal.append(torch.std(nStepErrorsMultimodal.t()[ind])/np.sqrt(len(nStepErrorsMultimodal.t()[ind])))
        meanUnimodal.append(torch.mean(nStepErrorsUnimodal.t()[ind]))
        stdUnimodal.append(torch.std(nStepErrorsUnimodal.t()[ind])/np.sqrt(len(nStepErrorsUnimodal.t()[ind])))
        if nsteps[ind] in nstepStatisticSteps:
            calcMeanStatistics(nStepErrorsMultimodal.t()[ind], nStepErrorsUnimodal.t()[ind], nsteps[ind])

    for ind in range(0, len(meanMultimodal)):
        print(ind)
        #print(meanMultimodal[ind] - meanUnimodal[ind], meanMultimodal[ind], meanUnimodal[ind])
        print(1 - (meanMultimodal[ind] - meanUnimodal[ind])/meanUnimodal[ind])
        print("---")

    rcParams['figure.figsize'] = 6.5, 6.5
    rcParams.update({'font.size': 25})

    meanMultimodal = np.array(meanMultimodal)
    meanUnimodal = np.array(meanUnimodal)
    stdMultimodal = np.array(stdMultimodal)
    stdUnimodal = np.array(stdUnimodal)

    significanceBarValue = 0.05
    
    #significance range restReference: 1-12
    #significance range cmtReference: 7-12

    plt.plot(nsteps, meanMultimodal, label=r'$\mathrm{mmPLRNN}$', color = colors[0], linewidth = 2.5)
    plt.plot(nsteps, meanUnimodal, label=r'$\mathrm{uniPLRNN}$', color = colors[1], linewidth = 2.5)
    plt.plot(np.arange(3,13, 1), np.repeat(significanceBarValue, 10),
              color='red', linestyle='--', linewidth=1.5, alpha=0.8)

    plt.fill_between(nsteps, meanMultimodal + stdMultimodal, meanMultimodal - stdMultimodal, color = colors[0], alpha=0.2)
    plt.fill_between(nsteps, meanUnimodal + stdUnimodal, meanUnimodal - stdUnimodal, color = colors[1], alpha=0.2)

    xTicks = [6,8,10,12,14]
    xTicks = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    plt.xticks(xTicks)

    '''Values for LSTM taken from Matlab LSTM Experiment (Philine)'''

    nsteps = [5, 7, 9, 11, 13, 15]
    meanLSTM = np.array([0.782, 0.854, 1.066, 1.173, 1.205, 1.22])
    semLSTM = np.array([0.057, 0.0724, 0.168, 0.202, 0.2047, 0.2037])
    plt.plot(nsteps, meanLSTM, label=r'$\mathrm{LSTM}$', color=colors[2], linewidth = 2.5)
    plt.fill_between(nsteps, meanLSTM + semLSTM, meanLSTM - semLSTM, color=colors[2], alpha=0.2)

    plt.gcf().subplots_adjust(bottom=0.2)
    plt.gcf().subplots_adjust(left=0.2)

    plt.xlabel(r'$n\mathrm{-steps}$', fontsize=32)
    plt.ylabel(r'$\mathrm{MSE}$', fontsize=32)
    plt.xlim(5, 15)
    #plt.xlim(0, 20)
    plt.ylim(0, 1.5)
    plt.legend(loc='lower right', frameon=False, fontsize=20)
    plt.savefig(PLOTPATH + 'Fig3A_.pdf')
    plt.close()

def getNstepErrorsFromTrajectories(experiment):
    nstepErrorList = []
    standardDeviationList = []
    klxList = []
    for idx, filename in enumerate(os.listdir(rootdir)):

        filepath = rootdir + '/' + filename


        if experiment == 'multimodal':
            filenameBeginning = "mmPL"
        if experiment == 'unimodal':
            filenameBeginning = "Sparse_PL"

        if filename.startswith(filenameBeginning) and filename in relevantRuns:
            print("Getting data from {}".format(filepath))
            data = load_matlab_data.loadmat(filepath)
            if experiment == 'multimodal':
                xTrue = data['xTrue']
                cTrue = data['cTrue']
                zInferred = data['zInf']
                A = data['A']
                B = data['B']
                W = data['W']
                h = data['h']
                mu0 = data['mu0']
                beta = data['beta']
                beta = torch.from_numpy(beta).t()

            if experiment == 'unimodal':
                xTrue = data['xTrue']
                cTrue = data['cTrue']

                zInferred = data['zInf']
                A = data['A']
                B = data['B']
                W = data['W']
                h = data['h']
                mu0 = data['mu0']
                beta = torch.rand(4, 20)

            A = torch.from_numpy(A)
            A = torch.diag(A)
            B = torch.from_numpy(B)
            W = torch.from_numpy(W)
            h = torch.from_numpy(h)
            mu0 = torch.from_numpy(mu0)
            zInferred = torch.from_numpy(zInferred).t().type(torch.FloatTensor)

            dim_x = 20
            dim_z = 20
            dim_c = 5

            args_dict = {'dim_ex': 5, 'dim_reg': 5, 'tau': 1, 'use_hrf': False, 'useExplicitHrf': False,
                         'repetitionTime': 6,
                         'useBaseExpansion': False, 'A': A, 'B': B, 'W': W, 'h': h, 'mu0': mu0, 'beta': beta}

            trained_mdl = datagenerator.DataGenerator(dim_x, dim_z, args_dict, dim_c,
                                                      args_dict, 'uniform',
                                                      False,
                                                      nonlinearity=torch.nn.functional.relu)


            xTrue = torch.from_numpy(xTrue).t()
            T = 360

            nstepError = []
            #nstepsList = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            nstepsList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

            xRecon, zTrue = trained_mdl.generate_timeseries(T, noise=False)

            klx = calc_kl_from_data(xRecon, xTrue)

            for nsteps in nstepsList:
                xNstep = trained_mdl.generate_nstep_timeseries_carlo(T, nsteps, zInferred)
                xTrueForNstep = xTrue[nsteps:].detach().clone()

                normalisingTerm = (T - nsteps) * dim_x
                mse_xNstep = (torch.abs(xNstep - xTrueForNstep) ** 2).sum() / normalisingTerm
                nstepError.append(mse_xNstep)

            nstepErrorList.append(nstepError)
            klxList.append(klx)

    return nstepErrorList, standardDeviationList, klxList

def calcMeanStatistics(meanListUnimodal, meanListMultimodal, nstep):

    meanListUnimodal = meanListUnimodal.numpy()
    meanListMultimodal = meanListMultimodal.numpy()

    print("Statistics for nstep {}".format(nstep))

    nStepStatisticList = []

    for ind in range(0, len(meanListUnimodal)):
        nStepStatisticList.append(meanListUnimodal[ind])
        nStepStatisticList.append(meanListMultimodal[ind])

    experiments = 2 #unimodal & multimodal

    #print(len(np.repeat(np.arange(1, int(len(relevantRuns)/2)+1), experiments)),
    #      len(np.tile(np.arange(1, experiments+1), int(len(relevantRuns)/2))),
    #      len(nStepStatisticList))

    #print(np.repeat(np.arange(1, int(len(relevantRuns)/2)+1), experiments),
    #      np.tile(np.arange(1, experiments+1), int(len(relevantRuns)/2)),
    #      nStepStatisticList)

    df = pd.DataFrame({'subject': np.repeat(np.arange(1, int(len(relevantRuns)/2) + 1), experiments),
                       'experiment': np.tile(np.arange(1, experiments + 1), int(len(relevantRuns)/2)),
                       'f1Score': nStepStatisticList})

    #print(df)

    anova = AnovaRM(data=df, depvar='f1Score', subject='subject', within=['experiment']).fit()
    print(anova)
    print(anova.summary())

    print("---------------------------------------")
    print("---------------------------------------")

    print("Tukey Test:")

    tukey = pairwise_tukeyhsd(endog=df['f1Score'], groups=df['experiment'], alpha=0.05)
    print(tukey)

    print("---------------------------------------")
    print("---------------------------------------")

    print("Mann-Whitney Test:")

    whitneyu, p = scipy.stats.mannwhitneyu(meanListUnimodal, meanListMultimodal)
    print(whitneyu, p)

    print("---------------------------------------")
    print("---------------------------------------")

    print("Wilcoxon Test:")
    wilcoxon = scipy.stats.wilcoxon(meanListUnimodal, meanListMultimodal)
    print(wilcoxon)

    print("---------------------------------------")
    print("---------------------------------------")

    print("Ttest:")

    res = scipy.stats.ttest_ind(meanListUnimodal, meanListMultimodal)
    print(res)

    print("---------------------------------------")
    print("---------------------------------------")

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
    vec=vec.float()
    precision = 1 / (std ** 2)
    precision = torch.diag_embed(precision).float()

    prec_vec = torch.einsum('zij,azj->azi', precision, vec)
    exponent = torch.einsum('abc,abc->ab', vec, prec_vec)
    sqrt_det_of_cov = torch.prod(std, dim=1)
    likelihood = torch.exp(-0.5 * exponent) / sqrt_det_of_cov
    return likelihood.sum(dim=1) / T


def calc_kl_mc(mu_inf, cov_inf, mu_gen, cov_gen):
    mc_n = 1000
    t = torch.randint(0, mu_inf.shape[0], (mc_n,))

    std_inf = torch.sqrt(cov_inf)
    std_gen = torch.sqrt(cov_gen)

    # print(mu_inf.shape)
    # print(std_inf.shape)

    z_sample = (mu_inf[t] + std_inf[t] * torch.randn(mu_inf[t].shape)).reshape((mc_n, 1, -1))

    prior = eval_likelihood_gmm_for_diagonal_cov(z_sample, mu_gen, std_gen)
    posterior = eval_likelihood_gmm_for_diagonal_cov(z_sample, mu_inf, std_inf)
    prior, posterior, outlier_ratio = clean_from_outliers(prior, posterior)
    kl_mc = torch.mean(torch.log(posterior) - torch.log(prior), dim=0)
    return kl_mc, outlier_ratio

def calc_kl_from_data(mu_gen, data_true):
    time_steps = min(len(data_true), 10000)
    mu_inf = data_true[:time_steps]

    mu_gen = mu_gen[:time_steps]

    scaling = 1.
    cov_inf = torch.ones(data_true.shape[-1]).repeat(time_steps, 1) * scaling
    cov_gen = torch.ones(data_true.shape[-1]).repeat(time_steps, 1) * scaling

    kl_mc1, _ = calc_kl_mc(mu_gen, cov_gen.detach(), mu_inf.detach(), cov_inf.detach())

    kl_mc2, _ = calc_kl_mc(mu_inf.detach(), cov_inf.detach(), mu_gen, cov_gen.detach())

    kl_mc = 1 / 2 * (kl_mc1 + kl_mc2)

    # scaling = 1
    # mu_inf = get_posterior_mean(model.rec_model, x)
    # cov_true = scaling * tc.ones_like(data_true)
    # mu_gen = get_prior_mean(model.gen_model, time_steps)
    # cov_gen = scaling * tc.ones_like(data_gen)

    # kl_mc, _ = calc_kl_mc(data_true, cov_true, data_gen, cov_gen)
    return kl_mc


if __name__ == "__main__":
    main()
