import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.font_manager
from matplotlib import rcParams
import pandas as pd
from sklearn import metrics
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Computer Modern Roman serif']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

sys.path.insert(0, "seqmvae_RestReference/modules/")

import load_matlab_data
import datagenerator
import scipy


plotpath = '../plots/'

'''Full TS Experiment with Rest as Reference Category'''

pwd = '../data/FullTS_restReference/'

rootdir = pwd

THIS_DIR = os.path.normpath(os.path.abspath(os.path.dirname(__file__)))
DIR = os.path.normpath(os.path.join(THIS_DIR, '..', '.'))



colorsCategories = ['black', 'gold', 'cyan', 'magenta', 'mediumseagreen']



labelsRowBeta = [r'$\mathrm{Rest}$', r'$\mathrm{Instr}$', r'$\mathrm{CRT}$', r'$\mathrm{CDRT}$', r'$\mathrm{CMT}$']

PLOTPATH = '../plots/'


#subject 5 excluded, cause of an artifact in the signal

'''Full TS Experiment with Rest as Reference Category'''

relevantRuns = ['mmPLRNN_m20pat_2_init_1.mat', 'mmPLRNN_m20pat_3_init_3.mat',
                'mmPLRNN_m20pat_4_init_4.mat', 'mmPLRNN_m20pat_6_init_2.mat',
                'mmPLRNN_m20pat_7_init_3.mat','mmPLRNN_m20pat_1_init_4.mat', 
                'mmPLRNN_m20pat_8_init_4.mat', 'mmPLRNN_m20pat_9_init_1.mat', 'mmPLRNN_m20pat_10_init_2.mat',
                'mmPLRNN_m20pat_11_init_3.mat', 'mmPLRNN_m20pat_12_init_5.mat', 'mmPLRNN_m20pat_13_init_3.mat',
                'mmPLRNN_m20pat_15_init_1.mat', 'mmPLRNN_m20pat_17_init_3.mat',
                'mmPLRNN_m20pat_18_init_3.mat', 'mmPLRNN_m20pat_19_init_5.mat', 'mmPLRNN_m20pat_21_init_2.mat',
                'mmPLRNN_m20pat_22_init_1.mat', 'mmPLRNN_m20pat_23_init_2.mat', 'mmPLRNN_m20pat_24_init_4.mat', 
                'mmPLRNN_m20pat_25_init_3.mat']

colors = []
cmap_temp = np.linspace(0.0, 1.0, len(relevantRuns))
for number in cmap_temp:
    cmap = matplotlib.cm.get_cmap('viridis')
    colors.append(cmap(number))

#'Sparse_mmPLRNN_m20pat_5_init_5.mat', removed, because of weird looking BOLD signal - artifact



xAxis = [r'$\mathrm{free}$', r'$\mathrm{randReset}$', r'$\mathrm{instrReset}$', r'$\mathrm{inferred}$']


def calculateSomeStatistics(experiment):

    cTrueList_full = []
    cReconListInf = []
    cReconResetList = []
    cReconRandResetList = []
    cReconFreeList = []

    infoNames = []

    fScoreFreeList = []
    fScoreRandResetList = []
    fScoreInstrResetList = []
    fScoreInferedList = []
    scoreList = []
    scoreListRandInstr = []

    scoreListSignTestRand = []
    scoreListSignTestInstr = []

    scoreListThirds = []


    continueCount = 0
    colorVariable = 0

    rcParams['figure.figsize'] = 9.5, 6.5
    rcParams.update({'font.size': 24})

    FileList = ['mmPLRNN_m20pat_10_init_2.mat', 'mmPLRNN_m20pat_11_init_3.mat', 'mmPLRNN_m20pat_12_init_5.mat', 'mmPLRNN_m20pat_13_init_3.mat',
                'mmPLRNN_m20pat_15_init_1.mat', 'mmPLRNN_m20pat_17_init_3.mat', 'mmPLRNN_m20pat_18_init_3.mat', 'mmPLRNN_m20pat_19_init_5.mat',
                'mmPLRNN_m20pat_1_init_4.mat', 'mmPLRNN_m20pat_21_init_2.mat', 'mmPLRNN_m20pat_22_init_1.mat', 'mmPLRNN_m20pat_23_init_2.mat',
                'mmPLRNN_m20pat_24_init_4.mat', 'mmPLRNN_m20pat_25_init_3.mat', 'mmPLRNN_m20pat_2_init_1.mat', 'mmPLRNN_m20pat_3_init_3.mat',
                'mmPLRNN_m20pat_4_init_4.mat', 'mmPLRNN_m20pat_6_init_2.mat', 'mmPLRNN_m20pat_7_init_3.mat', 'mmPLRNN_m20pat_8_init_4.mat',
                'mmPLRNN_m20pat_9_init_1.mat',]

    for numbers in FileList:
        for idx, filename in enumerate(os.listdir(rootdir)):

            filepath = rootdir + '/' + filename

            if experiment == 'multimodal':
                #filenameBeginning = "Sparse_mmPL"5
                filenameBeginning = "mmPL"

            if filename.startswith(numbers):
            #if filename.startswith(filenameBeginning) and filename in relevantRuns:
                # if filename.startswith("Sparse_mmPL"):
                print("Getting data from {}".format(filename))
                data = load_matlab_data.loadmat(filepath)
                if experiment == 'multimodal':
                    xTrue = data['xTrue']
                    cTrue = data['cTrue']
                    # print(data.keys())
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
                    # print(data.keys())

                    zInferred = data['zInf']
                    A = data['AG']
                    B = data['BG']
                    C = data['CG']
                    W = data['WG']
                    h = data['hG']
                    mu0 = data['mu0G']
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

                xReconstructed, zReconstructed = trained_mdl.generate_timeseries(T, noise=False)
                cTrue = torch.from_numpy(cTrue)
                cTrue = cTrue.t()
                cTrueIndices = torch.zeros(len(cTrue))
                for ind in range(0, len(cTrue)):
                    max, index = torch.max(cTrue[ind], 0)
                    cTrueIndices[ind] = index


                xReset_inf, zReset_inf = trained_mdl.generate_reset_timeseries_for_fmriExperiment_fromFullTS(360, zInferred, cTrueIndices)


                try:
                    cReconstructedInf = trained_mdl.calc_categorical_pdf(zInferred)
                    cReconstructedReset = trained_mdl.calc_categorical_pdf(zReset_inf)
                    cReconstructedFree = trained_mdl.calc_categorical_pdf(zReconstructed)
                except:
                    print("Something went wrong with calculating categorical pdfs... Skipping this file")
                    continueCount += 1
                    continue

                randomRuns = 10
                for j in range(0, randomRuns):
                    xReset_rand, zReset_rand = trained_mdl.generate_reset_timeseries_for_fmriExperiment_fromFullTS_randomResets(
                        360, zInferred)
                    cReconstructedRandReset = trained_mdl.calc_categorical_pdf(zReset_rand)
                    cReconRandResetList.append(cReconstructedRandReset)

                infoNames.append(filename)

                cTrueList_full.append(cTrueIndices)
                cReconListInf.append(cReconstructedInf)
                cReconResetList.append(cReconstructedReset)

                cReconFreeList.append(cReconstructedFree)


                fscore_inf = metrics.classification_report(cTrueIndices, cReconstructedInf, output_dict=True)
                fscore_free = metrics.classification_report(cTrueIndices, cReconstructedFree, output_dict=True)
                fscore_reset = metrics.classification_report(cTrueIndices, cReconstructedReset, output_dict=True)

                fscore_free_firstThird = metrics.classification_report(cTrueIndices[:120], cReconstructedFree[:120], output_dict=True)
                fscore_free_secondThird = metrics.classification_report(cTrueIndices[120:240], cReconstructedFree[120:240], output_dict=True)
                fscore_free_thirdThird = metrics.classification_report(cTrueIndices[240:], cReconstructedFree[240:], output_dict=True)

                randResetfScoreList = []
                for stack in range(0, randomRuns):
                    fscore_randReset = metrics.classification_report(cTrueIndices, cReconRandResetList[stack], output_dict=True)
                    randResetfScoreList.append(fscore_randReset['macro avg']['f1-score'])

                print(fscore_free['macro avg']['f1-score'], sum(randResetfScoreList)/len(randResetfScoreList),
                            fscore_reset['macro avg']['f1-score'], fscore_inf['macro avg']['f1-score'])

                plotdata = [fscore_free['macro avg']['f1-score'], sum(randResetfScoreList)/len(randResetfScoreList),
                            fscore_reset['macro avg']['f1-score'], fscore_inf['macro avg']['f1-score']]
                plt.plot(xAxis, plotdata, 'o-', label='Subject {}'.format(filename[-13:-12]), color=colors[colorVariable], linewidth=2.5, alpha=0.9)
                plt.gcf().subplots_adjust(bottom=0.1)
                plt.gcf().subplots_adjust(left=0.2)
                colorVariable += 1

                scoreList.append(fscore_free['macro avg']['f1-score'])
                scoreList.append(sum(randResetfScoreList)/len(randResetfScoreList))
                scoreList.append(fscore_reset['macro avg']['f1-score'])
                scoreList.append(fscore_inf['macro avg']['f1-score'])

                scoreListThirds.append(fscore_free_firstThird['macro avg']['f1-score'])
                scoreListThirds.append(fscore_free_secondThird['macro avg']['f1-score'])
                scoreListThirds.append(fscore_free_thirdThird['macro avg']['f1-score'])


                scoreListRandInstr.append(fscore_reset['macro avg']['f1-score'])
                scoreListRandInstr.append(sum(randResetfScoreList)/len(randResetfScoreList))

                scoreListSignTestRand.append(sum(randResetfScoreList)/len(randResetfScoreList))
                scoreListSignTestInstr.append(fscore_reset['macro avg']['f1-score'])

                fScoreFreeList.append(fscore_free['macro avg']['f1-score'])
                fScoreRandResetList.append(sum(randResetfScoreList)/len(randResetfScoreList))
                fScoreInstrResetList.append(fscore_reset['macro avg']['f1-score'])
                fScoreInferedList.append(fscore_inf['macro avg']['f1-score'])

    rcParams['figure.figsize'] = 8.5, 6.5

    plotdata = [np.mean(fScoreFreeList), np.mean(fScoreRandResetList),np.mean(fScoreInstrResetList),
                np.mean(fScoreInferedList)]
    #plt.plot(xAxis, plotdata, 'o-', label='mean', color='black',
    #         linewidth=5.5)

    plt.plot(xAxis, plotdata, 'o-', label='mean', color='red',
             linewidth=5.5)

    plt.ylabel(r'$\mathrm{F1} ~~ \mathrm{Score}$')
    plt.savefig(PLOTPATH + 'Fig4C.pdf')

    print("Skipped {} files".format(continueCount))


    experiments = 4

    print(len(np.repeat(np.arange(1, len(relevantRuns)+1), experiments)),
          len(np.tile(np.arange(1, experiments+1), len(relevantRuns))),
          len(scoreList))

    df = pd.DataFrame({'subject': np.repeat(np.arange(1, len(relevantRuns)+1), experiments),
                     'experiment': np.tile(np.arange(1, experiments+1), len(relevantRuns)),
                     'f1Score': scoreList})

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

    print("Ttest Test: free/randReset")

    res = scipy.stats.ttest_ind(fScoreFreeList, fScoreRandResetList)
    print(res)

    print("Ttest Test: randReset/InstrReset")

    res = scipy.stats.ttest_ind(fScoreRandResetList, fScoreInstrResetList)
    print(res)

    print("Ttest Test: instrReset/inferred")

    res = scipy.stats.ttest_ind(fScoreInstrResetList, fScoreInferedList)
    print(res)


    print("---------------------------------------")
    print("---------------------------------------")

    print("Wilcoxon Test: free/randReset")

    wilcoxon = scipy.stats.wilcoxon(fScoreFreeList, fScoreRandResetList)
    print(wilcoxon)

    print("Wilcoxon Test: randReset/InstrReset")

    wilcoxon = scipy.stats.wilcoxon(fScoreRandResetList, fScoreInstrResetList)
    print(wilcoxon)

    print("Wilcoxon Test: instrReset/inferred")

    wilcoxon = scipy.stats.wilcoxon(fScoreInstrResetList, fScoreInferedList)
    print(wilcoxon)


    print("---------------------------------------")
    print("---------------------------------------")

    print("Mann-Whitney Test:")

    whitneyu, p = scipy.stats.mannwhitneyu(scoreListSignTestRand, scoreListSignTestInstr)
    print(whitneyu, p)

    print("---------------------------------------")
    print("---------------------------------------")


    print("Wilcoxon Test:")
    wilcoxon = scipy.stats.wilcoxon(scoreListSignTestRand, scoreListSignTestInstr)
    print(wilcoxon)


    print("---------------------------------------")
    print("---------------------------------------")

    print("Ttest:")

    res = scipy.stats.ttest_ind(scoreListSignTestRand, scoreListSignTestInstr)
    print(res)

    print("---------------------------------------")
    print("---------------------------------------")

    print("Only between Rand & Instr:")


    df = pd.DataFrame({'subject': np.repeat(np.arange(1, len(relevantRuns)+1), experiments),
                     'experiment': np.tile(np.arange(1, experiments+1), len(relevantRuns)),
                     'f1Score': scoreList})

    anova = AnovaRM(data=df, depvar='f1Score', subject='subject', within=['experiment']).fit()
    print(anova)
    print(anova.summary())

    print("---------------------------------------")
    print("---------------------------------------")

    print("Thirds Test")


    df = pd.DataFrame({'subject': np.repeat(np.arange(1, len(relevantRuns)+1), experiments),
                     'thirds': np.tile(np.arange(1, experiments+1), len(relevantRuns)),
                     'f1Score': scoreList})

    anova = AnovaRM(data=df, depvar='f1Score', subject='subject', within=['thirds']).fit()
    print(anova)
    print(anova.summary())

    print("---------------------------------------")
    print("---------------------------------------")

    print("Tukey Test:")

    tukey = pairwise_tukeyhsd(endog=df['f1Score'], groups=df['thirds'], alpha=0.05)
    print(tukey)

    scoreListThirds = np.array(scoreListThirds)

    print("F1 scores: First third, second third, third third")

    print(sum(scoreListThirds[0::3])/len(scoreListThirds)*3,
          sum(scoreListThirds[1::3])/len(scoreListThirds)*3,
          sum(scoreListThirds[2::3])/len(scoreListThirds)*3)


def main():
    print("Getting files from " + pwd)

    calculateSomeStatistics('multimodal')


if __name__ == "__main__":
    main()
