import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
from matplotlib import rcParams
import matplotlib.font_manager
from sklearn import metrics
import pandas as pd
import matplotlib.colors as colors

sys.path.insert(0, "seqmvae/modules/")

import load_matlab_data
import datagenerator


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

cmap = plt.get_cmap('BrBG')
new_cmap = truncate_colormap(cmap, 0.5, 1.0)

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Computer Modern Roman serif']})
rc('text', usetex=True)

PLOTPATH = '../plots/'
DATAPATH = '../data/CrossValidation'

matrixLabels = [r'$\mathrm{Rest}$', r'$\mathrm{Instr}$', r'$\mathrm{CRT}$', r'$\mathrm{CDRT}$', r'$\mathrm{CMT}$']

def main():
    print("Getting files from " + DATAPATH)

    confusionMatrices_leftout = []
    confusionMatrices_full = []
    confusionMatrices_reset = []
    confusionMatrices_free = []
    reconSuccesses = []

    infoNames = []
    countList = []

    cTrueList_leftout = []
    cTrueList_full = []
    cReconList = []
    cReconResetList = []
    cReconFullList = []
    cReconFreeList = []

    continueCount = 0

    for idx, filename in enumerate(os.listdir(DATAPATH)):
        filepath = DATAPATH + '/' + filename

        if filename.startswith("CV_sparse") and not filename.endswith("all.mat"):

            fileIndex = int(filename[-5:-4]) - 1

            data = load_matlab_data.loadmat(filepath)

            xTrue = data['xTrue']
            xTrue = stackNumpyArray(xTrue, 20)

            cTrue = data['cTrue']
            cTrue = stackNumpyArray(cTrue, 5)

            zInferredLeftOut = data['zInfLeftout']
            zInferredOther = data['zInfOther']
            zInferredOther = stackNumpyArray(zInferredOther, 20)

            A = data['A']
            B = data['B']
            C = data['C']
            W = data['W']
            h = data['h']
            mu0 = data['mu0']
            beta = data['beta']

            zInferredLeftOut = torch.from_numpy(zInferredLeftOut).type(torch.FloatTensor).t()
            A = torch.from_numpy(A)
            A = torch.diag(A)
            B = torch.from_numpy(B)
            C = torch.from_numpy(C)
            W = torch.from_numpy(W)

            h = torch.from_numpy(h)
            mu0 = torch.from_numpy(mu0)
            beta = torch.from_numpy(beta).t()

            dim_x = 20
            dim_z = 20
            dim_c = 5

            args_dict = {'dim_ex': 5, 'dim_reg': 5, 'tau': 1, 'use_hrf': False, 'useExplicitHrf': False,
                         'repetitionTime': 6,
                         'useBaseExpansion': False, 'A': A, 'B': B, 'C': C, 'W': W, 'h': h, 'mu0': mu0, 'beta': beta}

            trained_mdl = datagenerator.DataGenerator(dim_x, dim_z, args_dict, dim_c,
                                                      args_dict, 'uniform',
                                                      False,
                                                      nonlinearity=torch.nn.functional.relu, externalInputs=None)
            xReconstructed, zReconstructed = trained_mdl.generate_timeseries(360, noise=False)



            zInferredFull = torch.zeros(360, dim_z, requires_grad=False)
            for ind in range(0, 360):
                if ind >= 0 and ind < fileIndex * 72:
                    zInferredFull[ind] = zInferredOther.t()[ind]
                if ind >= fileIndex * 72 and ind < (fileIndex + 1) * 72:
                    zInferredFull[ind] = zInferredLeftOut[ind - fileIndex * 72]
                if ind >= (fileIndex + 1) * 72:
                    zInferredFull[ind] = zInferredOther.t()[ind - (fileIndex + 1) * 72]

            cTrue = cTrue.t()
            cTrueIndices = torch.zeros(len(cTrue))
            for ind in range(0, len(cTrue)):
                max, index = torch.max(cTrue[ind], 0)
                cTrueIndices[ind] = index

            xReset, zReset = trained_mdl.generate_reset_timeseries_for_fmriExperiment_fromCV(360, zInferredLeftOut,
                                                                                      zInferredOther.t(), cTrueIndices, fileIndex)
            cIndicesLeftOut = cTrueIndices[fileIndex * 72:(fileIndex + 1) * 72]

            try:
                cReconstructed = trained_mdl.calc_categorical_pdf(zInferredLeftOut)
                cReconstructedFull = trained_mdl.calc_categorical_pdf(zInferredFull)
                cReconstructedReset = trained_mdl.calc_categorical_pdf(zReset)
                cReconstructedFree = trained_mdl.calc_categorical_pdf(zReconstructed)
            except:
                print("Something went wrong with calculating categorical pdfs... Skipping this file")
                continueCount += 1
                continue

            confusion_matrix_leftout = metrics.confusion_matrix(cIndicesLeftOut, cReconstructed)
            confusionMatrices_leftout = appendConfusionMatrix(confusionMatrices_leftout, confusion_matrix_leftout, filename)

            confusion_matrix_reset = metrics.confusion_matrix(cTrueIndices, cReconstructedReset)
            confusionMatrices_reset = appendConfusionMatrix(confusionMatrices_reset, confusion_matrix_reset, filename)

            confusion_matrix_full = metrics.confusion_matrix(cTrueIndices, cReconstructedFull)
            confusionMatrices_full = appendConfusionMatrix(confusionMatrices_full, confusion_matrix_full, filename)

            confusion_matrix_free = metrics.confusion_matrix(cTrueIndices, cReconstructedFree)
            confusionMatrices_free = appendConfusionMatrix(confusionMatrices_free, confusion_matrix_free, filename)

            count = 0
            for ind in range(0, len(cIndicesLeftOut)):
                if cIndicesLeftOut[ind] == cReconstructed[ind]:
                    count += 1

            countList.append(count)
            infoNames.append(filename)
            reconSucces = count / len(cIndicesLeftOut)
            reconSuccesses.append(reconSucces)

            cTrueList_leftout.append(cIndicesLeftOut)
            cTrueList_full.append(cTrueIndices)
            cReconList.append(cReconstructed)
            cReconResetList.append(cReconstructedReset)
            cReconFullList.append(cReconstructedFull)
            cReconFreeList.append(cReconstructedFree)

    print("Skipped {} files".format(continueCount))


    dictMultimodal = {'reconSuccess': reconSuccesses, 'confusionMatrices': confusionMatrices_leftout,
                      'confusionMatrices_reset': confusionMatrices_reset, 'confusionMatrices_full': confusionMatrices_full,
                      'confusionMatrices_free': confusionMatrices_free, 'names': infoNames, 'count': countList,
                      'cTrue': cTrueList_leftout,'cTrueFull': cTrueList_full, 'cReconstructed': cReconList,
                       'cReconstructedReset': cReconResetList, 'cReconstructedFull': cReconFullList,
                      'cReconstructedFree': cReconFreeList}


    infoNames = dictMultimodal['names']

    print(len(reconSuccesses), len(confusionMatrices_leftout), len(infoNames), len(cTrue), len(cReconList))

    dataFrame = pd.DataFrame(dictMultimodal)
    dataFrame = dataFrame.sort_values(by=['names'])

    runsToClean = ['pat_1_cv_1','pat_1_cv_2','pat_1_cv_3','pat_1_cv_4','pat_1_cv_5','pat_2_cv_3','pat_4_cv_4','pat_4_cv_5',
                   'pat_5_cv_1','pat_5_cv_2','pat_5_cv_4','pat_5_cv_5','pat_6_cv_2','pat_10_cv_2','pat_10_cv_4','pat_7_cv_2',
                   'pat_13_cv_1','pat_8_cv_5','pat_9_cv_1','pat_9_cv_2','pat_9_cv_5','pat_14_cv_1','pat_14_cv_2','pat_15_cv_1',
                   'pat_15_cv_3','pat_17_cv_1','pat_17_cv_2','pat_17_cv_3','pat_18_cv_2','pat_18_cv_3','pat_18_cv_5','pat_19_cv_1',
                   'pat_19_cv_2','pat_19_cv_4','pat_21_cv_2','pat_21_cv_3','pat_22_cv_5','pat_23_cv_1','pat_23_cv_3','pat_23_cv_4']

    patients = ['pat_2_', 'pat_3_', 'pat_4_', 'pat_5_', 'pat_6_', 'pat_7_', 'pat_8_',
                'pat_9_', 'pat_10_', 'pat_11_', 'pat_12_', 'pat_13_', 'pat_14_', 'pat_15_',
                'pat_17_', 'pat_18_', 'pat_19_', 'pat_21_', 'pat_22_', 'pat_23_', 'pat_25']

    sortedInfoNames = dataFrame['names'].values
    sortedConfusionMatrices = dataFrame['confusionMatrices'].values
    sorted_cTrue = dataFrame['cTrue'].values
    sorted_cRecon = dataFrame['cReconstructed'].values


    report_leftout = cleanAndProcessConfusionMatrices(patients, runsToClean, sortedConfusionMatrices,
                                     sortedInfoNames, sorted_cRecon, sorted_cTrue, 'leftout', plot=True)

    sortedInfoNames = dataFrame['names'].values
    sortedConfusionMatrices = dataFrame['confusionMatrices_free'].values
    sorted_cTrue = dataFrame['cTrueFull'].values
    sorted_cRecon = dataFrame['cReconstructedFree'].values


    report_free = cleanAndProcessConfusionMatrices(patients, runsToClean, sortedConfusionMatrices,
                                     sortedInfoNames, sorted_cRecon, sorted_cTrue, 'generated')

    sortedInfoNames = dataFrame['names'].values
    sortedConfusionMatrices = dataFrame['confusionMatrices_reset'].values
    sorted_cTrue = dataFrame['cTrueFull'].values
    sorted_cRecon = dataFrame['cReconstructedReset'].values


    report_reset = cleanAndProcessConfusionMatrices(patients, runsToClean, sortedConfusionMatrices,
                                     sortedInfoNames, sorted_cRecon, sorted_cTrue, 'reset')

    sortedInfoNames = dataFrame['names'].values
    sortedConfusionMatrices = dataFrame['confusionMatrices_full'].values
    sorted_cTrue = dataFrame['cTrueFull'].values
    sorted_cRecon = dataFrame['cReconstructedFull'].values


    report_inferred = cleanAndProcessConfusionMatrices(patients, runsToClean, sortedConfusionMatrices,
                                     sortedInfoNames, sorted_cRecon, sorted_cTrue, 'inferred')

    print("Skipped {} files".format(continueCount))

    print("Final statistics:")

    print("Left Out Plots & Statistics:")
    print(report_leftout)
    print("Free Plots & Statistics:")
    print(report_free)
    print("Reset Plots & Statistics:")
    print(report_reset)
    print("Inferred Full Plots & Statistics:")
    print(report_inferred)


def cleanAndProcessConfusionMatrices(patients, runsToClean, sortedConfusionMatrices,
                                     sortedInfoNames, sorted_cRecon, sorted_cTrue, experiment, plot=False):
    cleanedConfusionMatrices = []
    cleanedNames = []
    cleaned_cTrue = []
    cleaned_cRecon = []
    for ind in range(0, len(sortedConfusionMatrices)):
        savePatient = True
        infoname = sortedInfoNames[ind][:-4]
        for j in range(0, len(runsToClean)):
            if runsToClean[j] in infoname:
                savePatient = False
        if savePatient:
            cleanedConfusionMatrices.append(sortedConfusionMatrices[ind])
            cleanedNames.append(infoname)
            cleaned_cTrue.append(sorted_cTrue[ind])
            cleaned_cRecon.append(sorted_cRecon[ind])
    cleaned_cTrue = np.concatenate(cleaned_cTrue, axis=0)
    cleaned_cRecon = np.concatenate(cleaned_cRecon, axis=0)
    print(len(cleanedNames))
    patientConfusionMatricesCleaned = []
    for j in range(0, len(patients)):
        patientConfusionMatrix = torch.zeros(sortedConfusionMatrices[0].shape)
        for ind in range(0, len(cleanedConfusionMatrices)):
            if patients[j] in cleanedNames[ind]:
                patientConfusionMatrix += cleanedConfusionMatrices[ind]

        patientConfusionMatricesCleaned.append(patientConfusionMatrix)

    averagedConfusionCleaned = getAverageConfusionMatrix(patientConfusionMatricesCleaned)
    averagedConfusionCleaned = averagedConfusionCleaned.round(2)
    if plot:
        plotConfusionMatrix(averagedConfusionCleaned, "averaged_cleaned_" + experiment)
    torch.save(torch.tensor(averagedConfusionCleaned), '../data/confusionMatrix')
    report = metrics.classification_report(cleaned_cTrue, cleaned_cRecon)
    print(report)
    print(averagedConfusionCleaned)
    mcc = metrics.matthews_corrcoef(cleaned_cTrue, cleaned_cRecon)
    print(mcc)
    print("-----------------------------------------------")
    return report


def appendConfusionMatrix(confusionMatrices, confusion_matrix, filename):
    if torch.tensor(confusion_matrix).shape == torch.zeros(5, 5).shape:
        if not np.isnan(confusion_matrix).any():
            confusionMatrices.append(confusion_matrix)
    else:
        confusionMatrices.append(torch.zeros(5, 5))
        print("0s Confusion matrix added to {} because of nans or bad shape".format(filename))
    return confusionMatrices


def getAverageConfusionMatrix(confusionMatrices, normalise=True):
    averageConfusionMatrix = torch.zeros(torch.tensor(confusionMatrices[1]).shape)
    for ind in range(0, len(confusionMatrices)):
        averageConfusionMatrix += confusionMatrices[ind]
    averageConfusionMatrix = averageConfusionMatrix.numpy()
    if normalise==True:
        averageConfusionMatrix = averageConfusionMatrix.astype('float') / averageConfusionMatrix.sum(axis=0)[np.newaxis,:]

    return averageConfusionMatrix


def stackNumpyArray(array, dimArray):
    newarray = torch.zeros(dimArray, 360)
    for ind in range(0, len(array)):
        for j in range(0, dimArray):
            newarray[j][ind * 72:(ind + 1) * 72] = torch.tensor(array[ind][j])
    return newarray


def plotConfusionMatrix(averageConfusionMatrix, plotname):
    rcParams['figure.figsize']=7.5,7.5
    rcParams.update({'font.size': 26})
    ax = plt.subplot()
    #sns.set(rc={'text.usetex': True})
    sns.heatmap(averageConfusionMatrix, annot=True, ax=ax, cmap=new_cmap, xticklabels=matrixLabels,
                yticklabels=matrixLabels)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.15)
    plt.xlabel(r'$\mathrm{True}$', fontsize=26)
    plt.ylabel(r'$\mathrm{Predicted}$',fontsize=26)
    plt.savefig(PLOTPATH + 'Fig3C.pdf')
    print("Averaged Confusion Matrix plotted")
    plt.close()


if __name__ == "__main__":
    main()
