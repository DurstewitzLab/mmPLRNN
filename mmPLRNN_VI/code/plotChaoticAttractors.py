import os
import torch
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.collections import LineCollection
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sys
from matplotlib import rc
from matplotlib import rcParams

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Computer Modern Roman serif']})
## for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

sys.path.insert(0, "seqmvae/modules/")

import load_matlab_data
import datagenerator

colors = ['black', 'gold', 'cyan', 'magenta', 'mediumseagreen']
labelsRowBeta = ['Rest', 'Instr', 'CRT', 'CDRT', 'CMT']

DATAPATH = '../data/FullTS/'
PLOTPATH = '../plots/'

def plotAndSave2dLineCollection(true_data, id, colorlist, label, x1label, x2label):
    rcParams.update({'font.size': 26})
    points = true_data.reshape(-1, 1, 2)

    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, colors=colorlist, alpha=0.8)

    fig, ax = plt.subplots()

    ax.add_collection(lc)
    ax.autoscale()

    n_colors = len(labelsRowBeta)
    cmap = matplotlib.colors.ListedColormap(colors)
    bounds = [0, 1, 2, 3, 4, 5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    c_ticks = np.arange(n_colors) * ((n_colors+1) / (n_colors + 1)) + (2 / n_colors)
    cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, ticks=c_ticks)
    cbar.ax.set_yticklabels(labelsRowBeta)
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.gcf().subplots_adjust(left=0.1)
    #plt.yticks([3, 4])
    plt.xlabel(x1label)
    plt.ylabel(x2label)
    plt.savefig(PLOTPATH + 'Attractor_subspace_' + id + "_" + label + ".pdf")
    plt.close()

def plotAllSubSpaceLineCollections(z, cTrue, savepath, filename, scoreList, colorlist, u, v, w, labels):

    zSubspace = torch.cat((z.t()[u:u+1], z.t()[v:v+1], z.t()[w:w+1])).t().numpy()
    lda = LinearDiscriminantAnalysis()
    lda.fit(zSubspace, cTrue)
    score = lda.score(zSubspace, cTrue)

    categoriesInOrthants = []
    for ind in range(0, 8):
        categoriesInOrthants.append([])
    for ind in range(0, len(zSubspace)):
        x, y, z = zSubspace[ind]
        if (x > 0 and y > 0 and z > 0):
            categoriesInOrthants[0].append(cTrue[ind])
        if (x > 0 and y > 0 and z < 0):
            categoriesInOrthants[1].append(cTrue[ind])
        if (x > 0 and y < 0 and z < 0):
            categoriesInOrthants[2].append(cTrue[ind])
        if (x < 0 and y < 0 and z < 0):
            categoriesInOrthants[3].append(cTrue[ind])
        if (x < 0 and y < 0 and z > 0):
            categoriesInOrthants[4].append(cTrue[ind])
        if (x < 0 and y > 0 and z < 0):
            categoriesInOrthants[5].append(cTrue[ind])
        if (x > 0 and y < 0 and z > 0):
            categoriesInOrthants[6].append(cTrue[ind])
        if (x < 0 and y > 0 and z > 0):
            categoriesInOrthants[7].append(cTrue[ind])
    for ind in range(0, 8):
        values, counts = np.unique(categoriesInOrthants[ind], return_counts=True)

    plotpath = savepath + '_' + str(u) + '_' + str(v) + '_' + str(w) + '_' + filename
    id = '_' + str(u) + '_' + str(v) + '_' + str(w) + "_"

    scoreList[0].append(score)
    scoreList[1].append(id)
    for x in range(0, len(zSubspace)):
        if sum(zSubspace[x] == 0):
            print(zSubspace[x])
            print(x)
            print("Anomaly found")
            break

    plotAndSave2dLineCollection(zSubspace.T[:-1].T, id, colorlist, '1', x1label = labels[0], x2label= labels[1])
    plotAndSave2dLineCollection(zSubspace.T[1:].T, id, colorlist, '2', x1label = labels[1], x2label= labels[2])
    plotAndSave2dLineCollection(zSubspace.T[::2].T, id, colorlist, '3', x1label = labels[0], x2label= labels[2])

    return scoreList


def plotSomeChaoticTrajectories(experiment, datapath, savepath):
    scoreList = [[],[]]

    filenames = ['Sparse_mmPLRNN_m20pat_12_init_3.mat',
                 'Sparse_mmPLRNN_m20pat_3_init_1.mat',
                 'Sparse_mmPLRNN_m20pat_2_init_1.mat']

    dimensions = [[7, 10, 11], [0, 9, 11], [9, 14, 18]]

    labels = [[r'$z_7$', r'$z_{{10}}$', r'$z_{{11}}$'],
              [r'$z_0$', r'$z_{{9}}$', r'$z_{{11}}$'],
              [r'$z_{{9}}$', r'$z_{{14}}$', r'$z_{{18}}$']]

    for idx, filename in enumerate(os.listdir(datapath)):
        filepath = datapath + '/' + filename

        if experiment == 'multimodal':
            filenameBeginning = "Sparse_mmPL"
        if experiment == 'unimodal':
            filenameBeginning = "Sparse_PL"

        if filename.startswith(filenameBeginning):
            for j in range(0, len(filenames)):
                if filename.startswith(filenames[j]):
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

                        zInferred = data['zInf']
                        A = data['A']
                        B = data['B']
                        C = data['C']
                        W = data['W']
                        h = data['h']
                        mu0 = data['mu0']
                        beta = torch.rand(4, 20)

                    A = torch.from_numpy(A)
                    A = torch.diag(A)
                    B = torch.from_numpy(B)
                    W = torch.from_numpy(W)
                    h = torch.from_numpy(h)
                    cTrue = torch.from_numpy(cTrue).T
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
                    cIndices = torch.zeros((360))
                    for ind in range(0, len(cTrue)):
                        cIndices[ind] = torch.argmax(cTrue[ind])


                    xTrue = torch.from_numpy(xTrue).t()
                    T = 50000
                    xReconstructed, zReconstructed = trained_mdl.generate_timeseries(T, noise=False)

                    if torch.isnan(zReconstructed).any() or torch.isnan(cIndices).any():
                        print("Found nan values in zReconstructed or cTrue. Continuing...")
                        continue

                    timesteps = []
                    for ind in range(1, len(cIndices)):
                        if cIndices[ind] == 1 and (cIndices[ind] - cIndices[ind - 1]) > 0:
                            timesteps.append(ind)

                    probabilities = trained_mdl.calc_categorical_pdf(zReconstructed)
                    cReconstructed = probabilities

                    colorlist = []
                    cIndices = cReconstructed

                    for ind in range(0, len(cIndices)):
                        colorlist.append(colors[int(cIndices[ind])])

                    u, v, w = dimensions[j][0], dimensions[j][1], dimensions[j][2]

                    scoreList = plotAllSubSpaceLineCollections(zReconstructed[100:], cIndices[100:], savepath,
                                                               filename[:-4], scoreList, colorlist, u, v, w, labels[j])
                    print("Attractor plotted...")
    return scoreList

scoreListMultimodal = plotSomeChaoticTrajectories('multimodal', DATAPATH, PLOTPATH)

