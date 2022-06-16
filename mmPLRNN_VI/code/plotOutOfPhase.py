import os
import torch
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
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

colors = []
cmap_temp = np.linspace(0.0, 1.0, 5)
for number in cmap_temp:
    cmap = matplotlib.cm.get_cmap('viridis')
    colors.append(cmap(number))

colors = ['black', 'gold', 'cyan', 'magenta', 'mediumseagreen']
colorsCategories = ['black', 'gold', 'cyan', 'magenta', 'mediumseagreen']
PLOTPATH = '../plots/'

phases = 5

labelsRowBeta = ['Rest', 'Instr', 'CRT', 'CDRT', 'CMT']

ele = [5, 10, 20, 35]
az = [10, 20, 35, 50, 70, 110, 130, 150]


def plotAndSave2dLineCollection(true_data, file_path, colorlist, label, x1label, x2label):
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
    plt.xlabel(x1label)
    plt.ylabel(x2label)
    plt.savefig(file_path + '_subspace_' + label)

    plt.close()


def plotAndSaveTimeSeries(zReconstructed, zInferred, colorlist1, colorlist2, colorlistTrue, filename):
    rcParams.update({'font.size': 28})
    rcParams['figure.figsize'] = 16.5, 6.5

    rcParams['axes.titley'] = 1.0
    rcParams['axes.titlepad'] = -30

    zInferred = zInferred.T[0].T
    zInferred = np.array(zInferred)
    zReconstructed = zReconstructed.T[0].T
    zReconstructed = np.array(zReconstructed)

    fig, ax = plt.subplots(2, sharex = True)
    timesteps = np.arange(0, 360)


    points = np.array([timesteps, zReconstructed]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, colors=colorlist1, label=r'$\mathrm{free}$', linewidth=2.5)
    ax[0].add_collection(lc)
    ax[0].set_title(r'$\mathrm{free}$')

    points = np.array([timesteps, zInferred]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, colors=colorlist2, label=r'$\mathrm{instrReset}$', linewidth=2.5)
    ax[1].add_collection(lc)
    ax[1].set_title(r'$\mathrm{instrReset}$')

    colorcount = 0
    for ind in range(0, len(colorlistTrue)-1):
        if colorlistTrue[ind] == colorlistTrue[ind+1]:
            colorcount += 1
            continue
        else:
            ax[0].axvspan(timesteps[ind-colorcount], timesteps[ind+1], color = colorlistTrue[ind], alpha=0.3, linewidth=0)
            ax[1].axvspan(timesteps[ind-colorcount], timesteps[ind+1], color = colorlistTrue[ind], alpha=0.3, linewidth=0)
            colorcount = 0

    ax[0].autoscale()
    ax[1].autoscale()

    n_colors = len(labelsRowBeta)

    lower_lim = 184
    upper_lim = 324

    #cmap = matplotlib.colors.ListedColormap(colorsCategories)
    #bounds = [0, 1, 2, 3, 4, 5]
    #norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    #c_ticks = np.arange(n_colors) * ((n_colors+1) / (n_colors + 1)) + (2 / n_colors)
    #cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax[0], ticks=c_ticks, pad=10)
    #cbar.ax.set_yticklabels(labelsRowBeta)

    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.1)
    plt.xlim(lower_lim, upper_lim)
    plt.xlabel(r'$\mathrm{Timesteps}~ t$')
    ax[0].set_ylabel(r'$z_0$')
    ax[1].set_ylabel(r'$z_0$')
    ax[0].set_yticks((0,5))
    ax[1].set_yticks((0,5))
    #plt.legend(loc='lower left')
    plt.savefig(PLOTPATH + filename[:-4] + '_outOfPhase_single.pdf')

    '''
    cmapTrue = matplotlib.colors.ListedColormap(colorlistTrue[lower_lim:upper_lim])
    bounds = timesteps[lower_lim:upper_lim]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmapTrue.N)
    cbar2 = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmapTrue), ax=ax,
                         orientation='horizontal', pad=0.1)'''

    plt.savefig(PLOTPATH + filename[:-4] + '_outOfPhaseWithBar_single.pdf')
    plt.close()


def plotAndSave3dLineCollection(true_data, file_path, colorlist, x1label=1, x2label=2, x3label=3):
    fig = plt.figure(figsize=(16, 9))
    matplotlib.rcParams.update({'font.size': 20})
    true_data = (true_data - np.mean(true_data))/np.std(true_data)
    # `ax` is a 3D-aware axis instance, because of the projection='3d' keyword argument to add_subplot
    points = true_data.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    #lc = Line3DCollection(segments[100:], colors=colorlist)
    lc = Line3DCollection(segments, colors=colorlist)


    X, Y = np.meshgrid(np.arange(-2.5, 2.6, 0.5), np.arange(-2.5, 2.6, 0.5))
    Z = 0 * X

    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_xlabel(r"$x_{{x1label}}$".format(x1label=x1label), size=26)
    ax.set_ylabel(r'$x_{x2label}$'.format(x2label=x2label), size=26)
    ax.set_zlabel(r'$x_{x3label}$'.format(x3label=x3label), size=26)

    ax.add_collection(lc)

    lim = 2.5
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])

    n_colors = len(labelsRowBeta)
    cmap = matplotlib.colors.ListedColormap(colors)
    bounds = [0, 1, 2, 3, 4, 5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    c_ticks = np.arange(n_colors) * ((n_colors+1) / (n_colors + 1)) + (2 / n_colors)
    cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, ticks=c_ticks, alpha=0.9)
    cbar.ax.set_yticklabels(labelsRowBeta)

    for elevs in ele:
        for azus in az:
            ax.view_init(elevs, azus)
            plt.savefig(file_path + '_' + str(elevs) + '_' + str(azus))

    plt.close()

def plotAllSubSpaceLineCollections(z, cTrue, savepath, filename, scoreList, colorlist, u, v, w, labels):

    zSubspace = torch.cat((z.t()[u:u+1], z.t()[v:v+1], z.t()[w:w+1])).t().numpy()
    lda = LinearDiscriminantAnalysis()
    lda.fit(zSubspace, cTrue)
    score = lda.score(zSubspace, cTrue)


    plotpath = savepath + '_' + str(u) + '_' + str(v) + '_' + str(w) + '_' + filename
    id = filename + '_' + str(u) + '_' + str(v) + '_' + str(w)

    scoreList[0].append(score)
    scoreList[1].append(id)
    for x in range(0, len(zSubspace)):
        if sum(zSubspace[x] == 0):
            print(zSubspace[x])
            print(x)
            print("Anomaly found")
            break

    plotAndSave2dLineCollection(zSubspace.T[:-1].T, plotpath, colorlist, '1', x1label = labels[0], x2label= labels[1])
    plotAndSave2dLineCollection(zSubspace.T[1:].T, plotpath, colorlist, '2', x1label = labels[1], x2label= labels[2])
    plotAndSave2dLineCollection(zSubspace.T[::2].T, plotpath, colorlist, '3', x1label = labels[0], x2label= labels[2])

    return scoreList


def getDataAndScores(experiment, datapath, savepath):
    scoreList = [[],[]]

    filenames = ['Sparse_mmPLRNN_m20pat_12_init_3.mat']
    dimensions = [[7, 10, 11]]

    labels = [[r'$z_7$', r'$z_{{10}}$', r'$z_{{11}}$']]

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
                        #C = data['C']
                        W = data['W']
                        h = data['h']
                        mu0 = data['mu0']
                        beta = data['beta']
                        beta = torch.from_numpy(beta).t()

                    if experiment == 'unimodal':
                        xTrue = data['xTrue']
                        cTrue = data['cTrue']

                        zInferred = data['zInf']
                        A = data['AG']
                        B = data['BG']
                        #C = data['CG']
                        W = data['WG']
                        h = data['hG']
                        mu0 = data['mu0G']
                        beta = torch.rand(4, 20)

                    A = torch.from_numpy(A)
                    A = torch.diag(A)
                    B = torch.from_numpy(B)
                    #C = torch.from_numpy(C)
                    C = torch.rand(1)
                    W = torch.from_numpy(W)
                    h = torch.from_numpy(h)
                    cTrue = torch.from_numpy(cTrue).T
                    mu0 = torch.from_numpy(mu0)
                    zInferred = torch.from_numpy(zInferred).t().type(torch.FloatTensor)
                    print(mu0.shape)

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
                    for ind in range(0, len(cIndices)):
                        cIndices[ind] = torch.argmax(cTrue[ind])

                    cIndicesTrue = cIndices

                    xTrue = torch.from_numpy(xTrue).t()
                    T = 360
                    xReconstructed, zReconstructed = trained_mdl.generate_timeseries(T, noise=False)
                    xReset_inf, zReset_inf = trained_mdl.generate_reset_timeseries_for_fmriExperiment_fromFullTS(360,
                                                                                                                 zInferred,
                                                                                                                 cIndicesTrue)

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

                    zInferred = zInferred.T[:360].T
                    print(zInferred.shape, zReset_inf.shape)

                    colorlist2 = []
                    colorlistTrue = []
                    #probabilities = trained_mdl.calc_categorical_pdf(zInferred)
                    probabilities = trained_mdl.calc_categorical_pdf(zReset_inf)
                    cIndices = probabilities

                    for ind in range(0, len(cIndices)):
                        colorlist2.append(colors[int(cIndices[ind])])
                        colorlistTrue.append(colors[int(cIndicesTrue[ind])])

                    u, v, w = dimensions[j][0], dimensions[j][1], dimensions[j][2]

                    #scoreList = plotAllSubSpaceLineCollections(zReconstructed[100:], cIndices[100:], savepath,
                    #                                           filename[:-4], scoreList, colorlist, u, v, w, labels[j])

                    print(zReconstructed - zReset_inf)

                    plotAndSaveTimeSeries(zReconstructed, zReset_inf, colorlist, colorlist2, colorlistTrue, filename)


    return scoreList


#dataPath = '/zi-flstorage/group_theoretische_neuro/ZIFNASBackup/philine.bommer/Masters/PLRNN-multi-categorical-extension/BimodalPLRNN' \
#      '/Application/Experiments/GeneralizedVersion/Data/Evaluation/NetComparison/fMRI/FullTS/NoInp/m20/'

dataPath = '../data/FullTS/'

scoreList = [[],[]]

savePathSubSpaces = '/home/daniel.kramer/algorithms/analysis/subspaceAnalysis/'

scoreListMultimodal = getDataAndScores('multimodal', dataPath, savePathSubSpaces)

