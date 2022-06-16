import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
from matplotlib import rcParams
import matplotlib.font_manager

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Computer Modern Roman serif']})
rc('text', usetex=True)

"""globals"""

singleLables = [r'$\mathrm{mmPLRNN}$', r'$\mathrm{uniPLRNN}$']
colors = [[], []]
cmap_temp = np.linspace(0.0, 0.5, 2)

for number in cmap_temp:
    cmap = matplotlib.cm.get_cmap('viridis')
    colors[0].append(cmap(number))

for number in cmap_temp:
    cmap = matplotlib.cm.get_cmap('viridis')
    colors[1].append(cmap(number))

binNumber = 15
PLOTPATH = '../plots/'
DATAPATH = '../data/'


def makeDensityPlotCumulative(histogram_data, plotname, experiment):
    if experiment == 'VI':
        title = r'$\mathrm{VI}$'
    elif experiment == 'EM':
        title = r'$\mathrm{EM}$'
    else:
        "This experiment is not supported, aborting..."
        return

    rcParams['figure.figsize'] = 6.5, 6.5
    rcParams.update({'font.size': 26})
    for i in range(0, len(histogram_data)):
        data = np.array(histogram_data[i])
        data = data[:30]
        print(max(data))
        normValue = 18.38 #this Value is taken from all the KLx experiments
        data /= normValue

        print("Data has {} entries.".format(len(data)))
        kl = np.sort(data)
        plt.step(kl, np.linspace(0, kl.size, num=kl.size) / kl.size, color=colors[0][i], linewidth=2, label=singleLables[i], alpha=0.9)
        plt.fill_between(np.hstack((kl, np.array([1]))), 0, np.hstack((np.arange(1, len(kl) + 1) / (len(kl) - 0.8), np.array([1]))),
                         step='post', color=colors[0][i], alpha=0.4)
        median = np.median(data).round(2)
        percentile = np.percentile(data, 50).round(2)
        print(median, np.mean(data), percentile)
        median = r"${}$".format(median)
        plt.text(0.07, 0.65 - i * .08, r"$\bar{D}_\mathrm{KL}$ = " + median, color=colors[0][i])

    print("------------------------------------------")
    plt.tight_layout()
    plt.legend(loc='upper left', frameon=False)
    plt.xlabel(r'$D_\mathrm{KL}$')
    plt.ylabel(r'$\mathrm{Cumulative ~~ frequency}$')
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.title(title)
    plt.savefig(PLOTPATH + plotname + ".pdf")
    plt.close()


def main():
    '''Get data and reshape for plotting'''

    normValue = 18.38

    xAxis = [r'$5\%$', r'$10\%$', r'$15\%$', r'$20\%$', r'$30\%$']
    
    data_5p = torch.load(DATAPATH + 'KLs_noisyLorenz_5p_long')
    #makeDensityPlotCumulative(data_5p, 'noise5p', 'VI')

    data_10p = torch.load(DATAPATH + 'KLs_noisyLorenz_10p_long')
    #makeDensityPlotCumulative(data_10p, 'noise10p', 'VI')

    data_15p = torch.load(DATAPATH + 'KLs_noisyLorenz_15p_long')
    #makeDensityPlotCumulative(data_15p, 'noise15p', 'VI')


    data_20p = torch.load(DATAPATH + 'KLs_noisyLorenz_20p_long')
    #makeDensityPlotCumulative(data_20p, 'noise20p', 'VI')

    data_30p = torch.load(DATAPATH + 'KLs_noisyLorenz_30p_long')
      
    print("-------------------------------------------------------------")
    print("-------------------------------------------------------------")
    print("-------------------------------------------------------------")

    #dataList = [data_5p, data_10p, data_15p]
    dataList = [data_5p, data_10p, data_15p, data_20p]
    dataList = [data_5p, data_10p, data_15p, data_20p, data_30p]

    multimodalList = []
    unimodalList = []

    multimodalPercentileList = []
    unimodalPercentileList = []

    for ind in range(0, len(xAxis)):
        data_p = dataList[ind]
        for i in range(0, len(data_p)):
            data = np.array(data_p[i])
            data = data[:40]
            data /= normValue
            if i == 0:
                multimodalList.append(np.mean(data))
                #multimodalPercentileList.append(np.percentile(data, 25))
                multimodalPercentileList.append(np.std(data)/np.sqrt(len(data)))

            elif i == 1:
                unimodalList.append(np.mean(data))
                #unimodalPercentileList.append(np.percentile(data, 25))
                unimodalPercentileList.append(np.std(data)/np.sqrt(len(data)))

            elif i > 1:
                "Print something went wrong, breaking..."
                break

    multimodalList = np.array(multimodalList)
    multimodalPercentileList = np.array(multimodalPercentileList)
    unimodalList = np.array(unimodalList)
    unimodalPercentileList = np.array(unimodalPercentileList)


    differenceMultimodal = multimodalPercentileList
    differenceUnimodal = unimodalPercentileList

    plt.fill_between(xAxis, unimodalList + differenceUnimodal, unimodalList - differenceUnimodal, color=colors[0][1], alpha=0.5)
    plt.fill_between(xAxis, multimodalList + differenceMultimodal, multimodalList - differenceMultimodal, color=colors[0][0], alpha=0.5)

    plt.plot(xAxis, multimodalList, 'o-', color=colors[0][0], label=singleLables[0])
    plt.plot(xAxis, unimodalList, 'o-', color=colors[0][1], label=singleLables[1])

    plt.ylabel(r'$D_\mathrm{KL}$')
    plt.xlabel(r'$\mathrm{Noise}$')
    plt.legend()
    plt.ylim(0, 1)
    plt.savefig(PLOTPATH + "NoiseLevels.pdf")

    print(multimodalList)
    print(multimodalPercentileList)
    print(unimodalList)
    print(differenceUnimodal)


    plt.close()




if __name__ == "__main__":
    main()
