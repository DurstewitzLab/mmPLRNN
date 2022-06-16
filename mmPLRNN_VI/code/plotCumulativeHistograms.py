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
    rcParams.update({'font.size': 28})
    for i in range(0, len(histogram_data)):
        data = np.array(histogram_data[i])
        data = data[:100]
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


    data = torch.load(DATAPATH + 'KLs_noisyLorenz_EM')
    makeDensityPlotCumulative(data, 'Fig2_C', 'EM')

    data = torch.load(DATAPATH + 'KLs_noisyLorenz_VI')
    makeDensityPlotCumulative(data, 'Fig2_D', 'VI')

    data = torch.load(DATAPATH + 'KLs_missDim_EM')
    makeDensityPlotCumulative(data, 'Fig5_EM', 'EM')

    data = torch.load(DATAPATH + 'KLs_missDim_VI')
    makeDensityPlotCumulative(data, 'Fig5_VI', 'VI')






if __name__ == "__main__":
    main()
