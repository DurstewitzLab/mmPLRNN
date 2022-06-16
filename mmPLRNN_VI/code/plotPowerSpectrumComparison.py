import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.font_manager
from matplotlib import rcParams

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Computer Modern Roman serif']})
rc('text', usetex=True)


'''Parameters for smoothing Power-Spectrum'''
SMOOTHING_SIGMA = 1.05
FREQUENCY_CUTOFF = 80

'''Indexes chosen manually, to have a plot of well reconstructed Powerspectra for both multi and unimodal case.'''
index_Multimodal = 7
index_Unimodal = 15

colors = []
cmap_temp = np.linspace(0.0, 0.5, 2)
for number in cmap_temp:
    cmap = matplotlib.cm.get_cmap('viridis')
    colors.append(cmap(number))

def standardiseData(dim, data):
    data = data.t()
    for ind in range(0, dim):
        data[ind] = (data[ind] - torch.mean(data[ind])) / torch.std(data[ind])
    data = data.t()
    return data

def convert_to_decibel(x):
    x = 20 * np.log10(x)
    return x[0]


def ensure_length_is_even(x):
    n = len(x)
    if n % 2 != 0:
        x = x[:-1]
        n = len(x)
    x = np.reshape(x, (1, n))
    return x


def fft_in_decibel(x):
    """
    Originally by: Vlachas Pantelis, CSE-lab, ETH Zurich in https://github.com/pvlachas/RNN-RC-Chaos
    Calculate spectrum in decibel scale,
    scale the magnitude of FFT by window and factor of 2, because we are using half of FFT spectrum.
    :param x: input signal
    :return fft_decibel: spectrum in decibel scale
    """
    x = ensure_length_is_even(x)
    fft_real = np.fft.rfft(x)
    fft_magnitude = np.abs(fft_real) * 2 / len(x)
    fft_decibel = convert_to_decibel(fft_magnitude)

    fft_smoothed = kernel_smoothen(fft_decibel, kernel_sigma=SMOOTHING_SIGMA)
    return fft_smoothed


def fft_smoothed(x):
    x = ensure_length_is_even(x)
    fft_real = np.fft.rfft(x)
    fft_magnitude = np.abs(fft_real) * 2 / len(x[0])
    fft_smoothed = kernel_smoothen(fft_magnitude[0], kernel_sigma=SMOOTHING_SIGMA)
    return fft_smoothed


def get_average_spectrum(trajectories):
    spectrum = []
    for trajectory in trajectories:
        spectrum.append(trajectory)
    spectrum = np.array(spectrum).mean(axis=1)
    return spectrum

def power_spectrum_error_per_dim(x_gen, x_true):
    dim_x = x_gen.shape[1]
    pse_corrs_per_dim = []
    for dim in range(dim_x):
        spectrum_true = x_true[:FREQUENCY_CUTOFF, dim]
        spectrum_gen = x_gen[:FREQUENCY_CUTOFF, dim]
        pse_corr_per_dim = np.corrcoef(x=spectrum_gen, y=spectrum_true)[0, 1]
        pse_corrs_per_dim.append(pse_corr_per_dim)
    return pse_corrs_per_dim

def power_spectrum_error(x_gen, x_true):
    pse_errors_per_dim = power_spectrum_error_per_dim(x_gen, x_true)
    return np.array(pse_errors_per_dim).mean(axis=0)

def kernel_smoothen(data, kernel_sigma=1):
    """
    Smoothen data with Gaussian kernel
    @param kernel_sigma: standard deviation of gaussian, kernel_size is adapted to that
    @return: internal data is modified but nothing returned
    """
    kernel = get_kernel(kernel_sigma)
    data_final = data.copy()
    data_conv = np.convolve(data, kernel)
    pad = int(len(kernel) / 2)
    data_final = data_conv[pad:-pad]
    data = data_final[:, np.newaxis]
    return data


def gauss(x, sigma=1):
    return 1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-1 / 2 * (x / sigma) ** 2)


def get_kernel(sigma):
    size = int(sigma * 10 + 1)
    kernel = list(range(size))
    kernel = [float(k) - int(size / 2) for k in kernel]
    kernel = [gauss(k, sigma) for k in kernel]
    kernel = [k / np.sum(kernel) for k in kernel]
    return kernel

def smooth_data_dimensions(data):
    data = torch.tensor(data)
    data = standardiseData(20, data)
    data_smoothed = []
    dim = data.shape[1]
    for d in range(dim):
        data_smoothed.append(fft_smoothed(data[:, d]))
    data_smoothed = np.column_stack(data_smoothed)
    return data_smoothed

def plot_spectrum_comparison_3TS(s_true, s_gen, s_gen_uni):
    rcParams['figure.figsize'] = 7, 7
    rcParams.update({'font.size': 27})
    samplingRate = 1/3/100 #Hz
    xAxis = np.arange(0, 79.5*samplingRate, samplingRate)
    s_true, s_gen, s_gen_uni = s_true*samplingRate, s_gen*samplingRate, s_gen_uni*samplingRate
    plt.plot(xAxis, s_true, color='black', linewidth=2.5, label=r'$\mathrm{true}$', alpha=1.0)
    plt.plot(xAxis, s_gen, color=colors[0], linewidth=2.5, label=r'$\mathrm{mmPLRNN}$', alpha=0.8)
    plt.plot(xAxis, s_gen_uni, color=colors[1], linewidth=2.5, label=r'$\mathrm{uniPLRNN}$', alpha=0.8)
    plt.xlim(0,78*samplingRate)
    plt.gcf().subplots_adjust(left=0.2)
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.yticks((0, 0.001))
    plt.xlabel(r'$\mathrm{Frequency}~\mathrm{in}~\mathrm{Hz}$', fontsize=27)
    plt.ylabel(r'$\mathrm{Power}$', fontsize=27)
    plt.legend(frameon=False)
    plt.savefig('../plots/powerSpectrum.pdf')
    plt.close()


def main():

    xReconstructedListMultimodal = torch.load('../data/PowerSpectrum/xReconstructedList_EM_multimodal')
    xTrueListMultimodal = torch.load('../data/PowerSpectrum/xTrueList_EM_multimodal')

    xReconstructedListUnimodal = torch.load('../data/PowerSpectrum/xReconstructedList_EM_unimodal')
    xTrueListUnimodal = torch.load('../data/PowerSpectrum/xTrueList_EM_unimodal') 

    xReconstructedSmoothedMultimodal = smooth_data_dimensions(xReconstructedListMultimodal[index_Multimodal])
    xtrueSmoothedMultimodal = smooth_data_dimensions(xTrueListMultimodal[index_Multimodal])

    xAverage = get_average_spectrum(xReconstructedSmoothedMultimodal)
    xTrueAverage = get_average_spectrum(xtrueSmoothedMultimodal)

    xTrueForPlotting = xTrueAverage
    xRecForPlotting = xAverage

    xReconstructedSmoothedUnimodal = smooth_data_dimensions(xReconstructedListUnimodal[index_Unimodal])
    xAverage = get_average_spectrum(xReconstructedSmoothedUnimodal)

    xRecUniForPlotting = xAverage
       
    plot_spectrum_comparison_3TS(xTrueForPlotting[:FREQUENCY_CUTOFF], xRecForPlotting[:FREQUENCY_CUTOFF],
                                 xRecUniForPlotting[:FREQUENCY_CUTOFF])





if __name__ == "__main__":
    main()
