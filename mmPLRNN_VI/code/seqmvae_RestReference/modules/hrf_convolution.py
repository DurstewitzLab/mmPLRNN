"""Haemodynamic response function
FORMAT [hrf,p] = spm_hrf(RT,p,T)
RT   - scan repeat time
p    - parameters of the response function (two Gamma functions)
                                                           defaults
                                                          {seconds}
        p(1) - delay of response (relative to onset)          6
        p(2) - delay of undershoot (relative to onset)       16
        p(3) - dispersion of response                         1
        p(4) - dispersion of undershoot                       1
        p(5) - ratio of response to undershoot                6
        p(6) - onset {seconds}                                0
        p(7) - length of kernel {seconds}                    32

T    - microtime resolution [Default: 16]

hrf  - haemodynamic response function
p    - parameters of the response function
__________________________________________________________________________
Copyright (C) 1996-2015 Wellcome Trust Centre for Neuroimaging

Karl Friston
$Id: spm_hrf.m 6594 2015-11-06 18:47:05Z guillaume $
$edited Georgia Koppe for only stuff that I need


-Parameters of the response function
--------------------------------------------------------------------------
p = [6 16 1 1 6 0 32];
-Microtime resolution
--------------------------------------------------------------------------
fMRI_T=16;

-Modelled haemodynamic response function - {mixture of Gammas}
--------------------------------------------------------------------------"""

import torch
import math
from matplotlib import pyplot as plt

repetitionTime = 0.5
parameters = torch.tensor([6, 16, 1, 1, 6, 0, 32.]).type(torch.DoubleTensor)
fMRI_microtimeResolution = 16



def main():
    x = torch.linspace(0, 20, 500).type(torch.DoubleTensor)
    x1 = 2 * torch.sin(14 * x)
    x2 = 2 * torch.cos(14 * x)
    x1 = torch.zeros(500)
    x2 = torch.zeros(500)
    x1[4:75], x1[90:120], x1[190:270], x1[290:350], x1[385:425], x1[450:480] = 1, 1, 1, 1, 1, 1
    x2[50:85], x2[110:145], x2[200:225], x2[240:275], x2[315:355], x2[395:430] = 1, 1, 1, 1, 1, 1
    data = torch.stack((x1, x2))
    hrf = haemodynamicResponseFunction()
    print(hrf)
    print(len(hrf))

    plt.plot(torch.linspace(0, parameters[-1], len(hrf)), hrf)
    plt.savefig('hrf_convolution')

    convolvedData = torch.zeros(len(data), len(data[0]) + len(hrf)).type(torch.DoubleTensor)

    print(data.shape)

    for ind in range(0, len(data)):
        convolvedData[ind] = convolve_data(data[ind], hrf)

    xValues = torch.linspace(0, parameters[-1] * 3, len(data[0]))

    plt.plot(xValues, data[0], color='blue')
    plt.plot(xValues, convolvedData[0][:500], color='cyan')
    plt.show()


def convolve_data(data, convolveFunction):
    lengthData = len(data)
    lengthConvolveFunction = len(convolveFunction)

    ind = torch.tensor([0])
    convolvedData = torch.tensor([]).type(torch.DoubleTensor)

    for idx in range(0, lengthData + lengthConvolveFunction):
        temp = 0
        for j in ind:
            temp += convolveFunction[j] * data[idx - j]
        if idx < lengthConvolveFunction - 1:
            ind = torch.cat((ind, torch.tensor([idx + 1])))
        if idx >= lengthData - 1:
            ind = ind[1:]

        convolvedData = torch.cat((convolvedData, torch.tensor([temp]).type(torch.DoubleTensor)))
    return convolvedData


def gammaDistLikeGeorgia(x, h, l):
    dist = l.pow(h) * x ** (h - 1) * torch.exp(-l * x) / math.factorial(h - 1)
    return dist


def haemodynamicResponseFunction(repTime=repetitionTime):

    dt = repTime / fMRI_microtimeResolution

    x = (torch.arange(0, math.ceil(parameters[6] / dt) - 1) - parameters[5] / dt).type(torch.DoubleTensor)

    h1 = parameters[0] / parameters[2]
    l1 = dt / parameters[2]

    h2 = parameters[1] / parameters[3]
    l2 = dt / parameters[3]

    hrf = gammaDistLikeGeorgia(x, h1, l1) - gammaDistLikeGeorgia(x, h2, l2) / parameters[4]
    hrf = hrf[torch.arange(0, math.floor(parameters[6] / repTime)) * fMRI_microtimeResolution + 1]
    hrf = hrf / torch.sum(hrf)
    return hrf


if __name__ == "__main__":
    main()
