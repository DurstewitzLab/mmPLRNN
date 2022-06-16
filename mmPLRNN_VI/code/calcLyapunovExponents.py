import torch
import numpy as np
import sys
import nolds
import pickle

sys.path.insert(0, "seqmvae/modules")

import datagenerator

def calc_lyapunovExponent(z, A, W, T):

    dimension = 20
    z = z[200:]
    J_t_product = torch.ones(dimension, dimension).type(torch.DoubleTensor)

    for t in range(T, 1, -1):
        D_t1 = np.maximum(0, z[t-1])
        D_t1[D_t1 > 0] = 1
        J_t = (torch.diag(A) + W @ torch.diag(D_t1).type(torch.DoubleTensor))
        J_t_product = J_t @ J_t_product

    lyapunov_exponent = np.log(np.linalg.norm(J_t_product, ord=2))/(T+1)
    print(lyapunov_exponent)


def getDataAndScores():

    with open('../data/chaoticDict', 'rb') as f:
        loaded_dict = pickle.load(f)

    data = loaded_dict

    cTrue = data['C_']
    A = data['A']
    B = data['B']
    C = data['C']
    W = data['W']
    h = data['h']
    mu0 = data['mu0']
    beta = data['Beta']

    dim_x = 20
    dim_z = 20
    dim_c = 5

    args_dict = {'dim_ex': 5, 'dim_reg': 5, 'tau': 1, 'use_hrf': False, 'useExplicitHrf': False,
                 'repetitionTime': 6,
                 'useBaseExpansion': False, 'A': A, 'B': B, 'W': W, 'C': C, 'h': h, 'mu0': mu0, 'beta': beta}

    trained_mdl = datagenerator.DataGenerator(dim_x, dim_z, args_dict, dim_c,
                                              args_dict, 'uniform',
                                              False,
                                              nonlinearity=torch.nn.functional.relu)
    cIndices = torch.zeros((360))
    for ind in range(0, len(cTrue)):
        cIndices[ind] = torch.argmax(cTrue[ind])

    T = 150000
    xReconstructed, zReconstructed = trained_mdl.generate_timeseries(T, noise=False)

    print("Lyapunov Exponents:")

    calc_lyapunovExponent(zReconstructed, A, W, 1000)
    calc_lyapunovExponent(zReconstructed, A, W, 1500)
    calc_lyapunovExponent(zReconstructed, A, W, 10000)
    calc_lyapunovExponent(zReconstructed, A, W, 50000)
    calc_lyapunovExponent(zReconstructed, A, W, 60000)
    calc_lyapunovExponent(zReconstructed, A, W, 70000)
    calc_lyapunovExponent(zReconstructed, A, W, 80000)

    print("-------------------------------------------------------")

    z0 = zReconstructed.t()[0].t()
    z9 = zReconstructed.t()[9].t()
    z11 = zReconstructed.t()[11].t()

    print("Nolds Values: ")

    lyap_e = nolds.lyap_e(z0)
    lyap_r = nolds.lyap_r(z0)
    corr_dim = nolds.corr_dim(z0, 3)

    print(lyap_e, lyap_r, corr_dim)

    lyap_e = nolds.lyap_e(z9)
    lyap_r = nolds.lyap_r(z9)
    corr_dim = nolds.corr_dim(z9, 3)

    print(lyap_e, lyap_r, corr_dim)

    lyap_e = nolds.lyap_e(z11)
    lyap_r = nolds.lyap_r(z11)
    corr_dim = nolds.corr_dim(z11, 3)

    print(lyap_e, lyap_r, corr_dim)

    print("-------------------------------------------------------")

scoreListMultimodal = getDataAndScores()


