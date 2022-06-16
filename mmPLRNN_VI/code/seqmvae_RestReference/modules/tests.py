import torch
from sklearn import datasets
import math
import numpy as np

global BATCH_SIZE, DIM_LATENT, DIM_CATEGORICAL
BATCH_SIZE = 100
DIM_LATENT = 10
DIM_CATEGORICAL = 8


def klDivergence(mu_one, cov_one, mu_two, cov_two):
    """Calculate D_KL(q_one||q_two) in the case of non-diagonal batched covariance matrices"""
    difference_term = 0
    trace_term = 0
    for ind in range(0, len(mu_two)):
        difference_term += (mu_two[ind] - mu_one[ind]) @ torch.inverse(cov_two[ind]) @ (
                mu_two[ind] - mu_one[ind]).t()
        trace_term += torch.trace(torch.inverse(cov_two[ind]) @ cov_one[ind])

    klDiv = 0.5 * (trace_term + difference_term
                   - DIM_LATENT * BATCH_SIZE + torch.sum(torch.log(torch.det(cov_two)) - torch.log(torch.det(cov_one))))

    if (klDiv < 0):
        print("Calculated negative klDivergence, this should never happen.")

    test = torch.sum(torch.log(torch.det(cov_two)) - torch.log(torch.det(cov_one)))

    klDiv /= BATCH_SIZE * (3 + 8)

    return klDiv


def test_kl_divergence():
    def isPSD(A, tol=1e-8):
        E = np.linalg.eigvalsh(A)
        return np.all(E > -tol)

    tensor_list_one = []
    tensor_list_two = []

    for ind in range(0, BATCH_SIZE):
        temp1 = datasets.make_spd_matrix(DIM_LATENT)
        temp2 = datasets.make_spd_matrix(DIM_LATENT)
        tensor_list_one.append(torch.tensor(temp1).type(torch.FloatTensor))
        tensor_list_two.append(torch.tensor(temp2).type(torch.FloatTensor))

    print(isPSD(temp1))

    cov_one = torch.stack(tensor_list_one)
    cov_two = torch.stack(tensor_list_two)

    mu = torch.randn(BATCH_SIZE, DIM_LATENT)
    mu_two = torch.randn(BATCH_SIZE, DIM_LATENT)

    klDiv = klDivergence(mu, cov_one, mu, cov_one)
    print("expected result vs. real result")
    print(0, klDiv.data)
    assert (klDiv == 0)

    if (klDiv == 0):
        print("Kl divergence of two identical distributions is zero.")
    else:
        print("Kl divergence test failed")
    print("--------------------------------")


def categorical_log_likelihood(categorical_data, latent_data, weights):
    # TODO: make sure this calculation is correct via test & data!!
    result = 0

    for ind in range(0, BATCH_SIZE):
        normalizationTerm = 0
        for idx in range(0, DIM_CATEGORICAL):
            normalizationTerm += torch.exp(weights[idx] @ latent_data[ind])
        for idx in range(0, DIM_CATEGORICAL):
            if categorical_data[ind][idx] == 1:
                result += weights[idx] @ latent_data[ind] - torch.log(normalizationTerm)
    return result


def test_categorical_likelihood():
    """If categories are zero, categorical likelihood is expected to take value 0"""
    expected_result = 0

    categorical_input_batch = torch.zeros(BATCH_SIZE, DIM_CATEGORICAL)
    weights = torch.rand(DIM_CATEGORICAL, DIM_LATENT)
    z_data = torch.rand(BATCH_SIZE, DIM_LATENT)
    categorical_likelihood = categorical_log_likelihood(categorical_input_batch, z_data, weights)
    print("expected result vs. real result")
    print(expected_result, categorical_likelihood)
    print("--------------------------------")
    assert (categorical_likelihood == expected_result)

    idx = 0
    for ind in range(0, BATCH_SIZE):
        if (idx > DIM_CATEGORICAL - 1):
            idx = 0
        categorical_input_batch[ind][idx] = 1
        idx += 1

    """If latent_data or weights are zero, categorical_log_likelihood is expected to take value 
    BATCH_SIZE * (-log(DIM_CATEGORICAL)) """
    expected_result = torch.tensor(BATCH_SIZE * (-math.log(DIM_CATEGORICAL)))

    weights = torch.rand(DIM_CATEGORICAL, DIM_LATENT)
    z_data = torch.zeros(BATCH_SIZE, DIM_LATENT)
    categorical_likelihood = categorical_log_likelihood(categorical_input_batch, z_data, weights)
    print("expected result vs. real result")
    print(expected_result, categorical_likelihood)
    print("--------------------------------")
    assert (torch.abs(categorical_likelihood - expected_result) < 0.1)

    weights = torch.zeros(DIM_CATEGORICAL, DIM_LATENT)
    z_data = torch.rand(BATCH_SIZE, DIM_LATENT)
    categorical_likelihood = categorical_log_likelihood(categorical_input_batch, z_data, weights)
    print("expected result vs. real result")
    print(expected_result, categorical_likelihood)
    print("--------------------------------")
    assert (torch.abs(categorical_likelihood - expected_result) < 0.1)

    """If given all values we expect a result < 0"""

    weights = torch.rand(DIM_CATEGORICAL, DIM_LATENT)
    z_data = torch.rand(BATCH_SIZE, DIM_LATENT)
    categorical_likelihood = categorical_log_likelihood(categorical_input_batch, z_data, weights)
    print("expected result vs. real result")
    print(str("<= 0 ") + str(categorical_likelihood.data))
    print("--------------------------------")
    assert (categorical_likelihood <= 0)


test_kl_divergence()
test_categorical_likelihood()
