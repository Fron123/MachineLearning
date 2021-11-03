import numpy as np
from getLogLikelihood import getLogLikelihood


def MStep(gamma, X):
    # Maximization step of the EM Algorithm
    #
    # INPUT:
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.
    # X              : Input data (NxD matrix for N datapoints of dimension D).
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # means          : Mean for each gaussian (KxD).
    # weights        : Vector of weights of each gaussian (1xK).
    # covariances    : Covariance matrices for each component(DxDxK).

    N = len(X)
    K = gamma.shape[1]
    D = X.shape[1]

    N_dach = np.array([np.sum(gamma[:, j]) for j in range(K)])
    weights = np.array([N_dach[j] / N for j in range(K)])
    means = np.array([(1 / N_dach[j]) * np.sum(gamma[n, j] * X[n] for n in range(N)) for j in range(K)])

    covariances = np.zeros((D, D, K))
    for j in range(K):
        temp_mat = np.zeros((D, D))
        for n in range(N):
            temp = X[n] - means[j]
            temp_mat += gamma[n, j] * (temp.reshape(-1, 1) @ temp.reshape(1, -1))
        covariances[:, :, j] = temp_mat / N_dach[j]

    logLikelihood = getLogLikelihood(means, weights, covariances, X)

    return weights, means, covariances, logLikelihood
