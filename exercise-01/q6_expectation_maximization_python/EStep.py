import numpy as np
from getLogLikelihood import getLogLikelihood, normal


def EStep(means, covariances, weights, X):
    # Expectation step of the EM Algorithm
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each Gaussian DxDxK
    # X              : Input data NxD
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.

    logLikelihood = 0
    N = len(X)
    K = len(weights)
    gamma = np.zeros([N, K])

    for n in range(N):
        for j in range(K):
            temp_sum = 0
            for k in range(K):
                temp_sum += weights[k] * normal(X[n], means[k], covariances[:, :, k])
            gamma[n, j] = (weights[j] * normal(X[n], means[j], covariances[:, :, j])) / temp_sum

    return [logLikelihood, gamma]
