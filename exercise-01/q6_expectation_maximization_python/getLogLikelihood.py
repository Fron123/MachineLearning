import numpy as np


def getLogLikelihood(means, weights, covariances, X):
    # Log Likelihood estimation
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each gaussian DxDxK
    # X              : Input data NxD
    # where N is number of data points
    # D is the dimension of the data points
    # K is number of gaussians
    #
    # OUTPUT:
    # logLikelihood  : log-likelihood

    # logLikelihood = 0
    # D = 2
    N = len(X)
    K = len(weights)

    outer_sum = 0
    for i in range(N):
        inner_sum = 0
        for k in range(K):
            inner_sum += weights[k] * normal(X[i], means[k], covariances[:, :, k])
        outer_sum += np.log(inner_sum)

    return outer_sum


def normal(x_n, means, covariances):
    sigma_x = np.sqrt(covariances[0, 0])
    sigma_y = np.sqrt(covariances[1, 1])
    cor = covariances[1, 0] / (sigma_x * sigma_y)
    x = x_n[0]
    y = x_n[1]
    mu_x = means[0]
    mu_y = means[1]

    prefactor = 1 / (2 * np.pi * sigma_y * sigma_x * np.sqrt(1 - (cor ** 2)))
    z = ((((x - mu_x) / sigma_x) ** 2) -
         (2 * cor * ((x - mu_x) / sigma_x) * ((y - mu_y) / sigma_y)) +
         (((y - mu_y) / sigma_y) ** 2))
    rest = np.exp(-(z / (2 * (1 - (cor ** 2)))))
    return prefactor * rest
