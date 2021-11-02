from math import pi, exp, sqrt

import numpy as np


def kde(samples, h):
    # compute density estimation from samples with KDE
    # Input
    #  samples    : DxN matrix of data points
    #  h          : (half) window size/radius of kernel
    # Output
    #  estDensity : estimated density in the range of [-5,5]

    pos = np.arange(-5, 5.0, 0.1)  # Returns a 100 dimensional vector
    estDensity = np.zeros((100, 2))
    estDensity[:100, 0] = pos

    # h=0.5 This leads to better results in our opinion

    for i in range(len(pos)):
        for j in samples:
            estDensity[i, 1] += kernel(pos[i] - j, h) / len(pos)

    # Compute the number of samples created
    return estDensity


def kernel(u, h):
    return (1 / sqrt(2 * pi * (h ** 2))) * exp(-((u ** 2) / (2 * (h ** 2))))
