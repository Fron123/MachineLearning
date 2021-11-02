import numpy as np


def knn(samples, k):
    # compute density estimation from samples with KNN
    # Input
    #  samples    : DxN matrix of data points
    #  k          : number of neighbors
    # Output
    #  estDensity : estimated density in the range of [-5, 5]

    upper = 5.0
    lower = -5.0

    pos = np.arange(lower, upper, 0.1)  # Returns a 100 dimensional vector
    estDensity = np.zeros((100, 2))
    estDensity[:100, 0] = pos

    # p(x) ~ K/(NV) -> K fixed at 30, N fixed at 100, estimate V
    # To estimate V:
    # Contruct a "bubble" around each point in pos, and increase the length, until 30 items are collected

    for i in range(len(pos)):
        estDensity[i, 1] = k/(len(samples)*volume(samples, pos[i], k, upper, lower))

    # volume(samples, 0, k, upper, lower)

    # Compute the number of the samples created
    return estDensity


def volume(samples, pos: int, k, upper, lower):
    # center = pos
    # needed: [pos-x, pos+x], where x is chosen so that the interval results in length k

    dist_arr = [dist(pos, x) for x in samples]
    dist_arr = np.sort(dist_arr)

    print(len(dist_arr), dist_arr)
    print(dist_arr[0:30])

    return np.max(dist_arr[0:30]) * 2


def dist(a, b):
    return np.abs(a - b)
