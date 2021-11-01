import numpy as np


def knn(samples, k):
    # compute density estimation from samples with KNN
    # Input
    #  samples    : DxN matrix of data points
    #  k          : number of neighbors
    # Output
    #  estDensity : estimated density in the range of [-5, 5]

    ##### Insert your code here for subtask 5b ####

    pos = np.arange(-5, 5.0, 0.1)  # Returns a 100 dimensional vector
    estDensity = np.zeros((100, 2))
    estDensity[:100, 0] = pos

    # p(x) ~ K/(NV) -> K fixed at 30, N fixed at 100, estimate V
    # To estimate V:
    # Contruct a "bubble" around each point in pos, and increase the length, until 30 items are collected

    

    # Compute the number of the samples created
    return estDensity


def dist(a, b):
    return np.abs(a - b)
