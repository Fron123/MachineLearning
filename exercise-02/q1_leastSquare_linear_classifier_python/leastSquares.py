import numpy as np


def leastSquares(data, label):
    # Sum of squared error shoud be minimized
    #
    # INPUT:
    # data        : Training inputs  (num_samples x dim)
    # label       : Training targets (num_samples x 1)
    #
    # OUTPUT:
    # weights     : weights   (dim x 1)
    # bias        : bias term (scalar)

    # Extend each datapoint x as [1, x]
    # (Trick to avoid modeling the bias term explicitly)

    D = data.shape[1]  # + 1 for bias part
    K = 1

    sum1 = np.zeros((D, D))
    sum2 = np.zeros((D, K))

    for i in range(len(data)):
        x_i = data[i]
        y_i = label[i]
        sum1 += np.outer(x_i, x_i)  # x_i*x_i
        sum2 += np.outer(x_i, y_i)  # x_i*y_i

    weight = np.dot(np.linalg.inv(sum1), sum2)
    bias = (1/len(data)) * np.sum(np.dot(data, weight))

    return weight, bias
