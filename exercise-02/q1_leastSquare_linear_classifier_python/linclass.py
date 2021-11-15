import numpy as np
from random import randint


def linclass(weight, bias, data):
    # Linear Classifier
    #
    # INPUT:
    # weight      : weights                (dim x 1)
    # bias        : bias term              (scalar)
    # data        : Input to be classified (num_samples x dim)
    #
    # OUTPUT:
    # class_pred       : Predicted class (+-1) values  (num_samples x 1)

    # Perform linear classification i.e. class prediction

    class_pred = np.array([-1 for _ in range(data.shape[0])])
    x = data

    values = list(np.dot(x, weight))
    winners = [i for i, x in enumerate(values) if x == max(values)]
    winner = winners[randint(0, len(winners) - 1)]

    class_pred[winner] = 1

    return class_pred
