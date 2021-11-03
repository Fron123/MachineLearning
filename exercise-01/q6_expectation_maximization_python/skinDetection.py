import numpy as np
from estGaussMixEM import estGaussMixEM
from getLogLikelihood import getLogLikelihood, normal


def skinDetection(ndata, sdata, K, n_iter, epsilon, theta, img):
    # Skin Color detector
    #
    # INPUT:
    # ndata         : data for non-skin color
    # sdata         : data for skin-color
    # K             : number of modes
    # n_iter        : number of iterations
    # epsilon       : regularization parameter
    # theta         : threshold
    # img           : input image
    #
    # OUTPUT:
    # result        : Result of the detector for every image pixel

    # sdata
    s_weights, s_means, s_covariances = estGaussMixEM(sdata, K, n_iter, epsilon)

    # ndata
    n_weights, n_means, n_covariances = estGaussMixEM(ndata, K, n_iter, epsilon)

    # s_ll = getLogLikelihood(s_means, s_weights, s_covariances, sdata)
    # n_ll = getLogLikelihood(n_means, n_weights, n_covariances, ndata)

    # iterate over each pixel in img -> classify to n_skin oder skin
    width = img.shape[1]
    height = img.shape[0]
    result = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            s_ll = getLogLikelihood(s_means, s_weights, s_covariances, [img[y, x, :]])
            n_ll = getLogLikelihood(n_means, n_weights, n_covariances, [img[y, x, :]])
            result[y, x] = 255 if ((s_ll / n_ll) > theta) else 0

    # s_ll = getLogLikelihood(s_means, s_weights, s_covariances, [img[0, 0, :]])
    # n_ll = getLogLikelihood(n_means, n_weights, n_covariances, [img[0, 0, :]])
    # print((s_ll/n_ll) > theta)

    return result
