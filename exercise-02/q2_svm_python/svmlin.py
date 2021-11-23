import numpy as np
# might need to add path to mingw-w64/bin for cvxopt to work
# import os
# os.environ["PATH"] += os.pathsep + ...
import cvxopt

def svmlin(X, t, C):
    # Linear SVM Classifier
    #
    # INPUT:
    # X        : the dataset                  (num_samples x dim)
    # t        : labeling                     (num_samples x 1)
    # C        : penalty factor for slack variables (scalar)
    #
    # OUTPUT:
    # alpha    : output of quadprog function  (num_samples x 1)
    # sv       : support vectors (boolean)    (1 x num_samples)
    # w        : parameters of the classifier (1 x dim)
    # b        : bias of the classifier       (scalar)
    # result   : result of classification     (1 x num_samples)
    # slack    : points inside the margin (boolean)   (1 x num_samples)

    N = len(t)
    # print("X", X)
    # print("t", t)
    # print("C", C)
    # print("N", N)

    vec_C = np.array([C for _ in range(N)])
    vec_1 = np.ones(N)
    vec_0 = np.zeros(N)
    H = np.zeros(shape=(N, N))

    for n in range(N):
        for m in range(N):
            H[n][m] = t[n] * t[m] * np.inner(X[n], X[m])

    print(H)

    n = H.shape[1]
    LB = vec_0
    UB = vec_C

    q = cvxopt.matrix((-1) * np.ones(N))
    G = cvxopt.matrix(np.vstack([-np.eye(n), np.eye(n)]))
    A = cvxopt.matrix(t)
    b = cvxopt.matrix(0)
    P = cvxopt.matrix(X)
    h = cvxopt.matrix(np.hstack([-LB, UB]))

    a = cvxopt.solvers.qp(H, q, G, h, A, b)



    pass
    # return alpha, sv, w, b, result, slack
