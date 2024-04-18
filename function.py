import numpy as np

def RBF_kernel(X, l):
    X = np.array(X)
    Xsq = np.sum(np.square(X), 1)
    R2 = -2. * np.dot(X, X.T) + (Xsq[:, None] + Xsq[None, :])
    return np.exp(-R2 / (2 * l ** 2))

def linear_kernel(X):
    K = np.dot(X, X.T)
    return K / K.max()