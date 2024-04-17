import numpy as np

def RBF_kernel(X, l):
    X = np.array(X)
    Xsq = np.sum(np.square(X), 1)
    R2 = -2. * np.dot(X, X.T) + (Xsq[:, None] + Xsq[None, :])
    R2 = np.clip(R2, 1e-12, np.inf)
    return np.exp(-R2 / (2 * l ** 2))