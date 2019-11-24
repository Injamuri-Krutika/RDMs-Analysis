import scipy
import numpy as np


def epanechnikov_similarity(matrix, sigma=1):
    n_samples = matrix.shape[0]
    mat = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            feature_i, feature_j = matrix[i], matrix[j]
            feature_i, feature_j = feature_i / \
                np.linalg.norm(feature_i), feature_j/np.linalg.norm(feature_j)
            dist = np.linalg.norm(feature_i-feature_j)
            mat[i, j] = 1 - dist**2/sigma**2
    return mat
