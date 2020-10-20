import numpy as np


def euclidean_dist(v1, v2):
    diff = np.subtract(v1, v2)
    return diff.dot(diff)
