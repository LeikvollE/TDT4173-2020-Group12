import numpy as np


def euclidean_dist(v1, v2):
    return np.linalg.norm(v1 - v2)


def manhattan_dist(v1, v2):
    return np.sum(np.abs(v1 - v2))
