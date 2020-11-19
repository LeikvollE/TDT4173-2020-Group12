from distance.distance_functions import euclidean_dist
from distance.distance_functions import manhattan_dist
import numpy as np


def __loss(data, centroids, assignments, d_func):
    loss = 0
    for i, entry in enumerate(data):
        loss += np.power(d_func(entry, centroids[int(assignments[i])]), 2)
    return loss


def manhattan_loss(data, centroids, assignments):
    return __loss(data, centroids, assignments, manhattan_dist)


def euclidean_loss(data, centroids, assignments):
    return __loss(data, centroids, assignments, euclidean_dist)
