import numpy as np
from distance.distance_functions import euclidean_dist


def euclidean_loss(data, centroids, assignments):
    loss = 0
    for i, entry in enumerate(data):
        loss += euclidean_dist(entry, centroids[assignments[i]])
    return loss

