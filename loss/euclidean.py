import numpy as np


def euclidean(data, centroids, assignments):
    loss = 0
    for i, entry in enumerate(data):
        diff = np.subtract(entry, centroids[assignments[i]])
        loss += diff.dot(diff)
    return loss

