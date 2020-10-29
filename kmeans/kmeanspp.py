import numpy as np
import random
from distance.distance_functions import euclidean_dist
from loss.euclidean_loss import euclidean_loss


def kmeanspp(data, k):
    #centroids = np.random.normal(0.0, 1.0, [k, len(data[0])])
    #centroids = data[random.sample(range(len(data)), k)].copy()
    centroids = []
    indeces = range(len(data))
    centroids.append(data[np.random.choice(indeces, 1)[0]].copy())
    while len(centroids) < k:
        distances = np.full(len(data), np.inf)
        for i, point in enumerate(data):
            for j, centroid in enumerate(centroids):
                distances[i] = min(distances[i], euclidean_dist(point, centroid))
        dss = sum(np.power(distances, 2))
        p = np.divide(np.power(distances, 2), dss)
        centroids.append(data[np.random.choice(indeces, 1, p=p)[0]].copy())
    assignments = -np.ones(len(data))
    loss = []
    changed = True
    while changed:
        changed = False
        new_centroids = np.zeros_like(centroids)
        centroid_divisor = np.zeros((len(centroids), 1))
        for i, entry in enumerate(data):
            dist = float("inf")
            dist_j = -1
            for j, centroid in enumerate(centroids):
                new_dist = euclidean_dist(entry, centroid)
                if new_dist < dist:
                    dist = new_dist
                    dist_j = j
            if dist_j != assignments[i]:
                changed = True
                assignments[i] = dist_j
            centroid_divisor[int(assignments[i])] += 1
            new_centroids[int(assignments[i])] += entry
        if not changed:
            return centroids, assignments, loss
        else:
            for k, div in enumerate(centroid_divisor):
                if div == 0:
                    centroid_divisor[k] = 1
                    new_centroids[k] = centroids[k]
            centroids = np.divide(new_centroids, centroid_divisor)
            loss.append(euclidean_loss(data, centroids, assignments))
