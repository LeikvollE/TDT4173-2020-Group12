import numpy as np
from distance.distance_functions import euclidean_dist


def kmeans(data, k):
    centroids = np.random.normal(0.0, 1.0, [k, len(data[0])])
    assignments = -np.ones(len(data))
    changed = True
    while changed:
        changed = False
        new_centroids = np.zeros_like(centroids)
        centroid_divisor = np.zeros_like(centroids)
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
            centroid_divisor[int(assignments[i])] += 1 # TODO: div by zero error (tom cluster)
            new_centroids[int(assignments[i])] += entry
        if not changed:
            return centroids, assignments
        else:
            centroids = np.divide(new_centroids,
                                  centroid_divisor,
                                  out=np.zeros_like(new_centroids),
                                  where=centroid_divisor!=0)
