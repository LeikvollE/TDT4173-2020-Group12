import numpy as np
import random
from distance.distance_functions import euclidean_dist


def kmeans(data, k):
    #centroids = np.random.normal(0.0, 1.0, [k, len(data[0])])
    centroids = data[random.sample(range(len(data)), k)].copy()
    assignments = -np.ones(len(data))
    changed = True
    counter = 0
    while changed:
        counter+=1
        print(counter)
        changed = False
        new_centroids = np.zeros_like(centroids)
        centroid_divisor = np.zeros((len(centroids),1))
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
            # TODO: div by zero error (tom cluster)
            centroid_divisor[int(assignments[i])] += 1
            new_centroids[int(assignments[i])] += entry
        if not changed:
            return centroids, assignments
        else:
            for k, div in enumerate(centroid_divisor):
                if div == 0:
                    centroid_divisor[k] = 1
                    new_centroids[k] = centroids[k]
            centroids = np.divide(new_centroids, centroid_divisor)
