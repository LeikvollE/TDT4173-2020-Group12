from distance.distance_functions import euclidean_dist


def euclidean_loss(data, centroids, assignments):
    loss = 0
    for i, entry in enumerate(data):
        loss += np.power(euclidean_dist(entry, centroids[int(assignments[i])]), 2)
    return loss
