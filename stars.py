import numpy as np
from kmeans.basic import kmeans
from loss.euclidean_loss import euclidean_loss

def load_stars():
    stars = np.genfromtxt('data/stars.csv', delimiter=',', skip_header=1)
    stars = stars[:,:-3]
    return (stars - np.min(stars, axis=0)) / (np.max(stars) - np.min(stars))


stars = load_stars()

centroids, assignments = kmeans(stars, 6)
print(assignments)
print(euclidean_loss(stars, centroids, assignments))
