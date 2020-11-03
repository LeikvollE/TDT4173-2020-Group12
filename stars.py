import numpy as np
from kmeans.basic import kmeans
from loss.euclidean_loss import euclidean_loss
import matplotlib.pyplot as plt


def load_stars():
    stars = np.genfromtxt('data/stars.csv', delimiter=',', skip_header=1)
    stars = stars[:, :-3]
    return (stars - np.min(stars, axis=0)) / (np.max(stars) - np.min(stars))


if __name__ == '__main__':
    stars = load_stars()

    centroids, assignments, loss = kmeans(stars, 6, 1)
    plt.plot(loss)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()
    print(loss)
    print(euclidean_loss(stars, centroids, assignments))
