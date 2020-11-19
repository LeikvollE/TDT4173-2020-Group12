import numpy as np
from kmeans.basic import kmeans
from loss.euclidean_loss import euclidean_loss
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale


def load_stars():
    stars = np.genfromtxt('data/stars.csv', delimiter=',', skip_header=1)
    stars = stars[:, :-3]
    return minmax_scale(stars)


if __name__ == '__main__':
    stars = load_stars()

    centroids, assignments, loss = kmeans(stars, 6, 1)
    plt.plot(loss)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()
    print(loss)
    print(euclidean_loss(stars, centroids, assignments))
