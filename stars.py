import numpy as np
from kmeans.basic import kmeans
from loss.euclidean_loss import euclidean_loss
import matplotlib.pyplot as plt
import pandas
from accuracy import accuracy
from sklearn.preprocessing import minmax_scale


def load_stars():
    stars = np.genfromtxt('data/stars.csv', delimiter=',', skip_header=1)
    stars = stars[:, :-3]
    return minmax_scale(stars)


def get_accuracy(assignments):
    labels = pandas.read_csv('data/stars.csv')[['Star type']].to_numpy().flatten()
    return accuracy(labels, assignments, allow_miss=True)


if __name__ == '__main__':
    stars = load_stars()

    centroids, assignments, loss = kmeans(stars, 6, 1)
    plt.plot(loss)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()
    print(loss)
    print(euclidean_loss(stars, centroids, assignments))
