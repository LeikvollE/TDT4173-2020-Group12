import numpy as np
from kmeans.basic import kmeans
from loss.euclidean_loss import euclidean_loss
import matplotlib.pyplot as plt


def load_stars():
    stars = np.genfromtxt('data/stars.csv', delimiter=',', skip_header=1)
    stars = stars[:, :-3]
    return (stars - np.min(stars, axis=0)) / (np.max(stars, axis=0) - np.min(stars, axis=0))

def get_accuracy(assignments):
    labels = pandas.read_csv('data/stars.csv')[['Star type']].to_numpy().flatten()
    return accuracy(labels, assignments)


if __name__ == '__main__':
    stars = load_stars()
    labels = pandas.read_csv('data/stars.csv')[['Star type']].to_numpy().flatten()
    sil = metrics.silhouette_score(stars, labels, metric='euclidean')
    cal = metrics.calinski_harabasz_score(stars, labels)
    dav = metrics.davies_bouldin_score(stars, labels)

    print(sil)
    print(cal)
    print(dav)
    sil = 0
    cal = 0
    dav = 0
    for i in range(1000):
        centroids, labels, loss = kmeans(stars, 6, 1)
        sil += metrics.silhouette_score(stars, labels, metric='euclidean')
        cal += metrics.calinski_harabasz_score(stars, labels)
        dav += metrics.davies_bouldin_score(stars, labels)
    print(sil/1000)
    print(cal/1000)
    print(dav/1000)
