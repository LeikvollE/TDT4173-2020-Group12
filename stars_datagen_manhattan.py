#!/usr/bin/env python3
from kmeans.basic import kmeans
from kmeans.kmeanspp import kmeanspp
# import mnist
# import weka
from stars import load_stars
from stars import get_accuracy
# from loss.euclidean_loss import euclidean_loss
# import matplotlib.pyplot as plt
import numpy as np
import click
# from pprint import pp
from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans
from distance.distance_functions import manhattan_dist


@click.command()
@click.argument('output', type=str)
def main(output):
    d = load_stars()

    def get_acc(x):
        return get_accuracy(x)

    # basic
    c, a, loss = kmeans(d, 6, 1, d_func=manhattan_dist)
    acc = get_acc(a)
    print('basic: {:f}'.format(acc[1]))
    np.array(acc[0]).tofile('stars_output_manhattan/{}_basic_acc_0'.format(output))
    np.array(c).tofile('stars_output_manhattan/{}_basic_c'.format(output))
    np.array(a).tofile('stars_output_manhattan/{}_basic_a'.format(output))
    np.array(loss).tofile('stars_output_manhattan/{}_basic_loss'.format(output))

    # better
    c, a, loss = kmeans(d, 6, 1, init='better', d_func=manhattan_dist)
    acc = get_acc(a)
    print('better: {:f}'.format(acc[1]))
    np.array(acc[0]).tofile('stars_output_manhattan/{}_better_acc_0'.format(output))
    np.array(c).tofile('stars_output_manhattan/{}_better_c'.format(output))
    np.array(a).tofile('stars_output_manhattan/{}_better_a'.format(output))
    np.array(loss).tofile('stars_output_manhattan/{}_better_loss'.format(output))

    # kmeanspp
    c, a, loss = kmeanspp(d, 6, 1, d_func=manhattan_dist)
    acc = get_acc(a)
    print('kmeanspp: {:f}'.format(acc[1]))
    np.array(acc[0]).tofile('stars_output_manhattan/{}_kmeanspp_acc_0'.format(output))
    np.array(c).tofile('stars_output_manhattan/{}_kmeanspp_c'.format(output))
    np.array(a).tofile('stars_output_manhattan/{}_kmeanspp_a'.format(output))
    np.array(loss).tofile('stars_output_manhattan/{}_kmeanspp_loss'.format(output))

    # PCA
    for co in range(1, 5):
        pca = PCA(n_components=co)
        d = pca.fit_transform(load_stars())

        # basic
        c, a, loss = kmeans(d, 6, 1, d_func=manhattan_dist)
        acc = get_acc(a)
        print('pca={:d} basic: {:f}'.format(co, acc[1]))
        np.array(acc[0]).tofile(
                'stars_output_manhattan/{}_pca_{:d}_basic_acc_0'.format(output, co))
        np.array(c).tofile(
                'stars_output_manhattan/{}_pca_{:d}_basic_c'.format(output, co))
        np.array(a).tofile(
                'stars_output_manhattan/{}_pca_{:d}_basic_a'.format(output, co))
        np.array(loss).tofile(
                'stars_output_manhattan/{}_pca_{:d}_basic_loss'.format(output, co))

        # better
        c, a, loss = kmeans(d, 6, 1, init='better', d_func=manhattan_dist)
        acc = get_acc(a)
        print('pca={:d} better: {:f}'.format(co, acc[1]))
        np.array(acc[0]).tofile(
                'stars_output_manhattan/{}_pca_{:d}_better_acc_0'.format(output, co))
        np.array(c).tofile(
                'stars_output_manhattan/{}_pca_{:d}_better_c'.format(output, co))
        np.array(a).tofile(
                'stars_output_manhattan/{}_pca_{:d}_better_a'.format(output, co))
        np.array(loss).tofile(
                'stars_output_manhattan/{}_pca_{:d}_better_loss'.format(output, co))

        # kmeanspp
        c, a, loss = kmeanspp(d, 6, 1, d_func=manhattan_dist)
        acc = get_acc(a)
        print('pca={:d} kmeanspp: {:f}'.format(co, acc[1]))
        np.array(acc[0]).tofile(
                'stars_output_manhattan/{}_pca_{:d}_kmeanspp_acc_0'.format(output, co))
        np.array(c).tofile(
                'stars_output_manhattan/{}_pca_{:d}_kmeanspp_c'.format(output, co))
        np.array(a).tofile(
                'stars_output_manhattan/{}_pca_{:d}_kmeanspp_a'.format(output, co))
        np.array(loss).tofile(
                'stars_output_manhattan/{}_pca_{:d}_kmeanspp_loss'.format(output, co))


if __name__ == '__main__':
    main()

