#!/usr/bin/env python3
from kmeans.basic import kmeans
from kmeans.kmeanspp import kmeanspp
import mnist
# import weka
# from stars import load_stars
# from loss.euclidean_loss import euclidean_loss
# import matplotlib.pyplot as plt
import numpy as np
import click
# from pprint import pp
from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans


@click.command()
@click.argument('output', type=str)
def main(output):
    m = mnist.mnist()
    d = m.get_test_set()

    def get_acc(x):
        return m.get_accuracy(x, 'test')

    # basic
    c, a, loss = kmeans(d, 10, 1)
    acc = get_acc(a)
    print('basic: {:f}'.format(acc[1]))
    np.array(acc[0]).tofile('output/{}_basic_acc_0'.format(output))
    np.array(c).tofile('output/{}_basic_c'.format(output))
    np.array(a).tofile('output/{}_basic_a'.format(output))
    np.array(loss).tofile('output/{}_basic_loss'.format(output))

    # better
    c, a, loss = kmeans(d, 10, 1, init='better')
    acc = get_acc(a)
    print('better: {:f}'.format(acc[1]))
    np.array(acc[0]).tofile('output/{}_better_acc_0'.format(output))
    np.array(c).tofile('output/{}_better_c'.format(output))
    np.array(a).tofile('output/{}_better_a'.format(output))
    np.array(loss).tofile('output/{}_better_loss'.format(output))

    # kmeanspp
    c, a, loss = kmeanspp(d, 10, 1)
    acc = get_acc(a)
    print('kmeanspp: {:f}'.format(acc[1]))
    np.array(acc[0]).tofile('output/{}_kmeanspp_acc_0'.format(output))
    np.array(c).tofile('output/{}_kmeanspp_c'.format(output))
    np.array(a).tofile('output/{}_kmeanspp_a'.format(output))
    np.array(loss).tofile('output/{}_kmeanspp_loss'.format(output))

    # PCA
    for co in range(100, 800, 100):
        pca = PCA(n_components=co)
        d = pca.fit_transform(m.get_test_set())

        # basic
        c, a, loss = kmeans(d, 10, 1)
        acc = get_acc(a)
        print('pca={:d} basic: {:f}'.format(co, acc[1]))
        np.array(acc[0]).tofile(
                'output/{}_pca_{:d}_basic_acc_0'.format(output, co))
        np.array(c).tofile(
                'output/{}_pca_{:d}_basic_c'.format(output, co))
        np.array(a).tofile(
                'output/{}_pca_{:d}_basic_a'.format(output, co))
        np.array(loss).tofile(
                'output/{}_pca_{:d}_basic_loss'.format(output, co))

        # better
        c, a, loss = kmeans(d, 10, 1, init='better')
        acc = get_acc(a)
        print('pca={:d} better: {:f}'.format(co, acc[1]))
        np.array(acc[0]).tofile(
                'output/{}_pca_{:d}_better_acc_0'.format(output, co))
        np.array(c).tofile(
                'output/{}_pca_{:d}_better_c'.format(output, co))
        np.array(a).tofile(
                'output/{}_pca_{:d}_better_a'.format(output, co))
        np.array(loss).tofile(
                'output/{}_pca_{:d}_better_loss'.format(output, co))

        # kmeanspp
        c, a, loss = kmeanspp(d, 10, 1)
        acc = get_acc(a)
        print('pca={:d} kmeanspp: {:f}'.format(co, acc[1]))
        np.array(acc[0]).tofile(
                'output/{}_pca_{:d}_kmeanspp_acc_0'.format(output, co))
        np.array(c).tofile(
                'output/{}_pca_{:d}_kmeanspp_c'.format(output, co))
        np.array(a).tofile(
                'output/{}_pca_{:d}_kmeanspp_a'.format(output, co))
        np.array(loss).tofile(
                'output/{}_pca_{:d}_kmeanspp_loss'.format(output, co))


if __name__ == '__main__':
    main()
