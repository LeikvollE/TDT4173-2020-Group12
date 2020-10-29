#!/usr/bin/env python3
from kmeans.basic import kmeans
from kmeans.kmeanspp import kmeanspp
import mnist
import weka
from stars import load_stars
from loss.euclidean_loss import euclidean_loss
import matplotlib.pyplot as plt
import numpy as np
import click


@click.group()
def cli():
    pass


@click.command()
@click.argument('p', default=False)
def test_mnist(p):
    m = mnist.mnist()
    c, a, loss = kmeans(m.get_test_set(), 10)
    print(euclidean_loss(m.get_test_set(), c, a))
    print(m.get_accuracy(a, 'test'))
    if p:
        for i in range(10):
            plt.imshow(np.reshape(c[i], (28, 28)))
            plt.show()

    c, a, loss = kmeans(m.get_trainig_set(), 10)
    print(euclidean_loss(m.get_trainig_set(), c, a))
    print(m.get_accuracy(a, 'train'))
    if p:
        for i in range(10):
            plt.imshow(np.reshape(c[i], (28, 28)))
            plt.show()


@click.command()
def test_stars():
    stars = load_stars()
    c, a, loss = kmeans(stars, 6)
    print(a)
    print(euclidean_loss(stars, c, a))


@click.command()
def test_weka():
    w = weka.weka()
    c, a, loss = kmeans(w.get_two(), 2)
    print(euclidean_loss(w.get_two(), c, a))
    print(w.get_accuracy(a, 2))

    c, a, loss = kmeans(w.get_three(), 3)
    print(euclidean_loss(w.get_three(), c, a))
    print(w.get_accuracy(a, 3))


cli.add_command(test_mnist)
cli.add_command(test_stars)
cli.add_command(test_weka)


if __name__ == '__main__':
    cli()
