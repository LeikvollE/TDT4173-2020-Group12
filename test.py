#!/usr/bin/env python3
from kmeans.basic import kmeans
from loss.euclidean_loss import euclidean_loss
import mnist
from stars import load_stars


def main():
    stars = load_stars()
    c, a = kmeans(stars, 6)
    print(a)
    print(euclidean_loss(stars, c, a))

    m = mnist.mnist()
    c, a = kmeans(m.get_test_set(), 10)
    print(euclidean_loss(m.get_test_set(), c, a))


if __name__ == '__main__':
    main()
