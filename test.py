#!/usr/bin/env python3
from kmeans.basic import kmeans
import mnist
from stars import load_stars
from loss.euclidean_loss import euclidean_loss


def main():
    stars = load_stars()
    c, a = kmeans(stars, 6)
    print('loss %d' % euclidean_loss(stars, c, a))

    m = mnist.mnist()
    c, a = kmeans(m.get_test_set(), 10)
    print('loss %d' % euclidean_loss(m.get_test_set(), c, a))


if __name__ == '__main__':
    main()
