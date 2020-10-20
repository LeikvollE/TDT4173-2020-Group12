#!/usr/bin/env python3
import kmeans
import mnist
import numpy as np
import random
from pprint import pp
from stars import load_stars


def main():
    # stars = load_stars()
    # pp(stars)

    # centroids = stars[random.sample(range(len(stars)), 6)].copy()
    # assignments = np.zeros(len(stars))

    # pp(centroids)
    # pp(assignments)

    m = mnist.mnist()
    # pp(m.get_test_set())
    # pp(m.get_trainig_set())

    import sys
    np.set_printoptions(threshold=sys.maxsize)
    c, a = kmeans.kmeans(m.get_test_set(), 10)
    # print('c')
    # print(c)
    # pp(c)
    print('a')
    print(a)
    # pp(a)


if __name__ == '__main__':
    main()
