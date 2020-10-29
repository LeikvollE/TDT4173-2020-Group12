#!/usr/bin/env python3
import numpy as np
import itertools


def accuracy(labels, assignments):
    count = {}

    for l, c in np.c_[labels, assignments]:
        if l not in count:
            count[l] = {}
        if c not in count[l]:
            count[l][c] = 0
        count[l][c] += 1

    l = set(labels)
    c = set(assignments)

    pairs = []

    permut = itertools.permutations(l, len(c))
    for comb in permut:
        zipped = zip(comb, c)
        pairs.append(list(zipped))

    out = (None, 0)
    for p in pairs:
        i = 0
        miss = False

        for l, c in p:
            if c in count[l]:
                i += count[l][c]
            else:
                miss = True
                break

        if not miss and out[1] <= i:
            out = (p, i)

    return out[0], out[1] / len(assignments)
