#!/usr/bin/env python3
import numpy as np
import itertools


def accuracy(labels, assignments):
    count = {}

    # for each element in the dataset count all pairs of label to
    # cluster assignments
    #
    # NOTE: this is somewhat backwards, so if you pp(count) keep in mind
    # that labels are on the left-hand side, and the actual clusters are
    # on the right
    for l, c in np.c_[labels, assignments]:
        if l not in count:
            count[l] = {}
        if c not in count[l]:
            count[l][c] = 0
        count[l][c] += 1

    l = set(labels)
    c = set(assignments)

    # generate all possible label to cluster assignments mappings
    pairs = []
    permut = itertools.permutations(l, len(c))
    for comb in permut:
        zipped = zip(comb, c)
        pairs.append(list(zipped))

    # for every such mapping
    out = (None, 0)
    for p in pairs:
        i = 0

        # get the total count of correctly labeled assignments
        for l, c in p:
            if c in count[l]:
                i += count[l][c]

        # and keep the mapping+count if the count is greater (or equal)
        # to the current largest
        if out[1] <= i:
            out = (p, i)

    # return the resulting label to cluster assignment mapping and accuracy
    return out[0], out[1] / len(assignments)
