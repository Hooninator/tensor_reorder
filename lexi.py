from collections import defaultdict
import numpy as np
import functools


def lexi_order(inds_dict, nrows, **kwargs):

    def lexi_comp(x, y) -> int:

        if len(x)==0:
            if len(y)==0:
                return 0
            else:
                return -1 
        if len(y)==0:
            return 1

        x = list(x)
        y = list(y)
        x.sort()
        y.sort()
        done = False
        i, j = 0, 0
        while not done:
            xi = x[i]
            yj = y[j]

            i += 1
            j += 1

            if xi < yj:
                return -1
            if xi > yj:
                return 1

            if i == len(x):
                # They are equal
                if j == len(y):
                    return 0
                # y is longer than x, and therefore > x
                return -1
            if j == len(y):
                # x is longer than y, and therefore > y
                return 1
        return -1

    inds = {}
    sorted_inds = []
    for rid in inds_dict.keys():
        colinds = tuple(inds_dict[rid])
        inds[colinds] = rid
        sorted_inds.append(colinds)

    sorted_inds.sort(key=functools.cmp_to_key(lexi_comp))

    clusters = {}

    print(inds_dict)
    print(sorted_inds)
    for i in range(len(sorted_inds)):
        colinds = sorted_inds[i]
        rid = inds[colinds]
        clusters[rid] = i
    print(clusters)
    return clusters
