import numpy as np


def random_cluster(inds_dict, nrows, **kwargs):
    clusters = np.arange(nrows)
    return np.random.permutation(clusters)
