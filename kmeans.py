import numpy as np
from sklearn.cluster import KMeans

nclusters = 10


def kmeans_cluster(inds_dict, nrows, **kwargs):
    d = max(map(len, inds_dict.values()))
    n = nrows
    data = np.zeros(shape=(n, d))
    for i in inds_dict.keys():
        for j in inds_dict[i]:
            print(i, j)
            data[i, j] = 1
    _kmeans = KMeans(n_clusters=nclusters).fit(data)
    return _kmeans.labels_
