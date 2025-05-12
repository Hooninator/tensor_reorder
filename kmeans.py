from collections import defaultdict
import numpy as np


def kmeans_cluster(inds_dict, nrows, **kwargs):
    k = kwargs["nclusters"]
    maxiters = 30
    iters = 0
    assignments = np.zeros(nrows)

    cluster_inds = defaultdict(list)
    cluster_vals = defaultdict(list)

    nassigned = 0
    while nassigned < k:
        pid = np.random.randint(0, nrows)

        if cluster_inds[pid] != []:
            continue

        nassigned += 1

        cluster_inds[pid] = inds_dict[pid]
        cluster_vals[pid] = list(np.ones(len(cluster_inds[pid])))

    while iters < maxiters:
        print(f"Kmeans iteration {iters}")
        distances = pwdist_sparse(
            inds_dict, cluster_inds, cluster_vals, nrows, k)
        assignments, cluster_inds, cluster_vals = update_clusters(
            distances, inds_dict)
        iters+=1

    assignments_dict = {}
    for i in range(len(assignments)):
        if assignments[i] != 0:
            assignments_dict[i] = assignments[i]

    return assignments_dict


def pwdist_sparse(inds_dict, cluster_inds, cluster_vals, n, k):
    distances = np.zeros(shape=(n, k))
    for i in range(n):
        point = inds_dict[i]
        for j in range(k):
            cinds = cluster_inds[j]
            cvals = cluster_vals[j]
            distances[i, j] = euclidean_distance(point, cinds, cvals)
    return distances


def euclidean_distance(point, cinds, cvals):
    dist = 0
    intersect = set(np.intersect1d(point, cinds))
    for i in range(len(cinds)):
        idx = cinds[i]
        if idx in intersect:
            dist += np.square(1 - cvals[i])
        else:
            dist += np.square(cvals[i])
    for i in point:
        if i in intersect:
            continue
        else:
            dist += 1
    return dist


def update_clusters(distances, points):
    n, k = distances.shape
    assignments = np.argmin(distances, axis=1)
    new_cinds = defaultdict(list)
    new_cvals = defaultdict(list)
    clens = np.zeros(k)
    for i in range(n):
        cluster = assignments[i]
        point = points[i]
        new_cinds[cluster], new_cvals[cluster] = add_point(
            point, new_cinds[cluster], new_cvals[cluster])
        clens[cluster] += 1
    for i in range(k):
        new_cvals[i] = list(np.divide(clens[i], new_cvals[i]))
    return assignments, new_cinds, new_cvals


def add_point(point, cinds, cvals):
    for i in range(len(point)):
        idx = point[i]
        if idx in cinds:
            cvals[i] += 1
        else:
            cinds.append(idx)
            cvals.append(1)
    return cinds, cvals
