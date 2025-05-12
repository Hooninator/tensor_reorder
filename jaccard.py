from collections import defaultdict
import numpy as np


def jaccard_cluster(inds_dict, nrows, **kwargs):

    threshold = kwargs["threshold"]

    unassigned_rows = np.arange(0, nrows)
    np.random.permutation(unassigned_rows)
    unassigned_rows = list(unassigned_rows)
    #assignments = np.zeros(shape=nrows, dtype=int)
    #assignments.fill(-1)
    assignments = {}

    cid = 0
    while len(unassigned_rows) > 0:
        # Choose unassigned row
        v_id = unassigned_rows[0]
        unassigned_rows.remove(v_id)
        assignments[v_id] = cid
        v = inds_dict[v_id]
        pc = v

        # Merge other unassigned rows
        for w_id in unassigned_rows:
            w = inds_dict[w_id]
            d = jaccard_distance(w, pc)
            if d < threshold:
                unassigned_rows.remove(w_id)
                assignments[w_id] = cid
                pc = list(np.union1d(w, pc))
        cid += 1

    return assignments


def jaccard_distance(w, pc):
    if len(w) == 0 or len(pc) == 0:
        return 1
    return 1 - (len(np.intersect1d(w, pc)) / len(np.union1d(w, pc)))
