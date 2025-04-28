import numpy as np
from typing import List
from functools import reduce
from collections import defaultdict

from jaccard import jaccard_cluster
from kmeans import kmeans_cluster
from unif_random import random_cluster
from lexi import lexi_order


def prefix_sum(arr):
    sum_arr = np.zeros(len(arr)+1)
    for i in range(1, len(arr)+1):
        sum_arr[i] = sum_arr[i-1] + arr[i-1]
    return sum_arr


class Tensor:

    def __init__(self, data: np.ndarray, order: int, mode_sizes, bm, bn):
        self.data = data
        self.order = order
        self.nnz = len(data)
        self.mode_sizes = mode_sizes
        self.bm = bm
        self.bn = bn

        self.reorder_funcs = {
            "random": random_cluster,
            "kmeans": kmeans_cluster,
            "jaccard": jaccard_cluster,
            "lexi": lexi_order
        }

    def reorder(self, reorder_str: str, **kwargs):
        mode_nblocks = []
        for k in range(self.order):
            unfolding = self.unfold_modek(k)
            nrows = self.mode_sizes[k]
            unfolding_dict = self.make_unfolding_dict(unfolding)
            clusters = self.reorder_funcs[reorder_str](
                unfolding_dict, nrows, **kwargs)
            reordered_dict = self.reorder_from_clusters(
                unfolding_dict, clusters)
            nblocks = self.count_blocks_unfolded(reordered_dict)
            mode_nblocks.append(nblocks)
        return mode_nblocks

    def reorder_tiled(self, reorder_str: str, **kwargs):
        tile_size = kwargs["tilde_size"]
        mode_nblocks = []
        for k in range(self.order):
            unfolding = self.unfold_modek(k)
            nrows = self.mode_sizes[k]
            ntiles = self.get_unfolding_ncols(k)
            nblocks = 0
            for tile_id in range(ntiles // tile_size):
                unfolding_dict = self.make_unfolding_dict_tiled(
                    unfolding, tile_id, tile_size)
                clusters = self.reorder_funcs[reorder_str](
                    unfolding_dict, nrows, **kwargs)
                reordered_dict = self.reorder_from_clusters(
                    unfolding_dict, clusters)
                nblocks += self.count_blocks_unfolded(reordered_dict)
            mode_nblocks.append(nblocks)
        return mode_nblocks

    def reorder_from_clusters(self, unfolding_dict, clusters):
        cluster_sizes = defaultdict(lambda: 0)
        for c in clusters:
            cluster_sizes[c] += 1
        offsets = prefix_sum(cluster_sizes)

        new_dict = {}
        for i in range(len(clusters)):
            rid = offsets[clusters[i]]
            offsets[clusters[i]] += 1
            new_dict[rid] = unfolding_dict[i]

        return new_dict

    def get_bid_modek(self, value, k, bm, bn):
        i = value[k]
        offset = 1
        j = 0
        for l in range(self.order):
            if l == k:
                continue
            j += value[l]*offset
            offset *= self.mode_sizes[l]
        return (i // bm, j // bn)

    # Explicit column major coo representation of the mode-k unfolding
    def unfold_modek(self, k):
        unfolding = np.zeros(shape=(self.nnz, 3))
        h = 0
        for value in self.data:
            i = value[k]
            offset = 1
            j = 0
            for l in range(self.order):
                if l == k:
                    continue
                j += value[l]*offset
                offset *= self.mode_sizes[l]
            unfolding[h] = np.array([i, j, value[self.order]])
            h += 1
        return unfolding

    # d[rid] = list of column indices in row rid
    def make_unfolding_dict(self, unfolding):
        d = defaultdict(list)
        for i in range(len(unfolding)):
            value = unfolding[i]
            rid = int(value[0])
            d[rid].append(int(value[1]))
        return d

    def make_unfolding_dict_tiled(self, unfolding, tile_id, tile_size):
        lower = tile_id * tile_size
        upper = lower + tile_size
        d = defaultdict(list)
        for i in range(len(unfolding)):
            value = unfolding[i]
            rid = int(value[0])
            if lower <= value[1] and value[1] < upper:
                d[rid].append(int(value[1]))
        return d

    def idx_to_inds(self, idx):
        inds = np.zeros(self.order)
        for i in range(self.order):
            mode_offset = reduce(lambda x, y: x*y, self.mode_sizes[:i])
            inds[i] = (idx // mode_offset) % self.mode_sizes[i]
        return inds

    def count_blocks(self, bm: int, bn: int) -> List[int]:
        blocks = []
        for k in range(self.order):
            bids = set()
            for val in self.data:
                bid = self.get_bid_modek(val, k, bm, bn)
                bids.add(bid)
            blocks.append(len(bids))
        return blocks

    def count_blocks_unfolded(self, unfolded_dict):
        bids = set()
        for i in unfolded_dict.keys():
            for j in unfolded_dict[i]:
                bid = (i // self.bm, j // self.bn)
                bids.add(bid)
        return len(bids)

    def get_unfolding_ncols(self, k):
        return reduce(lambda x, y: x*y, [self.mode_sizes[i] for i in range(self.order) if i != k])
