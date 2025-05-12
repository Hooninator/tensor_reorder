import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
import argparse
import time

from tensor import Tensor


def read_tensor(filepath, mode_sizes, bm, bn):
    order = 0
    coo = []
    with open(filepath, 'r') as file:
        for line in file:
            bits = line.split(" ")
            if order == 0:
                order = len(bits)-1
            curr = []
            for i in range(order):
                curr.append(int(bits[i]) - 1) # 1 indexing 
            curr.append(float(bits[order]))
            coo.append(np.array(curr))
    nnz = len(coo)
    print(f"Tensor order: {order}, nnz: {nnz}, mode sizes: {mode_sizes}")
    return Tensor(np.array(coo), order, mode_sizes, bm, bn)


def pretty_print_config(args):
    print(
        f"Tensor: {args.file}, Reordering: {args.reorder}, Block Size: {args.bm}x{args.bn}, Tiled: {args.col_tiled}")


def print_separator():
    print("="*100)
    print("="*100)


def stime(msg: str) -> float:
    print(msg)
    return time.time()


def etime(stime: float) -> None:
    etime = time.time()
    print(f"Time elapsed: {etime-stime}s")


def get_tensor_name(args):
    fname = args.file
    return fname.split("/")[-1].split(".")[0]


def get_csv_fname(args):
    tname = get_tensor_name(args)
    if args.reorder == "kmeans":
        return f"nblocks_{tname}_{args.reorder}_{args.bm}x{args.bn}{'_tiled' if args.col_tiled else ''}_k{args.nclusters}.csv"
    if args.reorder == "jaccard":
        return f"nblocks_{tname}_{args.reorder}_{args.bm}x{args.bn}{'_tiled' if args.col_tiled else ''}_t{args.threshold}.csv"
    if args.reorder == "lexi":
        return f"nblocks_{tname}_{args.reorder}_{args.bm}x{args.bn}{'_tiled' if args.col_tiled else ''}.csv"
    return ""


def csv_write(args, blocks, og_blocks):
    assert len(blocks) == len(og_blocks)
    fname = get_csv_fname(args)
    with open(f"./data/{fname}", 'w') as file:
        file.write("reordered_blocks, original_blocks, mode\n")
        for i in range(len(blocks)):
            file.write(f"{blocks[i]}, {og_blocks[i]}, {i}\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str)
    parser.add_argument("--reorder", type=str)
    parser.add_argument("--bm", type=int)
    parser.add_argument("--bn", type=int)
    parser.add_argument("--tile_size", type=int)
    parser.add_argument("--nclusters", type=int)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--col_tiled", action='store_true')
    parser.add_argument("--mode_sizes", nargs='+', type=int)
    args = parser.parse_args()

    pretty_print_config(args)

    print_separator()
    t1 = stime("Reading tensor...")
    tensor = read_tensor(args.file, args.mode_sizes, args.bm, args.bn)
    etime(t1)
    print(f"Best possible blocks: {tensor.nnz // (args.bm * args.bn)}")

    print_separator()
    t2 = stime("Counting blocks...")
    og_blocks = tensor.count_blocks(args.bm, args.bn)
    print(f"Number of blocks before reordering: {og_blocks}")
    etime(t2)

    print_separator()
    t3 = stime("Reordering tensor...")
    if args.col_tiled:
        blocks = tensor.reorder_tiled(
            args.reorder, nclusters=args.nclusters, threshold=args.threshold, tile_size=args.tile_size)
    else:
        blocks = tensor.reorder(
            args.reorder, nclusters=args.nclusters, threshold=args.threshold)
    etime(t3)

    print(f"Number of blocks after reordering: {blocks}")
    print_separator()

    csv_write(args, blocks, og_blocks)
