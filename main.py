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
                curr.append(int(bits[i]))
            curr.append(float(bits[order]))
            coo.append(np.array(curr))
    nnz = len(coo)
    print(f"Tensor order: {order}, nnz: {nnz}, mode sizes: {mode_sizes}")
    return Tensor(np.array(coo), order, mode_sizes, bm, bn)


def pretty_print_config(args):
    print(
        f"Tensor: {args.file}, Reordering: {args.reorder}, Block Size: {args.bm}x{args.bn}")


def print_separator():
    print("="*100)
    print("="*100)


def stime(msg: str) -> float:
    print(msg)
    return time.time()


def etime(stime: float) -> None:
    etime = time.time()
    print(f"Time elapsed: {etime-stime}s")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str)
    parser.add_argument("--reorder", type=str)
    parser.add_argument("--bm", type=int)
    parser.add_argument("--bn", type=int)
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
    blocks = tensor.count_blocks(args.bm, args.bn)
    print(f"Number of blocks before reordering: {blocks}")
    etime(t2)

    print_separator()
    t3 = stime("Reordering tensor...")
    blocks = tensor.reorder(args.reorder)
    etime(t3)

    print(f"Number of blocks after reordering: {blocks}")
    print_separator()
