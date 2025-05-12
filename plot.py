import pandas as pd
import matplotlib.pyplot as plt

import argparse
import os

from dataclasses import dataclass


@dataclass
class Params:
    tensor: str
    reordering: str
    bmxbn: str
    tiled: bool
    tile_size: int
    threshold: float
    nclusters: int

    def to_str(self):
        if self.reordering=="jaccard":
            return f"{self.tensor}_{self.reordering}_{self.bmxbn}{'_tiled' if self.tiled else ''}{'_t' + str(self.threshold) if self.threshold > -1 else ''}"
        if self.reordering=="kmeans":
            return f"{self.tensor}_{self.reordering}_{self.bmxbn}{'_tiled' if self.tiled else ''}{'_k' + str(self.nclusters)}"

    def get_label(self):
        param = ""
        if self.reordering == "jaccard":
            param = ", "+f"t={str(self.threshold)}"
        elif self.reordering == "kmeans":
            param = ", "+f"k={str(self.nclusters)}"
        return f"{self.reordering}{param}, {self.bmxbn}"


def read_dfs(args):
    files = os.listdir(args.datadir)
    param_lst, df_lst = [], []
    for file in files:
        bits = file.split("/")[-1].split(".csv")[0].split("_")
        # print(bits)
        params = Params(bits[1], bits[3], bits[4], file.find(
            "tiled") != -1, -1, -1, 0)
        if params.reordering == "jaccard":
            params.threshold = float(bits[-1][1:])
        if params.reordering == "kmeans":
            params.nclusters = int(bits[-1][1])
        param_lst.append(params)
        df_lst.append(pd.read_csv(f"{args.datadir}/{file}"))
        df_lst[-1].columns = df_lst[-1].columns.str.strip()
    return param_lst, df_lst


def plot(params, df):
    width = 0.25
    plt.grid(True, axis='both', linestyle='-',
             color='gray', alpha=0.5, zorder=1)
    plt.bar(df["mode"] - width/2, df["original_blocks"], label="original",
            color='crimson', edgecolor='black', zorder=2, width=width)
    plt.bar(df["mode"] + width/2, df["reordered_blocks"], label="reordered",
            color='steelblue', edgecolor='black', zorder=2, width=width)
    plt.xlabel("Mode")
    plt.ylabel("Number of Blocks")
    plt.title(params.to_str())
    plt.xticks(df["mode"])
    plt.legend()
    print(params.to_str())
    plt.savefig(f"./plots/{params.to_str()}.png")
    plt.close()


# One plot per tensor, x-axis is modes, y-axis is number of blocks, one bar for each method+param combo
# We'll have separate plots for tiled and non-tiled
colors = ["crimson", "steelblue", "limegreen",
          "skyblue", "salmon", "firebrick", "navy", "yellow", "orange", "purple"]


def plot_single_tensor(param_lst, df_lst, tensor, reordering, tiled):
    assert len(param_lst) == len(df_lst)
    print([p.get_label() for p in param_lst])
    width = 0.15
    plt.figure(figsize=(10, 6))
    plt.grid(True, axis='both', linestyle='-',
             color='gray', alpha=0.5, zorder=1)
    offset = -1 * ((len(param_lst) + 1) // 2) * width 
    i = 0
    plt.bar(df_lst[0]["mode"] + offset, df_lst[0]["original_blocks"], label="original", 
                color=colors[i], edgecolor='black', zorder=2, width=width)
    offset += width
    i += 1

    for param, df in zip(param_lst, df_lst):
        x = df["mode"]
        y = df["reordered_blocks"]
        plt.bar(x + offset, y, label=param.get_label(),
                color=colors[i], edgecolor='black', zorder=2, width=width)
        offset += width
        i += 1
        plt.xticks(df["mode"])
    plt.xlabel("Mode")
    plt.ylabel("Number of Blocks")
    plt.title(
        f"{reordering} Reordering Performance on {tensor} {', Column Tiling' if tiled else ''}")
    plt.legend()
    plt.savefig(f"./plots/{tensor}_{reordering}{'_tiled' if tiled else ''}.png")
    plt.close()
    plt.clf()


def filter(param_lst, df_lst, tensor_name, reordering, tiled):
    param_lst_f, df_lst_f = [], []
    for param, df in zip(param_lst, df_lst):
        if param.tensor == tensor_name and param.tiled == tiled and param.reordering==reordering:
            param_lst_f.append(param)
            df_lst_f.append(df)
    return param_lst_f, df_lst_f


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str)
    args = parser.parse_args()
    param_lst, df_lst = read_dfs(args)
    tensors = ["uber", "matmul"]
    reorderings = ["jaccard", "lexi", "kmeans"]
    for tensor in tensors:
        for reordering in reorderings:
            param_lst_f, df_lst_f = filter(param_lst, df_lst, tensor, reordering, True)
            if len(param_lst_f) != 0:
                plot_single_tensor(param_lst_f, df_lst_f, tensor,  reordering, True)

            param_lst_f, df_lst_f = filter(param_lst, df_lst, tensor, reordering, False)
            if len(param_lst_f) != 0:
                plot_single_tensor(param_lst_f, df_lst_f, tensor, reordering, False)

    #for params, df in zip(param_lst, df_lst):
    #   print(df)
    #   plot(params, df)
