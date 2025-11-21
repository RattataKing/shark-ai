import pandas as pd
import glob
import matplotlib.pyplot as plt
import random, secrets
import os
from pathlib import Path
import math
import numpy as np

files = glob.glob('./dispatch_tuner/tuning_database/*.csv')
files = [
    f for f in files
    if all(pd.read_csv(f)[col].iloc[0] > 512 for col in ["knob_M", "knob_N", "knob_K"])
]
print(f"Found {len(files)} CSV files")



def is_pow2(x):
    return (x & (x - 1) == 0) and x > 0

def is_mult_simd_num(x: int, simd_num=4) -> bool:
    return x % simd_num == 0

def arith_intensity(x: int, y: int, z: int) -> float:
    num_flops = 2 * x * y * z
    num_byte_access = 2 * (x * y + y * z + x * z)
    return num_flops / num_byte_access

def quantization_inefficiency(M, tile_m, N, tile_n, cu_num: int = 304):
    num_workgroups = (M / tile_m) * (N / tile_n)
    ceil_val = np.ceil(num_workgroups / cu_num)
    q_ie = (ceil_val - num_workgroups / cu_num) / ceil_val
    return q_ie

def compute_sort_columns(df):
    df = df.copy()
    df["tile_k_is_pow2"] = df["knob_tile_k"].apply(is_pow2)

    df["is_mult_simd"] = (
        (df["knob_subgroup_m_cnt"] * df["knob_subgroup_n_cnt"])
        .apply(lambda x: is_mult_simd_num(x))
    )

    df["ai"] = arith_intensity(
        df["knob_intrinsic_mn"],
        df["knob_intrinsic_mn"],
        df["knob_intrinsic_k"],
    )

    df["q_ie"] = quantization_inefficiency(
        df["knob_M"],
        df["knob_tile_m"],
        df["knob_N"],
        df["knob_tile_n"],
    )

    return df


def draw(ax, true_rank, predicted_rank, label):
    ax.scatter(true_rank, predicted_rank, alpha=0.7, label=label)

script_dir = os.path.dirname(os.path.abspath(__file__))
rng = random.Random(secrets.randbits(64))
rng.shuffle(files)
for i, f in enumerate(files):
    df = pd.read_csv(f)
    # if 16 not in df["knob_intrinsic_mn"].values or \
    #     16 not in df["knob_intrinsic_k"].values:
    #     print("happy")
    #     exit()
    # if (~df["knob_tile_k"].apply(is_pow2)).any():
    #     print(f"{f}")
    #     # exit()
    # continue
    df2 = compute_sort_columns(df)

    df_sorted = df2.sort_values(
        by=["tile_k_is_pow2", "is_mult_simd", "ai", "q_ie"],
        ascending=[False, False, True, True]
    )
    # print(df_sorted.head(2000))
    pred_rank = list(range(1, len(df_sorted) + 1))
    true_rank = df_sorted["benchmark_rank_order"].tolist().copy()
    max_rank = int(max(r for r in true_rank if not pd.isna(r)))
    true_rank = [
        int(r) if not pd.isna(r) else max_rank + 1
        for r in true_rank
    ]
    # print(true_rank)
    # print()

    fig, ax = plt.subplots()

    draw(ax, true_rank, pred_rank, "heuristic")

    pred_rank = df.index.tolist()
    true_rank = df["benchmark_rank_order"].tolist().copy()
    true_rank = [
        int(r) if not pd.isna(r) else max_rank + 1
        for r in true_rank
    ]
    # print(pred_rank)
    # print(true_rank)
    draw(ax, true_rank, pred_rank, "shuffle")

    ax.plot([1, max_rank], [1, max_rank], 'r--')
    ax.set(xlim=(1, max_rank), ylim=(1, max_rank),
           xlabel="True Rank", ylabel="Predicted Rank",
           title=f"True vs Predicted Rank\n{Path(f).stem}")
    ax.legend(frameon=False)

    save_path = os.path.join(script_dir, f"true_vs_pred_sim.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {save_path}")
    exit()

