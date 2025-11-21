import pandas as pd
import glob
import matplotlib.pyplot as plt
import random, secrets
import os
from pathlib import Path
import math
import numpy as np

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
    
def geometric_mean(nums):
    product = 1
    n = len(nums)
    for x in nums:
        if x==0: continue
        product *= x
    return product ** (1/n)


files = glob.glob('./dispatch_tuner/tuning_database_small/*.csv')
files = [
    f for f in files
    if all(pd.read_csv(f)[col].iloc[0] > 512 for col in ["knob_M", "knob_N", "knob_K"])
]
files = files[:7] + files[10:]
print(f"Found {len(files)} CSV files")

results = []
for i, f in enumerate(files):
    df = pd.read_csv(f)
    df2 = compute_sort_columns(df)

    df_sorted = df2.sort_values(
        by=["tile_k_is_pow2", "is_mult_simd", "ai", "q_ie"],
        ascending=[False, False, True, True]
    )

    heuristic_pred_rank = list(range(1, len(df_sorted) + 1))
    heuristic_true_rank = df_sorted["benchmark_rank_order"].tolist().copy()
    if not [r for r in heuristic_true_rank if not pd.isna(r)]:
        continue
    max_rank = int(max(r for r in heuristic_true_rank if not pd.isna(r)))
    heuristic_true_rank = [
        int(r) if not pd.isna(r) else max_rank + 1
        for r in heuristic_true_rank
    ]




    shuffle_pred_rank = df.index.tolist()
    shuffle_true_rank = df["benchmark_rank_order"].tolist().copy()
    shuffle_true_rank = [
        int(r) if not pd.isna(r) else max_rank + 1
        for r in shuffle_true_rank
    ]


    df["shuffle_pred_rank"] = shuffle_pred_rank
    df["heuristic_pred_rank"] = (
        df_sorted
        .assign(heuristic_pred_rank=heuristic_pred_rank)
        .sort_index()
        ["heuristic_pred_rank"]
        .reindex(df.index)
        .values
    )


    optimal_tol = float(df["benchmark_speedup"].min() * 1.05)
    opt_candidates = df[df["benchmark_speedup"] <= optimal_tol]["candidate_id"]

    opt_candidates_pred_ranks = df[df["candidate_id"].isin(opt_candidates)]["shuffle_pred_rank"]
    shuffle_min_search_space  = opt_candidates_pred_ranks.min()
    # print(f"Shuffle min search space: {shuffle_min_search_space}")

    opt_candidates_pred_ranks = df[df["candidate_id"].isin(opt_candidates)]["heuristic_pred_rank"]
    heuristic_min_search_space = opt_candidates_pred_ranks.min()
    # print(f"Heuristic min search space: {heuristic_min_search_space}")

    # print(str((Path(f).stem).removeprefix("tuning_")))
    results.append({
        "dispatch_id": f,
        "shuffle_min_search_space_#": shuffle_min_search_space,
        "heuristic_min_search_space_#": heuristic_min_search_space,
        "candidate_num": len(df),
        "shuffle_min_search_space_%": round(shuffle_min_search_space / len(df), 5),
        "heuristic_min_search_space_%": round(heuristic_min_search_space / len(df), 5)
    })
    # exit()


out_df = pd.DataFrame(results)

# Arithmetic means (averages)
shuffle_avg = out_df["shuffle_min_search_space_%"].mean()
heuristic_avg = out_df["heuristic_min_search_space_%"].mean()

# Geometric means
shuffle_gmean = geometric_mean(out_df["shuffle_min_search_space_%"])
heuristic_gmean = geometric_mean(out_df["heuristic_min_search_space_%"])

base_path = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(base_path, f"sort_search_space.csv")
out_df["shuffle_avg"] = round(shuffle_avg,5)
out_df["heuristic_avg"] = round(heuristic_avg,5)
out_df["shuffle_gmean"] = round(shuffle_gmean,5)
out_df["heuristic_gmean"] = round(heuristic_gmean, 5)
print(f"shuffle_avg vs. heuristic_avg: {shuffle_avg:.2f} vs. {heuristic_avg:.2f}")
print(f"shuffle_gmean vs. heuristic_gmean: {shuffle_gmean:.2f} vs. {heuristic_gmean:.2f}")
out_df.to_csv(save_path, index=False)
print(f"Saved results to {save_path}")