import os, sys
import pandas as pd
import glob
from pathlib import Path
import random, secrets
import numpy as np
from scipy.stats import spearmanr, rankdata, pearsonr
import matplotlib.pyplot as plt

files = glob.glob('./dispatch_tuner/tuning_database_clean/*.csv')
excluded_files = [
    # Problem size too small 
    "tuning_square_gemm_128_128_128_f16_f32_tB.csv",
    "tuning_square_gemm_256_256_256_f16_f32_tB.csv",
    "tuning_square_gemm_512_512_512_f16_f32_tB.csv",
]
files = [f for f in files if os.path.basename(f) not in excluded_files]
files = [
    f for f in files
    if all(pd.read_csv(f)[col].iloc[0] > 512 for col in ["cfg.M", "cfg.N", "cfg.K"])
]
print(f"Found {len(files)} CSV files")

def get_rank(candidates: list):
    return rankdata(candidates, method=RANKMETHOD)

def draw(ax, true_rank, predicted_rank, label):
    ax.scatter(true_rank, predicted_rank, alpha=0.7, label=label)

def col(df, name: str):
    """Return a pandas Series for a feature name, mapping cfg_ → cfg."""
    mapped = name.replace("cfg_", "cfg.")
    if mapped not in df.columns:
        raise KeyError(f"Column '{mapped}' not found. Available: {list(df.columns)[:10]} ...")
    return df[mapped]

def compute_sr_speedup(df):
    k = col(df, "cfg_k").to_numpy(dtype=float)
    subgroup_size = col(df, "cfg_subgroup_size").to_numpy(dtype=float)
    mma_map = col(df, "cfg_mma_attr_map").to_numpy(dtype=float)
    # translate the symbolic expression to numpy
    # Original intent (readable decomposition):
    inner2 = (
        0.31579238
        * np.power(0.059571117, 0.059571117 * np.sin(k + subgroup_size))
        * np.sin(0.18980226 * np.power(k, 2.0))
        + np.sin(mma_map + 0.059571117)
    )
    inner = 1.0939728 * np.sin(2.1222813 * np.sin(inner2)) + 0.3447154
    score = np.sin(np.abs(inner) ** 1.2775977)
    # guard against NaN/Inf before ranking
    finite = np.isfinite(score)
    if not finite.all():
        score = score[finite]
        true_rank_f = get_rank(df.loc[finite, "norm_speedup"].tolist())
    else:
        true_rank_f = true_rank
    
    return score

def compute_rf_speedup(df):
    # --- Convenience accessors ---
    K_intr   = df["cfg.intrinsic_k"].astype(float)
    MN_intr  = df["cfg.intrinsic_mn"].astype(float)
    mma_attr = df["cfg.mma_attr_map"].astype(float)
    sg_n_cnt = df["cfg.sg_n_cnt"].astype(float)
    k_pow2   = df["k_pow2"].astype(float)             # works for bool or 0/1
    lds_util = df["cfg.lds_utilization"].astype(float)

    # --- Rule masks ---
    r1 = (K_intr > 24) & (mma_attr <= 2.5)
    r2 = (K_intr <= 24) & (sg_n_cnt <= 4.5) & (k_pow2 <= 0.5)
    r3 = (K_intr <= 24) & (MN_intr > 24) & (mma_attr <= 2.5) & (k_pow2 > 0.5)
    r4 = (K_intr <= 24) & (mma_attr > 1) & (mma_attr <= 2.5) & (k_pow2 > 0.5)
    r5 = (K_intr <= 24) & (lds_util > 0.215) & (sg_n_cnt <= 4.5) & (k_pow2 <= 0.5)

    # --- Weighted sum + intercept ---
    speedup = (
        0.849 * r1 +
        0.847 * r2 +
        0.592 * r3 +
        0.257 * r4 +
        0.001 * r5 +
        0.150
    ).astype(float)

    return speedup

RANKMETHOD = "dense"
from functools import cmp_to_key


def cmp_rows(row1, row2) -> bool:
    get_tile_size = lambda x: x["cfg.m"] * x["cfg.n"] * x["cfg.k"]

    is_pow2 = lambda x: (int(x) != 0) and (int(x) & (int(x) - 1))
    is_tile_pow2 = lambda x: is_pow2(x["cfg.m"]) and is_pow2(x["cfg.n"]) and is_pow2(x["cfg.k"])
    area = lambda x, y, z: 2 * (x * y + y * z + x * z)
    volume = lambda x, y, z: x * z * y
    get_v_a = lambda x: area(x["cfg.m"], x["cfg.n"], x["cfg.k"]) \
                        / volume(x["cfg.m"], x["cfg.n"], x["cfg.k"])    
    
    return get_v_a(row1) < get_v_a(row2)


def stable_sort(items, less):
    if len(items) <= 1:
        return items[:]

    def merge(left, right):
        i = j = 0
        out = []
        while i < len(left) and j < len(right):
            # Take from left if left < right, otherwise from right.
            # For "ties" (neither < the other), we take from left to keep stability.
            if less(left[i], right[j]):
                out.append(left[i]); i += 1
            elif less(right[j], left[i]):
                out.append(right[j]); j += 1
            else:
                out.append(left[i]); i += 1  # tie → left first (stable)
        if i < len(left):  out.extend(left[i:])
        if j < len(right): out.extend(right[j:])
        return out

    mid = len(items) // 2
    left = stable_sort(items[:mid], less)
    right = stable_sort(items[mid:], less)
    return merge(left, right)


def handwritten(df):
    """
    Assigns 0-1 ranks based on can_prior() strategy.
    0 = best, 1 = worst.
    """
    # print(df)

    # rows_sorted = sorted(df.to_dict("records"), key=cmp_to_key(can_prior))
    rows_sorted = stable_sort(df.to_dict("records"), cmp_rows)
    df_sorted = pd.DataFrame(rows_sorted)
    # print(df_sorted["cfg.intrinsic_mn"])

    # exit()

    # Back to DataFrame
    df_sorted = pd.DataFrame(rows_sorted)
    n = len(df_sorted)
    df_sorted["norm_speedup_score"] = [i / (n - 1) if n > 1 else 0 for i in range(n)]
    print(df_sorted[["candidate_id","cfg.intrinsic_mn","norm_speedup_score"]])
    
    df_final = df_sorted.set_index("candidate_id").loc[df["candidate_id"]].reset_index()
    # print(df_final[["candidate_id","norm_speedup_score", "cfg.intrinsic_mn"]])

    return df_final["norm_speedup_score"].tolist().copy()

rng = random.Random(secrets.randbits(64))
rng.shuffle(files)
script_dir = os.path.dirname(os.path.abspath(__file__))
for i, f in enumerate(files):
    df = pd.read_csv(f)

    rank_max = []
    true_rank = get_rank(df["norm_speedup"].tolist())
    rank_max.append(true_rank.max())
    fig, ax = plt.subplots()

    # Shuffle
    l = df["norm_speedup"].tolist().copy()
    rng.shuffle(l)
    predicted_rank = rankdata(l, method=RANKMETHOD)
    rank_max.append(predicted_rank.max())
    draw(ax, true_rank, predicted_rank, "shuffle")

    # q_ineff
    # l = df["cfg.quantization_inefficiency"].tolist().copy()
    # predicted_rank = rankdata(l, method=RANKMETHOD)
    # rank_max.append(predicted_rank.max())
    # draw(ax, true_rank, predicted_rank, "low_q_ineff")

    # symbolic regressor formula
    # speedup = compute_sr_speedup(df)
    # predicted_rank = rankdata(speedup, method=RANKMETHOD)
    # rank_max.append(predicted_rank.max())
    # draw(ax, true_rank, predicted_rank, "symbolic_exp")

    ## RF formula
    # speedup = compute_rf_speedup(df)
    # predicted_rank = rankdata(speedup, method=RANKMETHOD)
    # rank_max.append(predicted_rank.max())
    # draw(ax, true_rank, predicted_rank, "rf_formula")

    speedup = handwritten(df)
    predicted_rank = rankdata(speedup, method=RANKMETHOD)
    rank_max.append(predicted_rank.max())
    draw(ax, true_rank, predicted_rank, "jakub")



    max_val = max(rank_max)
    ax.plot([1, max_val], [1, max_val], 'r--')
    ax.set(xlim=(1, max_val), ylim=(1, max_val),
           xlabel="True Rank", ylabel="Predicted Rank",
           title=f"True vs Predicted Rank\n{Path(f).stem}")
    ax.legend(frameon=False)

    # save_path = os.path.join(script_dir, f"true_vs_pred_sim{i}.png")
    save_path = os.path.join(script_dir, f"true_vs_pred_sim.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {save_path}")

    exit()