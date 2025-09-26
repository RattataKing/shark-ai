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
print(f"Found {len(files)} CSV files")

def get_rank(candidates: list):
    return rankdata(candidates, method="dense")

def draw(ax, true_rank, predicted_rank, label):
    ax.scatter(true_rank, predicted_rank, alpha=0.7, label=label)

def col(df, name: str):
    """Return a pandas Series for a feature name, mapping cfg_ → cfg."""
    mapped = name.replace("cfg_", "cfg.")
    if mapped not in df.columns:
        raise KeyError(f"Column '{mapped}' not found. Available: {list(df.columns)[:10]} ...")
    return df[mapped]

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
    predicted_rank = rankdata(l, method="ordinal")
    rank_max.append(predicted_rank.max())
    draw(ax, true_rank, predicted_rank, "shuffle")

    # q_ineff
    l = df["cfg.quantization_inefficiency"].tolist().copy()
    predicted_rank = rankdata(l, method="ordinal")
    rank_max.append(predicted_rank.max())
    draw(ax, true_rank, predicted_rank, "q_ineff")

    # symbolic regressor formula
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

    predicted_rank = rankdata(score, method="ordinal")
    rank_max.append(predicted_rank.max())
    draw(ax, true_rank_f, predicted_rank, "symbolic_exp")



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