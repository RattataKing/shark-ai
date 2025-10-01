import os, sys
import pandas as pd
import glob
from pathlib import Path
import random, secrets
import numpy as np
from scipy.stats import rankdata
import matplotlib.pyplot as plt
from functools import cmp_to_key

RANKMETHOD = "ordinal"


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
    if all(pd.read_csv(f)[col].iloc[0] > 640 for col in ["cfg.M", "cfg.N", "cfg.K"])
]
print(f"Found {len(files)} CSV files")

# def geometric_mean(nums):
#     product = 1
#     n = len(nums)
#     for x in nums:
#         product *= x
#     return product ** (1/n)

def get_rank(candidates: list):
    return rankdata(candidates, method=RANKMETHOD)

def adjusted_true(true_rank, pred_rank, bench_mask):
    print(f"{len(true_rank[bench_mask])} / {len(bench_mask)}")
    last_valid_rank = pred_rank[bench_mask].max()
    print(last_valid_rank)
    for i, pred in enumerate(pred_rank):
        if bench_mask[i] == False:
            if pred > last_valid_rank:
                true_rank[i] = pred
            # else:
            #     true_rank[i] = last_valid_rank+i+1
    return true_rank

def draw(ax, true_rank, pred_rank, label, df, color=None):
    if color is None:
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][
            len(ax.collections) % len(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        ]

    true_rank = np.array(true_rank)
    pred_rank = np.array(pred_rank)

    # Masks
    mask_valid = df["benchmark_status"].to_numpy()
    optimal_time = df["benchmark_time_ms"].min()
    mask_near = (df["benchmark_time_ms"].to_numpy() <= optimal_time * 1.5)

    # valid points
    ax.scatter(true_rank[mask_valid], pred_rank[mask_valid],
               label=label, color=color, alpha=0.5)

    # 95+% optimal points
    ax.scatter(true_rank[mask_near], pred_rank[mask_near],
               color=color, alpha=0.9)

    # benchmark failed
    ax.scatter(true_rank[~mask_valid], pred_rank[~mask_valid],
               color=color, alpha=0.1)


def cmp_rows(row1, row2) -> bool:
    get_tile_size = lambda x: x["cfg.m"] * x["cfg.n"] * x["cfg.k"]

    is_pow2 = lambda x: 1 if (int(x) != 0 and (int(x) & (int(x)-1)) == 0) else 0    # Zero = False, Non-zero = True
    m_pow2 = lambda x: is_pow2(x["cfg.m"])
    n_pow2 = lambda x: is_pow2(x["cfg.n"])
    k_pow2 = lambda x: is_pow2(x["cfg.k"])
    # is_tile_pow2 = lambda x: m_pow2(x) and n_pow2(x) and k_pow2(x)

    num_flops = lambda x, y, z: 2* x * y * z
    num_byte_access = lambda x, y, z: 2 * (x * y + y * z + x * z)
    arith_intensity = lambda x,y,z: num_flops(x,y,z)/num_byte_access(x,y,z)
    p_ai = lambda x: arith_intensity(x["cfg.M"], x["cfg.N"], x["cfg.K"])
    t_ai = lambda x: arith_intensity(x["cfg.m"], x["cfg.n"], x["cfg.k"])
    intrinsic_ai = lambda x: arith_intensity(x["cfg.intrinsic_mn"], x["cfg.intrinsic_mn"], x["cfg.intrinsic_k"])

    return k_pow2(row1) > k_pow2(row2)

def stable_sort(rows: list[dict], cmp_func):
    if len(rows) <= 1:
        return rows[:]

    def merge(left, right):
        i = j = 0
        out = []
        while i < len(left) and j < len(right):
            if cmp_func(left[i], right[j]):
                out.append(left[i])
                i += 1
            elif cmp_func(right[j], left[i]):
                out.append(right[j])
                j += 1
            else:
                out.append(left[i])
                i += 1
        out.extend(left[i:])
        out.extend(right[j:])
        return out

    mid = len(rows) // 2
    left = stable_sort(rows[:mid], cmp_func)
    right = stable_sort(rows[mid:], cmp_func)
    return merge(left, right)

def test_cmp_rows(r1, r2):
    # prefer tile_k that is pow2 
    is_pow2 = lambda x: 1 if (int(x) != 0 and (int(x) & (int(x)-1)) == 0) else 0    # Zero = False, Non-zero = True
    return is_pow2(r1["cfg.k"]) > is_pow2(r2["cfg.k"])

def test_sort():
    dummy = [
        {"candidate_id": 1, "cfg.m": 4, "cfg.n": 4, "cfg.k": 7, "cfg.M": 16, "cfg.N": 16, "cfg.K": 16, "cfg.intrinsic_mn": 4, "cfg.intrinsic_k": 4},
        {"candidate_id": 2, "cfg.m": 4, "cfg.n": 4, "cfg.k": 8, "cfg.M": 16, "cfg.N": 16, "cfg.K": 16, "cfg.intrinsic_mn": 4, "cfg.intrinsic_k": 4},
        {"candidate_id": 3, "cfg.m": 4, "cfg.n": 4, "cfg.k": 6, "cfg.M": 16, "cfg.N": 16, "cfg.K": 16, "cfg.intrinsic_mn": 4, "cfg.intrinsic_k": 4},
        {"candidate_id": 4, "cfg.m": 4, "cfg.n": 4, "cfg.k": 16, "cfg.M": 16, "cfg.N": 16, "cfg.K": 16, "cfg.intrinsic_mn": 4, "cfg.intrinsic_k": 4},
        {"candidate_id": 5, "cfg.m": 8, "cfg.n": 4, "cfg.k": 16, "cfg.M": 16, "cfg.N": 16, "cfg.K": 16, "cfg.intrinsic_mn": 4, "cfg.intrinsic_k": 4},
    ]

    print("Test stable_sort():")
    print(pd.DataFrame(dummy))

    rows_sorted = stable_sort(dummy, test_cmp_rows)
    print(pd.DataFrame(rows_sorted))

    pred_id = [row["candidate_id"] for row in rows_sorted]
    expect_id = [2, 4, 5, 1, 3]  
    assert pred_id == expect_id, f"Expected {expect_id}, got {pred_id}"

    print("stable_sort() passed!")

def manual_rank(df):
    rows_sorted = stable_sort(df.to_dict("records"), cmp_rows)
    df_sorted = pd.DataFrame(rows_sorted)
    df_sorted["pred_rank"] = [i+1 for i in range(len(df_sorted))]
    df_final = df_sorted.set_index("candidate_id").loc[df["candidate_id"]].reset_index()
    return np.array(df_final["pred_rank"].tolist().copy())




test_sort()

print("Ranking...")
rng = random.Random(secrets.randbits(64))
rng.shuffle(files)
script_dir = os.path.dirname(os.path.abspath(__file__))
results = []
for i, f in enumerate(files):
    df = pd.read_csv(f)

    true_rank = get_rank(df["norm_speedup"].tolist())
    benchmark_status=df["benchmark_status"].to_numpy()

    optimal_tol = float(df["benchmark_time_ms"].min() * 1.05)
    opt_candidates = df[df["benchmark_time_ms"] <= optimal_tol]["candidate_id"]

    shuffle_pred_rank = true_rank.copy()
    rng.shuffle(shuffle_pred_rank)

    df["shuffle_pred_rank"] = shuffle_pred_rank
    opt_candidates_pred_ranks = df[df["candidate_id"].isin(opt_candidates)]["shuffle_pred_rank"]
    shuffle_min_search_space  = opt_candidates_pred_ranks.min()
    # print(f"Shuffle min search space: {shuffle_min_search_space}")

    pred_rank = manual_rank(df)
    df["heuristic_pred_rank"] = pred_rank
    opt_candidates_pred_ranks = df[df["candidate_id"].isin(opt_candidates)]["heuristic_pred_rank"]
    heuristic_min_search_space = opt_candidates_pred_ranks.min()
    # print(f"Heuristic min search space: {heuristic_min_search_space}")

    results.append({
        "dispatch_id": df["dispatch_id"].iloc[0],
        "shuffle_min_search_space_#": shuffle_min_search_space,
        "heuristic_min_search_space_#": heuristic_min_search_space,
        "candidate_num": len(df),
        "shuffle_min_search_space_%": round(shuffle_min_search_space / len(df), 5),
        "heuristic_min_search_space_%": round(heuristic_min_search_space / len(df), 5)
    })


def geometric_mean(nums):
    product = 1
    n = len(nums)
    for x in nums:
        if x==0: continue
        product *= x
    return product ** (1/n)

out_df = pd.DataFrame(results)

# Arithmetic means (averages)
shuffle_avg = out_df["shuffle_min_search_space_%"].mean()
heuristic_avg = out_df["heuristic_min_search_space_%"].mean()

# Geometric means
shuffle_gmean = geometric_mean(out_df["shuffle_min_search_space_%"])
heuristic_gmean = geometric_mean(out_df["heuristic_min_search_space_%"])

save_path = os.path.join(script_dir, f"sort_search_space.csv")
out_df["shuffle_avg"] = round(shuffle_avg,5)
out_df["heuristic_avg"] = round(heuristic_avg,5)
out_df["shuffle_gmean"] = round(shuffle_gmean,5)
out_df["heuristic_gmean"] = round(heuristic_gmean, 5)
print(f"shuffle_avg vs. heuristic_avg: {shuffle_avg:.2f} vs. {heuristic_avg:.2f}")
print(f"shuffle_gmean vs. heuristic_gmean: {shuffle_gmean:.2f} vs. {heuristic_gmean:.2f}")
out_df.to_csv(save_path, index=False)
print(f"Saved results to {save_path}")


f = files[0]
df = pd.read_csv(f)
# df = df[df["benchmark_status"] == True]

rank_max = []
true_rank = get_rank(df["norm_speedup"].tolist())
benchmark_status=df["benchmark_status"].to_numpy()
rank_max.append(true_rank.max())
fig, ax = plt.subplots()

# Shuffle
pred_rank = true_rank.copy()
rng.shuffle(pred_rank)
rank_max.append(pred_rank.max())
draw(ax, true_rank, pred_rank, "shuffle", df, color="C0")

pred_rank = manual_rank(df)
rank_max.append(pred_rank.max())
draw(ax, true_rank, pred_rank, "heuristic", df, color="C1")

max_val = max(rank_max)
ax.plot([1, max_val], [1, max_val], 'r--')
ax.set(xlim=(1, max_val), ylim=(1, max_val),
    xlabel="True Rank", ylabel="Predicted Rank",
    title=f"True vs Predicted Rank\n{Path(f).stem}")
ax.legend(frameon=False)

save_path = os.path.join(script_dir, f"true_vs_pred_sim.png")
fig.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved plot to {save_path}")

exit()