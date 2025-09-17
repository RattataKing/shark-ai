import os
import pandas as pd
import glob
from pathlib import Path
import random

input_csv = Path('./dispatch_tuner/tuning_database/tuning_compute_gemm_4096_4096_8192_f16_f32_tB.csv')
df = pd.read_csv(input_csv)
df = df.copy().iloc[1:] # Skip candidate 0, the baseline

WINNER_TOP_RANK = 10
SORT_TOP_RANK = 20

winners = (
    df[df["benchmark_result_order"] <= WINNER_TOP_RANK]
    .sort_values("benchmark_result_order")["candidate_id"]
    .tolist()
)
print("Real Winner Candidates:")
print(winners)

# build key function for sorting
by_cid = df.set_index("candidate_id")

# sort() function example
def candidate_priority(cid: int):
    ### Prefer greater cfg.workgroup_tile_size_x value here
    row = by_cid.loc[cid]
    wg_x = row["cfg.workgroup_tile_size_x"]
    # negative so bigger values come first; None/NaN sorted last
    return (pd.isna(wg_x), -wg_x if not pd.isna(wg_x) else None)

# get list of candidates
candidate_list = df["candidate_id"].tolist()
candidate_list.sort(key=candidate_priority)

perf_cols = ["benchmark_result_order", "benchmark_time_ms", "benchmark_speedup"]

print(f"\nWith sort(), first {SORT_TOP_RANK} candidates:")
sorted_top = candidate_list[:SORT_TOP_RANK]
print(by_cid.loc[sorted_top, perf_cols])

# ---- evaluation ----
# 1) hit rate: how many true winners appear in sorted first 10
hits = len(set(sorted_top) & set(winners))
print(f"\nWinner hit rate in first {SORT_TOP_RANK} sorted: {hits}/{len(sorted_top)} = {hits/len(sorted_top):.2%}")

# 2) benchmark_status rate in sorted first 10
bench_true = by_cid.loc[sorted_top, "benchmark_status"].fillna(False).astype(bool)
count_true = bench_true.sum()
benchmark_rate = count_true / len(sorted_top)
print(f"Benchmark success rate in first {SORT_TOP_RANK} sorted: {count_true}/{len(sorted_top)} = {benchmark_rate:.2%}")


# 3) rate: how many in sorted top have benchmark_speedup < 1
if "benchmark_speedup" in by_cid.columns:
    speedup_mask = by_cid.loc[sorted_top, "benchmark_speedup"] < 1
    slow_rate = speedup_mask.mean()
    count_slow = speedup_mask.sum()
    print(f"Benchmark_speedup < 1 in first {SORT_TOP_RANK} sorted: {count_slow}/{len(sorted_top)} = {slow_rate:.2%}")
else:
    print("Column 'benchmark_speedup' not found, skipped check.")

# 4) positions of true winners in the sorted list
rank_by_cid = {cid: i + 1 for i, cid in enumerate(candidate_list)}
print("\nPositions of true winners in sorted list:")
for cid in winners:  # winners are ordered by real benchmark_result_order
    sorted_order = rank_by_cid.get(cid)
    real_rank = int(by_cid.loc[cid, "benchmark_result_order"]) if cid in by_cid.index else None
    print(f"candidate_id: {cid}, sorted order: {sorted_order}, real rank: {real_rank}")