import os, sys
import pandas as pd
import glob
from pathlib import Path
import random
import numpy as np

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <input_csv>")
    sys.exit(1)

input_csv_dir = Path(sys.argv[1])
files = glob.glob(str(input_csv_dir / "*.csv"))
if not files:
    print("No CSVs found.")
    sys.exit(1)

dfs = []
for f in files:
    df = pd.read_csv(f)
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

# Drop rows where candidate_id == 0
df = df[df["candidate_id"] != 0]

# Sanity check
if "dispatch_id" not in df.columns:
    print("ERROR: Column 'dispatch_id' not found in input CSVs.")
    sys.exit(1)

WINNER_TOP_RANK = 10
SORT_TOP_RANK = 10
MAX_DISPATCH_SEARCH = 0 # set to None or 0 for no limit
OPTIMAL_THR_RATIO=0.95 # e.g. 0.95-> 95% of optimal, optimal time * 1.05

perf_cols = ["benchmark_result_order", "benchmark_time_ms", "benchmark_speedup"]

# For aggregate stats across dispatches
agg_hits = []
agg_bench_true_rates = []
agg_fast_rates = []
num_dispatches = 0
agg_optimal_gap_pct = []
agg_thr_gap_pct = []

# Iterate per dispatch_id
for dispatch_idx, (dispatch_id, g) in enumerate(df.groupby("dispatch_id", sort=True), start=1):
    if MAX_DISPATCH_SEARCH and dispatch_idx > MAX_DISPATCH_SEARCH:
        break

    print("\n" + "=" * 80)
    print(f"Dispatch: {dispatch_id} ({dispatch_idx}/{len(df['dispatch_id'].unique())})")

    # winners within this dispatch (based on raw group)
    winners = (
        g[g["benchmark_result_order"] <= WINNER_TOP_RANK]
        .sort_values("benchmark_result_order")["candidate_id"]
        .tolist()
    )
    print("Real Winner Candidates:")
    print(winners)

    # ---- pick exactly one row per candidate_id within this dispatch ----
    rep = (
        g.sort_values(["benchmark_result_order", "candidate_id"], na_position="last")
         .drop_duplicates(subset="candidate_id", keep="first")
    )

    # index by candidate_id (now unique)
    by_cid = rep.set_index("candidate_id")

    # sorting key: prefer greater cfg.workgroup_tile_size_x; NaN last
    def candidate_priority(cid: int):
        row = by_cid.loc[cid]
        M=row.get("cfg.M")
        N=row.get("cfg.N")
        m=row.get("cfg.workgroup_tile_size_x")
        n=row.get("cfg.workgroup_tile_size_y")
        k=row.get("cfg.reduction_tile_size_3")
        wg = (M / m) * (N / n)
        q_ie = float(np.ceil(wg / 304.0) - (wg / 304.0) / np.ceil(wg / 304.0))
        shared_mem = (2 * m * k) + (2 * n * k)
        shared_mem_ratio = float(shared_mem / 65536)

        # Heuristic: lower q_ie is better; give bonus if shared_mem is 8KB-aligned and ratio in [0.3, 0.8]
        aligned = (shared_mem % (8 * 1024) == 0)
        ratio_ok = (0.3 <= shared_mem_ratio <= 0.8)

        # final score (lower sorts earlier)
        score = q_ie * 100
        # if not aligned:
        #     score += 20
        # if not ratio_ok:
        #     score += 10
        # # tie-breakers: prefer larger tiles (m, then n)
        return (False, score, -float(m), -float(n))

    # sort candidate list for this dispatch
    candidate_list = rep["candidate_id"].tolist()
    candidate_list.sort(key=candidate_priority)

    # show first SORT_TOP_RANK rows for this dispatch
    sorted_top = candidate_list[:SORT_TOP_RANK]
    cols_present = [c for c in perf_cols if c in by_cid.columns]
    if cols_present:
        table = by_cid.loc[sorted_top, cols_present].copy()
        table["benchmark_result_order"] = table["benchmark_result_order"].astype("Int64")
        print(f"\nWith sort(), first {len(sorted_top)} candidates:")
        print(table)
    else:
        print("\nPerformance columns not present to display.")

    # ---- evaluation (within dispatch) ----
    # 1) hit rate: how many true winners appear in first SORT_TOP_RANK
    hits = len(set(sorted_top) & set(winners))
    hit_rate = hits / max(1, len(sorted_top))
    print(f"\nWinner hit rate in first {len(sorted_top)} sorted: {hits}/{len(sorted_top)} = {hit_rate:.2%}")

    # 2) benchmark_status rate in first SORT_TOP_RANK
    bench_true = by_cid.loc[sorted_top, "benchmark_status"].fillna(False).astype(bool)
    count_true = int(bench_true.sum())
    benchmark_rate = count_true / max(1, len(sorted_top))
    print(f"Benchmark success rate in first {len(sorted_top)} sorted: {count_true}/{len(sorted_top)} = {benchmark_rate:.2%}")

    # 3) fraction with benchmark_speedup < 1 in first SORT_TOP_RANK
    speedup_mask = (by_cid.loc[sorted_top, "benchmark_speedup"] < 1).fillna(False)
    count_fast = int(speedup_mask.sum())
    fast_rate = count_fast / max(1, len(sorted_top))
    print(f"Benchmark_speedup < 1 in first {len(sorted_top)} sorted: {count_fast}/{len(sorted_top)} = {fast_rate:.2%}")

    # 4) positions of true winners in the sorted list (within this dispatch)
    rank_by_cid = {cid: i + 1 for i, cid in enumerate(candidate_list)}
    print("\nPositions of true winners in sorted list:")
    for cid in winners:
        sorted_order = rank_by_cid.get(cid)
        real_rank = int(by_cid.loc[cid, "benchmark_result_order"]) if cid in by_cid.index else None
        print(f"candidate_id: {cid}, sorted order: {sorted_order}, real rank: {real_rank}")

    # 5) distance of best candidate benchmark result from the optimal benchmark time
    best_sorted_time = pd.to_numeric(
        by_cid.loc[sorted_top, "benchmark_time_ms"], errors="coerce"
    ).min()
    optimal_time = pd.to_numeric(
        g["benchmark_optimal_time_ms"], errors="coerce"
    ).dropna().min()

    if pd.isna(best_sorted_time) or pd.isna(optimal_time) or optimal_time == 0:
        # fallback: treat as worst-case
        distance_ms = np.nan
        distance_pct = 1.0   # 100%
        thr_optimal_time = np.nan
        thr_distance_pct = 1.0
    else:
        distance_ms = best_sorted_time - optimal_time
        distance_pct = distance_ms / optimal_time
        thr_optimal_time = optimal_time * (2 - OPTIMAL_THR_RATIO)
        thr_distance_pct = (best_sorted_time - thr_optimal_time) / thr_optimal_time

    print(f"\nBest top benchmark vs optimal:\n"
        f"best = {best_sorted_time if not pd.isna(best_sorted_time) else 'n/a'} ms, "
        f"optimal = {optimal_time if not pd.isna(optimal_time) else 'n/a'} ms, "
        f"gap = {distance_ms if not pd.isna(distance_ms) else 'n/a'} ms "
        f"({distance_pct:.2%})")

    print(f"Best top benchmark vs {OPTIMAL_THR_RATIO*100}% optimal:\n"
        f"best = {best_sorted_time if not pd.isna(best_sorted_time) else 'n/a'} ms, "
        f"{OPTIMAL_THR_RATIO*100}% optimal = {thr_optimal_time if not pd.isna(thr_optimal_time) else 'n/a'} ms, "
        f"gap = {(best_sorted_time - thr_optimal_time) if not (pd.isna(best_sorted_time) or pd.isna(thr_optimal_time)) else 'n/a'} ms "
        f"({thr_distance_pct:.2%})")


    # collect aggregates
    num_dispatches += 1
    agg_hits.append(hit_rate)
    agg_bench_true_rates.append(benchmark_rate)
    agg_fast_rates.append(fast_rate)
    agg_optimal_gap_pct.append(distance_pct)
    agg_thr_gap_pct.append(thr_distance_pct)

# ---- simple aggregate view across dispatches ----
print("\n" + "#" * 80)
print("Aggregate (macro) averages across dispatches:")
if num_dispatches > 0:
    if agg_hits:
        print(f"- Avg winner hit rate@{SORT_TOP_RANK}: {sum(agg_hits)/len(agg_hits):.2%} (over {len(agg_hits)} dispatches)")
    if agg_bench_true_rates:
        print(f"- Avg benchmark success rate@{SORT_TOP_RANK}: {sum(agg_bench_true_rates)/len(agg_bench_true_rates):.2%} (over {len(agg_bench_true_rates)} dispatches)")
    if agg_fast_rates:
        print(f"- Avg fast (speedup<1) rate@{SORT_TOP_RANK}: {sum(agg_fast_rates)/len(agg_fast_rates):.2%} (over {len(agg_fast_rates)} dispatches)")
    if agg_optimal_gap_pct:
        print(f"- Avg best-top gap vs optimal: {sum(agg_optimal_gap_pct)/len(agg_optimal_gap_pct):.2%}")
    if agg_thr_gap_pct:
        print(f"- Avg best-top gap vs {OPTIMAL_THR_RATIO}optimal: {sum(agg_thr_gap_pct)/len(agg_thr_gap_pct):.2%}")
else:
    print("No dispatches found.")
