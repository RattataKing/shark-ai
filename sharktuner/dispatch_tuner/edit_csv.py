import pandas as pd
import glob
import os
import math
import numpy as np

files = glob.glob("./dispatch_tuner/tuning_database/*.csv")

FP_BYTEWIDTH = 2
CU = 304
LDS=65536

for f in files:
    df = pd.read_csv(f)

    speedup = df["benchmark_speedup"]
    missing_mask = speedup.isna()
    min_val = speedup[~missing_mask].min()
    max_val = speedup[~missing_mask].max()
    eps=1e-6
    normed = (speedup - min_val) / (max_val - min_val + eps)
    normed *= (1 - eps)     # shrink top so max < 1
    normed[missing_mask] = 1.0  # assign missing to 1.0
    df["norm_speedup"] = normed

    # Save back to the same file (overwrite)
    df.to_csv(f, index=False)

    print(f"Updated: {os.path.basename(f)}")

    continue

    if (
        (
            ((df["cfg.M"] % df["cfg.workgroup_tile_size_x"]) != 0) |
            ((df["cfg.N"] % df["cfg.workgroup_tile_size_y"]) != 0) |
            ((df["cfg.K"] % df["cfg.reduction_tile_size_3"]) != 0)
        ) & (df["benchmark_status"] == True)
    ).any():
        print("HELLO")
    else:
        continue

    old_len = len(df)
    df = df[
        ((df["cfg.M"] % df["cfg.workgroup_tile_size_x"]) == 0) &
        ((df["cfg.N"] % df["cfg.workgroup_tile_size_y"]) == 0) &
        ((df["cfg.K"] % df["cfg.reduction_tile_size_3"]) == 0)
    ]
    df = df[df["candidate_id"] != 0]
    print(f"Before: {old_len} rows, After dropna: {len(df)} rows")

    # exit()

    # WG = M/m * N/n
    df["cfg.WG"] = (
        (df["cfg.M"] / df["cfg.workgroup_tile_size_x"]) *
        (df["cfg.N"] / df["cfg.workgroup_tile_size_y"])
    )



    # num subgroups
    df["cfg.num_subgroups"] = df["cfg.sg_m_cnt"] * df["cfg.sg_n_cnt"]
    # quantization Inefficency = [ceil(WG/CU) - WG/CU] / ceil(WG/CU), ~0 is good
    df["cfg.quantization_inefficiency"] = (np.ceil(df["cfg.WG"]/CU) - df["cfg.WG"]/CU) / np.ceil(df["cfg.WG"]/CU)
    # lhs_tile_size = m * k
    df["cfg.lhs_tile_size"] = df["cfg.workgroup_tile_size_x"] * df["cfg.reduction_tile_size_3"]
    # rhs_tile_size = n * k
    df["cfg.rhs_tile_size"] = df["cfg.workgroup_tile_size_y"] * df["cfg.reduction_tile_size_3"]
    # flat_wg_size
    df["cfg.flat_wg_size"] = df["cfg.wg_x"] * df["cfg.wg_y"] * df["cfg.wg_z"]
    # lds utilization = (FP16 bytewidth * m * k + FP16 bytewidth * n * k) / LDS mem size
    df["cfg.lds_utilization"] = (
        (FP_BYTEWIDTH * df["cfg.workgroup_tile_size_x"] * df["cfg.reduction_tile_size_3"]) + 
        (FP_BYTEWIDTH * df["cfg.workgroup_tile_size_y"] * df["cfg.reduction_tile_size_3"])
    ) / LDS
    
    # Norm speedup, speedup < 1 means faster than baseline
    # --- Robust per-CSV normalization (handles failures / missing speedup) ---
    FAIL_ALPHA = 1.10     # failure penalty must be > any successful norm by this factor
    FAIL_MIN   = 5.0      # minimum penalty for failures (relative to best)
    success = df["benchmark_status"] == True
    # -------- norm_speedup (>=1): successes = speedup / best_success; failures = penalty --------
    best = df.loc[success, "benchmark_speedup"].min()
    df.loc[success, "norm_speedup"] = df.loc[success, "benchmark_speedup"] / best
    worst_success = df.loc[success, "norm_speedup"].max()
    penalty = max(FAIL_MIN, worst_success * FAIL_ALPHA)
    df.loc[~success, "norm_speedup"] = penalty
    
    # -------- norm_rank in [0,1]: successes ranked by speedup (lower is better); failures tie at worst --------
    max_rank = int(success.sum())
    if max_rank >= 2:
        ranks = df.loc[success, "benchmark_speedup"].rank(method="min", ascending=True)
        df.loc[success, "norm_rank"] = (ranks - 1) / (max_rank - 1) # 0..1 among successes
        df.loc[~success, "norm_rank"] = 1.0  # failures at worst
    elif K == 1:
        df.loc[success, "norm_rank"] = 0.0
        df.loc[~success, "norm_rank"] = 1.0
    else:
        df["norm_rank"] = 1.0  # all failed → all worst

    # Save back to the same file (overwrite)
    df.to_csv(f, index=False)

    print(f"Updated: {os.path.basename(f)}")