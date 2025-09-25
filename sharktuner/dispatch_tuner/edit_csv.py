import pandas as pd
import glob
import os
import math
import numpy as np

input_dir = "./dispatch_tuner/tuning_database"
output_dir = "./dispatch_tuner/tuning_database_clean"
os.makedirs(output_dir, exist_ok=True)
files = glob.glob(os.path.join(input_dir, "*.csv"))

FP_BYTEWIDTH = 2
CU = 304
LDS=65536

for f in files:
    df = pd.read_csv(f)

    # Drop invalid candidates and baseline
    old_len = len(df)
    df = df[
        ((df["cfg.M"] % df["cfg.m"]) == 0) &
        ((df["cfg.N"] % df["cfg.n"]) == 0) &
        ((df["cfg.K"] % df["cfg.k"]) == 0)
    ]
    df = df[df["candidate_id"] != 0]
    print(f"Before: {old_len} rows, After drop: {len(df)} rows")

    # Add engineered features
    # WG = M/m * N/n
    df["cfg.WG"] = (
        (df["cfg.M"] / df["cfg.m"]) *
        (df["cfg.N"] / df["cfg.n"])
    )
    # num subgroups
    df["cfg.num_subgroups"] = df["cfg.sg_m_cnt"] * df["cfg.sg_n_cnt"]
    # quantization Inefficency = [ceil(WG/CU) - WG/CU] / ceil(WG/CU), ~0 is good
    df["cfg.quantization_inefficiency"] = (np.ceil(df["cfg.WG"]/CU) - df["cfg.WG"]/CU) / np.ceil(df["cfg.WG"]/CU)
    # lhs_tile_size = m * k
    df["cfg.lhs_tile_size"] = df["cfg.m"] * df["cfg.k"]
    # rhs_tile_size = n * k
    df["cfg.rhs_tile_size"] = df["cfg.n"] * df["cfg.k"]
    # flat_wg_size
    df["cfg.flat_wg_size"] = df["cfg.wg_x"] * df["cfg.wg_y"] * df["cfg.wg_z"]
    # lds utilization = (FP16 bytewidth * m * k + FP16 bytewidth * n * k) / LDS mem size
    df["cfg.lds_utilization"] = (
        (FP_BYTEWIDTH * df["cfg.m"] * df["cfg.k"]) + 
        (FP_BYTEWIDTH * df["cfg.n"] * df["cfg.k"])
    ) / LDS

    # Normalize speedup to [0,1]: 0 for the best benchmarked candidate, 1 for failures
    speedup = df["benchmark_speedup"]
    missing_mask = speedup.isna()
    min_val = speedup[~missing_mask].min()
    max_val = speedup[~missing_mask].max()
    eps=1e-6
    normed = (speedup - min_val) / (max_val - min_val + eps)
    normed *= (1 - eps)     # shrink top so max < 1
    normed[missing_mask] = 1  # assign missing to 1.0
    df["norm_speedup"] = normed

    out_path = os.path.join(output_dir, os.path.basename(f))
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")