import pandas as pd
import glob
import os
import math
import numpy as np
import re

input_dir = "./dispatch_tuner/tuning_database"
output_dir = "./dispatch_tuner/tuning_database_clean"
os.makedirs(output_dir, exist_ok=True)
files = glob.glob(os.path.join(input_dir, "*.csv"))

FP_BYTEWIDTH = 2
CU = 304
LDS=65536

mma_attr_map = {
    "#iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>": 0,
    "#iree_gpu.mma_layout<MFMA_F32_16x16x32_F16>": 1,
    "#iree_gpu.mma_layout<MFMA_F32_32x32x16_F16>": 2,
    "#iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>": 3,
}
MMA_ATTR_UNSEEN = 4
mma_dims_map = {
    "#iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>": (16, 16, 16),
    "#iree_gpu.mma_layout<MFMA_F32_16x16x32_F16>": (16, 16, 32),
    "#iree_gpu.mma_layout<MFMA_F32_32x32x16_F16>": (32, 32, 16),
    "#iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>" : (32, 32,  8),
}

# Check how many different mma_attr cross all CSVs
# vals = set()
# for f in files:
#     vals |= set(pd.read_csv(f, usecols=['cfg.mma_attr'])['cfg.mma_attr'].dropna().astype(str).unique())
# print(f"{len(vals)} unique values in cfg.mma_attr:")
# print(*sorted(vals), sep="\n")
# exit()

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

    # Encode mma_attr from strings to class IDs
    df["cfg.mma_attr_map"] = (
        df["cfg.mma_attr"].astype(str)
        .map(lambda s: mma_attr_map.get(s, MMA_ATTR_UNSEEN))
        .astype(int)
    )
    # pattern = re.compile(r'_(\d+)x(\d+)x(\d+)_')
    # def extract_shape(val):
    #     if pd.isna(val):
    #         return (None, None, None)
    #     m = pattern.search(str(val))
    #     if m:
    #         return tuple(map(int, m.groups()))
    #     return (None, None, None)
    # df[["cfg.mma_a", "cfg.mma_b", "cfg.mma_c"]] = df["cfg.mma_attr"].apply(extract_shape).apply(pd.Series)

    # Indicators
    cols = ["cfg.m", "cfg.n", "cfg.k", "cfg.num_subgroups"]
    for col in cols:
        # convert to numeric, invalid → NaN → <NA> when cast to Int64
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    # if tile size is power of 2
    df["m_pow2"] = ((df["cfg.m"] > 0) & ((df["cfg.m"] & (df["cfg.m"] - 1)) == 0)).astype(int)
    df["n_pow2"] = ((df["cfg.n"] > 0) & ((df["cfg.n"] & (df["cfg.n"] - 1)) == 0)).astype(int)
    df["k_pow2"] = ((df["cfg.k"] > 0) & ((df["cfg.k"] & (df["cfg.k"] - 1)) == 0)).astype(int)
    # Tilze size perfect square check
    df["m_square"] = (np.sqrt(df["cfg.m"]).astype(int) ** 2 == df["cfg.m"]).astype(int)
    df["n_square"] = (np.sqrt(df["cfg.n"]).astype(int) ** 2 == df["cfg.n"]).astype(int)
    df["k_square"] = (np.sqrt(df["cfg.k"]).astype(int) ** 2 == df["cfg.k"]).astype(int)
    # Tile size perfect cube check
    df["m_cube"] = (np.round(df["cfg.m"] ** (1/3)) ** 3 == df["cfg.m"]).astype(int)
    df["n_cube"] = (np.round(df["cfg.n"] ** (1/3)) ** 3 == df["cfg.n"]).astype(int)
    df["k_cube"] = (np.round(df["cfg.k"] ** (1/3)) ** 3 == df["cfg.k"]).astype(int)
    # num of subgroups is a multiple of 4 (number of SIMDs in a CU)
    df["num_subgroups_mult4"] = (df["cfg.num_subgroups"] % 4 == 0).astype(int)


    out_path = os.path.join(output_dir, os.path.basename(f))
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")