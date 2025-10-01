#!/usr/bin/env python3
# build_features_and_extract_rules.py

import os, glob, math, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# ------------------------------- CONFIG ---------------------------------------

INPUT_DIR  = "./dispatch_tuner/tuning_database"
CLEAN_DIR  = "./dispatch_tuner/tuning_database_exp"
os.makedirs(CLEAN_DIR, exist_ok=True)

# HW-ish assumptions (tweak if needed)
CU_COUNT        = 304            # total CUs on device
LDS_BYTES       = 64 * 1024
WAVE_SIZE       = 64
OUT_BYTEWIDTH   = 4              # C writeback bytes (f32)
DEFAULT_LHS_BW  = 16             # bits for A if missing
DEFAULT_RHS_BW  = 16             # bits for B if missing

# L1 strength for sparse rule set
LOGIT_C = 1.0                    # larger -> less sparse; try 0.3..3.0
MAX_ITER = 4000

# ----------------------------- UTIL HELPERS -----------------------------------

# def  (x, dtype="float64"):
#     """Return a numeric array/Series with dtype, accepting scalars/ndarray/Series/Index."""
#     if isinstance(x, (pd.Series, pd.Index)):
#         return pd.to_numeric(x, errors="coerce").astype(dtype)
#     if isinstance(x, np.ndarray):
#         return x.astype(dtype, copy=False)
#     # scalar or other python object
#     return np.asarray(x, dtype=dtype)

def _safe_int(x):
    if isinstance(x, (pd.Series, pd.Index)):
        return pd.to_numeric(x, errors="coerce").astype("Int64")
    if isinstance(x, np.ndarray):
        return x.astype(np.int64, copy=False)
    return np.asarray(x, dtype=np.int64)

def _is_pow2(x):
    """Return 0/1 (same index as input if Series/Index)."""
    if isinstance(x, (pd.Series, pd.Index)):
        v = pd.to_numeric(x, errors="coerce").fillna(0).to_numpy(dtype=np.int64, copy=False)
        out = ((v > 0) & ((v & (v - 1)) == 0)).astype(int)
        return pd.Series(out, index=x.index)
    v = np.asarray(x, dtype=np.int64)
    return ((v > 0) & ((v & (v - 1)) == 0)).astype(int)

def _pow2_gap(x):
    """|x - nearest power of two| (preserve index if input is Series/Index)."""
    if isinstance(x, (pd.Series, pd.Index)):
        v = pd.to_numeric(x, errors="coerce").fillna(0).to_numpy(dtype=np.int64, copy=False)
        v_clip = np.where(v < 1, 1, v)
        lg = np.rint(np.log2(v_clip)).astype(np.int64)
        p2 = np.left_shift(np.ones_like(lg, dtype=np.int64), lg)  # elementwise 1<<lg
        gap = np.abs(v - p2)
        return pd.Series(gap, index=x.index)
    # scalar / ndarray path
    v = np.asarray(x, dtype=np.int64)
    v_clip = np.where(v < 1, 1, v)
    lg = np.rint(np.log2(v_clip)).astype(np.int64)
    p2 = np.left_shift(np.ones_like(lg, dtype=np.int64), lg)
    return np.abs(v - p2)

def _ceil_div(a, b):
    a =  (a)
    b =  (b)
    # replace zeros with NaN in a vectorized way for either Series/ndarray/scalar
    if isinstance(b, (pd.Series, pd.Index)):
        b = b.replace(0, np.nan)
    else:
        b = np.where(b == 0, np.nan, b)
    return np.ceil(a / b)

def _mod0(a, b):
    a =  (a)
    b =  (b)
    if isinstance(b, (pd.Series, pd.Index)):
        b2 = b.replace(0, np.nan)
    else:
        b2 = np.where(b == 0, np.nan, b)
    return (np.mod(a, b2) == 0).astype(int)

def _nearest_gap(x, base):
    x =  (x)
    base =  (base)
    if isinstance(base, (pd.Series, pd.Index)):
        base = base.replace(0, np.nan)
    else:
        base = np.where(base == 0, np.nan, base)
    mod = np.mod(x, base)
    return np.minimum(mod, base - mod)

def normalize_speedup(series: pd.Series) -> pd.Series:
    """Map to [0,1]: 0 = best (min), 1 = failures/missing."""
    s = series.copy()
    missing = s.isna()
    if (~missing).sum() == 0:
        return pd.Series(np.ones(len(s)), index=s.index)
    mn = s[~missing].min()
    mx = s[~missing].max()
    eps = 1e-6
    norm = (s - mn) / (mx - mn + eps)
    norm *= (1 - eps)
    norm[missing] = 1.0
    return norm

# ------------------------- FEATURE ENGINEERING --------------------------------

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interpretable, ranking-friendly features given your base schema."""
    # Cast core ints
    int_cols = ["cfg.M","cfg.N","cfg.K","cfg.m","cfg.n","cfg.k",
                "cfg.wg_x","cfg.wg_y","cfg.wg_z",
                "cfg.sg_m_cnt","cfg.sg_n_cnt","cfg.subgroup_size",
                "cfg.intrinsic_mn","cfg.intrinsic_k",
                "cfg.lhs_type_bitwidth","cfg.rhs_type_bitwidth"]
    for c in int_cols:
        if c in df: df[c] = _safe_int(df[c])

    # Shorthands
    M,N,K = df["cfg.M"], df["cfg.N"], df["cfg.K"]
    m,n,k = df["cfg.m"], df["cfg.n"], df["cfg.k"]
    wg_x, wg_y, wg_z = df["cfg.wg_x"], df["cfg.wg_y"], df["cfg.wg_z"]
    sg_m, sg_n = df["cfg.sg_m_cnt"], df["cfg.sg_n_cnt"]
    sgsz = df["cfg.subgroup_size"]
    intr_mn = df["cfg.intrinsic_mn"]
    intr_k  = df["cfg.intrinsic_k"]
    bw_a = df.get("cfg.lhs_type_bitwidth", pd.Series(DEFAULT_LHS_BW, index=df.index)).fillna(DEFAULT_LHS_BW)
    bw_b = df.get("cfg.rhs_type_bitwidth", pd.Series(DEFAULT_RHS_BW, index=df.index)).fillna(DEFAULT_RHS_BW)

    # WG geometry
    df["cfg.WG"] = ( (M)/ (m)) * ( (N)/ (n))
    df["cfg.num_subgroups"] =  (sg_m) *  (sg_n)
    df["cfg.quantization_inefficiency"] = (
        (np.ceil( (df["cfg.WG"])/CU_COUNT) -  (df["cfg.WG"])/CU_COUNT)
        / np.ceil( (df["cfg.WG"])/CU_COUNT)
    ).replace([np.inf,-np.inf], np.nan)

    # LDS usage (A/B staged)
    BYTES_A = ( (bw_a)/8.0) *  (m) *  (k)
    BYTES_B = ( (bw_b)/8.0) *  (n) *  (k)
    df["cfg.lds_utilization"] = (BYTES_A + BYTES_B) / float(LDS_BYTES)

    # Tile sizes & arithmetic intensity
    df["cfg.lhs_tile_size"] =  (m) *  (k)
    df["cfg.rhs_tile_size"] =  (n) *  (k)
    df["cfg.out_tile_size"] =  (m) *  (n)

    bytes_c = OUT_BYTEWIDTH *  (m) *  (n)
    df["tile_bytes_rw"] = BYTES_A + BYTES_B + bytes_c
    df["tile_flops"]    = 2.0 *  (m) *  (n) *  (k)
    df["arith_intensity"] = (df["tile_flops"] / df["tile_bytes_rw"]).replace([np.inf,-np.inf], np.nan)

    # Problem×Config tails (zero/small/tiles)
    for P,C in (("cfg.M","cfg.m"), ("cfg.N","cfg.n"), ("cfg.K","cfg.k")):
        tail = ( (df[P]) %  (df[C])).astype(float)
        df[f"{P}_tail"] = tail
        df[f"{P}_tail_zero"] = (tail == 0).astype(int)
        frac = (tail /  (df[C]).replace(0, np.nan)).astype(float)
        df[f"{P}_tail_frac"] = frac
        df[f"{P}_tail_small"] = (frac <= 0.125).astype(int)
        df[f"{P}_tiles"] = _ceil_div(df[P], df[C])

    df["perfect_tiling_all"] = (
        (df["cfg.M_tail_zero"]==1) & (df["cfg.N_tail_zero"]==1) & (df["cfg.K_tail_zero"]==1)
    ).astype(int)

    # Divisibility at common granularities
    for col in ["cfg.M","cfg.N","cfg.K","cfg.m","cfg.n","cfg.k"]:
        for b in (16,32,64,128,256):
            df[f"{col}_mod{b}_0"] = _mod0(df[col], b)

    # Pow2 predicates
    for col in ["cfg.M","cfg.N","cfg.K","cfg.m","cfg.n","cfg.k"]:
        df[f"{col}_pow2"] = _is_pow2(df[col])
        df[f"{col}_pow2_close"] = (_pow2_gap(df[col]) <= 8).astype(int)
        df[f"{col}_pow2_gap"] = _pow2_gap(df[col])

    # Intrinsic (MMA) compatibility
    df["cfg.M_mod_intrinsicMN_0"] = _mod0(M, intr_mn)
    df["cfg.N_mod_intrinsicMN_0"] = _mod0(N, intr_mn)
    df["cfg.m_mod_intrinsicMN_0"] = _mod0(m, intr_mn)
    df["cfg.n_mod_intrinsicMN_0"] = _mod0(n, intr_mn)

    df["cfg.K_mod_intrinsicK_0"]  = _mod0(K, intr_k)
    df["cfg.k_mod_intrinsicK_0"]  = _mod0(k, intr_k)

    df["cfg.K_to_intrinsicK_gap"] = _nearest_gap(K, intr_k)
    df["cfg.k_to_intrinsicK_gap"] = _nearest_gap(k, intr_k)
    df["cfg.M_to_intrinsicMN_gap"] = _nearest_gap(M, intr_mn)
    df["cfg.N_to_intrinsicMN_gap"] = _nearest_gap(N, intr_mn)
    df["cfg.m_to_intrinsicMN_gap"] = _nearest_gap(m, intr_mn)
    df["cfg.n_to_intrinsicMN_gap"] = _nearest_gap(n, intr_mn)

    # WG / waves / occupancy-ish
    wg_threads =  (wg_x) *  (wg_y) *  (wg_z) *  (df["cfg.subgroup_size"])
    df["wg_threads"] = wg_threads
    df["waves_per_wg"] = (wg_threads /  (df["cfg.subgroup_size"]).replace(0, np.nan)).replace([np.inf,-np.inf], np.nan)
    df["wg_threads_pow2"] = _is_pow2(wg_threads)

    grid_x =  (M)/ (m); grid_y =  (N)/ (n)
    df["grid_x"] = grid_x; df["grid_y"] = grid_y; df["grid_size"] = grid_x * grid_y
    df["wg_per_cu"] = (df["grid_size"] / max(CU_COUNT,1)).astype(float)
    df["wg_per_cu_frac"] = df["wg_per_cu"] - np.floor(df["wg_per_cu"])

    # Aspect ratio (tile vs problem)
    df["tile_aspect_m_over_n"] = ( (m) /  (n).replace(0, np.nan)).astype(float)
    df["prob_aspect_M_over_N"] = ( (M) /  (N).replace(0, np.nan)).astype(float)
    df["aspect_match_close"] = (
        np.abs(np.log2(
            (df["tile_aspect_m_over_n"].replace(0, np.nan)) /
            (df["prob_aspect_M_over_N"].replace(0, np.nan))
        )) <= 0.585  # ≈within 1.5×
    ).astype(int)

    # Simple vectorization proxies
    df["cfg.k_mod4_0"] = _mod0(k, 4)
    df["cfg.n_mod4_0"] = _mod0(n, 4)
    df["cfg.m_mod4_0"] = _mod0(m, 4)

    return df

# ------------------------- PAIRWISE CONSTRUCTION ------------------------------

def make_pairwise_boolean(df: pd.DataFrame, P: pd.DataFrame,
                          y_col="norm_speedup", qid_col="qid"):
    """Build (Xdiff, ysign) for pairwise training. Smaller y is better."""
    Xd, Y = [], []
    for _, g in df[[qid_col, y_col]].join(P).groupby(qid_col):
        idx = g.index.to_numpy()
        y   = g[y_col].to_numpy()
        # i better than j if y_i < y_j
        ii, jj = np.where(y[:, None] < y[None, :])
        if ii.size == 0:
            continue
        Pi, Pj = P.loc[idx[ii]].to_numpy(), P.loc[idx[jj]].to_numpy()
        Xd.append(Pi - Pj);  Y.append(np.ones(ii.size, dtype=int))   # i wins
        Xd.append(Pj - Pi);  Y.append(np.zeros(ii.size, dtype=int))  # j wins
    if not Xd:
        raise ValueError("No pairwise preferences (all ties?).")
    return np.vstack(Xd), np.concatenate(Y)

# ------------------------------- MAIN PIPE ------------------------------------

def main():
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))
    if not files:
        print(f"No CSVs found in {INPUT_DIR}")
        return

    all_frames = []
    for f in files:
        df = pd.read_csv(f)

        # Optional: drop baseline and enforce valid tiles (as in your original script)
        if "candidate_id" in df.columns:
            df = df[df["candidate_id"] != 0]
        if set(["cfg.M","cfg.N","cfg.K","cfg.m","cfg.n","cfg.k"]).issubset(df.columns):
            valid = ((df["cfg.M"] % df["cfg.m"] == 0) &
                     (df["cfg.N"] % df["cfg.n"] == 0) &
                     (df["cfg.K"] % df["cfg.k"] == 0))
            df = df[valid]

        # Add group id (qid) = per-file by default; override to dispatch_id if present
        qid = Path(f).stem
        df["qid"] = df.get("dispatch_id", qid)

        # Normalize target within the file (only order matters per qid)
        if "benchmark_speedup" in df.columns:
            df["norm_speedup"] = normalize_speedup(df["benchmark_speedup"])
        elif "norm_speedup" not in df.columns:
            raise ValueError("Need 'benchmark_speedup' or 'norm_speedup' in the CSVs.")

        # Engineer features
        df = add_engineered_features(df)

        # Save a cleaned copy
        out = os.path.join(CLEAN_DIR, os.path.basename(f))
        df.to_csv(out, index=False)
        print(f"Saved cleaned: {out} ({len(df)} rows)")

if __name__ == "__main__":
    main()
