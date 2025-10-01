import os, re, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------- load all CSVs ---------
files = glob.glob('./dispatch_tuner/tuning_database_clean/*.csv')
dfs = []
for f in files:
    try:
        dfs.append(pd.read_csv(f))
    except Exception as e:
        print(f"Error reading {f}: {e}")

if not dfs:
    raise SystemExit("No CSVs found.")

full_df = pd.concat(dfs, ignore_index=True)

# --------- helpers ---------
def safe_name(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(s))

def make_speed_bins(df: pd.DataFrame, speed_col: str, n_bins: int = 10) -> pd.Categorical:
    s = df[speed_col].astype(float)
    if s.empty or np.isclose(s.max(), s.min()):
        raise ValueError(f"{speed_col} has no variation or is empty")
    try:
        return pd.qcut(s, q=n_bins, duplicates="drop")
    except Exception:
        edges = np.linspace(s.min(), s.max(), n_bins + 1)
        return pd.cut(s, bins=edges, include_lowest=True)

def plot_feature_distribution(
    df: pd.DataFrame,
    speed_col: str,
    feature_col: str,
    n_bins: int = 10,
    top_k: int = 10,
    out_dir: str = "plots_feature_x"
):
    """
    One plot for this feature:
      - X-axis = feature values (top_k only, rest dropped or aggregated)
      - Y-axis = speed bins
      - Each column normalized to 100% (distribution of that feature value across bins)
    """
    os.makedirs(out_dir, exist_ok=True)

    if speed_col not in df.columns or feature_col not in df.columns:
        print(f"[skip] missing column: {feature_col}")
        return

    sub = df[[speed_col, feature_col]].dropna().copy()
    if sub.empty:
        print(f"[skip] {feature_col}: no data")
        return

    sub["speed_bin"] = make_speed_bins(sub, speed_col, n_bins=n_bins)
    sub[feature_col] = sub[feature_col].astype(str)

    # choose top_k feature values
    vc = sub[feature_col].value_counts()
    chosen = list(vc.head(top_k).index)
    sub = sub[sub[feature_col].isin(chosen)].copy()

    # counts per (bin, feature-value)
    counts = (
        sub.groupby(["speed_bin", feature_col], observed=True)
           .size()
           .unstack(feature_col, fill_value=0)
    )
    counts = counts.reindex(index=sub["speed_bin"].cat.categories, columns=chosen, fill_value=0)

    # normalize columns to 100%
    col_sums = counts.sum(axis=0).replace(0, np.nan)
    perc = counts.divide(col_sums, axis=1) * 100.0
    perc = perc.fillna(0.0)

    # save tidy CSV
    tidy = (
        counts.stack().rename("count").to_frame()
        .join(perc.stack().rename("percent_col_norm"))
        .reset_index()
        .rename(columns={"speed_bin": "speed_bin_interval", feature_col: "feature_value"})
    )
    base = f"heatmap_{safe_name(feature_col)}"
    csv_path = os.path.join(out_dir, base + ".csv")
    tidy.to_csv(csv_path, index=False)

    # --- Plot heatmap ---
    fig, ax = plt.subplots(figsize=(max(8, len(chosen)*0.8), max(5, n_bins*0.5)))
    im = ax.imshow(perc.values, aspect='auto')

    ax.set_xticks(range(len(chosen)))
    ax.set_xticklabels(chosen, rotation=45, ha="right")
    ax.set_yticks(range(len(perc.index)))
    ax.set_yticklabels([str(c) for c in perc.index])

    ax.set_xlabel(feature_col)
    ax.set_ylabel(f"{speed_col} bins")
    ax.set_title(f"Distribution across {speed_col} bins for {feature_col} values")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("% within column")

    ax.set_xticks(np.arange(-.5, len(chosen), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(perc.index), 1), minor=True)
    ax.grid(which="minor", linestyle="--", alpha=0.25)
    ax.tick_params(which="minor", bottom=False, left=False)

    png_path = os.path.join(out_dir, base + ".png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {png_path}, {csv_path}")

# --------- run for your features ---------
SPEED_COL = "norm_speedup"
features = [
    "cfg.intrinsic_k","k_pow2","cfg.mma_attr","cfg.lhs_type_bitwidth","n_pow2",
    "cfg.num_subgroups","cfg.rhs_tile_size","cfg.intrinsic_mn","cfg.m","cfg.n",
    "num_subgroups_mult4","cfg.sg_n_cnt","cfg.k","cfg.sg_m_cnt","cfg.lds_utilization",
    "n_square","cfg.N","cfg.M","cfg.quantization_inefficiency","k_square",
    "cfg.lhs_tile_size","m_square","m_cube","cfg.K","n_cube","k_cube"
]

for feat in features:
    plot_feature_distribution(full_df, speed_col=SPEED_COL, feature_col=feat, n_bins=10, top_k=10)
