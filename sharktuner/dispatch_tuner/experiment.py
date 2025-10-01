import os, re, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------- load all CSVs -------------
files = glob.glob('./dispatch_tuner/tuning_database_clean/*.csv')
dfs = []
for f in files:
    try:
        df = pd.read_csv(f)
        dfs.append(df)
    except Exception as e:
        print(f"Error reading {f}: {e}")
full_df = pd.concat(dfs, ignore_index=True)

# ------------- helpers -------------
def safe_name(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', s)

def plot_feature_share_by_speed_bins(
    df: pd.DataFrame,
    speed_col: str,
    feature_col: str,
    n_bins: int = 10,
    top_k: int = 8,
    out_dir: str = "plots"
):
    os.makedirs(out_dir, exist_ok=True)

    sub = df[[speed_col, feature_col]].dropna().copy()
    if sub.empty:
        print(f"[skip] {feature_col}: no data")
        return

    sub[feature_col] = sub[feature_col].astype(str)

    # equal-width bins on speed_col
    mn, mx = sub[speed_col].min(), sub[speed_col].max()
    if np.isclose(mx, mn):
        print(f"[skip] {feature_col}: {speed_col} has no variation")
        return
    bins = np.linspace(mn, mx, n_bins + 1)
    # sub["speed_bin"] = pd.cut(sub[speed_col], bins=bins, include_lowest=True) # same width
    sub["speed_bin"] = pd.qcut(sub[speed_col], q=n_bins, duplicates="drop") # same count

    # % share per bin
    share = (
        sub.groupby(["speed_bin", feature_col], observed=True)
           .size()
           .groupby(level=0, observed=True)
           .apply(lambda s: s / s.sum() * 100.0)
           .unstack(fill_value=0.0)
    )

    # keep top_k classes overall; aggregate the rest
    overall = sub[feature_col].value_counts()
    top_classes = list(overall.head(top_k).index)
    others = [c for c in share.columns if c not in top_classes]
    if others:
        share["Others"] = share[others].sum(axis=1)
        share = share[top_classes + ["Others"]]

    # turn index into simple bin numbers 0..n_bins-1
    plot_df = share.reset_index(drop=True)
    plot_df.insert(0, "bin_idx", range(len(plot_df)))

    # plot
    plt.figure(figsize=(10, 6))
    x = plot_df["bin_idx"]
    for col in plot_df.columns:
        if col == "bin_idx":
            continue
        plt.plot(x, plot_df[col], marker="o", linewidth=1, label=str(col))

    plt.xticks(range(len(plot_df)))
    plt.ylabel("% within speed bin")
    plt.xlabel(f"{speed_col} bin index (0 to {len(plot_df)-1})")
    plt.title(f"Class share of '{feature_col}' across {speed_col} bins (n={n_bins})")
    plt.legend(title="Class", fontsize=8, title_fontsize=9, ncol=2)
    plt.grid(True, linestyle="--", alpha=0.4)

    save_path = os.path.join(out_dir, f"share_{safe_name(feature_col)}_by_{safe_name(speed_col)}_bins.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ------------- run for your features -------------
features = [
    "cfg.intrinsic_k","k_pow2","cfg.mma_attr","cfg.lhs_type_bitwidth","n_pow2",
    "cfg.num_subgroups","cfg.rhs_tile_size","cfg.intrinsic_mn","cfg.m","cfg.n",
    "num_subgroups_mult4","cfg.sg_n_cnt","cfg.k","cfg.sg_m_cnt","cfg.lds_utilization",
    "n_square","cfg.N","cfg.M","cfg.quantization_inefficiency","k_square",
    "cfg.lhs_tile_size","m_square","m_cube","cfg.K","n_cube","k_cube"
]

os.makedirs("plots", exist_ok=True)
for feat in features:
    if feat in full_df.columns:
        plot_feature_share_by_speed_bins(
            full_df, speed_col="norm_speedup", feature_col=feat,
            n_bins=10, top_k=8, out_dir="plots"
        )
    else:
        print(f"[warn] column not found: {feat}")
