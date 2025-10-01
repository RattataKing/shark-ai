import os, re, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------- load all CSVs -------------
files = glob.glob('./dispatch_tuner/tuning_database_clean/*.csv')
dfs = []
for f in files[:-5]:
    try:
        df = pd.read_csv(f)
        dfs.append(df)
    except Exception as e:
        print(f"Error reading {f}: {e}")
full_df = pd.concat(dfs, ignore_index=True)

import numpy as np
import pandas as pd

def fit_nb_from_bins(df, speed_col, feature_cols, n_bins=10, binning="quantile", alpha=1.0):
    # 1) make speed bins (store the labels so we can present consistent bin ordering)
    x = df[[speed_col] + feature_cols].dropna()
    if binning == "quantile":
        bins = pd.qcut(x[speed_col], q=n_bins, duplicates="drop")
    else:
        mn, mx = x[speed_col].min(), x[speed_col].max()
        edges = np.linspace(mn, mx, n_bins+1)
        bins = pd.cut(x[speed_col], bins=edges, include_lowest=True)

    x = x.assign(speed_bin=bins)
    bin_order = x["speed_bin"].cat.categories  # pd.IntervalIndex (ordered)

    # 2) priors P(bin)
    bin_counts = x["speed_bin"].value_counts().reindex(bin_order, fill_value=0)
    priors = (bin_counts / bin_counts.sum()).astype(float)
    log_priors = np.log(priors.replace(0, 1e-12))

    # 3) per-feature likelihood tables: log P(value | bin)
    feature_models = {}
    for f in feature_cols:
        # treat everything as categorical (recommend pre-bucketing large-numeric features)
        s = x[["speed_bin", f]].astype({f: "category"})
        ct = s.groupby(["speed_bin", f], observed=True).size().unstack(fill_value=0)

        # Laplace smoothing: (count + α) / (bin_total + α*|V|)
        V = ct.shape[1]
        denom = ct.sum(axis=1).values[:, None] + alpha * V
        probs = (ct.values + alpha) / denom
        log_lik = pd.DataFrame(np.log(probs), index=ct.index, columns=ct.columns)

        feature_models[f] = {
            "classes": list(ct.columns.astype(str)),
            "log_likelihood": log_lik
        }

    return {
        "speed_col": speed_col,
        "feature_cols": feature_cols,
        "bins": list(bin_order),
        "log_priors": log_priors,
        "feature_models": feature_models,
        "alpha": alpha,
    }

def predict_bin_nb(model, df_new):
    # For each row, sum log-likelihoods across available features + log prior; pick argmax
    bin_labels = model["bins"]
    log_priors = model["log_priors"]
    feats = model["feature_cols"]

    preds = []
    posteriors = []

    for _, row in df_new.iterrows():
        # start with priors
        log_score = pd.Series(log_priors.copy())

        for f in feats:
            if f not in row or pd.isna(row[f]):
                continue
            v = str(row[f])
            fm = model["feature_models"][f]
            log_lik = fm["log_likelihood"]

            if v in log_lik.columns:
                log_score = log_score.add(log_lik[v], fill_value=0.0)
            else:
                # unseen value: back off to uniform smoothed probability mass
                V = len(fm["classes"])
                backoff = np.log(model["alpha"] / (np.exp(log_priors)*0 + 1))  # not used
                # build proper uniform backoff: log( α / (bin_total + α*V) )
                # We approximate by using column-wise mean log-likelihood as a fallback:
                col_mean = log_lik.mean(axis=1)
                log_score = log_score.add(col_mean, fill_value=0.0)

        # normalize to probabilities (optional, for inspecting confidences)
        max_log = log_score.max()
        probs = np.exp(log_score - max_log)
        probs = probs / probs.sum()

        # pick argmax
        best_bin = probs.idxmax()
        preds.append(best_bin)
        posteriors.append(probs)

    return pd.Series(preds, index=df_new.index, name="pred_bin"), pd.DataFrame(posteriors)

# ---- usage ----
feature_cols = [
    "cfg.intrinsic_k","k_pow2","cfg.mma_attr","cfg.lhs_type_bitwidth","n_pow2",
    "cfg.num_subgroups","cfg.rhs_tile_size","cfg.intrinsic_mn","cfg.m","cfg.n",
    "num_subgroups_mult4","cfg.sg_n_cnt","cfg.k","cfg.sg_m_cnt","cfg.lds_utilization",
    "n_square","cfg.quantization_inefficiency","k_square",
    "cfg.lhs_tile_size","m_square","m_cube","n_cube","k_cube"
]
feature_cols = [f for f in feature_cols if f in full_df.columns]

nb_model = fit_nb_from_bins(full_df, speed_col="norm_speedup", feature_cols=feature_cols, n_bins=10, binning="width", alpha=1.0)

# suppose df_unseen contains the same feature columns (no speed)
df_unseen = df = pd.read_csv(files[-2])
pred_bins, bin_posteriors = predict_bin_nb(nb_model, df_unseen)

# print(pred_bins)

# ===== Evaluate on the unseen problem and plot =====

def _bin_edges_from_model_bins(model_bins):
    left0 = model_bins[0].left
    rights = [iv.right for iv in model_bins]
    return np.array([left0] + rights, dtype=float)

def assign_bin_indices_for_speed(speeds, model_bins):
    edges = _bin_edges_from_model_bins(model_bins)
    idx = np.searchsorted(edges, np.asarray(speeds), side="right") - 1
    return np.clip(idx, 0, len(edges) - 2)

# 1) predicted bin indices (handle Interval labels or ints)
if pd.api.types.is_integer_dtype(pred_bins.dtype):
    pred_idx_all = pred_bins.to_numpy()
else:
    bin_to_idx = {b: i for i, b in enumerate(nb_model["bins"])}
    pred_idx_all = pred_bins.map(bin_to_idx).to_numpy()

# 2) true bin indices from norm_speedup (smaller=faster ⇒ bin 0 fastest)
true_idx_all = assign_bin_indices_for_speed(df_unseen["norm_speedup"].to_numpy(), nb_model["bins"])

# 3) keep only rows with valid norm_speedup and valid predictions
valid_mask = ~pd.isna(df_unseen["norm_speedup"]) & ~pd.isna(pred_idx_all)
true_idx = true_idx_all[valid_mask]
pred_idx = pred_idx_all[valid_mask]

# 4) scatter: each point = one candidate in the unseen problem
script_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(script_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

scatter_path = os.path.join(plots_dir, "pred_vs_true_bins_scatter_unseen.png")
plt.figure(figsize=(6, 6))
plt.plot(true_idx, pred_idx, ".", alpha=0.25)  # default color
B = int(max(true_idx.max(), pred_idx.max())) + 1
plt.plot([0, B-1], [0, B-1])  # y = x reference
plt.xlim(-0.5, B-0.5); plt.ylim(-0.5, B-0.5)
plt.xticks(range(B)); plt.yticks(range(B))
plt.xlabel("True bin (0 = fastest)")
plt.ylabel("Predicted bin (0 = fastest)")
plt.title("Predicted vs True bins – Unseen Problem")
plt.grid(True, linestyle="--", alpha=0.4)
plt.savefig(scatter_path, dpi=300, bbox_inches="tight"); plt.close()
print(f"Saved: {scatter_path}")