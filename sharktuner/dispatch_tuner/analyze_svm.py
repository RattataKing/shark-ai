import glob
import os
import ast
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr, rankdata, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils import check_random_state

# Core
import numpy as np

# Optional (if reading CSVs / using DataFrames)
import pandas as pd

# scikit-learn bits used by the RankSVM implementation
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils import check_random_state

# Optional (for group-aware cross-validation)
from sklearn.model_selection import GroupKFold
# --- RankSVM (pairwise) minimal implementation --------------------------------
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils import check_random_state

# --- RankSVM (pairwise) with symmetric +/- pairs + diagnostics ---------------
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils import check_random_state

def _make_pairwise(X, y, qids, *, sample_pairs=None, random_state=None, tie_policy="skip"):
    """
    Build pairwise differences within each qid.
      +1 for (xi - xj) where yi > yj
      -1 for (xj - xi)
    tie_policy: "skip" (default) or "jitter" (add tiny noise *within qid* to break ties).
    """
    rng = check_random_state(random_state)
    X = np.asarray(X); y = np.asarray(y); qids = np.asarray(qids)
    pos_diffs = []

    uniq_q = np.unique(qids)
    # --- per-qid sanity (variability of labels) ---
    qid_nuniq = {}
    for q in uniq_q:
        idx = np.where(qids == q)[0]
        yi = y[idx]
        # drop NaNs/inf
        mask = np.isfinite(yi)
        idx, yi = idx[mask], yi[mask]
        if tie_policy == "jitter" and idx.size > 1:
            # tiny noise relative to qid scale to break exact ties deterministically
            eps = 1e-12 if yi.std() == 0 else 1e-6 * yi.std()
            yi = yi + rng.normal(0.0, eps, size=yi.shape)
        qid_nuniq[q] = np.unique(yi).size

        if np.unique(yi).size < 2:
            continue

        ii, jj = np.where(yi[:, None] > yi[None, :])   # strict preferences
        if ii.size == 0:
            continue
        if sample_pairs is not None and ii.size > sample_pairs:
            sel = rng.choice(ii.size, size=sample_pairs, replace=False)
            ii, jj = ii[sel], jj[sel]
        pos_diffs.append(X[idx[ii]] - X[idx[jj]])

    if not pos_diffs:
        raise ValueError(
            "No pairwise preferences found. Within every qid, labels appear tied "
            "or degenerate. qid -> nunique(y): " + str(qid_nuniq)
        )

    Xpos = np.vstack(pos_diffs)
    Xneg = -Xpos
    Xdiff = np.vstack([Xpos, Xneg])
    ysign = np.hstack([
        np.ones(len(Xpos), dtype=np.int8),
        -np.ones(len(Xpos), dtype=np.int8),
    ])

    # global sanity: must have both classes
    classes, counts = np.unique(ysign, return_counts=True)
    if classes.size < 2:
        raise ValueError(f"Pairwise build produced one class only: classes={classes}, counts={counts}.")
    return Xdiff, ysign, qid_nuniq

class RankSVM(BaseEstimator, RegressorMixin):
    def __init__(self, C=1.0, max_iter=10_000, sample_pairs=None, random_state=None, dual=True, tie_policy="skip"):
        self.C = C; self.max_iter = max_iter
        self.sample_pairs = sample_pairs
        self.random_state = random_state
        self.dual = dual
        self.tie_policy = tie_policy
        self._pipe = make_pipeline(
            StandardScaler(with_mean=True),
            LinearSVC(C=C, fit_intercept=True, max_iter=max_iter, dual=dual)
        )

    def fit(self, X, y, qids):
        Xdiff, ysign, qid_nuniq = _make_pairwise(
            X, y, qids,
            sample_pairs=self.sample_pairs,
            random_state=self.random_state,
            tie_policy=self.tie_policy,
        )
        # extra safety: recheck class balance
        cls, cnt = np.unique(ysign, return_counts=True)
        if cls.size < 2:
            raise ValueError(f"Only one class in pairwise labels: {dict(zip(cls, cnt))}")
        # (Optional) print quick diagnostics
        # print("pairwise label counts:", dict(zip(cls, cnt)))
        # print("qid -> nunique(y):", qid_nuniq)

        self._pipe.fit(Xdiff, ysign)
        self._scaler = self._pipe.steps[0][1]
        self._clf = self._pipe.steps[1][1]
        return self

    def decision_function(self, X):
        Z = self._scaler.transform(np.asarray(X))
        s = Z @ self._clf.coef_.ravel()
        return s + float(self._clf.intercept_[0])

    def predict(self, X):
        return self.decision_function(X)

    def rank_by_query(self, X, qids):
        qids = np.asarray(qids)
        scores = self.decision_function(X)
        return {
            q: (idx := np.where(qids == q)[0])[np.argsort(-scores[idx])]
            for q in np.unique(qids)
        }
# ------------------------------------------------------------------------------


# Load CSVs from tuning_database_clean
files = glob.glob('./dispatch_tuner/tuning_database_clean/*.csv')
excluded_files = [
    # Problem size too small 
    "tuning_square_gemm_128_128_128_f16_f32_tB.csv",
    "tuning_square_gemm_256_256_256_f16_f32_tB.csv",
    "tuning_square_gemm_512_512_512_f16_f32_tB.csv",
]
files = [f for f in files if os.path.basename(f) not in excluded_files]
print(f"Found {len(files)} CSV files")

# Split at the file level
train_files, test_files = train_test_split(
    files, test_size=0.2, random_state=42
)
print(f"{len(train_files)} Train files:")
for f in train_files:
    print("  ", f)
print(f"{len(test_files)} Test files:")
for f in test_files:
    print("  ", f)

# Excluded columns
excluded_list = [
    # Problem size
    "cfg.M",
    "cfg.N",
    "cfg.K",

    # Categorical features 
    "cfg.mma_attr", # use int cfg.mma_attr_map instead

    # Random Forest Importances = 0
    "cfg.rhs_type_bitwidth",
    "cfg.lhs_type_bitwidth",
    "cfg.subgroup_size,",
    "cfg.subgroup_tile_k",
    "cfg.wg_z",
    "cfg.subgroup_tile_m",
    "cfg.subgroup_tile_n",
    "cfg.promote_operands",
    "cfg.subgroup_size",
    "cfg.codegen_pipeline",
    "cfg.pipeline_options_search_space",
    "cfg.allowed_waves_per_eu",

    # Highly correlated with other features
    "cfg.wg_y", "cfg.wg_x", "cfg.flat_wg_size", "cfg.WG"
]

# feature_cols = ['cfg.intrinsic_mn', 'cfg.intrinsic_k', 'cfg.m', 'cfg.n', 'cfg.k',
#        'cfg.sg_m_cnt', 'cfg.sg_n_cnt', 'cfg.num_subgroups',
#        'cfg.quantization_inefficiency', 'cfg.lhs_tile_size',
#        'cfg.rhs_tile_size', 'cfg.lds_utilization', 'cfg.mma_attr_map']
feature_cols = ["m_pow2", "n_pow2", "k_pow2", 
                "m_square", "n_square", "k_square",
                "m_cube", "n_cube", "k_cube",
                "num_subgroups_mult4",
                "cfg.lds_utilization", "cfg.quantization_inefficiency",
                'cfg.m', 'cfg.n', 'cfg.k',
                'cfg.M', 'cfg.N', 'cfg.K',
                ]
# feature_cols = ['cfg.intrinsic_mn', 'cfg.intrinsic_k', 'cfg.m', 'cfg.n', 'cfg.k',
#        'cfg.sg_m_cnt', 'cfg.sg_n_cnt', 'cfg.num_subgroups',
#        'cfg.quantization_inefficiency', 'cfg.lhs_tile_size',
#        'cfg.rhs_tile_size', 'cfg.lds_utilization', 'cfg.mma_attr_map',
#        "m_pow2", "n_pow2", "k_pow2", 
#         "m_square", "n_square", "k_square",
#         "m_cube", "n_cube", "k_cube",
#         "num_subgroups_mult4",
#         "cfg.lds_utilization", "cfg.quantization_inefficiency"]


def sanitize_df(df):
    old_shape = old_shape = df.shape
    df = df.dropna(axis=1, how="all")
    print(f"Dataset shape after sanitized: {old_shape} -> {df.shape}")
    return df

def prepare_features(df):
    X_all = df[feature_cols]
    y = df["norm_speedup"]
    return X_all, y


all_X = []
all_y = []
all_qids = []
for i,f in enumerate(train_files, start=1):
    train_df = pd.read_csv(f)
    train_df = sanitize_df(train_df)
    X_train, y_train = prepare_features(train_df)
    X_train = train_df[X_train.columns].to_numpy(dtype=float)
    y_train = y_train.to_numpy(dtype=float)
    y_train = -y_train
    qids = np.full(len(train_df), i, dtype=int)

    all_X.append(X_train)
    all_y.append(y_train)
    all_qids.append(qids)
X_train = np.vstack(all_X)
y_train = np.concatenate(all_y)
qids = np.concatenate(all_qids)

print("Training RankSVM...")
ranksvm = RankSVM(C=1.0, random_state=0).fit(X_train, y_train, qids)


w_std = ranksvm._clf.coef_.ravel()
b_std = float(ranksvm._clf.intercept_[0])
mean  = ranksvm._scaler.mean_
scale = ranksvm._scaler.scale_

# Back to original feature units
w_orig = w_std / scale
b_orig = b_std - float(np.dot(w_orig, mean))

print("w (original units):", w_orig)
print("b (original units):", b_orig)

# Pretty-print the formula
terms = " ".join(f"{w:+.6f}*{name}" for w, name in zip(w_orig, feature_cols) if abs(w) > 1e-8)
formula = " ".join([terms, f"{b_orig:+.6f}"])
print("score(x) =", formula)

script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, "SVM_output.txt")
with open(save_path, "w") as f:
    # Save w and b
    f.write(f"w (original units): {w_orig}\n")
    f.write(f"b (original units): {b_orig}\n\n")

    # Save the pretty-printed formula
    f.write(f"score(x) = {formula}\n")

print(f"Saved output to {save_path}")

for i,f in enumerate(test_files, start=1):
    test_df = pd.read_csv(f)
    test_df = sanitize_df(test_df)
    X_test, y_test = prepare_features(test_df)
    X_test = test_df[X_test.columns].to_numpy(dtype=float)
    y_test = y_test.to_numpy(dtype=float)
    y_test = y_test

    y_pred = ranksvm.decision_function(X_test)
    y_pred = -y_pred

    y_df = pd.DataFrame({
        "y_test": y_test,
        "y_pred": y_pred
    })
    y_df_sorted = y_df.sort_values("y_test", ascending=True).reset_index(drop=True)
    y_df_sorted["true_rank"] = rankdata(y_df_sorted["y_test"], method="dense")
    y_df_sorted["pred_rank"] = rankdata(y_df_sorted["y_pred"], method="dense")

    # --- fresh figure each loop ---
    fig, ax = plt.subplots()
    ax.scatter(y_df_sorted["true_rank"], y_df_sorted["pred_rank"], alpha=0.7)
    max_val = max(y_df_sorted["true_rank"].max(), y_df_sorted["pred_rank"].max())
    ax.plot([1, max_val], [1, max_val], 'r--')
    ax.set(xlim=(1, max_val), ylim=(1, max_val),
        xlabel="True Rank", ylabel="Predicted Rank",
        title=f"True vs Predicted Rank\n{Path(f).stem}")

    save_path = os.path.join(script_dir, f"SVM_true_vs_pred_rank_{i}.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)  # <<< important to avoid overlay + memory growth
    print(f"Saved plot to {save_path}")
## norm_speedup(x) ≈ 1.25·[m is pow2] + 0.80·[K % 64 == 0] - 0.55·[mma=2] + 0.30·[sg_m_cnt ≥ 2] + b

# # If you want the weights as a Series:
# coef_series = pd.Series(w_orig, index=feature_cols).sort_values(key=np.abs, ascending=False)
# print(coef_series)