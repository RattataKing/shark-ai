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
from sklearn.linear_model import LassoCV
from sklearn.tree import _tree
from rulefit import RuleFit

# Load CSVs from tuning_database_clean
files = glob.glob('./dispatch_tuner/tuning_database_clean/*.csv')
# excluded_files = [
#     # Problem size too small 
#     "tuning_square_gemm_128_128_128_f16_f32_tB.csv",
#     "tuning_square_gemm_256_256_256_f16_f32_tB.csv",
#     "tuning_square_gemm_512_512_512_f16_f32_tB.csv",
# ]
# files = [f for f in files if os.path.basename(f) not in excluded_files]
files = [
    f for f in files
    if all(pd.read_csv(f)[col].iloc[0] > 512 for col in ["cfg.M", "cfg.N", "cfg.K"])
]
print(f"Found {len(files)} CSV files")

# Split at the file level
train_files, test_files = train_test_split(
    files, test_size=0.2,
)
print(f"{len(train_files)} Train files:")
# for f in train_files:
#     print("  ", f)
print(f"{len(test_files)} Test files:")
# for f in test_files:
#     print("  ", f)

# Excluded columns
excluded_list = [
    # Problem size
    # "cfg.M",
    # "cfg.N",
    # "cfg.K",

    # Categorical features 
    # "cfg.mma_attr",
    "cfg.mma_attr_map",
    "cfg.mma_a",
    "cfg.mma_b",
    "cfg.mma_c",

    # Random Forest Importances = 0
    "cfg.rhs_type_bitwidth",
    # "cfg.lhs_type_bitwidth",
    "m_pow2",
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

def sanitize_df(df):
    old_shape = old_shape = df.shape
    df = df.dropna(axis=1, how="all")
    print(f"Dataset shape after sanitized: {old_shape} -> {df.shape}")
    return df

def prepare_features(df):
    cfg_cols = [c for c in df.columns if c.startswith("cfg.") and c not in excluded_list]
    # numeric subset
    numeric_cols = df[cfg_cols].select_dtypes(include="number").columns
    feature_cols = ["m_pow2", "n_pow2", "k_pow2", 
                "m_square", "n_square", "k_square",
                "m_cube", "n_cube", "k_cube",
                "num_subgroups_mult4",
                "cfg.M", "cfg.N", "cfg.K",
                "cfg.m", "cfg.n", "cfg.k",
                "cfg.quantization_inefficiency", "cfg.lds_utilization"]
    numeric_cols = numeric_cols.union(feature_cols)
    numeric_cols = [c for c in numeric_cols if c not in excluded_list]
    # categorical subset (strings)
    cat_cols = [c for c in cfg_cols if c not in numeric_cols]

    # Encode categories as integer labels (skip cleanly if none)
    if cat_cols:
        enc = OrdinalEncoder()
        X_cat = pd.DataFrame(
            enc.fit_transform(df[cat_cols].astype(str)),
            columns=cat_cols,
            index=df.index
        )
        print(f"Encoded features in {cat_cols}")
    else:
        enc = None
        X_cat = pd.DataFrame(index=df.index)  # (n_rows x 0) placeholder

    numeric_cols = ["m_pow2", "n_pow2", "k_pow2", 
                "m_square", "n_square", "k_square",
                "m_cube", "n_cube", "k_cube",
                "num_subgroups_mult4",
                "cfg.M", "cfg.N", "cfg.K",
                "cfg.m", "cfg.n", "cfg.k",
                "p_ai", "t_ai", "intrinsic_ai",
                "mn_ratio",
                "cfg.quantization_inefficiency", "cfg.lds_utilization"]

    X_num = df[numeric_cols]
    # X_all = pd.concat([X_num, X_cat], axis=1)
    X_all=X_num
    y = df["norm_speedup"]

    return X_all, y, numeric_cols


train_dfs = [pd.read_csv(f) for f in train_files]
train_df = pd.concat(train_dfs, ignore_index=True)
train_df = sanitize_df(train_df)

X_train, y_train, feature_cols = prepare_features(train_df)
print(f"Select Features: {X_train.columns}")

# Build Random Forest Regressor
rf = RandomForestRegressor(
    n_estimators=500,   # number of trees
    max_depth=None,     # let trees grow fully (can tune)
    n_jobs=-1,          # use all cores
    random_state=42
)

rf.fit(X_train, y_train)
# Feature importance
importances = pd.Series(rf.feature_importances_, index=X_train.columns)
print("\nRandom Forest Importances:")
print(importances.sort_values(ascending=False))

# Spearman correlation among features
if len(feature_cols) > 1:
    corr = X_train[feature_cols].corr(method="spearman")
    # Top correlated pairs
    tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    top_pairs = (
        tri.stack()
        .rename("|rho|")
        .abs()
        .sort_values(ascending=False)
        .head(30)
    )
    print("\nTop numeric-numeric Spearman |rho| pairs:")
    print(top_pairs)
else:
    print("No numeric-numeric correlation to compute.")

script_dir = os.path.dirname(os.path.abspath(__file__))
for i,f in enumerate(test_files):
    test_df = pd.read_csv(f)
    test_df = sanitize_df(test_df)

    X_test, y_test, _ = prepare_features(test_df)
    y_pred = rf.predict(X_test)
    # y_pred = np.expm1(rf.predict(X_test))


    y_df = pd.DataFrame({
        "y_test": y_test,
        "y_pred": y_pred
    })
    print(f"Len of y_test = {len(y_test)}")
    y_df_sorted = y_df.sort_values("y_test", ascending=True).reset_index(drop=True)
    y_df_sorted["true_rank"] = rankdata(y_df_sorted["y_test"], method="dense")
    y_df_sorted["pred_rank"] = rankdata(y_df_sorted["y_pred"], method="dense")
    spearman_corr, pval = spearmanr(y_df_sorted["true_rank"], y_df_sorted["pred_rank"])
    print(f"Spearman correlation: {spearman_corr:.4f} (p={pval:.4g})")
    rank_rmse = np.sqrt(mean_squared_error(y_df_sorted["true_rank"], y_df_sorted["pred_rank"]))
    print(f"Rank RMSE: {rank_rmse:.2f}")

    # --- fresh figure each loop ---
    fig, ax = plt.subplots()
    ax.scatter(y_df_sorted["true_rank"], y_df_sorted["pred_rank"], alpha=0.7)
    max_val = max(y_df_sorted["true_rank"].max(), y_df_sorted["pred_rank"].max())
    ax.plot([1, max_val], [1, max_val], 'r--')
    ax.set(xlim=(1, max_val), ylim=(1, max_val),
        xlabel="True Rank", ylabel="Predicted Rank",
        title=f"True vs Predicted Rank\n{Path(f).stem}")

    save_path = os.path.join(script_dir, f"RF_true_vs_pred_rank_{i}.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)  # <<< important to avoid overlay + memory growth
    print(f"Saved plot to {save_path}")

rf_output_path = os.path.join(script_dir, "rf.pkl")
joblib.dump(rf, rf_output_path)
print(f"Model saved as {rf_output_path}")
exit()


def extract_rules_from_tree(tree, feature_names, max_rule_len=None):
    """
    Returns a list of rules. Each rule is a list of conditions (feature, op, threshold).
    op is '<=' or '>'.
    """
    t = tree.tree_
    feat = t.feature
    thr = t.threshold
    rules = []

    def recurse(node, conds):
        if feat[node] != _tree.TREE_UNDEFINED:
            name = feature_names[feat[node]]
            threshold = thr[node]
            # left
            recurse(t.children_left[node], conds + [(name, "<=", float(threshold))])
            # right
            recurse(t.children_right[node], conds + [(name, ">", float(threshold))])
        else:
            rule = conds
            if (max_rule_len is None) or (len(rule) <= max_rule_len):
                rules.append(rule)

    recurse(0, [])
    return rules

# ---- 2) Merge rules from a whole RandomForest, prune duplicates, and simplify intervals ----
def simplify_rule(rule):
    """
    Merge multiple constraints on the same feature into a single interval [low, high].
    Returns canonical tuple like ('f1', low, high) & ('f2', low, high) & ...
    """
    bounds = {}
    for feat, op, thr in rule:
        low, high = bounds.get(feat, (-np.inf, np.inf))
        if op == "<=":
            high = min(high, thr)
        else:
            low = max(low, np.nextafter(thr, np.inf))  # open interval for '>'
        bounds[feat] = (low, high)
    # discard impossible intervals
    for f, (lo, hi) in list(bounds.items()):
        if lo > hi:
            return None
    # canonical sorted form
    return tuple(sorted((f, lo, hi) for f, (lo, hi) in bounds.items()))

def rules_from_forest(rf, feature_names, max_rule_len=4, min_coverage=0.02, X=None):
    """
    Extract, simplify, and filter rules from a RandomForest*Regressor*.
    - max_rule_len: limit number of conjuncts per rule (keeps rules human-sized)
    - min_coverage: drop rules that fire on too few samples
    - X: if provided, compute coverage on X; otherwise skip coverage filter
    """
    # Collect and simplify all rules
    simple_rules = []
    for est in rf.estimators_:
        for r in extract_rules_from_tree(est, feature_names, max_rule_len=max_rule_len):
            sr = simplify_rule(r)
            if sr is not None:
                simple_rules.append(sr)

    # Deduplicate
    simple_rules = list(dict.fromkeys(simple_rules))

    # Optional coverage filtering
    if X is not None:
        mask_list = []
        keep = []
        for rule in simple_rules:
            m = rule_to_mask(rule, X)
            cov = m.mean()
            if cov >= min_coverage:
                keep.append(rule)
                mask_list.append(m)
        return keep, mask_list
    else:
        return simple_rules, None

# ---- 3) Apply a simplified rule to data -> boolean mask ----
def rule_to_mask(rule, Xdf):
    m = np.ones(len(Xdf), dtype=bool)
    for feat, lo, hi in rule:
        v = Xdf[feat].values
        m &= (v > lo) & (v <= hi)
    return m

# ---- 4) Build the rule design matrix and fit a sparse linear model ----
def fit_rule_linear_model(rf, X, y, max_rule_len=4, min_coverage=0.02, add_linear_terms=False):
    """
    Returns: (lasso_model, rule_list, intercept, coefs, design_matrix)
    """
    feature_names = list(X.columns)
    rules, masks = rules_from_forest(rf, feature_names, max_rule_len=max_rule_len,
                                     min_coverage=min_coverage, X=X)
    if not rules:
        raise ValueError("No rules survived filtering. Relax constraints or check data.")

    # Build rule matrix R (n_samples x n_rules)
    R = np.column_stack(masks).astype(float)

    # Optionally add linear terms (standardized) for a hybrid rule+linear model
    if add_linear_terms:
        Xz = (X - X.mean()) / (X.std(ddof=0) + 1e-12)
        Z = np.column_stack([R, Xz.values])
        colnames = [f"[RULE {i}]" for i in range(len(rules))] + list(X.columns)
    else:
        Z = R
        colnames = [f"[RULE {i}]" for i in range(len(rules))]

    # L1-regularized linear regression selects a small set of rules
    lasso = LassoCV(cv=5, random_state=0, n_jobs=None).fit(Z, y)

    return lasso, rules, lasso.intercept_, lasso.coef_, colnames

# ---- 5) Pretty-print top rules as a formula ----
def format_rule(rule):
    parts = []
    for feat, lo, hi in rule:
        # choose compact wording; hide bounds near -/+inf
        if np.isfinite(lo) and np.isfinite(hi):
            parts.append(f"({lo:.3g} < {feat} ≤ {hi:.3g})")
        elif np.isfinite(lo):
            parts.append(f"({feat} > {lo:.3g})")
        else:
            parts.append(f"({feat} ≤ {hi:.3g})")
    return " and ".join(parts)

def summarize_rule_model(lasso, rules, intercept, coefs, colnames, k=10, tol=1e-6):
    # keep only nonzero-ish coefficients
    items = [(coef, name) for coef, name in zip(coefs, colnames) if abs(coef) > tol]

    # sort by importance
    items = sorted(items, key=lambda t: -abs(t[0]))[:k]

    # build human-readable terms
    lines = []
    for coef, name in items:
        if name.startswith("[RULE"):
            idx = int(name.split()[1][:-1])
            rule_str = format_rule(rules[idx])
            lines.append(f"{coef:+.3f}·[{rule_str}]")
        else:
            lines.append(f"{coef:+.3f}·{name}")

    # join terms + intercept
    formula = " ".join(lines) + f" {intercept:+.3f}"
    return "norm_speedup(x) ≈ " + formula

lasso, rules, intercept, coefs, names = fit_rule_linear_model(
    rf, X_train, y_train,
    max_rule_len=99,        # shorter rules -> more readable
    min_coverage=0.03,     # drop rules that fire on <3% of samples
    add_linear_terms=False # set True to allow a few raw-feature terms too
)

output = summarize_rule_model(lasso, rules, intercept, coefs, names, k=12)
print(output)

save_path = os.path.join(script_dir, "RF_output.txt")
with open(save_path, "w") as f:

    # Save the pretty-printed formula
    f.write(f"{output}\n")