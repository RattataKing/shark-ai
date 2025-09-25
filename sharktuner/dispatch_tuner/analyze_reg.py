import glob
import os
import json
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

# Load all CSVs from tuning_database
files = glob.glob('./dispatch_tuner/tuning_database/*.csv')
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
    # Useless/redundant features
    "cfg.workgroup_tile_sizes",
    "cfg.reduction_tile_sizes",
    "cfg.subgroup_tile_sizes",
    "cfg.pipeline_options_search_space",
    "cfg.codegen_pipeline",
    "cfg.allowed_waves_per_eu",
    "cfg.pipeline_prefetch_shared_memory",
    "cfg.pipeline_no_reduce_shared_memory_bank_conflicts",
    "cfg.pipeline_use_igemm_convolution",

    # Problem size
    "cfg.M",
    "cfg.N",
    "cfg.K",

    # No importance
    "cfg.wg.z",
    "cfg.subgroup_size",
    "cfg.workgroup_tile_size_z",
    "cfg.subgroup_tile_size_x",
    "cfg.subgroup_tile_size_y",
    "cfg.subgroup_tile_size_z",
    "cfg.promote_operand_a",
    "cfg.promote_operand_b",
    "cfg.reduction_tile_size_x",
    "cfg.reduction_tile_size_y",

    # MMA attr: string class, need to do one-hot or label
    # Highly correlated with other features
    "cfg.wg_y", "cfg.wg_x", "cfg.flat_wg_size", "cfg.WG"
]

def sanitize_df(df):
    df = df.dropna(axis=1, how="all")
    df = df[df["candidate_id"] != 0]
    df = df[df["cfg.M"] % df["cfg.workgroup_tile_size_x"] == 0]
    return df


def prepare_features(df):
    cfg_cols = [c for c in df.columns if c.startswith("cfg.") and c not in excluded_list]
    # numeric subset
    numeric_cols = df[cfg_cols].select_dtypes(include="number").columns
    # categorical subset (strings)
    cat_cols = [c for c in cfg_cols if c not in numeric_cols]

    # Encode categories as integer labels instead of one-hot
    enc = OrdinalEncoder()
    X_cat = pd.DataFrame(
        enc.fit_transform(df[cat_cols].astype(str)),
        columns=cat_cols,
        index=df.index
    )

    X_num = df[numeric_cols]
    X_all = pd.concat([X_num, X_cat], axis=1)
    y = df["norm_speedup"]

    return X_all, y


train_dfs = [pd.read_csv(f) for f in train_files]
train_df = pd.concat(train_dfs, ignore_index=True)
old_len = len(train_df)
train_df = sanitize_df(train_df)
print(f"Train set size after sanitized: {old_len} -> {len(train_df)} rows")

X_train, y_train = prepare_features(train_df)

# Build Random Forest Regressor
rf = RandomForestRegressor(
    n_estimators=1000,   # number of trees
    max_depth=None,     # let trees grow fully (can tune)
    n_jobs=-1,          # use all cores
    random_state=42
)
# y_log = np.log1p(y_train)
rf.fit(X_train, y_train)
# rf.fit(X_train, y_log)


for i,f in enumerate(test_files):
    test_df = pd.read_csv(f)
    old_len = len(test_df)
    test_df = sanitize_df(test_df)
    print(f"Test set [{i}] size after sanitized: {old_len} -> {len(test_df)} rows")

    X_test, y_test = prepare_features(test_df)
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
    print(y_df_sorted.head(30))
    spearman_corr, pval = spearmanr(y_df_sorted["true_rank"], y_df_sorted["pred_rank"])
    print(f"Spearman correlation: {spearman_corr:.4f} (p={pval:.4g})")
    rank_rmse = np.sqrt(mean_squared_error(y_df_sorted["true_rank"], y_df_sorted["pred_rank"]))
    print(f"Rank RMSE: {rank_rmse:.2f}")

    # --- fresh figure each loop ---
    fig, ax = plt.subplots()
    ax.scatter(y_df_sorted["true_rank"], y_df_sorted["pred_rank"], alpha=0.7)
    ax.plot([1, len(y_df_sorted)], [1, len(y_df_sorted)], 'r--')
    ax.set_xlabel("True Rank")
    ax.set_ylabel("Predicted Rank")
    ax.set_title(f"True vs Predicted Rank\n{Path(f).stem}")

    # save next to script (or use Path.cwd() if __file__ is unavailable)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, f"true_vs_pred_rank_{i}.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)  # <<< important to avoid overlay + memory growth
    print(f"Saved plot to {save_path}")

exit()
# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
spearman_corr, _ = spearmanr(y_test, y_pred)

# print(f"ytest: {y_test[:10]}")
# print(f"ypred: {y_pred[:10]}")

y_df = pd.DataFrame({
    "y_test": y_test,
    "y_pred": y_pred
})
print(f"Len of y_test = {len(y_test)}")
y_df_sorted = y_df.sort_values("y_test", ascending=True).reset_index(drop=True)
y_df_sorted["true_rank"] = rankdata(y_df_sorted["y_test"], method="dense")
y_df_sorted["pred_rank"] = rankdata(y_df_sorted["y_pred"], method="dense")
print(y_df_sorted.head(30))
spearman_corr, pval = spearmanr(y_df_sorted["true_rank"], y_df_sorted["pred_rank"])
print(f"Spearman correlation: {spearman_corr:.4f} (p={pval:.4g})")
rank_rmse = np.sqrt(mean_squared_error(y_df_sorted["true_rank"], y_df_sorted["pred_rank"]))
print(f"Rank RMSE: {rank_rmse:.2f}")

plt.scatter(y_df_sorted["true_rank"], y_df_sorted["pred_rank"], alpha=0.7)
plt.plot([1, len(y_df_sorted)], [1, len(y_df_sorted)], 'r--')  # perfect diagonal
plt.xlabel("True Rank")
plt.ylabel("Predicted Rank")
plt.title("True vs Predicted Rank")

# Save in the same directory as the script
script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, "true_vs_pred_rank.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"Save plot to {save_path}")

exit()

print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")
print(f"Spearman: {spearman_corr:.4f}")

# Feature importance
importances = pd.Series(rf.feature_importances_, index=X.columns)
print(importances.sort_values(ascending=False).head(10))

# Spearman correlation among numeric features
if len(numeric_cols) > 1:
    corr = X_train[numeric_cols].corr(method="spearman")
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

# Directory where script is running
script_dir = os.path.dirname(os.path.abspath(__file__))

# Number of unique mma_attr classes
n_classes = df["cfg.mma_attr"].nunique()
print("Number of unique mma_attr classes:", n_classes)

# Boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(x="cfg.mma_attr", y="norm_speedup", data=df)
plt.xticks(rotation=45)
plt.title("Distribution of norm_speedup per mma_attr")
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "mma_attr_boxplot.png"))
plt.close()

