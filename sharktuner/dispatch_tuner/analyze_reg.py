import glob
import os
import json
import ast
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# Load all CSVs from tuning_database
files = glob.glob('./dispatch_tuner/tuning_database/*.csv')
dfs = []
for f in files:
    df = pd.read_csv(f)
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)

old_len = len(df)
# Drop useless rows
df = df.dropna(axis=1, how="all")
df = df[df["candidate_id"] != 0]
df = df[(df["cfg.M"] != 0) & (df["cfg.workgroup_tile_size.x"] != 0)]
print(f"Before: {old_len} rows, After dropna: {len(df)} rows")

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
    "cfg.workgroup_tile_size.z",
    "cfg.subgroup_tile_size.x",
    "cfg.subgroup_tile_size.y",
    "cfg.subgroup_tile_size.z",
    "cfg.promote_operand_a",
    "cfg.promote_operand_b",
    "cfg.reduction_tile_size.x",
    "cfg.reduction_tile_size.y",

    # MMA attr: string class, need to do one-hot or label
    # Highly correlated with other features
    "cfg.wg.y", "cfg.wg.x", "cfg.flat_wg_size", "cfg.NC"
]

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

# Define target
y = df["norm_speedup"]

# Combine numeric + categorical
X = X_all

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build Random Forest Regressor
rf = RandomForestRegressor(
    n_estimators=200,   # number of trees
    max_depth=None,     # let trees grow fully (can tune)
    n_jobs=-1,          # use all cores
    random_state=42
)
rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_test)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
spearman_corr, _ = spearmanr(y_test, y_pred)

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
