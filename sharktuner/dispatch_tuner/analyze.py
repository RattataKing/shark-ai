from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import numpy as np
import math
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

files = glob.glob("./dispatch_tuner/tuning_database/*.csv")
dfs = []
for f in files:
    df = pd.read_csv(f)
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)

# df["winners"] = (
#     # (df["benchmark_status"] == False)
#     (df["benchmark_result_order"] <= 10) &
#     (df["benchmark_speedup"] < 1)
# )
old_len = len(df)
df = df.dropna(axis=1, how="all")
df = df[df["candidate_id"] != 0]
print(f"Before: {old_len} rows, After dropna: {len(df)} rows")
df["winners"] = df["norm_speedup"] <= df["norm_speedup"].quantile(0.2)
df=df




# exit()
excluded_list = [
    # Useless/redundant feature
    'cfg.workgroup_tile_sizes',
    'cfg.reduction_tile_sizes',
    'cfg.subgroup_tile_sizes',
    'cfg.promote_operands',
    'cfg.pipeline_options_search_space',
    'cfg.codegen_pipeline', 
    'cfg.allowed_waves_per_eu', 
    'cfg.pipeline_prefetch_shared_memory', 
    'cfg.pipeline_no_reduce_shared_memory_bank_conflicts', 
    'cfg.pipeline_use_igemm_convolution',

    # Problem size
    'cfg.M',
    'cfg.N',
    'cfg.K',

    # No importance
    'cfg.wg_z',
    'cfg.subgroup_size',
    'cfg.workgroup_tile_size_z',
    'cfg.subgroup_tile_size_x',
    'cfg.subgroup_tile_size_y',
    'cfg.subgroup_tile_size_z',
    'cfg.promote_operand_1',
    'cfg.promote_operand_2',
    'cfg.reduction_tile_size_1',
    'cfg.reduction_tile_size_2',

    # 'cfg.mma_attr', # Str Class, need to do one-hot or label

    # Highly correlated with other features
    'cfg.wg_x','cfg.wg_y','cfg.flat_wg_size','cfg.WG',
]

cfg_cols = [c for c in df.columns if c.startswith("cfg.") and c not in excluded_list]
# numeric subset
numeric_cols = df[cfg_cols].select_dtypes(include="number").columns
# categorical subset (strings)
cat_cols = [c for c in cfg_cols if c not in numeric_cols]

# selected_subset_cols = numeric_cols.tolist() + cat_cols
# old_len = len(df)
# df = df.dropna(subset=selected_subset_cols)
# print(f"Before: {old_len} rows, After dropna: {len(df)} rows")

# Encode categories as integer labels instead of one-hot
enc = OrdinalEncoder()
X_cat = pd.DataFrame(
    enc.fit_transform(df[cat_cols].astype(str)),
    columns=cat_cols,
    index=df.index
)

X_num = df[numeric_cols]
X_all = pd.concat([X_num, X_cat], axis=1)

y = df["winners"].astype(int)

# --- split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y, test_size=0.2, random_state=0, stratify=y
)

clf = RandomForestClassifier(
    n_estimators=500, 
    random_state=0, 
    class_weight="balanced",
    n_jobs=-1
)
clf.fit(X_train, y_train)

print("Test set class balance:", y_test.value_counts(normalize=True).to_dict())

importances = pd.Series(clf.feature_importances_, index=X_all.columns)
print("Random Forest Feature Importances:")
print(importances.sort_values(ascending=False))

# exit()

# Spearman correlation among numeric features
if len(numeric_cols) > 1:
    corr = X_train[numeric_cols].corr(method="spearman")
    # Top correlated pairs
    tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    top_pairs = (
        tri.stack()
           .rename("rho")
           .abs()
           .sort_values(ascending=False)
           .head(30)
    )
    print("\nTop numeric-numeric Spearman |rho| pairs:")
    print(top_pairs)
else:
    print("No numeric-numeric correlation to compute.")