import matplotlib.pyplot as plt
import os, glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

os.environ["OPENBLAS_NUM_THREADS"] = "32"
os.environ["OMP_NUM_THREADS"] = "32"

files = glob.glob("./dispatch_tuner/tuning_database/*.csv")
dfs = []
for f in files:
    df = pd.read_csv(f)
    dfs.append(df)

big_df = pd.concat(dfs, ignore_index=True)

big_df["winners"] = (
    (big_df["benchmark_result_order"] <= 20) &
    (big_df["benchmark_speedup"] < 1)
)
df=big_df

# 0) Drop columns that are entirely empty
df = df.dropna(axis=1, how="all")

excluded_list = [
    'cfg.workgroup_tile_sizes',
    'cfg.reduction_tile_sizes',
    'cfg.subgroup_tile_sizes',
    'cfg.promote_operands',
    'cfg.pipeline_options_search_space',

    'cfg.M',
    'cfg.N',
    'cfg.K',

    # 'cfg.mma_attr',
]


# 1) Identify columns
cfg_cols = [c for c in df.columns if c.startswith("cfg.") and c not in excluded_list]
numeric_cols = df[cfg_cols].select_dtypes(include="number").columns.tolist()
cat_cols = [c for c in cfg_cols if c not in numeric_cols]

# 2) Drop rows with NaNs in numeric
df_clean = df.dropna(subset=numeric_cols + cat_cols).copy()
# df_clean = df.dropna(subset=cat_cols).copy()
print(f"Before: {len(df)} rows, After dropna: {len(df_clean)} rows")

# 3) Preprocess
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),  # safe even if no NaNs left
        ("sc", StandardScaler()),
    ]), numeric_cols),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore")),
    ]), cat_cols),
])

# 4) KMeans pipeline
kmeans = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("cluster", KMeans(n_clusters=10, random_state=42, n_init="auto"))
])

# 5) Fit + assign clusters
df_clean["cluster"] = kmeans.fit_predict(df_clean[cfg_cols])

# 6) Winner ratio by cluster
cluster_summary = df_clean.groupby("cluster")["winners"].mean().sort_values(ascending=False)
print(cluster_summary)