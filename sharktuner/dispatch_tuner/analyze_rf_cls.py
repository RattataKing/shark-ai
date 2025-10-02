from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import numpy as np
import math
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

files = glob.glob("./dispatch_tuner/tuning_database_clean/*.csv")
dfs = []
for f in files[0]:
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
df["winners"] = df["norm_speedup"] <= df["norm_speedup"].quantile(0.1)

feature_cols = ["m_pow2", "n_pow2", "k_pow2", 
                "mnk_cube","closeness_to_cube_volume",
                "num_subgroups_mult4",
                "cfg.M", "cfg.N", "cfg.K",
                "cfg.m", "cfg.n", "cfg.k",
                "p_ai", "t_ai", "intrinsic_ai",
                "mn_ratio",
                "cfg.quantization_inefficiency", "cfg.lds_utilization"]
X_num = df[feature_cols]
X_all = X_num
# X_all = pd.concat([X_num, X_cat], axis=1)

y = df["winners"].astype(int)

# --- split ---
# X_train, X_test, y_train, y_test = train_test_split(
#     X_all, y, test_size=0.2, random_state=0, stratify=y
# )

clf = RandomForestClassifier(
    n_estimators=500, 
    random_state=0, 
    class_weight="balanced",
    n_jobs=-1
)
clf.fit(X_all, y)

# print("Test set class balance:", y_test.value_counts(normalize=True).to_dict())

importances = pd.Series(clf.feature_importances_, index=X_all.columns)
print("Random Forest Feature Importances:")
print(importances.sort_values(ascending=False))

exit()

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