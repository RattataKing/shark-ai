from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('./dispatch_tuner/tuning_compute_gemm_4096_4096_8192_f16_f32_tA.csv')

df["winners"] = (df["benchmark_result_order"] <= 20) & (df["benchmark_speedup"] < 1)


cfg_cols = [c for c in df.columns if c.startswith("cfg.")]
X = df[cfg_cols]

# numeric subset
numeric_cols = df[cfg_cols].select_dtypes(include="number").columns

# categorical subset (strings)
cat_cols = [c for c in cfg_cols if c not in numeric_cols]

# one-hot encode categories
X_cat = pd.get_dummies(df[cat_cols].astype(str))
# X_cat = pd.get_dummies(df[cat_cols].astype(str), prefix=cat_cols, drop_first=True)
# X_cat = X_cat.astype(int)

X_num = df[numeric_cols]
X_all = pd.concat([X_num, X_cat], axis=1)

# corr = pd.concat([X_num, df["winners"]], axis=1).corr()
# print(corr["winners"].sort_values(ascending=False))
corr = pd.concat([X_all, df["winners"]], axis=1).corr()
print("Correlation Matrix:")
print(corr["winners"].sort_values(ascending=False).head(10))

# Pick top-N features most correlated with winners (absolute value)
head=10
top_features = corr["winners"].abs().sort_values(ascending=False).head(head).index
corr_subset = corr.loc[top_features, top_features]

# Plot
plt.figure(figsize=(14, 12))
sns.heatmap(
    corr_subset,
    annot=True,        # show numbers
    fmt=".2f",         # 2 decimal places
    cmap="coolwarm",
    linewidths=0.5,
)

plt.title(f"Correlation Heatmap (Top {head} Features vs Winners)", fontsize=14)
plt.tight_layout()

# Save directly to file
plt.savefig("./dispatch_tuner/correlation_heatmap.png", dpi=300)
plt.close()
print("Correlation heatmap saved to ./dispatch_tuner/correlation_heatmap.png")





y = df["winners"].astype(int)
clf = RandomForestClassifier(
    n_estimators=500, 
    random_state=0, 
    class_weight="balanced"
)
clf.fit(X_all, y)

importances = pd.Series(clf.feature_importances_, index=X_all.columns)
print("Random Forest Feature Importances:")
print(importances.sort_values(ascending=False).head(10))




# X = df[["wg_x","wg_y","wg_z","subgroup_size","sg_m_cnt","sg_n_cnt"]]  # features
# y = df["latency"]

# model = RandomForestRegressor().fit(X, y)
# importances = pd.Series(model.feature_importances_, index=X.columns)
# print(importances.sort_values(ascending=False))
