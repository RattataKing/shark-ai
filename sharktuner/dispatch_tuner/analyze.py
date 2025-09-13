from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob

files = glob.glob("./dispatch_tuner/tuning_database/*.csv")
dfs = []
for f in files:
    df = pd.read_csv(f)
    # add a dispatch identifier (from filename or dispatch_id column)
    # df["dispatch_file"] = f
    dfs.append(df)

big_df = pd.concat(dfs, ignore_index=True)

big_df["winners"] = (
    # (big_df["benchmark_status"] == False)
    (big_df["benchmark_result_order"] <= 50) &
    (big_df["benchmark_speedup"] < 1)
)
df=big_df

# filename = "tuning_square_gemm_256_256_256_f16_f32_tB"
# df = pd.read_csv(f'./dispatch_tuner/single_gemm/{filename}.csv')
# df["winners"] = (df["benchmark_result_order"] <= 20) & (df["benchmark_speedup"] < 1)



# cfg_cols = [c for c in big_df.columns if c.startswith("cfg.")]
# numeric_cols = big_df[cfg_cols].select_dtypes(include="number").columns
# cat_cols = [c for c in cfg_cols if c not in numeric_cols]

# X_cat = pd.get_dummies(big_df[cat_cols].astype(str))
# X_all = pd.concat([big_df[numeric_cols].fillna(0), X_cat], axis=1)
# y = big_df["is_winner"].astype(int)

# clf = RandomForestClassifier(n_estimators=500, random_state=0, class_weight="balanced")
# clf.fit(X_all, y)
# importances = pd.Series(clf.feature_importances_, index=X_all.columns)
# print(importances.sort_values(ascending=False).head(20))

# exit()
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

cfg_cols = [c for c in df.columns if c.startswith("cfg.") and c not in excluded_list]
X = df[cfg_cols]

# numeric subset
numeric_cols = df[cfg_cols].select_dtypes(include="number").columns

# categorical subset (strings)
cat_cols = [c for c in cfg_cols if c not in numeric_cols]
# print(cat_cols)
# exit()

# one-hot encode categories
X_cat = pd.get_dummies(df[cat_cols].astype(str))
# X_cat = pd.get_dummies(df[cat_cols].astype(str), prefix=cat_cols, drop_first=True)
# X_cat = X_cat.astype(int)

X_num = df[numeric_cols]
X_all = pd.concat([X_num, X_cat], axis=1)

# corr = pd.concat([X_num, df["winners"]], axis=1).corr()
# print(corr["winners"].sort_values(ascending=False))
corr = pd.concat([X_all, df["winners"]], axis=1).corr()
# print("Correlation Matrix:")
# print(corr["winners"].sort_values(ascending=False).head(10))

# exit()

# Pick top-N features most correlated with winners (absolute value)
head=10
top_features = corr["winners"].abs().sort_values(ascending=False).head(head).index
corr_subset = corr.loc[top_features, top_features]

# # Plot
# plt.figure(figsize=(14, 12))
# sns.heatmap(
#     corr_subset,
#     annot=True,        # show numbers
#     fmt=".2f",         # 2 decimal places
#     cmap="coolwarm",
#     linewidths=0.5,
# )

# plt.title(f"Correlation Heatmap (Top {head} Features vs Winners)", fontsize=14)
# plt.tight_layout()

# # Save directly to file
# # plt.savefig(f"./dispatch_tuner/single_gemm/{filename}_correlation_heatmap.png", dpi=300)
# plt.close()
# # print(f"Correlation heatmap saved to ./dispatch_tuner/single_gemm/{filename}_correlation_heatmap.png")





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
