from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from sklearn.tree import export_text
import numpy as np
from rulefit import RuleFit

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
    (big_df["benchmark_result_order"] <= 10) &
    (big_df["benchmark_speedup"] < 1)
)
df=big_df

# 0) Drop columns that are entirely empty
df = df.dropna(axis=1, how="all")


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

    'cfg.mma_attr', # Str Class, need to do one-hot or label
]

cfg_cols = [c for c in df.columns if c.startswith("cfg.") and c not in excluded_list]
# numeric subset
numeric_cols = df[cfg_cols].select_dtypes(include="number").columns
# categorical subset (strings)
cat_cols = [c for c in cfg_cols if c not in numeric_cols]

selected_subset_cols = numeric_cols.tolist() + cat_cols
old_len = len(df)
df = df.dropna(subset=selected_subset_cols)
print(f"Before: {old_len} rows, After dropna: {len(df)} rows")

# one-hot encode categories
X_cat = pd.get_dummies(df[cat_cols].astype(str))
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


# Assume rf is trained RandomForestClassifier
# for i, tree in enumerate(clf.estimators_[:5]):  # first 5 trees
#     print(f"Tree {i}")
#     print(export_text(tree, feature_names=X_all.columns))

# exit()



# rf = RuleFit(tree_generator=clf)  # can pass your RF
# rf.fit(X_all.values, y.values, feature_names=X_all.columns)
# rules = rf.get_rules()
# print(rules[rules.coef != 0].sort_values("importance", ascending=False).head(10))

exit()




feature_names = np.array(X_all.columns)
X_mat = X_all.values
y_arr = y.values.astype(int)
base_rate = y_arr.mean()

def rules_from_tree(estimator, X_mat, y_arr, feature_names, max_depth=6):
    """
    Extract rules (one per leaf reachable within max_depth) from a single DecisionTree.
    Returns a list of dicts with: conditions, coverage, positives, precision, lift.
    """
    tree = estimator.tree_
    cl = tree.children_left
    cr = tree.children_right
    feat = tree.feature
    thr  = tree.threshold

    paths, path = [], []

    def recurse(node_id, depth):
        # stop if depth exceeded
        if depth > max_depth:
            return
        # leaf?
        if cl[node_id] == cr[node_id]:
            paths.append(list(path))  # record the path to this leaf
            return
        f = feat[node_id]
        t = thr[node_id]

        # left: <=
        path.append((f, "<=", t))
        recurse(cl[node_id], depth + 1)
        path.pop()

        # right: >
        path.append((f, ">", t))
        recurse(cr[node_id], depth + 1)
        path.pop()

    recurse(0, 0)

    rules = []
    for cond_triplets in paths:
        # Build mask
        mask = np.ones(len(X_mat), dtype=bool)
        conds_text = []
        for f_idx, op, t in cond_triplets:
            fname = feature_names[f_idx]
            if op == "<=":
                mask &= (X_mat[:, f_idx] <= t)
                conds_text.append(f"{fname} <= {t:.6g}")
            else:
                mask &= (X_mat[:, f_idx] >  t)
                conds_text.append(f"{fname} > {t:.6g}")

        cover = int(mask.sum())
        if cover == 0:
            continue
        pos = int(y_arr[mask].sum())
        prec = pos / cover
        lift = prec / base_rate if base_rate > 0 else np.inf

        rules.append({
            "conditions": conds_text,   # list[str]
            "coverage": cover,
            "positives": pos,
            "precision": prec,
            "lift": lift,
        })
    return rules

# Collect rules across the whole forest
all_rules = []
for est in clf.estimators_:
    all_rules.extend(rules_from_tree(est, X_mat, y_arr, feature_names, max_depth=6))

rules_df = pd.DataFrame(all_rules)
print(f"Extracted {len(rules_df)} raw rules")