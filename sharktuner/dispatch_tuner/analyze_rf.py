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
    files, test_size=0.2,
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

def sanitize_df(df):
    old_shape = old_shape = df.shape
    df = df.dropna(axis=1, how="all")
    print(f"Dataset shape after sanitized: {old_shape} -> {df.shape}")
    return df

def prepare_features(df):
    cfg_cols = [c for c in df.columns if c.startswith("cfg.") and c not in excluded_list]
    # numeric subset
    numeric_cols = df[cfg_cols].select_dtypes(include="number").columns
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

    X_num = df[numeric_cols]
    X_all = pd.concat([X_num, X_cat], axis=1)
    y = df["norm_speedup"]

    return X_all, y


train_dfs = [pd.read_csv(f) for f in train_files]
train_df = pd.concat(train_dfs, ignore_index=True)
train_df = sanitize_df(train_df)

X_train, y_train = prepare_features(train_df)
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

# Spearman correlation among numeric features
df=train_df
cfg_cols = [c for c in df.columns if c.startswith("cfg.") and c not in excluded_list]
numeric_cols = df[cfg_cols].select_dtypes(include="number").columns
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

script_dir = os.path.dirname(os.path.abspath(__file__))
for i,f in enumerate(test_files):
    test_df = pd.read_csv(f)
    test_df = sanitize_df(test_df)

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