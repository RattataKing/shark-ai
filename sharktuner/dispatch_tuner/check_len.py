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

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils import check_random_state

# Core
import numpy as np

# Optional (if reading CSVs / using DataFrames)
import pandas as pd

# scikit-learn bits used by the RankSVM implementation
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils import check_random_state

# Optional (for group-aware cross-validation)
from sklearn.model_selection import GroupKFold
# --- RankSVM (pairwise) minimal implementation --------------------------------
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils import check_random_state

# --- RankSVM (pairwise) with symmetric +/- pairs + diagnostics ---------------
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils import check_random_state

import glob
import pandas as pd


files = glob.glob('./dispatch_tuner/tuning_database_clean/*.csv')
# features = ['cfg.intrinsic_mn', 'cfg.intrinsic_k',
#         'cfg.m', 'cfg.n', 'cfg.k',
#         'cfg.sg_m_cnt', 'cfg.sg_n_cnt', 'cfg.num_subgroups',
#         'cfg.lhs_tile_size', 'cfg.rhs_tile_size']
# for f in files:
#     df = pd.read_csv(f)
    
#     if len(df) >= 2000:
#         print(len(df))
#         print(os.path.basename(f))
#         # dfb = df[df["benchmark_status"]==True]
#         # print(len(dfb))
#         # if len(dfb)>2000:
#         #     print(len(dfb))
#         #     print(os.path.basename(f))
#         # print(len(df))
#         continue

# exit()


# files = glob.glob('./dispatch_tuner/tuning_database_clean/*.csv')

# dfs = []
# for f in files:
#     try:
#         df = pd.read_csv(f)
#         # df["source_file"] = Path(f).stem
#         dfs.append(df)
#     except Exception as e:
#         print(f"Error reading {f}: {e}")

# full_df = pd.concat(dfs, ignore_index=True)

# x_feature = "cfg.intrinsic_k"
# y_feature = "norm_speedup"
# plt.figure(figsize=(8,6))
# plt.scatter(full_df[x_feature], full_df[y_feature], alpha=0.5, s=15)
# plt.xlabel(x_feature)
# plt.ylabel(y_feature)
# plt.title(f"{x_feature} vs {y_feature} across CSVs")
# plt.grid(True, linestyle="--", alpha=0.5)

# script_dir = os.path.dirname(os.path.abspath(__file__))
# save_path = os.path.join(script_dir, f"x_feature_vs_speedup.png")
# plt.savefig(save_path, dpi=300, bbox_inches="tight")
# plt.close()

# print(f"Saved scatter plot to {save_path}")

# exit()




# all_values = set()

# for f in files:
#     try:
#         df = pd.read_csv(f)
#         if "cfg.mma_attr" in df.columns:
#             all_values.update(df["cfg.mma_attr"].dropna().unique())
#     except Exception as e:
#         print(f"Error reading {f}: {e}")

# print(f"Total distinct cfg.mma_attr values: {len(all_values)}")
# print("Values:", all_values)

# exit()



features = [
    'cfg.intrinsic_mn','cfg.intrinsic_k','cfg.m','cfg.n','cfg.k',
    'cfg.sg_m_cnt','cfg.sg_n_cnt','cfg.num_subgroups',
    'cfg.lhs_tile_size','cfg.rhs_tile_size'
]
rank_col = 'benchmark_result_order'

dfs, lookups = {}, {}
for f in files:
    df = pd.read_csv(f).copy()
    df['__key__'] = df[features].astype(str).agg('|'.join, axis=1)
    # best (min) rank per feature key in this file (drop NaNs)
    best = (df[['__key__', rank_col]]
            .dropna(subset=[rank_col])
            .groupby('__key__', as_index=True)[rank_col]
            .min())
    lookups[f] = best  # Series: key -> best rank
    dfs[f] = df.set_index('__key__')

for f, df in dfs.items():
    # winners (top-5), deduped by feature key using best/min rank
    winners_best = (df[df[rank_col] <= 5][[rank_col]]
                    .dropna()
                    .groupby(level=0)[rank_col]
                    .min())

    for key, origin_rank in winners_best.items():
        ranks = []
        for other_f, other_best in lookups.items():
            if other_f == f: 
                continue
            r = other_best.get(key, None)
            if pd.notna(r):
                ranks.append(int(r))  # only keep found, non-NaN ranks
        print(int(origin_rank), ranks)

exit()


