# rf_global.py
import glob
import os
import json
import ast
import numpy as np
import pandas as pd

from typing import List, Optional, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import make_scorer
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr

# ----------------------------
# Configuration you can tweak
# ----------------------------

# 1) Where your CSVs are
CSV_GLOB = "./dispatch_tuner/tuning_database/*.csv"  # change to your pattern or use an explicit list

# 2) What to exclude from cfg.* (your list)
EXCLUDED_CFG = [
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

    'cfg.M', 'cfg.N', 'cfg.K',          # keep excluded if you want "tuning-only" analysis
    'cfg.wg_z', 'cfg.subgroup_size',
    'cfg.workgroup_tile_size_z',
    'cfg.subgroup_tile_size_x',
    'cfg.subgroup_tile_size_y',
    'cfg.subgroup_tile_size_z',
    'cfg.promote_operand_1',
    'cfg.promote_operand_2',
    'cfg.reduction_tile_size_1',
    'cfg.reduction_tile_size_2',
    # if you want to *include* any of the above, just comment them out.
]

# 3) Which columns identify a "group/problem" for leakage control
GROUP_COL = "dispatch_id"  # or "candidate_spec_mlir", or add a fallback to the CSV file path

# 4) Target column
TARGET_COL = "norm_speedup"

# 5) Keep only rows that actually have a benchmark result
REQUIRE_BENCHMARK = True

# ----------------------------
# Helpers
# ----------------------------

def safe_bool_series(s: pd.Series) -> pd.Series:
    """Convert strings like 'TRUE'/'FALSE'/True/False/1/0/None to clean booleans if present."""
    if s.dtype == bool:
        return s
    mapping = {"TRUE": True, "True": True, "true": True, "1": True,
               "FALSE": False, "False": False, "false": False, "0": False,
               "None": None, "": None, "nan": None}
    return s.astype(str).map(mapping).astype("boolean")

def try_parse_listlike(x):
    """If a cell looks like a python list '[1,2,3]' or '[None]', parse it; else return as-is."""
    if isinstance(x, str) and x.startswith('[') and x.endswith(']'):
        try:
            return ast.literal_eval(x)
        except Exception:
            return x
    return x

def read_and_tag(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    df["__source_file__"] = os.path.basename(path)

    # Normalize obvious boolean-ish columns (optional)
    for col in ["compile_status", "to_benchmark", "benchmark_status"]:
        if col in df.columns:
            df[col] = safe_bool_series(df[col])

    # Parse list-like strings only if you plan to use them (we excluded most of them)
    # Here we leave them as strings to avoid exploding the schema.
    return df

def build_feature_matrix(
    df: pd.DataFrame,
    excluded_cfg: List[str]
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    # choose cfg.* columns excluding your list
    cfg_cols = [c for c in df.columns if c.startswith("cfg.") and c not in excluded_cfg]

    # Optional: include a few non-cfg categorical controls (e.g., arch/device/op_kind)
    for extra in ["arch", "op_kind", "device"]:
        if extra in df.columns and extra not in cfg_cols:
            cfg_cols.append(extra)

    # Strongly recommend including 'cfg.mma_attr' (string category)
    # If it's not already in, add it if present
    if "cfg.mma_attr" in df.columns and "cfg.mma_attr" not in cfg_cols:
        cfg_cols.append("cfg.mma_attr")

    # Figure out numeric vs categorical
    numeric_cols = df[cfg_cols].select_dtypes(include=['number', 'boolean']).columns.tolist()
    cat_cols = [c for c in cfg_cols if c not in numeric_cols]

    # Keep only columns we actually use
    X = df[cfg_cols].copy()

    return X, numeric_cols, cat_cols

def spearman_scorer(y_true, y_pred):
    # if y_true has no variance, Spearman is undefined; fallback to 0
    if np.all(y_true == y_true[0]):
        return 0.0
    rho, _ = spearmanr(y_true, y_pred)
    # spearmanr may return nan with ties—treat as 0
    return 0.0 if np.isnan(rho) else float(rho)

def build_pipeline(numeric_cols: List[str], cat_cols: List[str]) -> Pipeline:
    numeric_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        # No scaling for trees
    ])
    cat_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    pre = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", cat_pipe, cat_cols),
    ], remainder="drop")

    rf = RandomForestRegressor(
        n_estimators=800,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=0,
        oob_score=False,
    )

    return Pipeline([
        ("pre", pre),
        ("rf", rf),
    ])

def per_group_weights(groups: pd.Series) -> np.ndarray:
    """Inverse-frequency weights so each group contributes ~equally."""
    vc = groups.value_counts()
    return groups.map(lambda g: 1.0 / vc[g]).values

# ----------------------------
# Main
# ----------------------------

def main():
    paths = sorted(glob.glob(CSV_GLOB))
    if not paths:
        raise SystemExit(f"No CSVs matched pattern: {CSV_GLOB}")

    frames = [read_and_tag(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)

    # Keep only rows with valid target
    if TARGET_COL not in df.columns:
        # Try to build norm_speedup from baseline and measured times if available
        # speedup = baseline_time / time
        if {"baseline_benchmark_time_ms", "benchmark_time_ms"}.issubset(df.columns):
            df[TARGET_COL] = df["baseline_benchmark_time_ms"] / df["benchmark_time_ms"]
        else:
            raise SystemExit(f"Target {TARGET_COL} not found and cannot be derived.")

    if REQUIRE_BENCHMARK and "benchmark_status" in df.columns:
        df = df[df["benchmark_status"] == True]

    # Drop impossible targets
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[TARGET_COL])

    # Group key for CV
    group_key = GROUP_COL if GROUP_COL in df.columns else "__source_file__"
    if group_key not in df.columns:
        group_key = "__source_file__"  # guaranteed present from read_and_tag

    # Build features
    X_raw, numeric_cols, cat_cols = build_feature_matrix(df, EXCLUDED_CFG)

    # Impute/encode through pipeline; no manual dropna on features—let imputers handle it
    y = df[TARGET_COL].astype(float).values
    groups = df[group_key].astype(str)

    # Optional: rebalance groups via sample weights
    sample_weight = per_group_weights(groups)

    pipe = build_pipeline(numeric_cols, cat_cols)

    # Grouped CV with Spearman
    gkf = GroupKFold(n_splits=min(5, groups.nunique()))
    spearman_cv = make_scorer(spearman_scorer, greater_is_better=True)

    scores = []
    for fold, (tr, te) in enumerate(gkf.split(X_raw, y, groups=groups), start=1):
        pipe.fit(X_raw.iloc[tr], y[tr], rf__sample_weight=sample_weight[tr])
        y_pred = pipe.predict(X_raw.iloc[te])
        s = spearman_scorer(y[te], y_pred)
        scores.append(s)
        print(f"[Fold {fold}] Spearman={s:.3f} (n_test={len(te)})")

    print(f"\nMean Spearman across folds: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

    # Fit final on all data for importances
    pipe.fit(X_raw, y, rf__sample_weight=sample_weight)

    # Report impurity-based importances
    rf: RandomForestRegressor = pipe.named_steps["rf"]
    # Get the transformed feature names to map importances
    pre: ColumnTransformer = pipe.named_steps["pre"]

    def transformed_feature_names(pre: ColumnTransformer) -> List[str]:
        names = []
        for name, trans, cols in pre.transformers_:
            if name == "remainder" and trans == "drop":
                continue
            if isinstance(cols, slice):
                # should not happen here, but handle for completeness
                cols = pre._feature_names_in_[cols]
            if hasattr(trans, "get_feature_names_out"):
                # mostly for OneHot; for us, we keep original column names
                try:
                    feats = trans.get_feature_names_out(cols)
                    names.extend(list(feats))
                except Exception:
                    names.extend(list(cols))
            else:
                names.extend(list(cols))
        return names

    feat_names = transformed_feature_names(pre)
    imp = pd.Series(rf.feature_importances_, index=feat_names).sort_values(ascending=False)
    print("\nTop 25 impurity-based feature importances:")
    print(imp.head(25))

    # Permutation importance on a held-out split (first fold)
    tr, te = next(iter(gkf.split(X_raw, y, groups=groups)))
    pipe.fit(X_raw.iloc[tr], y[tr], rf__sample_weight=sample_weight[tr])
    perm = permutation_importance(
        pipe, X_raw.iloc[te], y[te],
        n_repeats=10, random_state=0, n_jobs=-1, scoring=spearman_cv
    )
    perm_imp = pd.Series(perm.importances_mean, index=feat_names).sort_values(ascending=False)
    print("\nTop 25 permutation importances (Spearman on held-out fold):")
    print(perm_imp.head(25))

if __name__ == "__main__":
    main()
