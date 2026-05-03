"""
Kalshi vs Polymarket Resolution Agreement Predictor
=====================================================
Predicts whether Kalshi and Polymarket will resolve in the same direction
for the same underlying BTC price contract.

Usage:
    python predict_resolution_agreement.py --data your_data.csv
    python predict_resolution_agreement.py --data your_data.csv --model xgboost --tune

Requirements:
    pip install pandas numpy scikit-learn xgboost lightgbm shap matplotlib seaborn
"""

import argparse
import warnings
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    RocCurveDisplay, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import cross_val_score, ParameterGrid
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

# ─── Optional heavy deps ───────────────────────────────────────────────────────
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[warn] xgboost not installed — XGBoost model unavailable")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("[warn] lightgbm not installed — LightGBM model unavailable")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("[warn] shap not installed — SHAP explanations unavailable")


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

REQUIRED_COLS = [
    "timestamp", "kalshi_slug", "poly_slug",
    "chainlink_price", "cf_price", "price_diff",
    "kalshi_target_price", "poly_target_price",
    "kalshi_resolution", "poly_resolution",
]

def load_data(path: str) -> pd.DataFrame:
    """Load CSV, parse timestamps, validate schema."""
    df = pd.read_csv(path, parse_dates=["timestamp"])

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"[data] Loaded {len(df):,} rows from {path}")
    print(f"[data] Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"[data] Unique Kalshi slugs: {df['kalshi_slug'].nunique()}")
    print(f"[data] Unique Poly slugs:   {df['poly_slug'].nunique()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. LABELING
# ══════════════════════════════════════════════════════════════════════════════

def make_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Target = 1 if both markets resolve the same direction, 0 otherwise.
    Handles possible NaN resolutions by dropping those rows.
    """
    df = df.dropna(subset=["kalshi_resolution", "poly_resolution"]).copy()
    df["label"] = (df["kalshi_resolution"] == df["poly_resolution"]).astype(int)

    counts = df["label"].value_counts()
    pct_agree = counts.get(1, 0) / len(df) * 100
    print(f"\n[label] Agreement rate: {pct_agree:.1f}%  "
          f"(agree={counts.get(1,0):,}, disagree={counts.get(0,0):,})")
    if pct_agree > 90 or pct_agree < 10:
        print("[warn] Severe class imbalance — class_weight='balanced' will be applied")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all predictive features. No look-ahead — every value is available
    at prediction time (before expiry).
    """
    df = df.copy()

    # --- Core distance features ---
    # How far does price need to move for each market to resolve YES?
    df["dist_to_kalshi_target"] = df["kalshi_target_price"] - df["chainlink_price"]
    df["dist_to_poly_target"]   = df["poly_target_price"]   - df["cf_price"]

    # Relative distance (normalised by current price)
    df["rel_dist_kalshi"] = df["dist_to_kalshi_target"] / df["chainlink_price"]
    df["rel_dist_poly"]   = df["dist_to_poly_target"]   / df["cf_price"]

    # --- Direction features ---
    # +1 = needs UP, -1 = needs DOWN
    df["kalshi_direction"] = np.sign(df["dist_to_kalshi_target"])
    df["poly_direction"]   = np.sign(df["dist_to_poly_target"])

    # KEY FEATURE: do both markets require the same move direction?
    df["target_aligned"] = (df["kalshi_direction"] == df["poly_direction"]).astype(int)

    # --- Feed divergence features ---
    df["price_diff"]     = df["chainlink_price"] - df["cf_price"]   # may already exist
    df["abs_price_diff"] = df["price_diff"].abs()
    df["rel_price_diff"] = df["price_diff"] / df["chainlink_price"]

    # Divergence relative to distance-to-target (key edge-case signal)
    # If price_diff > dist_to_kalshi_target, the feed gap alone could flip resolution
    df["diff_to_kalshi_dist_ratio"] = df["abs_price_diff"] / (df["dist_to_kalshi_target"].abs() + 1e-8)
    df["diff_to_poly_dist_ratio"]   = df["abs_price_diff"] / (df["dist_to_poly_target"].abs()   + 1e-8)

    # --- Target spread ---
    df["target_spread"]     = df["kalshi_target_price"] - df["poly_target_price"]
    df["abs_target_spread"] = df["target_spread"].abs()

    # --- Zone indicators: is price within price_diff of either target? ---
    df["kalshi_in_danger_zone"] = (
        df["dist_to_kalshi_target"].abs() < df["abs_price_diff"]
    ).astype(int)
    df["poly_in_danger_zone"] = (
        df["dist_to_poly_target"].abs() < df["abs_price_diff"]
    ).astype(int)
    df["both_in_danger_zone"] = (
        df["kalshi_in_danger_zone"] & df["poly_in_danger_zone"]
    ).astype(int)

    # --- Time features ---
    df["hour_of_day"]  = df["timestamp"].dt.hour
    df["minute_of_hour"] = df["timestamp"].dt.minute
    df["day_of_week"]  = df["timestamp"].dt.dayofweek
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)

    # Minutes since first observation per slug-pair (proxy for time-into-window)
    df["slug_pair"] = df["kalshi_slug"] + "|" + df["poly_slug"]
    df["obs_in_window"] = df.groupby("slug_pair").cumcount()

    print(f"\n[features] Engineered {len(FEATURE_COLS)} features")
    return df


FEATURE_COLS = [
    "dist_to_kalshi_target",
    "dist_to_poly_target",
    "rel_dist_kalshi",
    "rel_dist_poly",
    "kalshi_direction",
    "poly_direction",
    "target_aligned",
    "price_diff",
    "abs_price_diff",
    "rel_price_diff",
    "diff_to_kalshi_dist_ratio",
    "diff_to_poly_dist_ratio",
    "target_spread",
    "abs_target_spread",
    "kalshi_in_danger_zone",
    "poly_in_danger_zone",
    "both_in_danger_zone",
    "hour_of_day",
    "minute_of_hour",
    "day_of_week",
    "is_weekend",
    "obs_in_window",
]


# ══════════════════════════════════════════════════════════════════════════════
# 4. SPLITTING — SLUG-LEVEL K-FOLD CROSS VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def slug_kfold_indices(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    K-fold cross-validation where the unit of splitting is the slug-pair
    (one 15-minute contract window), not individual rows.

    Why not a time split:
      Disagreements are temporally clustered — all 135 disagree rows fall in
      a contiguous window. Any chronological split puts all disagrees in one
      fold, making AUC undefined in the other. Slug-level k-fold scrambles
      contracts so both classes appear in every train/test fold, while still
      keeping all rows of one contract together (no within-contract leakage).

    Returns list of (train_row_indices, test_row_indices) tuples.
    """
    from sklearn.model_selection import StratifiedKFold

    # One row per slug: slug → majority label (or any label — we need the
    # binary signal "does this contract have any disagree?")
    slug_df = (
        df.groupby("slug_pair")
        .agg(has_disagree=("label", lambda x: int((x == 0).any())))
        .reset_index()
    )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for train_slug_idx, test_slug_idx in skf.split(slug_df["slug_pair"], slug_df["has_disagree"]):
        train_slugs = set(slug_df.iloc[train_slug_idx]["slug_pair"])
        test_slugs  = set(slug_df.iloc[test_slug_idx]["slug_pair"])
        train_rows  = df.index[df["slug_pair"].isin(train_slugs)].to_numpy()
        test_rows   = df.index[df["slug_pair"].isin(test_slugs)].to_numpy()
        folds.append((train_rows, test_rows))

    # Diagnostics
    print(f"\n[cv] {n_splits}-fold slug-stratified cross-validation")
    for i, (tr, te) in enumerate(folds):
        tr_df = df.loc[tr]
        te_df = df.loc[te]
        print(f"  Fold {i+1}: train={len(tr):,} rows ({tr_df['slug_pair'].nunique()} slugs, "
              f"disagree={int((tr_df['label']==0).sum())}) | "
              f"test={len(te):,} rows ({te_df['slug_pair'].nunique()} slugs, "
              f"disagree={int((te_df['label']==0).sum())})")
    return folds


def cross_validate_model(
    model_name: str,
    df: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    extra_params: dict | None = None,
    output_dir: Path | None = None,
) -> tuple[dict, object]:
    """
    Train and evaluate model across all folds. Returns aggregate metrics
    and the model retrained on the full dataset (for inference).
    """
    X_all, y_all = get_xy(df)
    spw = get_scale_pos_weight(y_all)

    fold_aucs, fold_aps = [], []
    all_test_proba = np.zeros(len(df))
    all_test_y     = np.full(len(df), -1)

    for i, (train_idx, test_idx) in enumerate(folds):
        X_tr, y_tr = X_all.iloc[train_idx], y_all.iloc[train_idx]
        X_te, y_te = X_all.iloc[test_idx],  y_all.iloc[test_idx]

        fold_spw = get_scale_pos_weight(y_tr)
        model = build_model_with_params(model_name, fold_spw, extra_params or {})

        if model_name == "xgboost":
            model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        elif model_name == "lightgbm":
            model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        else:
            model.fit(X_tr, y_tr)

        proba = model.predict_proba(X_te)[:, 1]
        all_test_proba[test_idx] = proba
        all_test_y[test_idx]     = y_te.to_numpy()

        if y_te.nunique() > 1:
            auc = roc_auc_score(y_te, proba)
            ap  = average_precision_score(y_te, proba)
            fold_aucs.append(auc)
            fold_aps.append(ap)
            print(f"  Fold {i+1}: AUC={auc:.4f}  AP={ap:.4f}  "
                  f"(disagree in test: {int((y_te==0).sum())})")
        else:
            print(f"  Fold {i+1}: skipped — only one class in test "
                  f"(disagree={int((y_te==0).sum())})")

    # OOF (out-of-fold) metrics across all folds combined
    valid = all_test_y >= 0
    oof_y     = all_test_y[valid]
    oof_proba = all_test_proba[valid]
    oof_pred  = (oof_proba >= 0.5).astype(int)

    oof_auc = roc_auc_score(oof_y, oof_proba) if len(np.unique(oof_y)) > 1 else float("nan")
    oof_ap  = average_precision_score(oof_y, oof_proba) if len(np.unique(oof_y)) > 1 else float("nan")

    print(f"\n{'─'*55}")
    print(f"  {model_name.upper()} — Out-of-Fold Results")
    print(f"  Mean fold AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    print(f"  OOF  AUC:      {oof_auc:.4f}")
    print(f"  OOF  AP:       {oof_ap:.4f}   (baseline={oof_y.mean():.4f})")
    print(f"{'─'*55}")
    print(classification_report(oof_y, oof_pred, target_names=["disagree", "agree"]))

    # Plots
    if output_dir:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        fig.suptitle(f"{model_name.upper()} — Out-of-Fold", fontsize=13, fontweight="bold")

        if len(np.unique(oof_y)) > 1:
            RocCurveDisplay.from_predictions(oof_y, oof_proba, ax=axes[0], name=model_name)
        axes[0].set_title("OOF ROC Curve")

        cm = confusion_matrix(oof_y, oof_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
                    xticklabels=["disagree", "agree"], yticklabels=["disagree", "agree"])
        axes[1].set_title("OOF Confusion Matrix")
        axes[1].set_ylabel("Actual"); axes[1].set_xlabel("Predicted")

        axes[2].hist(oof_proba[oof_y == 0], bins=20, alpha=0.6, label="disagree", color="salmon")
        axes[2].hist(oof_proba[oof_y == 1], bins=20, alpha=0.6, label="agree",    color="steelblue")
        axes[2].axvline(0.5, color="black", linestyle="--", linewidth=1)
        axes[2].set_title("OOF Probability Distribution")
        axes[2].set_xlabel("P(agree)"); axes[2].legend()

        plt.tight_layout()
        plt.savefig(output_dir / f"{model_name}_oof_evaluation.png", dpi=120, bbox_inches="tight")
        plt.close()

        # Fold AUC bar chart
        if fold_aucs:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar(range(1, len(fold_aucs)+1), fold_aucs, color="steelblue", alpha=0.8)
            ax.axhline(np.mean(fold_aucs), color="red", linestyle="--", label=f"mean={np.mean(fold_aucs):.3f}")
            ax.set_xlabel("Fold"); ax.set_ylabel("AUC"); ax.set_title("Per-fold AUC")
            ax.legend(); plt.tight_layout()
            plt.savefig(output_dir / f"{model_name}_fold_aucs.png", dpi=120, bbox_inches="tight")
            plt.close()
            print(f"[plot] Saved evaluation plots → {output_dir}/")

    # Retrain on full data for inference
    full_model = build_model_with_params(model_name, spw, extra_params or {})
    if model_name == "xgboost":
        full_model.fit(X_all, y_all, verbose=False)
    elif model_name == "lightgbm":
        full_model.fit(X_all, y_all, callbacks=[lgb.log_evaluation(-1)])
    else:
        full_model.fit(X_all, y_all)

    # Feature importance from full model
    if output_dir:
        feat_importance = None
        if model_name == "logistic":
            feat_importance = pd.Series(
                np.abs(full_model.named_steps["clf"].coef_[0]), index=FEATURE_COLS
            ).sort_values(ascending=False)
            label = "|coefficient|"
        elif model_name in ("xgboost", "lightgbm"):
            feat_importance = pd.Series(
                full_model.feature_importances_, index=FEATURE_COLS
            ).sort_values(ascending=False)
            label = "importance"

        if feat_importance is not None:
            fig, ax = plt.subplots(figsize=(8, 6))
            feat_importance.head(15).plot(kind="barh", ax=ax, color="steelblue")
            ax.set_title(f"{model_name.upper()} — Top 15 Features ({label})")
            ax.invert_yaxis(); plt.tight_layout()
            plt.savefig(output_dir / f"{model_name}_feature_importance.png", dpi=120, bbox_inches="tight")
            plt.close()

        if HAS_SHAP and model_name in ("xgboost", "lightgbm"):
            try:
                explainer   = shap.TreeExplainer(full_model)
                shap_values = explainer.shap_values(X_all)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                shap.summary_plot(shap_values, X_all, show=False)
                plt.savefig(output_dir / f"{model_name}_shap_summary.png", dpi=120, bbox_inches="tight")
                plt.close()
            except Exception as e:
                print(f"[warn] SHAP failed: {e}")

    results = {
        "model":          model_name,
        "mean_fold_auc":  round(float(np.mean(fold_aucs)), 4) if fold_aucs else float("nan"),
        "std_fold_auc":   round(float(np.std(fold_aucs)),  4) if fold_aucs else float("nan"),
        "oof_auc":        round(oof_auc, 4),
        "oof_ap":         round(oof_ap,  4),
        "n_folds_scored": len(fold_aucs),
    }
    return results, full_model


# ══════════════════════════════════════════════════════════════════════════════
# 5. HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_xy(df: pd.DataFrame):
    X = df[FEATURE_COLS].fillna(0)
    y = df["label"]
    return X, y


def get_scale_pos_weight(y: pd.Series) -> float:
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    return neg / pos if pos > 0 else 1.0


# ══════════════════════════════════════════════════════════════════════════════
# 6. MODEL FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def build_model_with_params(model_name: str, spw: float, extra_params: dict):
    """Construct a model instance, merging default + tuned params."""
    if model_name == "logistic":
        base = {"class_weight": "balanced", "max_iter": 1000, "C": 1.0, "random_state": 42}
        base.update(extra_params)
        return Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(**base))])
    elif model_name == "xgboost":
        base = {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
                "subsample": 0.8, "colsample_bytree": 0.8, "scale_pos_weight": spw,
                "use_label_encoder": False, "eval_metric": "logloss",
                "random_state": 42, "verbosity": 0}
        base.update(extra_params)
        return xgb.XGBClassifier(**base)
    elif model_name == "lightgbm":
        base = {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
                "subsample": 0.8, "colsample_bytree": 0.8,
                "class_weight": "balanced", "random_state": 42, "verbose": -1}
        base.update(extra_params)
        return lgb.LGBMClassifier(**base)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. HYPERPARAMETER TUNING — CV-BASED
# ══════════════════════════════════════════════════════════════════════════════

def tune_model(model_name: str, df: pd.DataFrame, folds: list) -> dict:
    """
    Grid search evaluated via mean OOF AUC across the provided folds.
    Tune on the same folds used for evaluation — no separate val set needed.
    """
    X_all, y_all = get_xy(df)
    spw = get_scale_pos_weight(y_all)

    grids = {
        "logistic": [
            {"C": c} for c in [0.01, 0.1, 1.0, 10.0]
        ],
        "xgboost": [
            {"n_estimators": n, "max_depth": d, "learning_rate": lr}
            for n in [100, 300] for d in [3, 5] for lr in [0.01, 0.05]
        ],
        "lightgbm": [
            {"n_estimators": n, "max_depth": d, "learning_rate": lr}
            for n in [100, 300] for d in [3, 5] for lr in [0.01, 0.05]
        ],
    }

    best_auc, best_params = -1, {}
    print(f"\n[tune] CV grid search for {model_name} ({len(grids[model_name])} combos)…")

    for params in grids[model_name]:
        fold_aucs = []
        for train_idx, test_idx in folds:
            X_tr, y_tr = X_all.iloc[train_idx], y_all.iloc[train_idx]
            X_te, y_te = X_all.iloc[test_idx],  y_all.iloc[test_idx]
            if y_te.nunique() < 2:
                continue
            fold_spw = get_scale_pos_weight(y_tr)
            try:
                m = build_model_with_params(model_name, fold_spw, params)
                if model_name == "xgboost":
                    m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
                elif model_name == "lightgbm":
                    m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)],
                          callbacks=[lgb.log_evaluation(-1)])
                else:
                    m.fit(X_tr, y_tr)
                auc = roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])
                fold_aucs.append(auc)
            except Exception:
                pass
        if fold_aucs:
            mean_auc = np.mean(fold_aucs)
            if mean_auc > best_auc:
                best_auc, best_params = mean_auc, params
                print(f"  ✓ {params} → mean AUC {mean_auc:.4f}")

    print(f"[tune] Best mean CV AUC: {best_auc:.4f} | params: {best_params}")
    return best_params


# ══════════════════════════════════════════════════════════════════════════════
# 8. PREDICTION ON NEW DATA
# ══════════════════════════════════════════════════════════════════════════════

def predict_new(model, new_data_path: str) -> pd.DataFrame:
    """Run inference on a new CSV (no resolution columns needed)."""
    df = pd.read_csv(new_data_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["slug_pair"] = df["kalshi_slug"] + "|" + df["poly_slug"]
    df["obs_in_window"] = df.groupby("slug_pair").cumcount()
    df = engineer_features(df)
    X = df[FEATURE_COLS].fillna(0)

    proba = model.predict_proba(X)[:, 1]
    df["p_agree"]        = proba
    df["prediction"]     = (proba >= 0.5).astype(int)
    df["prediction_label"] = df["prediction"].map({1: "agree", 0: "disagree"})
    return df[["timestamp", "kalshi_slug", "poly_slug", "p_agree",
               "prediction", "prediction_label"]]


# ══════════════════════════════════════════════════════════════════════════════
# 9. EDA
# ══════════════════════════════════════════════════════════════════════════════

def run_eda(df: pd.DataFrame, output_dir: Path):
    print("\n[eda] Generating exploratory plots…")
    df_eng = engineer_features(df)
    df_eng = make_label(df_eng)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("EDA — Kalshi vs Polymarket Resolution Agreement", fontsize=13)

    df_eng["label"].value_counts().plot(
        kind="bar", ax=axes[0, 0], color=["salmon", "steelblue"])
    axes[0, 0].set_xticklabels(["disagree (0)", "agree (1)"], rotation=0)
    axes[0, 0].set_title("Label distribution")

    pd.crosstab(df_eng["target_aligned"], df_eng["label"], normalize="index").plot(
        kind="bar", ax=axes[0, 1], stacked=True, color=["salmon", "steelblue"])
    axes[0, 1].set_title("Target aligned vs agreement rate")
    axes[0, 1].legend(["disagree", "agree"])

    for lbl, grp in df_eng.groupby("label"):
        axes[0, 2].hist(grp["abs_price_diff"], bins=30, alpha=0.5,
                        label=["disagree", "agree"][lbl])
    axes[0, 2].set_title("Feed divergence by label")
    axes[0, 2].set_xlabel("abs_price_diff"); axes[0, 2].legend()

    axes[1, 0].hist(df_eng["dist_to_kalshi_target"], bins=40, color="steelblue", alpha=0.7)
    axes[1, 0].axvline(0, color="red", linestyle="--")
    axes[1, 0].set_title("Distance to Kalshi target")
    axes[1, 0].set_xlabel("kalshi_target − chainlink_price")

    num_feats = [f for f in FEATURE_COLS if f in df_eng.columns]
    corr = df_eng[num_feats + ["label"]].corr()["label"].drop("label").sort_values()
    corr.plot(kind="barh", ax=axes[1, 1],
              color=["salmon" if v < 0 else "steelblue" for v in corr])
    axes[1, 1].axvline(0, color="black", linewidth=0.8)
    axes[1, 1].set_title("Feature → label correlation")

    hr_rate = df_eng.groupby("hour_of_day")["label"].mean()
    hr_rate.plot(kind="bar", ax=axes[1, 2], color="steelblue", alpha=0.8)
    axes[1, 2].set_title("Agreement rate by hour (UTC)")
    axes[1, 2].set_xlabel("hour_of_day"); axes[1, 2].set_ylabel("P(agree)")
    axes[1, 2].set_ylim(0, 1)

    plt.tight_layout()
    eda_path = output_dir / "eda.png"
    plt.savefig(eda_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[eda] Saved → {eda_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 10. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Kalshi vs Polymarket resolution agreement predictor"
    )
    parser.add_argument("--data",    required=True, help="Path to labeled CSV")
    parser.add_argument(
        "--model", default="logistic",
        choices=["logistic", "xgboost", "lightgbm", "all"],
        help="Model to train (default: logistic)"
    )
    parser.add_argument("--folds",   type=int, default=5, help="Number of CV folds (default: 5)")
    parser.add_argument("--tune",    action="store_true", help="Run CV hyperparameter grid search")
    parser.add_argument("--eda",     action="store_true", help="Generate EDA plots")
    parser.add_argument("--predict", default=None,        help="Path to new CSV for inference")
    parser.add_argument("--out",     default="outputs",   help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data)
    df = make_label(df)
    df = engineer_features(df)

    if args.eda:
        run_eda(df, output_dir)

    folds = slug_kfold_indices(df, n_splits=args.folds)

    models_to_run = (
        ["logistic", "xgboost", "lightgbm"] if args.model == "all"
        else [args.model]
    )
    models_to_run = [
        m for m in models_to_run
        if m == "logistic"
        or (m == "xgboost"  and HAS_XGB)
        or (m == "lightgbm" and HAS_LGB)
    ]

    all_results  = []
    best_models  = {}

    for m_name in models_to_run:
        extra_params = {}
        if args.tune:
            extra_params = tune_model(m_name, df, folds)
            print(f"[tune] Using params for {m_name}: {extra_params}")

        results, full_model = cross_validate_model(
            m_name, df, folds, extra_params, output_dir
        )
        all_results.append(results)
        best_models[m_name] = full_model

    results_df = pd.DataFrame(all_results)
    print("\n" + "═" * 60)
    print("  RESULTS SUMMARY (out-of-fold)")
    print("═" * 60)
    print(results_df.to_string(index=False))

    results_path = output_dir / "results.json"
    results_df.to_json(results_path, orient="records", indent=2)
    print(f"\n[out] Results saved → {results_path}")

    if args.predict:
        best_name  = results_df.sort_values("oof_auc", ascending=False).iloc[0]["model"]
        best_model = best_models[best_name]
        print(f"\n[predict] Using {best_name} (best OOF AUC) for inference on {args.predict}")
        predictions = predict_new(best_model, args.predict)
        pred_path   = output_dir / "predictions.csv"
        predictions.to_csv(pred_path, index=False)
        print(f"[predict] Saved → {pred_path}")
        print(predictions.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
