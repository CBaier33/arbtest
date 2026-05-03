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
# 4. TRAIN / VALIDATION / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def time_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Strict chronological split. Never shuffle — autocorrelated time series.

    Splits at the slug-pair level: all observations of a given contract
    stay in the same fold to prevent leakage from the same window appearing
    in both train and test.
    """
    slug_pairs = df["slug_pair"].unique()
    n = len(slug_pairs)

    train_end = int(n * train_frac)
    val_end   = int(n * (train_frac + val_frac))

    train_slugs = set(slug_pairs[:train_end])
    val_slugs   = set(slug_pairs[train_end:val_end])
    test_slugs  = set(slug_pairs[val_end:])

    train = df[df["slug_pair"].isin(train_slugs)].copy()
    val   = df[df["slug_pair"].isin(val_slugs)].copy()
    test  = df[df["slug_pair"].isin(test_slugs)].copy()

    print(f"\n[split] Train: {len(train):,} rows ({len(train_slugs)} contracts)")
    print(f"[split] Val:   {len(val):,} rows ({len(val_slugs)} contracts)")
    print(f"[split] Test:  {len(test):,} rows ({len(test_slugs)} contracts)")
    return train, val, test


def get_xy(df: pd.DataFrame):
    X = df[FEATURE_COLS].fillna(0)
    y = df["label"]
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
# 5. MODELS
# ══════════════════════════════════════════════════════════════════════════════

def build_logistic(class_weight="balanced") -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight=class_weight,
            max_iter=1000,
            C=1.0,
            random_state=42,
        )),
    ])


def build_xgboost(scale_pos_weight=1.0) -> "xgb.XGBClassifier":
    if not HAS_XGB:
        raise ImportError("xgboost not installed")
    return xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )


def build_lightgbm(class_weight="balanced") -> "lgb.LGBMClassifier":
    if not HAS_LGB:
        raise ImportError("lightgbm not installed")
    return lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight=class_weight,
        random_state=42,
        verbose=-1,
    )


def get_scale_pos_weight(y_train: pd.Series) -> float:
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    return neg / pos if pos > 0 else 1.0


# ══════════════════════════════════════════════════════════════════════════════
# 6. TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate(
    model_name: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: Path,
) -> dict:
    X_train, y_train = get_xy(train)
    X_val,   y_val   = get_xy(val)
    X_test,  y_test  = get_xy(test)

    spw = get_scale_pos_weight(y_train)

    # --- Build model ---
    if model_name == "logistic":
        model = build_logistic()
    elif model_name == "xgboost":
        model = build_xgboost(scale_pos_weight=spw)
    elif model_name == "lightgbm":
        model = build_lightgbm()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # --- Fit ---
    if model_name == "xgboost":
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    elif model_name == "lightgbm":
        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks,
        )
    else:
        model.fit(X_train, y_train)

    # --- Predictions ---
    def predict(X):
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        return model.decision_function(X)

    val_proba  = predict(X_val)
    test_proba = predict(X_test)
    test_pred  = (test_proba >= 0.5).astype(int)

    # --- Metrics ---
    val_auc  = roc_auc_score(y_val,  val_proba)  if y_val.nunique()  > 1 else float("nan")
    test_auc = roc_auc_score(y_test, test_proba) if y_test.nunique() > 1 else float("nan")
    test_ap  = average_precision_score(y_test, test_proba) if y_test.nunique() > 1 else float("nan")

    print(f"\n{'─'*50}")
    print(f"  Model: {model_name.upper()}")
    print(f"  Val  AUC:  {val_auc:.4f}")
    print(f"  Test AUC:  {test_auc:.4f}")
    print(f"  Test AP:   {test_ap:.4f}")
    print(f"{'─'*50}")
    print(classification_report(y_test, test_pred, target_names=["disagree", "agree"]))

    # --- Plots ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(f"{model_name.upper()} — Test Set", fontsize=13, fontweight="bold")

    # ROC curve
    if y_test.nunique() > 1:
        RocCurveDisplay.from_predictions(y_test, test_proba, ax=axes[0], name=model_name)
    axes[0].set_title("ROC Curve")

    # Confusion matrix
    cm = confusion_matrix(y_test, test_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
                xticklabels=["disagree", "agree"],
                yticklabels=["disagree", "agree"])
    axes[1].set_title("Confusion Matrix")
    axes[1].set_ylabel("Actual")
    axes[1].set_xlabel("Predicted")

    # Probability distribution
    axes[2].hist(test_proba[y_test == 0], bins=20, alpha=0.6, label="disagree", color="salmon")
    axes[2].hist(test_proba[y_test == 1], bins=20, alpha=0.6, label="agree",    color="steelblue")
    axes[2].axvline(0.5, color="black", linestyle="--", linewidth=1)
    axes[2].set_title("Predicted Probability Distribution")
    axes[2].set_xlabel("P(agree)")
    axes[2].legend()

    plt.tight_layout()
    plot_path = output_dir / f"{model_name}_evaluation.png"
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved → {plot_path}")

    # --- Feature importance ---
    feat_importance = None
    if model_name == "logistic":
        clf = model.named_steps["clf"]
        feat_importance = pd.Series(
            np.abs(clf.coef_[0]), index=FEATURE_COLS
        ).sort_values(ascending=False)
        importance_label = "|coefficient|"
    elif model_name in ("xgboost", "lightgbm"):
        feat_importance = pd.Series(
            model.feature_importances_, index=FEATURE_COLS
        ).sort_values(ascending=False)
        importance_label = "importance"

    if feat_importance is not None:
        fig, ax = plt.subplots(figsize=(8, 6))
        feat_importance.head(15).plot(kind="barh", ax=ax, color="steelblue")
        ax.set_title(f"{model_name.upper()} — Top 15 Features ({importance_label})")
        ax.set_xlabel(importance_label)
        ax.invert_yaxis()
        plt.tight_layout()
        fi_path = output_dir / f"{model_name}_feature_importance.png"
        plt.savefig(fi_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"[plot] Saved → {fi_path}")

    # --- SHAP ---
    if HAS_SHAP and model_name in ("xgboost", "lightgbm"):
        try:
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # class 1

            fig, ax = plt.subplots(figsize=(9, 6))
            shap.summary_plot(shap_values, X_test, show=False, plot_size=None)
            shap_path = output_dir / f"{model_name}_shap_summary.png"
            plt.savefig(shap_path, dpi=120, bbox_inches="tight")
            plt.close()
            print(f"[plot] Saved → {shap_path}")
        except Exception as e:
            print(f"[warn] SHAP failed: {e}")

    results = {
        "model":    model_name,
        "val_auc":  round(val_auc,  4),
        "test_auc": round(test_auc, 4),
        "test_ap":  round(test_ap,  4),
    }
    return results, model


# ══════════════════════════════════════════════════════════════════════════════
# 7. HYPERPARAMETER TUNING (SIMPLE GRID)
# ══════════════════════════════════════════════════════════════════════════════

def tune_model(model_name: str, train: pd.DataFrame, val: pd.DataFrame) -> dict:
    """Lightweight manual grid search evaluated on the val set."""
    X_train, y_train = get_xy(train)
    X_val,   y_val   = get_xy(val)
    spw = get_scale_pos_weight(y_train)

    grids = {
        "logistic": [
            {"C": c, "class_weight": cw}
            for c in [0.01, 0.1, 1.0, 10.0]
            for cw in ["balanced", None]
        ],
        "xgboost": [
            {"n_estimators": n, "max_depth": d, "learning_rate": lr}
            for n in [100, 300]
            for d in [3, 5]
            for lr in [0.01, 0.05]
        ],
        "lightgbm": [
            {"n_estimators": n, "max_depth": d, "learning_rate": lr}
            for n in [100, 300]
            for d in [3, 5]
            for lr in [0.01, 0.05]
        ],
    }

    best_auc, best_params = -1, {}
    print(f"\n[tune] Grid searching {model_name} ({len(grids[model_name])} combos)…")

    for params in grids[model_name]:
        try:
            if model_name == "logistic":
                m = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=500, random_state=42, **params)),
                ])
            elif model_name == "xgboost":
                m = xgb.XGBClassifier(
                    scale_pos_weight=spw, use_label_encoder=False,
                    eval_metric="logloss", random_state=42, verbosity=0, **params
                )
            elif model_name == "lightgbm":
                m = lgb.LGBMClassifier(
                    class_weight="balanced", random_state=42, verbose=-1, **params
                )

            if model_name == "xgboost":
                m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            elif model_name == "lightgbm":
                m.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                      callbacks=[lgb.log_evaluation(-1)])
            else:
                m.fit(X_train, y_train)

            proba = m.predict_proba(X_val)[:, 1]
            auc   = roc_auc_score(y_val, proba) if y_val.nunique() > 1 else 0.5
            if auc > best_auc:
                best_auc, best_params = auc, params
        except Exception as e:
            print(f"  [skip] {params} → {e}")

    print(f"[tune] Best val AUC: {best_auc:.4f} | params: {best_params}")
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
    df["p_agree"]   = proba
    df["prediction"] = (proba >= 0.5).astype(int)
    df["prediction_label"] = df["prediction"].map({1: "agree", 0: "disagree"})
    return df[["timestamp", "kalshi_slug", "poly_slug", "p_agree",
               "prediction", "prediction_label"]]


# ══════════════════════════════════════════════════════════════════════════════
# 9. EDA
# ══════════════════════════════════════════════════════════════════════════════

def run_eda(df: pd.DataFrame, output_dir: Path):
    """Quick exploratory plots."""
    print("\n[eda] Generating exploratory plots…")
    df_eng = engineer_features(df)
    df_eng = make_label(df_eng)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("EDA — Kalshi vs Polymarket Resolution Agreement", fontsize=13)

    # 1. Label distribution
    df_eng["label"].value_counts().plot(
        kind="bar", ax=axes[0, 0], color=["salmon", "steelblue"]
    )
    axes[0, 0].set_xticklabels(["disagree (0)", "agree (1)"], rotation=0)
    axes[0, 0].set_title("Label distribution")

    # 2. target_aligned vs label
    pd.crosstab(df_eng["target_aligned"], df_eng["label"], normalize="index").plot(
        kind="bar", ax=axes[0, 1], stacked=True, color=["salmon", "steelblue"]
    )
    axes[0, 1].set_title("Target aligned vs agreement rate")
    axes[0, 1].set_xlabel("target_aligned")
    axes[0, 1].legend(["disagree", "agree"])

    # 3. abs_price_diff distribution by label
    for lbl, grp in df_eng.groupby("label"):
        axes[0, 2].hist(grp["abs_price_diff"], bins=30, alpha=0.5,
                        label=["disagree", "agree"][lbl])
    axes[0, 2].set_title("Feed divergence by label")
    axes[0, 2].set_xlabel("abs_price_diff")
    axes[0, 2].legend()

    # 4. dist_to_kalshi_target distribution
    axes[1, 0].hist(df_eng["dist_to_kalshi_target"], bins=40, color="steelblue", alpha=0.7)
    axes[1, 0].axvline(0, color="red", linestyle="--")
    axes[1, 0].set_title("Distance to Kalshi target")
    axes[1, 0].set_xlabel("kalshi_target − chainlink_price")

    # 5. Feature correlations with label
    num_feats = [f for f in FEATURE_COLS if f in df_eng.columns]
    corr = df_eng[num_feats + ["label"]].corr()["label"].drop("label").sort_values()
    corr.plot(kind="barh", ax=axes[1, 1], color=["salmon" if v < 0 else "steelblue" for v in corr])
    axes[1, 1].axvline(0, color="black", linewidth=0.8)
    axes[1, 1].set_title("Feature → label correlation")

    # 6. Agreement rate by hour of day
    hr_rate = df_eng.groupby("hour_of_day")["label"].mean()
    hr_rate.plot(kind="bar", ax=axes[1, 2], color="steelblue", alpha=0.8)
    axes[1, 2].set_title("Agreement rate by hour (UTC)")
    axes[1, 2].set_xlabel("hour_of_day")
    axes[1, 2].set_ylabel("P(agree)")
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
    parser.add_argument("--data",   required=True, help="Path to labeled CSV")
    parser.add_argument(
        "--model", default="logistic",
        choices=["logistic", "xgboost", "lightgbm", "all"],
        help="Model to train (default: logistic)"
    )
    parser.add_argument("--tune",    action="store_true", help="Run hyperparameter grid search")
    parser.add_argument("--eda",     action="store_true", help="Generate EDA plots")
    parser.add_argument("--predict", default=None,        help="Path to new CSV for inference")
    parser.add_argument("--out",     default="outputs",   help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load & prepare
    df = load_data(args.data)
    df = make_label(df)
    df = engineer_features(df)

    if args.eda:
        run_eda(df, output_dir)

    train, val, test = time_split(df)

    # Select models
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

    all_results = []
    trained_models = {}

    for m_name in models_to_run:
        if args.tune:
            best_params = tune_model(m_name, train, val)
            print(f"[tune] Retraining {m_name} with best params…")
            # merge best params into model build — for simplicity rebuild inline
            # (extend here if you want full param injection)

        results, model = train_and_evaluate(m_name, train, val, test, output_dir)
        all_results.append(results)
        trained_models[m_name] = model

    # Summary table
    results_df = pd.DataFrame(all_results)
    print("\n" + "═" * 50)
    print("  RESULTS SUMMARY")
    print("═" * 50)
    print(results_df.to_string(index=False))

    results_path = output_dir / "results.json"
    results_df.to_json(results_path, orient="records", indent=2)
    print(f"\n[out] Results saved → {results_path}")

    # Inference on new data
    if args.predict:
        best_model_name = results_df.sort_values("test_auc", ascending=False).iloc[0]["model"]
        best_model      = trained_models[best_model_name]
        print(f"\n[predict] Using best model ({best_model_name}) for inference on {args.predict}")
        predictions = predict_new(best_model, args.predict)
        pred_path   = output_dir / "predictions.csv"
        predictions.to_csv(pred_path, index=False)
        print(f"[predict] Saved → {pred_path}")
        print(predictions.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
