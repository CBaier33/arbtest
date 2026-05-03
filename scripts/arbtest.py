from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "final_btc_dataset.csv"
MODEL_PATH = BASE_DIR / "market_matching_model.joblib"
OOF_PATH = BASE_DIR / "market_matching_oof_predictions.csv"
RANDOM_STATE = 42


@dataclass
class EvalResult:
    model_name: str
    auc: float
    ap: float
    brier: float
    logloss: float


def add_market_state_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["kalshi_slug", "timestamp"]).reset_index(drop=True)

    grp = df.groupby("kalshi_slug", group_keys=False)
    eps = 1e-9

    df["progress_ratio"] = (
        df["time_progress"] / (df["time_progress"] + df["time_remaining"] + eps)
    ).clip(0, 1)
    df["oracle_to_target_ratio"] = df["abs_oracle_gap"] / (df["abs_target_gap"] + 1.0)
    df["error_balance"] = df["abs_kalshi_error"] - df["abs_poly_error"]
    df["error_product"] = df["kalshi_error"] * df["poly_error"]
    df["is_late_market"] = (df["time_remaining"] < 180).astype(int)

    for col in ["oracle_gap", "target_gap", "relative_disagreement"]:
        df[f"{col}_lag1"] = grp[col].shift(1)
        df[f"{col}_ma3"] = grp[col].transform(lambda s: s.rolling(3, min_periods=1).mean())

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    base = [
        "time_remaining",
        "time_progress",
        "progress_ratio",
        "kalshi_error",
        "poly_error",
        "oracle_gap",
        "target_gap",
        "abs_kalshi_error",
        "abs_poly_error",
        "abs_oracle_gap",
        "abs_target_gap",
        "same_direction",
        "relative_disagreement",
        "rel_kalshi_error",
        "rel_poly_error",
        "oracle_gap_delta",
        "target_gap_delta",
        "oracle_volatility",
        "oracle_stress",
        "oracle_to_target_ratio",
        "error_balance",
        "error_product",
        "is_late_market",
        "cf_price",
        "chainlink_price",
        "oracle_gap_lag1",
        "oracle_gap_ma3",
        "target_gap_lag1",
        "target_gap_ma3",
        "relative_disagreement_lag1",
        "relative_disagreement_ma3",
    ]
    return [c for c in base if c in df.columns]


def build_models(features: list[str]) -> dict[str, Pipeline | CalibratedClassifierCV]:
    numeric_scaled = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    numeric_plain = Pipeline([("imputer", SimpleImputer(strategy="median"))])

    logreg = Pipeline(
        [
            ("prep", ColumnTransformer([("num", numeric_scaled, features)])),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    rf_base = Pipeline(
        [
            ("prep", ColumnTransformer([("num", numeric_plain, features)])),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=220,
                    min_samples_leaf=4,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    rf_calibrated = CalibratedClassifierCV(rf_base, method="sigmoid", cv=3)

    return {
        "logreg_balanced": logreg,
        "random_forest": rf_base,
        "random_forest_calibrated": rf_calibrated,
    }


def evaluate_grouped_cv(
    model: Pipeline | CalibratedClassifierCV,
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
) -> tuple[EvalResult, np.ndarray]:
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_prob = np.zeros(len(X), dtype=float)
    fold_stats = []

    for train_idx, test_idx in cv.split(X, y, groups=groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        oof_prob[test_idx] = prob

        fold_stats.append(
            (
                roc_auc_score(y_test, prob),
                average_precision_score(y_test, prob),
                brier_score_loss(y_test, prob),
                log_loss(y_test, prob, labels=[0, 1]),
            )
        )

    arr = np.asarray(fold_stats)
    result = EvalResult(
        model_name=model_name,
        auc=float(arr[:, 0].mean()),
        ap=float(arr[:, 1].mean()),
        brier=float(arr[:, 2].mean()),
        logloss=float(arr[:, 3].mean()),
    )
    return result, oof_prob


def print_time_bin_reliability(df: pd.DataFrame, oof_prob: np.ndarray) -> None:
    bins = pd.cut(
        df["time_remaining"],
        bins=[-1, 60, 180, 420, 900],
        labels=["<=1m", "1-3m", "3-7m", "7-15m"],
    )

    view = df.assign(pred_prob=oof_prob, time_bin=bins).groupby("time_bin", observed=True).apply(
        lambda g: pd.Series(
            {
                "rows": len(g),
                "actual_match_rate": g["matching"].mean(),
                "avg_predicted_prob": g["pred_prob"].mean(),
                "brier": brier_score_loss(g["matching"], g["pred_prob"])
                if g["matching"].nunique() > 1
                else np.nan,
            }
        )
    )

    print("\n===== RELIABILITY BY TIME TO RESOLUTION =====")
    print(view)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find input dataset at: {DATA_PATH.resolve()}")

    raw = pd.read_csv(DATA_PATH)
    df = add_market_state_features(raw)

    target = "matching"
    features = get_feature_columns(df)
    groups = df["kalshi_slug"].astype(str)

    for col in features + [target]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[target]).reset_index(drop=True)
    X = df[features]
    y = df[target].astype(int)

    print("\n===== DATA PROFILE =====")
    print(f"rows: {len(df)}")
    print(f"markets: {df['kalshi_slug'].nunique()}")
    print(f"features: {len(features)}")
    print("class distribution:")
    print(y.value_counts(normalize=True).sort_index())

    model_map = build_models(features)
    all_results = []
    oof_by_model: dict[str, np.ndarray] = {}

    print("\n===== GROUPED CV MODEL COMPARISON (NO MARKET LEAKAGE) =====")
    for name, model in model_map.items():
        result, oof = evaluate_grouped_cv(model, name, X, y, groups)
        all_results.append(result)
        oof_by_model[name] = oof
        print(
            f"{name:26s} "
            f"AUC={result.auc:.4f} AP={result.ap:.4f} "
            f"Brier={result.brier:.4f} LogLoss={result.logloss:.4f}"
        )

    best = sorted(all_results, key=lambda r: (r.logloss, r.brier))[0]
    print("\n===== SELECTED MODEL =====")
    print(f"{best.model_name} (best probability quality by LogLoss/Brier)")

    best_model = model_map[best.model_name]
    best_oof = oof_by_model[best.model_name]

    y_pred = (best_oof >= 0.5).astype(int)
    print("\n===== OOF CLASSIFICATION REPORT (THRESHOLD=0.5) =====")
    print(classification_report(y, y_pred, digits=4))
    print("\n===== OOF CONFUSION MATRIX =====")
    print(confusion_matrix(y, y_pred))

    print_time_bin_reliability(df, best_oof)

    df_out = df[["timestamp", "kalshi_slug", "poly_slug", "matching"]].copy()
    df_out["pred_prob_match"] = best_oof
    df_out.to_csv(OOF_PATH, index=False)
    print(f"\nSaved OOF predictions to: {OOF_PATH}")

    best_model.fit(X, y)
    joblib.dump(
        {
            "model_name": best.model_name,
            "features": features,
            "model": best_model,
        },
        MODEL_PATH,
    )
    print(f"Saved trained model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
