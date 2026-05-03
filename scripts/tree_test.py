import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, log_loss, confusion_matrix


# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("final_btc_dataset.csv")
df.columns = df.columns.str.strip()

# =========================
# 2. FEATURES
# =========================
features = [
    # core oracle structure
    "kalshi_error",
    "poly_error",
    "oracle_gap",
    "oracle_gap_abs",

    # market disagreement
    "market_gap",
    "abs_market_gap",

    # time pressure
    "time_remaining",

    # regime / structure
    "same_direction",
    "oracle_volatility",

    # dynamics
    "oracle_gap_delta",
    "market_gap_delta",
    "oracle_stress",
    "relative_disagreement",

    # anchors (weak but sometimes useful)
    "cf_price",
    "chainlink_price"
]

target = "matching"


# =========================
# 3. CLEAN DATA
# =========================
for c in features + [target]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=features + [target])

X = df[features]
y = df[target].astype(int)

# =========================
# 4. CLASS DISTRIBUTION
# =========================
print("\n===== CLASS DISTRIBUTION =====")
print(y.value_counts(normalize=True))

# =========================
# 5. TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nTRAIN SIZE:", len(X_train))
print("TEST SIZE :", len(X_test))


# =========================
# 6. TREE MODEL
# =========================
model = HistGradientBoostingClassifier(
    max_depth=6,
    learning_rate=0.05,
    max_iter=400,
    random_state=42
)

model.fit(X_train, y_train)


# =========================
# 7. PREDICTIONS
# =========================
y_prob = model.predict_proba(X_test)[:, 1]

threshold = 0.5
print(f"\nthreshold: {threshold}")

y_pred = (y_prob >= threshold).astype(int)


# =========================
# 8. METRICS
# =========================
print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred, digits=4))

print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(y_test, y_pred))

print("\n===== LOG LOSS =====")
print(log_loss(y_test, y_prob))


# =========================
# 9. FEATURE IMPORTANCE
# =========================
print("\n===== FEATURE IMPORTANCE =====")

importances = model.feature_importances_

for f, w in sorted(zip(features, importances), key=lambda x: -x[1]):
    print(f"{f:25s} {w:.4f}")


# =========================
# 10. MODEL BEHAVIOR
# =========================
print("\n===== MODEL BEHAVIOR =====")

print("Avg predicted match prob:", y_prob.mean())
print("Trades avoided (low confidence):", (y_prob < threshold).sum())
print("Trades allowed:", (y_prob >= threshold).sum())
