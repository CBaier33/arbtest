import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, classification_report
from sklearn.preprocessing import StandardScaler

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("final_btc_dataset.csv")
df.columns = df.columns.str.strip()

# =========================
# 2. UPDATED FEATURE SET
# =========================
features = [
    # raw prices
    "chainlink_price",
    "cf_price",
    "kalshi_target_price",
    "poly_target_price",

    # oracle structure (IMPORTANT SIGNAL)
    "kalshi_error",
    "poly_error",
    "oracle_gap",

    # disagreement structure
    "market_gap",
    "abs_market_gap",

    # regime signal
    "same_direction",

    # time signal
    "time_remaining"
]

target = "matching"

# =========================
# 3. NUMERIC CLEANING
# =========================
def to_num(col):
    return pd.to_numeric(df[col], errors="coerce")

for c in features + [target]:
    if c in df.columns:
        df[c] = to_num(c)

# =========================
# 4. CLEAN DATA
# =========================
df = df.dropna(subset=features + [target])

X = df[features]
y = df[target].astype(int)

# =========================
# 5. TRAIN/TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    shuffle=True
)

print("TRAIN SIZE:", len(X_train))
print("TEST SIZE :", len(X_test))

# =========================
# 6. SCALE FEATURES (IMPORTANT for logreg)
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 7. TRAIN MODEL
# =========================
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# =========================
# 8. PREDICTIONS
# =========================
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

# =========================
# 9. METRICS
# =========================
print("\n===== METRICS =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Log Loss:", log_loss(y_test, y_prob))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# =========================
# 10. FEATURE IMPORTANCE
# =========================
print("\n===== FEATURE WEIGHTS =====")

for f, w in zip(features, model.coef_[0]):
    print(f"{f:25s} {w:.4f}")

# =========================
# 11. SIMPLE EV MODEL
# =========================
profit = 0.04
loss = 0.04

evs = y_prob * profit - (1 - y_prob) * loss

print("\n===== TRADE STATS =====")
print("Avg EV:", evs.mean())
print("Trades (EV>0):", (evs > 0).sum())
print("Trade Rate:", (evs > 0).mean())
