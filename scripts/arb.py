import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, classification_report

# =========================
# 1. LOAD DATA SAFELY
# =========================
df = pd.read_csv(
    "backtest-results.csv",
)
# strip whitespace from headers
df.columns = df.columns.str.strip()

# =========================
# 1. DEFINE FEATURE SET
# =========================
features = [
    "abs_start_price_diff",
    "prob_diff",
    "abs_prob_diff",
    "btc_return",
    "btc_volatility",
    "btc_last_minute_trend",
    "btc_last_minute_volatility",
    "end_time_diff_seconds",
]

target = "same_resolution"

# =========================
# 2. FORCE NUMERIC CONVERSION
# =========================
def to_num(col):
    return pd.to_numeric(df[col], errors="coerce")

for c in features:
    df[c] = to_num(c)

df[target] = pd.to_numeric(df[target], errors="coerce")

# =========================
# 3. OPTIONAL: sanity check (VERY important)
# =========================
print("\nNULL COUNTS AFTER CONVERSION:")
print(df[features + [target]].isnull().sum())

# =========================
# 4. DROP INVALID ROWS
# =========================
#before = len(df)
#
#df = df.dropna(subset=features + [target])
#
#after = len(df)
#
#print(f"\nROWS BEFORE: {before}")
#print(f"ROWS AFTER : {after}")
#
#if after == 0:
#    raise ValueError("Everything got dropped → check numeric conversion or column names.")
# =========================
# 6. SPLIT DATA
# =========================
X = df[features]
y = df[target].astype(int)

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
# 7. TRAIN MODEL
# =========================
model = LogisticRegression(max_iter=3000)
model.fit(X_train, y_train)

# =========================
# 8. PREDICT
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
# 10. FEATURE WEIGHTS
# =========================
print("\n===== FEATURE WEIGHTS =====")

for f, w in zip(features, model.coef_[0]):
    print(f"{f:35s} {w:.4f}")

# =========================
# 11. SIMPLE EV SIMULATION
# =========================
profit = 0.04
loss = 0.04

evs = []

for p in y_prob:
    ev = p * profit - (1 - p) * loss
    evs.append(ev)

evs = np.array(evs)

print("\n===== TRADE STATS =====")
print("Avg EV:", evs.mean())
print("Trades (EV>0):", (evs > 0).sum())
print("Trade Rate:", (evs > 0).mean())
