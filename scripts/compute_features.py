
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

def parse_kalshi_time(slug: str):
    dt_part = slug.split("-")[1]  # 26APR291645
    return datetime.strptime(dt_part, "%y%b%d%H%M").replace(
        tzinfo=ZoneInfo("America/New_York")
    )

def compute_features(df):

    # --- timestamps ---
    df["event_time"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("America/New_York")
    df["kalshi_time"] = df["kalshi_slug"].apply(parse_kalshi_time)

    # --- time features ---
    df["time_remaining"] = (
        (df["kalshi_time"] - df["event_time"]).dt.total_seconds()
    )

    df["time_progress"] = 900 - df["time_remaining"]  # 15min bucket = 900s

    # Kalshi anchored to CF
    df["kalshi_error"] = df["kalshi_target_price"] - df["cf_price"]

    # Polymarket anchored to Chainlink
    df["poly_error"] = df["poly_target_price"] - df["chainlink_price"]

    # Oracle divergence (core exogenous signal)
    df["oracle_gap"] = df["cf_price"] - df["chainlink_price"]
    df["oracle_gap_abs"] = (df["cf_price"] - df["chainlink_price"]).abs()

    # Market-to-market gap (secondary derived signal)
    df["target_gap"] = df["kalshi_target_price"] - df["poly_target_price"]

    # Absolute versions (magnitude matters more than direction often)
    df["abs_kalshi_error"] = df["kalshi_error"].abs()
    df["abs_poly_error"] = df["poly_error"].abs()
    df["abs_oracle_gap"] = df["oracle_gap"].abs()
    df["abs_target_gap"] = df["target_gap"].abs()

    # --- regime alignment feature ---
    df["same_direction"] = (
        df["kalshi_error"] * df["poly_error"] > 0
    ).astype(int)

    # --- optional normalized features ---
    df["rel_kalshi_error"] = df["kalshi_error"] / df["cf_price"]
    df["rel_poly_error"] = df["poly_error"] / df["chainlink_price"]
    df["relative_disagreement"] = abs(df["kalshi_error"] - df["poly_error"])

    df["oracle_gap_delta"] = df["oracle_gap"].diff()
    df["target_gap_delta"] = df["target_gap"].diff()
    df["oracle_volatility"] = df["oracle_gap"].rolling(10).std()
    df["oracle_stress"] = df["same_direction"] * df["oracle_gap_abs"]

    # --- target label (corrected) ---
    df["matching"] = (
        df["kalshi_resolution"] == df["poly_resolution"]
    ).astype(int)

    # --- cleanup ---
    df = df.drop(columns=["event_time", "kalshi_time"])

    return df


# ----------------------------
# Load + run
# ----------------------------
#data = pd.read_csv("enriched_btc_dataset.csv")
data = pd.read_csv("enriched.csv")
final = compute_features(data)
final.to_csv("final_btc_dataset.csv", index=False)
