import pandas as pd

df = pd.read_csv("enriched_btc_dataset.csv", parse_dates=["timestamp"])
df["label"] = (df["kalshi_resolution"] == df["poly_resolution"]).astype(int)

# Look at disagree contracts grouped by time
disagree = df[df["label"] == 0].copy()
print(disagree.groupby("kalshi_slug")[
    ["chainlink_price","cf_price","price_diff","kalshi_target_price","poly_target_price"]
].first().to_string())

# Are the disagrees clustered in time?
print(disagree["timestamp"].dt.floor("15min").value_counts().sort_index())
