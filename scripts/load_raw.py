import pandas as pd

# Load both CSV files
df1 = pd.read_csv("latest.csv")
df2 = pd.read_csv("btc_market_dataset_new.csv")

# Option 1: rows in df1 that are NOT in df2
diff_df = df1.merge(df2, how="outer", indicator=True)

# Keep only rows unique to either side
diff_df = diff_df[diff_df["_merge"] != "both"]

# Drop helper column
diff_df = diff_df.drop(columns=["_merge"])

# Save result to third CSV
diff_df.to_csv("difference.csv", index=False)

print("Created difference.csv")

