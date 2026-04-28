import pandas as pd

prices = pd.from_csv("backtest-results.csv")

kalshi_columns = [
    'kalshi_ticket',
    'poly_slug',
    'bucket_start_iso'
    'bucket_end_iso',
    'kalshi_yes_sub_title'
]

markets = prices[kalshi_columns]




