import pandas as pd
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta
import requests
from tqdm import tqdm
import time
import random
import re

data = pd.read_csv('difference.csv')
data = data.dropna()

def floor_to_15m(dt):
    return dt - timedelta(
        minutes=dt.minute % 15,
        seconds=dt.second,
        microseconds=dt.microsecond
    )

def ceil_to_15m(dt):
    floored = floor_to_15m(dt)
    return floored + timedelta(minutes=15)


def build_kalshi_ticker(ts):
    # ensure timezone consistency
    ts = pd.to_datetime(ts).tz_convert("America/New_York")

    bucket_end = ceil_to_15m(ts)

    ticker = (
        "KXBTC15M-"
        f"{bucket_end.strftime('%y%b%d').upper()}"
        f"{bucket_end.strftime('%H%M')}"
        f"-{bucket_end.strftime('%M')}"
    )

    return ticker

data["kalshi_slug"] = data["timestamp"].apply(build_kalshi_ticker)
kalshi_slugs = data['kalshi_slug'].unique()

def generate_poly_slug(kalshi_slug: str) -> str:
    dt_part = kalshi_slug.split("-")[1]
    dt = datetime.strptime(dt_part, "%y%b%d%H%M")
    dt = dt.replace(tzinfo=ZoneInfo("America/New_York"))
    earlier = dt - timedelta(minutes=15)
    return "btc-updown-15m-" + str(int(earlier.timestamp()))

data["poly_slug"] = data["kalshi_slug"].apply(generate_poly_slug) # correct slug names

import time
import random
import requests

def get_with_retries(url, max_retries=5, timeout=10):
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=timeout)

            if r.status_code in [429, 500, 502, 503, 504]:
                raise requests.exceptions.HTTPError(f"{r.status_code}")

            r.raise_for_status()
            return r.json()

        except (requests.exceptions.RequestException, ValueError):
            if attempt == max_retries - 1:
                return None

            time.sleep((2 ** attempt) + random.uniform(0, 0.5))

    return None

def fetch_kalshi(slug: str):
    # replace with real API call
    url = "https://api.elections.kalshi.com/trade-api/v2/markets/" + slug

    data = get_with_retries(url)

    if data == None:
        return {
            "target_price": None,
            "resolution": None
        }

			
    market = data['market']

    ## target price
    subtitle = market["yes_sub_title"]  # "Target Price: $76,361.00"

    match = re.search(r"\$([\d,]+\.\d+)", subtitle)

    target_price = float(match.group(1).replace(",", ""))

    ## resolution
    resolution = 0 if float(market['last_price_dollars']) < 0.5 else 1

    return {
        "target_price": target_price,
        "resolution": resolution
    }

def fetch_poly(slug: str):
    # replace with real API call
    import requests

    url = "https://gamma-api.polymarket.com/markets/slug/" + slug

    market = get_with_retries(url)

    if market == None:
        return {
            "target_price": None,
            "resolution": None
        }

    tp = (
        market.get("events", [{}])[0]
        .get("eventMetadata", {})
        .get("priceToBeat")
    )

    resolution = 0 if float(market.get('lastTradePrice')) < 0.5 else 1

    return {
        "target_price": tp,
        "resolution": resolution
    }

unique_pairs = data[["kalshi_slug", "poly_slug"]].drop_duplicates()

cache = {}

for row in tqdm(unique_pairs.itertuples(index=False), total=len(unique_pairs)):

    kalshi_data = fetch_kalshi(row.kalshi_slug)
    poly_data = fetch_poly(row.poly_slug)

    cache[row.kalshi_slug] = {
        "kalshi": kalshi_data,
        "poly": poly_data
    }

def enrich(row):
    k_slug = row["kalshi_slug"]
    return {
        "kalshi_target_price": cache[k_slug]["kalshi"].get("target_price"),
        "poly_target_price": cache[k_slug]["poly"].get("target_price"),
        "kalshi_resolution": cache[k_slug]["kalshi"].get("resolution"),
        "poly_resolution": cache[k_slug]["poly"].get("resolution"),
    }

market_df = pd.DataFrame({
    "kalshi_slug": list(cache.keys()),

    "kalshi_target_price": [cache[k]["kalshi"]["target_price"] for k in cache],
    "poly_target_price": [cache[k]["poly"]["target_price"] for k in cache],

    "kalshi_resolution": [cache[k]["kalshi"]["resolution"] for k in cache],
    "poly_resolution": [cache[k]["poly"]["resolution"] for k in cache],
})

enriched_df = data.merge(market_df, on="kalshi_slug", how="left")

enriched_df.to_csv('enriched_btc_dataset_latest.csv', index=False)
