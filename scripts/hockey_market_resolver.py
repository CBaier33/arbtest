import pandas as pd
import requests
import re

## pull kalshi markets
#def get_poly_markets(cursor=None):
#    print("requesting..")
#    base_url = "https://gamma-api.polymarket.com/markets/"
#
#    params = {
#        "limit": 1000,
#        "closed": "true",
#        "sports_market_types": "moneyline"
#    }
#
#    if cursor:
#        params["next_cursor"] = cursor
#
#    response = requests.get(base_url, params=params)
#    data = response.json()
#
#    markets = data.get("markets", [])
#    next_cursor = data.get("next_cursor")
#
#    if next_cursor:
#        return markets + get_kalshi_markets(next_cursor)
#    else:
#        return markets

def fetch_polymarket_markets(slugs):
    url = "https://gamma-api.polymarket.com/markets/"

    all_markets = []
    after_cursor = None

    while True:
        params = {
            "slug": slugs[:10],
            "limit": 1000,
            "closed": "true"
        }

        if after_cursor:
            params["after_cursor"] = after_cursor

        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()

        all_markets.extend(data.get("markets", []))

        after_cursor = data.get("next_cursor")

        if not after_cursor:
            break

    return all_markets

def chunk(lst, size=50):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]


def fetch_poly_markets(slugs):
    all_markets = []

    for i, batch in enumerate(chunk(slugs, 100)):
        print(f'requesting poly {i}')
        r = requests.get(
            "https://gamma-api.polymarket.com/markets/keyset",
            params={
                "slug": batch,
                "limit": 1000,
                "closed": "true",
                "sports_market_types": "moneyline"
            }
        )
        r.raise_for_status()
        all_markets.extend(r.json().get("markets", []))

    return all_markets

def get_kalshi_markets(cursor=None, count=0):
    print(f"requesting kalshi: {count}")
    base_url = "https://api.elections.kalshi.com/trade-api/v2/historical/markets"
    
    params = {
        "limit": 1000,
        "series_ticker": "KXNHLGAME"
    }

    if cursor:
        params["cursor"] = cursor

    response = requests.get(base_url, params=params)
    data = response.json()

    markets = data.get("markets", [])
    next_cursor = data.get("cursor")

    if next_cursor:
        return markets + get_kalshi_markets(next_cursor, count+1)
    else:
        return markets

MONTHS = {
    "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
    "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
    "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12"
}

TEAM_MAP = {
    "MTL": "MON",
}

def normalize_team(t):
    return TEAM_MAP.get(t, t).lower()

# Receives Kalshi Ticker and coverts to Poly Ticker
def kalshi_to_slug(ticker: str) -> str:
    parts = ticker.split("-")
    if len(parts) < 3:
        raise ValueError(f"Bad ticker: {ticker}")

    core = parts[1]         # e.g. 26FEB05LAVGK
    team2 = parts[2]        # trusted (e.g. VGK)

    match = re.match(r"(\d{2})([A-Z]{3})(\d{2})([A-Z]+)", core)
    if not match:
        raise ValueError(f"Unrecognized format: {ticker}")

    year, mon, day, teams_str = match.groups()

    year = f"20{year}"
    month = MONTHS[mon]

    # remove known team from blob to get the other
    if team2 in teams_str:
        team1 = teams_str.replace(team2, "")
    else:
        raise ValueError(f"Unable to parse: {ticker}")

    t1 = normalize_team(team1)
    t2 = normalize_team(team2)

    teams = sorted([t1, t2])

    return f"nhl-{teams[0]}-{teams[1]}-{year}-{month}-{day}"    

kalshi_cols = [
    "event_ticker",
    "ticker",
    "open_time",
    "close_time",
    "last_price_dollars",
    "result"
]

kalshi_markets = pd.DataFrame(get_kalshi_markets())

kalshi_markets = kalshi_markets[kalshi_cols]
#poly_markets = pd.DataFrame(get_poly_markets())
#poly_markets.to_csv('poly_markets.csv')


poly_slugs = kalshi_markets["ticker"].astype(str).apply(kalshi_to_slug)

poly_markets = pd.DataFrame(fetch_poly_markets(poly_slugs))

#poly_markets = poly_markets[poly_slugs]

poly_cols = [
    "id", 
    "slug",
    "startDate",
    "lastTradePrice"
]

poly_markets = poly_markets[poly_cols]

poly_markets.to_csv("poly_markets.csv", index=False)

kalshi_markets.to_csv("kalshi_markets.csv")

## derive poly names

## pull poly markets enrich with clobids

## pull kalshi markets

## join markets and export csv

## use ids to pull pull prices data per market, enrich with candlestick data, export csv

## use data set to run arb simulation
