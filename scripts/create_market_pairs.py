import pandas as pd
import re

## init
kalshi = pd.read_csv("kalshi_markets.csv")
poly = pd.read_csv("poly_markets.csv")

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


def kalshi_to_slug(ticker: str) -> str:
    parts = ticker.split("-")
    if len(parts) < 3:
        raise ValueError(f"Bad ticker: {ticker}")

    core = parts[1]   # e.g. 26FEB05LAVGK
    team2 = parts[2]  # trusted (e.g. VGK)

    match = re.match(r"(\d{2})([A-Z]{3})(\d{2})([A-Z]+)", core)
    if not match:
        raise ValueError(f"Unrecognized format: {ticker}")

    year, mon, day, teams_str = match.groups()

    year = f"20{year}"
    month = MONTHS[mon]

    if team2 not in teams_str:
        raise ValueError(f"Unable to parse teams: {ticker}")

    team1 = teams_str.replace(team2, "")

    t1 = normalize_team(team1)
    t2 = normalize_team(team2)

    teams = sorted([t1, t2])

    return f"nhl-{teams[0]}-{teams[1]}-{year}-{month}-{day}"


## =========================
## Poly preprocessing
## =========================

poly = poly.copy()

poly["poly_result"] = poly["lastTradePrice"].apply(
    lambda x: "yes" if float(x) > 0.5 else "no"
)

poly["lastTradePrice"] = poly["lastTradePrice"].astype(float)


## =========================
## Kalshi preprocessing
## =========================

kalshi = kalshi.copy()

kalshi["poly_slug"] = kalshi["ticker"].apply(kalshi_to_slug)


## =========================
## Merge
## =========================

merged = kalshi.merge(
    poly,
    left_on="poly_slug",
    right_on="slug",
    how="outer"
)


## =========================
## Final dataset
## =========================

final = merged[[
    "ticker",            # kalshi ticker
    "slug",              # poly slug
    "startDate",
    "poly_result",
    "result",
    "lastTradePrice"
]].copy()

final = final.rename(columns={
    "ticker": "kalshi_ticker",
    "slug": "poly_slug",
    "startDate": "date",
    "result": "kalshi_result",
    "lastTradePrice": "poly_price"
})


## =========================
## Matching logic
## =========================

final["matching"] = (
    (final["kalshi_result"] == "yes") & (final["poly_result"] == "yes")
) | (
    (final["kalshi_result"] == "no") & (final["poly_result"] == "no")
)

final["matching"] = final["matching"].astype(int)

#final = final.loc[final["matching"] == 1]


## =========================
## Output
## =========================

final.to_csv("outer_hockey_pairs.csv", index=False)
