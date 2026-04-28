import pmxt
import pandas as pd
import datetime
from zoneinfo import ZoneInfo
import asyncio
from playwright.async_api import async_playwright
import re
import time


# ----------------------------
# Time helpers
# ----------------------------

def floor_to_15_minutes(ts: datetime.datetime) -> datetime.datetime:
    minute = (ts.minute // 15) * 15
    return ts.replace(minute=minute, second=0, microsecond=0)


def ceil_to_15_minutes(ts: datetime.datetime) -> datetime.datetime:
    minute = ((ts.minute // 15) + (1 if ts.minute % 15 else 0)) * 15

    if minute == 60:
        return ts.replace(
            hour=ts.hour + 1,
            minute=0,
            second=0,
            microsecond=0
        )

    return ts.replace(minute=minute, second=0, microsecond=0)


def build_tickers(ts: datetime.datetime):
    start = floor_to_15_minutes(ts)
    end = ceil_to_15_minutes(ts)

    poly_epoch = int(start.timestamp())
    poly_slug = f"btc-updown-15m-{poly_epoch}"

    kalshi_slug = (
        f"KXBTC15M-"
        f"{end.strftime('%d%b%y').upper()}"
        f"{end.strftime('%H%M')}-{end.strftime('%M')}"
    )

    return kalshi_slug, poly_slug


# ----------------------------
# Price extraction
# ----------------------------

async def get_chainlink_price():
    url = "https://data.chain.link/streams/btc-usd-cexprice-streams"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        await page.goto(url, timeout=30000)
        html = await page.content()

        await browser.close()

    match = re.search(r"\$([0-9,]+\.[0-9]+)", html)
    if not match:
        return None

    return float(match.group(1).replace(",", ""))


async def get_cf_price():
    url = "https://www.cfbenchmarks.com/data/assets/BTC"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        await page.goto(url, wait_until="networkidle", timeout=30000)
        await page.wait_for_selector("span.tabular-nums", timeout=15000)

        html = await page.content()

        await browser.close()

    match = re.search(r'tabular-nums[^>]*>\s*\$([0-9,]+\.[0-9]+)', html)
    if not match:
        return None

    return float(match.group(1).replace(",", ""))


# ----------------------------
# Row builder
# ----------------------------

def build_row(ts, kalshi_slug, poly_slug, chainlink_price, cf_price):
    price_diff = None
    price_diff_abs = None

    if chainlink_price is not None and cf_price is not None:
        price_diff = chainlink_price - cf_price
        price_spread = abs(price_diff)

    return {
        "timestamp": ts.isoformat(),
        "kalshi_slug": kalshi_slug,
        "poly_slug": poly_slug,
        "chainlink_price": chainlink_price,
        "cf_price": cf_price,
        "price_diff": price_diff,
        "price_spread": price_spread
    }


# ----------------------------
# Single tick
# ----------------------------

async def run_tick(output_file):
    now = datetime.datetime.now(ZoneInfo("America/New_York"))

    kalshi_slug, poly_slug = build_tickers(now)

    chainlink_price, cf_price = await asyncio.gather(
        get_chainlink_price(),
        get_cf_price()
    )

    row = build_row(now, kalshi_slug, poly_slug, chainlink_price, cf_price)

    df = pd.DataFrame([row])

    try:
        existing = pd.read_csv(output_file)
        df = pd.concat([existing, df], ignore_index=True)
    except FileNotFoundError:
        pass

    df.to_csv(output_file, index=False)

    print(f"[{now.strftime('%H:%M:%S')}] saved")


# ----------------------------
# Sleep until next minute
# ----------------------------

def sleep_to_next_minute():
    now = datetime.datetime.now()
    next_min = (now + datetime.timedelta(minutes=1)).replace(second=0, microsecond=0)
    time.sleep((next_min - now).total_seconds())


# ----------------------------
# Main loop
# ----------------------------

async def main():
    output_file = "btc_market_dataset.csv"

    while True:
        try:
            await run_tick(output_file)
        except Exception as e:
            print("tick error:", e)

        sleep_to_next_minute()


if __name__ == "__main__":
    asyncio.run(main())
