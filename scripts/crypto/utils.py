from __future__ import annotations

import re
import time
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import requests
from playwright.async_api import async_playwright


NY_TZ = ZoneInfo("America/New_York")


def floor_to_15m(ts: datetime) -> datetime:
    return ts - timedelta(
        minutes=ts.minute % 15,
        seconds=ts.second,
        microseconds=ts.microsecond,
    )


def ceil_to_15m(ts: datetime) -> datetime:
    return floor_to_15m(ts) + timedelta(minutes=15)


def build_slugs_for_timestamp(ts: datetime) -> tuple[str, str]:
    bucket_start = floor_to_15m(ts)
    bucket_end = ceil_to_15m(ts)

    poly_slug = f"btc-updown-15m-{int(bucket_start.timestamp())}"
    kalshi_slug = (
        "KXBTC15M-"
        f"{bucket_end.strftime('%y%b%d').upper()}"
        f"{bucket_end.strftime('%H%M')}-{bucket_end.strftime('%M')}"
    )
    return kalshi_slug, poly_slug


def parse_kalshi_event_time(slug: str) -> datetime:
    dt_part = slug.split("-")[1]
    return datetime.strptime(dt_part, "%y%b%d%H%M").replace(tzinfo=NY_TZ)


async def get_chainlink_price() -> float | None:
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


async def get_cf_price() -> float | None:
    url = "https://www.cfbenchmarks.com/data/assets/BTC"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        await page.goto(url, wait_until="networkidle", timeout=30000)
        await page.wait_for_selector("span.tabular-nums", timeout=15000)
        html = await page.content()

        await browser.close()

    match = re.search(r"tabular-nums[^>]*>\s*\$([0-9,]+\.[0-9]+)", html)
    if not match:
        return None

    return float(match.group(1).replace(",", ""))


def get_with_retries_json(url: str, timeout: int = 8, retries: int = 3) -> dict[str, Any] | None:
    for _ in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception:  # noqa: BLE001
            time.sleep(0.25)
    return None


def fetch_kalshi_target(slug: str) -> float | None:
    url = f"https://api.elections.kalshi.com/trade-api/v2/markets/{slug}"
    payload = get_with_retries_json(url)
    if not payload or "market" not in payload:
        return None

    subtitle = payload["market"].get("yes_sub_title", "")
    if "$" not in subtitle:
        return None

    try:
        return float(subtitle.split("$")[-1].replace(",", "").strip())
    except ValueError:
        return None


def fetch_poly_target(slug: str) -> float | None:
    url = f"https://polymarket.com/event/{slug}"

    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        html = r.text

        matches = re.findall(r"text-heading-2xl[^>]*>\s*\$([0-9,]+(?:\.[0-9]+)?)\s*<", html)
        if not matches:
            matches = re.findall(r"\$([0-9,]+(?:\.[0-9]+)?)", html)

        if matches:
            return float(matches[0].replace(",", ""))
    except Exception:  # noqa: BLE001
        return None

    return None