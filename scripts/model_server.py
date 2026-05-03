from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import requests
from zoneinfo import ZoneInfo

from build_btc_dataset import get_cf_price, get_chainlink_price


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "market_matching_model.joblib"
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


@dataclass
class MarketState:
    oracle_gap_hist: deque[float] = field(default_factory=lambda: deque(maxlen=50))
    target_gap_hist: deque[float] = field(default_factory=lambda: deque(maxlen=50))
    rel_disagreement_hist: deque[float] = field(default_factory=lambda: deque(maxlen=50))


@dataclass
class CachedValue:
    value: Any
    expires_at: float


class ModelRuntime:
    def __init__(self, model_path: Path) -> None:
        bundle = joblib.load(model_path)
        self.model_name = bundle["model_name"]
        self.features: list[str] = bundle["features"]
        self.model = bundle["model"]
        self.market_state: dict[str, MarketState] = {}
        self.target_cache: dict[str, CachedValue] = {}
        self.spot_cache: dict[str, CachedValue] = {}
        # One-time lookup cache for Polymarket target price per market slug.
        self.poly_target_once: dict[str, float | None] = {}
        self.lock = threading.Lock()

        self.block_threshold = float(os.getenv("P_MATCH_BLOCK_THRESHOLD", "0.40"))
        self.green_threshold = float(os.getenv("P_MATCH_GREENLIGHT_THRESHOLD", "0.90"))
        self.window_start_ratio = float(os.getenv("WINDOW_START_PROGRESS_RATIO", "0.50"))
        self.window_end_ratio = float(os.getenv("WINDOW_END_PROGRESS_RATIO", "0.98"))

    def _get_cached(self, cache: dict[str, CachedValue], key: str) -> Any:
        item = cache.get(key)
        if item and item.expires_at > time.time():
            return item.value
        if item:
            del cache[key]
        return None

    def _set_cached(self, cache: dict[str, CachedValue], key: str, value: Any, ttl: int) -> None:
        cache[key] = CachedValue(value=value, expires_at=time.time() + ttl)

    def _fetch_chainlink_price(self) -> float | None:
        cached = self._get_cached(self.spot_cache, "chainlink_price")
        if cached is not None:
            return cached

        try:
            value = asyncio.run(get_chainlink_price())
        except Exception:  # noqa: BLE001
            value = None

        if value is not None:
            self._set_cached(self.spot_cache, "chainlink_price", float(value), ttl=5)
        return value

    def _fetch_cf_price(self) -> float | None:
        cached = self._get_cached(self.spot_cache, "cf_price")
        if cached is not None:
            return cached

        try:
            value = asyncio.run(get_cf_price())
        except Exception:  # noqa: BLE001
            value = None

        if value is not None:
            self._set_cached(self.spot_cache, "cf_price", float(value), ttl=5)
        return value

    def _get_with_retries(self, url: str, timeout: int = 8) -> Any:
        for _ in range(3):
            try:
                r = requests.get(url, timeout=timeout)
                r.raise_for_status()
                return r.json()
            except Exception:  # noqa: BLE001
                time.sleep(0.25)
        return None

    def _fetch_kalshi_target(self, slug: str) -> float | None:
        cache_key = f"kalshi:{slug}"
        cached = self._get_cached(self.target_cache, cache_key)
        if cached is not None:
            return cached

        url = f"https://api.elections.kalshi.com/trade-api/v2/markets/{slug}"
        payload = self._get_with_retries(url)
        value = None

        if payload and "market" in payload:
            subtitle = payload["market"].get("yes_sub_title", "")
            # Expected format: "Target Price: $76,361.00"
            if "$" in subtitle:
                try:
                    value = float(subtitle.split("$")[-1].replace(",", "").strip())
                except ValueError:
                    value = None

        if value is not None:
            self._set_cached(self.target_cache, cache_key, value, ttl=30)
        return value

    def _fetch_poly_target(self, slug: str) -> float | None:
        # Per request: never fetch more than once for the same market slug.
        if slug in self.poly_target_once:
            return self.poly_target_once[slug]

        url = f"https://polymarket.com/event/{slug}"
        value: float | None = None

        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            html = r.text

            # Example target in page markup:
            # <span class="... text-heading-2xl">$78,829.92</span>
            matches = re.findall(r"text-heading-2xl[^>]*>\s*\$([0-9,]+(?:\.[0-9]+)?)\s*<", html)
            if not matches:
                # Fallback: first large currency value rendered in HTML.
                matches = re.findall(r"\$([0-9,]+(?:\.[0-9]+)?)", html)

            if matches:
                value = float(matches[0].replace(",", ""))
        except Exception:  # noqa: BLE001
            value = None

        # Cache result permanently for this process lifetime, including misses.
        self.poly_target_once[slug] = value
        return value

    def _resolve_snapshot(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        row = dict(snapshot)

        ts_value = row.get("timestamp")
        if ts_value is None:
            event_time = datetime.now(NY_TZ)
            row["timestamp"] = event_time.isoformat()
        else:
            event_time = pd.to_datetime(ts_value)
            if event_time.tzinfo is None:
                event_time = event_time.tz_localize(NY_TZ)
            else:
                event_time = event_time.tz_convert(NY_TZ)

        kalshi_slug = row.get("kalshi_slug")
        poly_slug = row.get("poly_slug")
        if not kalshi_slug or not poly_slug:
            auto_kalshi, auto_poly = build_slugs_for_timestamp(event_time.to_pydatetime())
            kalshi_slug = kalshi_slug or auto_kalshi
            poly_slug = poly_slug or auto_poly
            row["kalshi_slug"] = kalshi_slug
            row["poly_slug"] = poly_slug

        if row.get("chainlink_price") is None:
            row["chainlink_price"] = self._fetch_chainlink_price()
        if row.get("cf_price") is None:
            row["cf_price"] = self._fetch_cf_price()

        if row.get("kalshi_target_price") is None:
            row["kalshi_target_price"] = self._fetch_kalshi_target(kalshi_slug)
        if row.get("poly_target_price") is None:
            row["poly_target_price"] = self._fetch_poly_target(poly_slug)

        if row.get("chainlink_price") is None or row.get("cf_price") is None:
            raise ValueError("Missing chainlink/cf prices; include them in payload if live fetch fails")
        if row.get("kalshi_target_price") is None or row.get("poly_target_price") is None:
            raise ValueError("Missing target prices; include them in payload if market fetch fails")

        event_dt = parse_kalshi_event_time(kalshi_slug)
        row["time_remaining"] = float((event_dt - event_time.to_pydatetime()).total_seconds())
        row["time_progress"] = 900.0 - row["time_remaining"]

        return row

    def _compute_feature_row(self, row: dict[str, Any]) -> dict[str, Any]:
        out = dict(row)

        cf_price = float(out["cf_price"])
        chainlink_price = float(out["chainlink_price"])
        kalshi_target = float(out["kalshi_target_price"])
        poly_target = float(out["poly_target_price"])

        out["kalshi_error"] = kalshi_target - cf_price
        out["poly_error"] = poly_target - chainlink_price
        out["oracle_gap"] = cf_price - chainlink_price
        out["target_gap"] = kalshi_target - poly_target

        out["abs_kalshi_error"] = abs(out["kalshi_error"])
        out["abs_poly_error"] = abs(out["poly_error"])
        out["abs_oracle_gap"] = abs(out["oracle_gap"])
        out["abs_target_gap"] = abs(out["target_gap"])

        out["same_direction"] = int(out["kalshi_error"] * out["poly_error"] > 0)
        out["rel_kalshi_error"] = out["kalshi_error"] / cf_price if cf_price else None
        out["rel_poly_error"] = out["poly_error"] / chainlink_price if chainlink_price else None
        out["relative_disagreement"] = abs(out["kalshi_error"] - out["poly_error"])

        eps = 1e-9
        out["progress_ratio"] = (
            out["time_progress"] / (out["time_progress"] + out["time_remaining"] + eps)
        )
        out["oracle_to_target_ratio"] = out["abs_oracle_gap"] / (out["abs_target_gap"] + 1.0)
        out["error_balance"] = out["abs_kalshi_error"] - out["abs_poly_error"]
        out["error_product"] = out["kalshi_error"] * out["poly_error"]
        out["is_late_market"] = int(out["time_remaining"] < 180)

        market_key = str(out["kalshi_slug"])
        state = self.market_state.setdefault(market_key, MarketState())

        prev_oracle = state.oracle_gap_hist[-1] if state.oracle_gap_hist else None
        prev_target = state.target_gap_hist[-1] if state.target_gap_hist else None
        prev_rel = state.rel_disagreement_hist[-1] if state.rel_disagreement_hist else None

        out["oracle_gap_delta"] = (
            out["oracle_gap"] - prev_oracle if prev_oracle is not None else None
        )
        out["target_gap_delta"] = (
            out["target_gap"] - prev_target if prev_target is not None else None
        )

        oracle_window = list(state.oracle_gap_hist)[-9:] + [out["oracle_gap"]]
        if len(oracle_window) >= 2:
            out["oracle_volatility"] = float(pd.Series(oracle_window).std())
        else:
            out["oracle_volatility"] = None

        out["oracle_stress"] = out["same_direction"] * out["abs_oracle_gap"]

        out["oracle_gap_lag1"] = prev_oracle
        out["target_gap_lag1"] = prev_target
        out["relative_disagreement_lag1"] = prev_rel

        oracle_ma_vals = list(state.oracle_gap_hist)[-2:] + [out["oracle_gap"]]
        target_ma_vals = list(state.target_gap_hist)[-2:] + [out["target_gap"]]
        rel_ma_vals = list(state.rel_disagreement_hist)[-2:] + [out["relative_disagreement"]]

        out["oracle_gap_ma3"] = float(pd.Series(oracle_ma_vals).mean())
        out["target_gap_ma3"] = float(pd.Series(target_ma_vals).mean())
        out["relative_disagreement_ma3"] = float(pd.Series(rel_ma_vals).mean())

        state.oracle_gap_hist.append(float(out["oracle_gap"]))
        state.target_gap_hist.append(float(out["target_gap"]))
        state.rel_disagreement_hist.append(float(out["relative_disagreement"]))

        return out

    def _policy_decision(self, p_match: float, progress_ratio: float, config: dict[str, Any]) -> tuple[str, bool]:
        block_threshold = float(config.get("hard_block_p_match", self.block_threshold))
        green_threshold = float(config.get("greenlight_p_match", self.green_threshold))
        window_start = float(config.get("window_start_progress_ratio", self.window_start_ratio))
        window_end = float(config.get("window_end_progress_ratio", self.window_end_ratio))

        should_evaluate = window_start <= progress_ratio <= window_end
        if not should_evaluate:
            return "wait", False

        if p_match <= block_threshold:
            return "block", True
        if p_match >= green_threshold:
            return "greenlight", True
        return "hold", True

    def predict(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not rows:
            return []

        frame = pd.DataFrame(rows)
        # Keep feature order identical to training.
        for col in self.features:
            if col not in frame.columns:
                frame[col] = None

        x = frame[self.features]
        probs = self.model.predict_proba(x)[:, 1]

        results: list[dict[str, Any]] = []
        for p in probs:
            if p <= self.block_threshold:
                action = "block"
            elif p >= self.green_threshold:
                action = "greenlight"
            else:
                action = "hold"

            results.append(
                {
                    "p_match": float(p),
                    "p_mismatch": float(1.0 - p),
                    "action": action,
                }
            )

        return results

    def predict_live(self, snapshots: list[dict[str, Any]], config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if not snapshots:
            return []

        cfg = config or {}
        results: list[dict[str, Any]] = []

        with self.lock:
            for snapshot in snapshots:
                resolved = self._resolve_snapshot(snapshot)
                feature_row = self._compute_feature_row(resolved)
                frame = pd.DataFrame([feature_row])

                for col in self.features:
                    if col not in frame.columns:
                        frame[col] = None

                p_match = float(self.model.predict_proba(frame[self.features])[:, 1][0])
                action, should_evaluate = self._policy_decision(
                    p_match=p_match,
                    progress_ratio=float(feature_row.get("progress_ratio", 0.0)),
                    config=cfg,
                )

                results.append(
                    {
                        "timestamp": resolved["timestamp"],
                        "kalshi_slug": resolved["kalshi_slug"],
                        "poly_slug": resolved["poly_slug"],
                        "p_match": p_match,
                        "p_mismatch": float(1.0 - p_match),
                        "action": action,
                        "should_evaluate_now": should_evaluate,
                        "progress_ratio": float(feature_row.get("progress_ratio", 0.0)),
                        "time_remaining": float(feature_row.get("time_remaining", 0.0)),
                    }
                )

        return results


class InferenceHandler(BaseHTTPRequestHandler):
    runtime: ModelRuntime | None = None

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            assert self.runtime is not None
            self._send_json(
                200,
                {
                    "ok": True,
                    "model_name": self.runtime.model_name,
                    "features": self.runtime.features,
                    "market_states": len(self.runtime.market_state),
                },
            )
            return

        self._send_json(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:
        if self.path not in {"/predict", "/predict-live"}:
            self._send_json(404, {"ok": False, "error": "not found"})
            return

        assert self.runtime is not None

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length)
            payload = json.loads(raw.decode("utf-8"))

            if "row" in payload:
                rows = [payload["row"]]
            elif "rows" in payload:
                rows = payload["rows"]
            else:
                self._send_json(
                    400,
                    {
                        "ok": False,
                        "error": "request must contain 'row' or 'rows'",
                    },
                )
                return

            if not isinstance(rows, list):
                self._send_json(400, {"ok": False, "error": "rows must be a list"})
                return

            if self.path == "/predict-live":
                preds = self.runtime.predict_live(rows, config=payload.get("config"))
            else:
                preds = self.runtime.predict(rows)
            self._send_json(200, {"ok": True, "predictions": preds})
        except Exception as exc:  # noqa: BLE001
            self._send_json(500, {"ok": False, "error": str(exc)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve sklearn market model over HTTP")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    args = parser.parse_args()

    runtime = ModelRuntime(Path(args.model_path))
    InferenceHandler.runtime = runtime

    server = ThreadingHTTPServer((args.host, args.port), InferenceHandler)
    print(f"Serving model {runtime.model_name} on http://{args.host}:{args.port}")
    print("Endpoints: GET /health, POST /predict, POST /predict-live")
    server.serve_forever()


if __name__ == "__main__":
    main()
