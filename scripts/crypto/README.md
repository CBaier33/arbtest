# Crypto Model Bundle (Portable)

This folder is a portable package for integrating the BTC market-agreement model into a trading bot.

It is designed so you can copy this folder into another repo and run it with minimal changes.

## Folder Contents

- `model_server.py`
  - Python HTTP server for model inference.
  - Computes model features internally from raw/live snapshots.
  - Maintains per-market rolling state for lag/rolling features.
  - Applies trading policy thresholds and returns an action.

- `modelClient.ts`
  - TypeScript client helpers for calling the server.
  - Includes policy helper functions for bot decisions.

- `market_matching_model.joblib`
  - Trained calibrated random forest model bundle.

- `market_matching_oof_predictions.csv`
  - Offline OOF prediction artifact from training/evaluation.

- `run_test.py`
  - Local smoke test for server-side logic and model prediction.

## What The Model Predicts

- `p_match`: Probability that Kalshi and Polymarket resolve the same.
- `p_mismatch`: Probability that they resolve differently (`1 - p_match`).

Server policy maps probability to an action:

- `wait`: Outside configured trading window.
- `block`: High mismatch risk (`p_match` below hard-block threshold).
- `hold`: In-between region.
- `greenlight`: High confidence they match.

## Server API

### Start server

Run from this folder (or point to this file path):

```bash
uv run model_server.py --host 127.0.0.1 --port 8080
```

### Health endpoint

```http
GET /health
```

Returns server/model metadata.

### Generic feature endpoint

```http
POST /predict
```

Payload:

```json
{
  "row": {
    "time_remaining": 120,
    "time_progress": 780
  }
}
```

Use this only if you already compute model feature columns externally.

### Live snapshot endpoint (recommended)

```http
POST /predict-live
```

Payload supports minimal raw snapshot fields plus optional config:

```json
{
  "row": {
    "timestamp": "2026-05-02T12:34:56-04:00",
    "kalshi_slug": "KXBTC15M-26MAY021245-45",
    "poly_slug": "btc-updown-15m-1777725600",
    "chainlink_price": 76350.1,
    "cf_price": 76340.7,
    "kalshi_target_price": 76361.0,
    "poly_target_price": 76355.2
  },
  "config": {
    "window_start_progress_ratio": 0.5,
    "window_end_progress_ratio": 0.98,
    "hard_block_p_match": 0.4,
    "greenlight_p_match": 0.9
  }
}
```

Notes:

- If some fields are omitted, server tries to fetch them.
- Polymarket target price is scraped from the event page and cached once per market slug for process lifetime.
- Rolling/lag features are tracked in-memory by `kalshi_slug`.

Response example:

```json
{
  "ok": true,
  "predictions": [
    {
      "timestamp": "2026-04-28T17:27:35.892570-04:00",
      "kalshi_slug": "KXBTC15M-26APR281730-30",
      "poly_slug": "btc-updown-15m-1777410900",
      "p_match": 0.93,
      "p_mismatch": 0.07,
      "action": "greenlight",
      "should_evaluate_now": true,
      "progress_ratio": 0.84,
      "time_remaining": 144.1
    }
  ]
}
```

## TypeScript Integration

Use `modelClient.ts`:

- `predictLiveSnapshot(baseUrl, snapshot, cfg)`
  - Calls `/predict-live`.

- `predictMarketAgreement(baseUrl, row)`
  - Calls `/predict`.

- `shouldEvaluateNow(tick, cfg)`
  - Local progress-ratio window gate.

- `decideTrade(prediction, cfg)`
  - Local threshold mapping to `block | hold | greenlight`.

Recommended bot flow:

1. Build a snapshot from your market data tick.
2. Call `predictLiveSnapshot`.
3. Respect returned `action`.
4. Add your own execution constraints (spread, depth, slippage, position limits).

## Recommended Default Config

For second-half trading window:

```json
{
  "window_start_progress_ratio": 0.5,
  "window_end_progress_ratio": 0.98,
  "hard_block_p_match": 0.4,
  "greenlight_p_match": 0.9
}
```

Interpretation:

- Evaluate only in second half of market (`>= 50%` progress).
- Hard reject high-risk mismatch regime (`p_match <= 0.40`).
- Green-light only high confidence (`p_match >= 0.90`).

## Quick Validation

Run:

```bash
python run_test.py
```

This loads the copied model, builds a sample snapshot from local dataset, and prints a prediction/action.

## Portability Checklist

When copying to another bot repo:

1. Copy entire `crypto` folder.
2. Ensure Python env has dependencies used by `model_server.py`.
3. Start server process where bot can reach it.
4. Import or copy `modelClient.ts` into your TypeScript runtime.
5. Wire bot config to the policy fields shown above.

## Important Runtime Notes

- Server keeps market rolling state in memory. Restarting the process resets lag/rolling history.
- If fetch/scrape sources fail, include prices/targets directly in payload for deterministic operation.
- Keep host local (`127.0.0.1`) unless you intentionally expose the service.
