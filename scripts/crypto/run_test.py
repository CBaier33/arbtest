from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from model_server import DEFAULT_MODEL_PATH, ModelRuntime


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir.parent / "final_btc_dataset.csv"

    runtime = ModelRuntime(DEFAULT_MODEL_PATH)

    if not data_path.exists():
        raise FileNotFoundError(f"Missing test dataset: {data_path}")

    row = pd.read_csv(data_path).iloc[0]

    snapshot = {
        "timestamp": row["timestamp"],
        "kalshi_slug": row["kalshi_slug"],
        "poly_slug": row["poly_slug"],
        "chainlink_price": float(row["chainlink_price"]),
        "cf_price": float(row["cf_price"]),
        "kalshi_target_price": float(row["kalshi_target_price"]),
        "poly_target_price": float(row["poly_target_price"]),
    }

    config = {
        "window_start_progress_ratio": 0.50,
        "window_end_progress_ratio": 0.98,
        "hard_block_p_match": 0.40,
        "greenlight_p_match": 0.90,
    }

    prediction = runtime.predict_live([snapshot], config=config)[0]
    print(json.dumps(prediction, indent=2))


if __name__ == "__main__":
    main()
