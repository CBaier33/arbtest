export type MarketModelFeatures = Record<string, number | null | undefined>;

export type MarketModelPrediction = {
  p_match: number;
  p_mismatch: number;
  action: "block" | "hold" | "greenlight" | "wait";
  should_evaluate_now?: boolean;
  progress_ratio?: number;
  time_remaining?: number;
};

export type TradeDecisionConfig = {
  windowStartProgressRatio: number; // 0.50 = second half of 15m market
  hardBlockPMatch: number; // e.g. 0.40
  greenlightPMatch: number; // e.g. 0.90
};

export type TickContext = {
  timeRemainingSeconds: number;
  marketDurationSeconds: number;
};

export type LiveSnapshotInput = {
  timestamp?: string;
  kalshi_slug?: string;
  poly_slug?: string;
  chainlink_price?: number;
  cf_price?: number;
  kalshi_target_price?: number;
  poly_target_price?: number;
};

export async function predictMarketAgreement(
  baseUrl: string,
  row: MarketModelFeatures,
): Promise<MarketModelPrediction> {
  const response = await fetch(`${baseUrl}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ row }),
  });

  if (!response.ok) {
    throw new Error(`Model API request failed: ${response.status}`);
  }

  const json = (await response.json()) as {
    ok: boolean;
    predictions: MarketModelPrediction[];
    error?: string;
  };

  if (!json.ok || !json.predictions?.length) {
    throw new Error(json.error ?? "Model API returned no prediction");
  }

  return json.predictions[0];
}

export async function predictLiveSnapshot(
  baseUrl: string,
  snapshot: LiveSnapshotInput,
  cfg: TradeDecisionConfig,
): Promise<MarketModelPrediction> {
  const response = await fetch(`${baseUrl}/predict-live`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      row: snapshot,
      config: {
        window_start_progress_ratio: cfg.windowStartProgressRatio,
        hard_block_p_match: cfg.hardBlockPMatch,
        greenlight_p_match: cfg.greenlightPMatch,
      },
    }),
  });

  if (!response.ok) {
    throw new Error(`Model API request failed: ${response.status}`);
  }

  const json = (await response.json()) as {
    ok: boolean;
    predictions: MarketModelPrediction[];
    error?: string;
  };

  if (!json.ok || !json.predictions?.length) {
    throw new Error(json.error ?? "Model API returned no prediction");
  }

  return json.predictions[0];
}

export function shouldEvaluateNow(
  tick: TickContext,
  cfg: TradeDecisionConfig,
): boolean {
  const progressRatio =
    (tick.marketDurationSeconds - tick.timeRemainingSeconds) /
    tick.marketDurationSeconds;
  return progressRatio >= cfg.windowStartProgressRatio;
}

export function decideTrade(
  prediction: MarketModelPrediction,
  cfg: TradeDecisionConfig,
): "block" | "hold" | "greenlight" {
  if (prediction.p_match <= cfg.hardBlockPMatch) {
    return "block";
  }
  if (prediction.p_match >= cfg.greenlightPMatch) {
    return "greenlight";
  }
  return "hold";
}

// Example usage in a bot loop:
// const cfg: TradeDecisionConfig = {
//   windowStartProgressRatio: 0.5,
//   hardBlockPMatch: 0.4,
//   greenlightPMatch: 0.9,
// };
//
// if (!shouldEvaluateNow({ timeRemainingSeconds, marketDurationSeconds: 900 }, cfg)) return;
// const pred = await predictMarketAgreement("http://127.0.0.1:8080", featureRow);
// const decision = decideTrade(pred, cfg);
// if (decision === "greenlight") {
//   // place trade
// }
