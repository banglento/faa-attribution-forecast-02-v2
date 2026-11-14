#!/usr/bin/env python3
"""Forecaster for polymarket-faa-attribution-02 (Python 3.13, stdlib only).

Model, in one paragraph: prediction-market prices here are step processes --
long flat stretches, small ticks, occasional jumps -- so a Gaussian random
walk calibrated on all history (the house baseline) is far too wide, and a
raw empirical-quantile replay of old returns drags in dead regimes.  This
entry therefore keeps the persistence median (the last price is the best
centre we found; drift and mean-reversion terms both scored worse) and
builds the spread from the market's own recent absolute one-minute moves:

  * volatility: mean absolute 1-minute change, converted to a Gaussian
    sigma by the usual 1.2533 factor, blended over three lookbacks
    (90 / 360 / 1440 minutes, weights 0.30 / 0.40 / 0.30) so a single jump
    or a single dead hour cannot dominate, then clamped to a sane range;
  * shape: symmetric quantile multipliers [-2.3, -0.52, 0, 0.52, 2.3] --
    a narrow waist (most minutes nothing happens) with fat tails (when a
    jump comes it is several ticks), between a Laplace and a Student-t;
  * growth: spread widens like h**0.42, slower than a random walk's
    sqrt(h), because these prices are bounded and partly stationary over
    a 4-hour horizon.

Everything is computed from the case file alone, deterministically, in
well under a second per case.
"""

import json
import math
import os
import sys

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
MULT = [-2.3, -0.52, 0.0, 0.52, 2.3]
WINDOWS = [(90, 0.30), (360, 0.40), (1440, 0.30)]
MEANABS_TO_SIGMA = 1.2533
SIGMA_FLOOR = 0.00045
SIGMA_CAP = 0.012
SIGMA_FALLBACK = 0.0025
GROWTH_POWER = 0.42


def price6(x):
    """Clamp to [0, 1] and round to 6 decimals (never -0.0)."""
    if x < 0.0:
        x = 0.0
    elif x > 1.0:
        x = 1.0
    return round(x, 6) + 0.0


def market_sigma(prices):
    """Blended recent mean-absolute-move volatility of one market.

    Window means come from a prefix-sum table over the absolute one-minute
    changes, so each lookback costs O(1) however long the history is.
    """
    n = len(prices)
    if n < 2:
        return SIGMA_FALLBACK
    prefix = [0.0]
    for i in range(1, n):
        prefix.append(prefix[-1] + abs(prices[i] - prices[i - 1]))
    total_weight = 0.0
    acc = 0.0
    for window, weight in WINDOWS:
        take = window if window < len(prefix) - 1 else len(prefix) - 1
        if take > 0:
            acc += weight * ((prefix[-1] - prefix[-1 - take]) / take)
            total_weight += weight
    if total_weight == 0.0:
        return SIGMA_FALLBACK
    sigma = MEANABS_TO_SIGMA * acc / total_weight
    if sigma < SIGMA_FLOOR:
        return SIGMA_FLOOR
    if sigma > SIGMA_CAP:
        return SIGMA_CAP
    return sigma


def forecast_market(prices, horizon):
    """240 x 5 quantile rows for one market, centred on its last price."""
    if not prices:
        raise ValueError("market with no prices")
    x = prices[-1]
    sigma = market_sigma(prices)
    rows = []
    for h in range(1, horizon + 1):
        spread = sigma * (h ** GROWTH_POWER)
        row = []
        prev = 0.0
        for mult in MULT:
            q = price6(x + mult * spread)
            if q < prev:  # keep the row non-decreasing after rounding
                q = prev
            row.append(q)
            prev = q
        rows.append(row)
    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inputs = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    if not inputs:
        raise ValueError(f"no .in files in {INPUT_DIR}")
    for name in inputs:
        with open(os.path.join(INPUT_DIR, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        if list(case["quantiles"]) != QUANTILES:
            raise ValueError(f"{name} asks for quantiles {case['quantiles']}")
        horizon = int(case["horizon_minutes"])
        forecasts = [
            {"id": m["id"], "q": forecast_market(m["prices"], horizon)}
            for m in case["markets"]
        ]
        out_name = name[: -len(".in")] + ".out"
        with open(os.path.join(OUTPUT_DIR, out_name), "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"forecasts": forecasts}))
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
