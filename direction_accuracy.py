"""
TimesFM directional-accuracy test (Phase 4), fixed.

Bug in the original: it forecast with horizon=24 but compared the forecast
against the actual price only `lookahead=8` candles ahead -> the forecast
horizon and the evaluation horizon disagreed, so the reported 4h/8h/24h
numbers couldn't all come from this script.

Here the forecast horizon and the evaluation lookahead are the SAME value, and
we compare the forecast AT that horizon (not the terminal point of a longer
horizon) to the realised move. We also report a simple binomial confidence
interval so "47%" can be judged against 50%.
"""

import argparse
import math

from data_loader import load_gold_data
from indicators import add_indicators
from forecast_engine import forecast_direction


def run(horizon, period="2y", interval="1h", start=300, step=10):
    df = load_gold_data(period=period, interval=interval)
    df = add_indicators(df)

    correct = wrong = 0
    for i in range(start, len(df) - horizon, step):
        hist = df.iloc[: i + 1]
        fc = forecast_direction(hist, horizon=horizon)  # horizon == lookahead

        cur = float(df.iloc[i]["Close"])
        fut = float(df.iloc[i + horizon]["Close"])
        actual = (fut - cur) / cur

        pred = fc["forecast_change_pct"]
        if pred == 0:
            continue
        if (pred > 0) == (actual > 0):
            correct += 1
        else:
            wrong += 1

    n = correct + wrong
    acc = correct / n if n else 0.0
    se = math.sqrt(acc * (1 - acc) / n) if n else 0.0
    lo, hi = acc - 1.96 * se, acc + 1.96 * se
    print(f"horizon={horizon:>3}h  n={n:<4}  accuracy={acc:6.2%}  "
          f"95% CI [{lo:.2%}, {hi:.2%}]  "
          f"{'(indistinguishable from 50%)' if lo <= 0.5 <= hi else ''}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", default="4,8,24")
    ap.add_argument("--period", default="2y")
    args = ap.parse_args()
    print("=== TimesFM DIRECTION TEST (forecast horizon == eval horizon) ===")
    for h in [int(x) for x in args.horizons.split(",")]:
        run(h, period=args.period)
