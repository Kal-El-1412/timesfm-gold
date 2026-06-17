"""
Build a clean macro dataset for gold forecasting research.

Improvements over the original macro_dataset.py:
  * DAILY bars over many years (default 10y) instead of 6 months of hourly
    bars. This gives ~2,500 rows instead of ~780 and, crucially, lets every
    market be sampled on the SAME calendar (no stale-equity / 24h-futures
    merge_asof misalignment).
  * Clean same-resolution alignment: all series are joined on the trading
    date with an exact inner join, so a feature never carries a stale value
    from a market that was closed.
  * Explicit, documented targets:
      - future_return_{H}d  : forward return over the horizon
      - target_up           : 1 if future_return > 0          (direction)
      - target_big_move     : 1 if |future_return| > median   (volatility,
                              balanced by construction)
      - target_class        : 0/1/2 DOWN/MID/UP by in-sample tertiles
  * HORIZON is configurable. With horizon=1 (daily) the direction/again
    targets do NOT overlap between consecutive rows, which removes the
    autocorrelated-label problem that inflated the original sample size.
    For horizon>1, evaluation.py applies purging+embargo instead.

Tertile thresholds for target_class are computed on the TRAIN portion only
inside evaluation.py; here we store the raw forward return so the splitter
can label without leakage.
"""

import argparse
import os

import pandas as pd
import yfinance as yf


TICKERS = {
    "gold": "GC=F",
    "silver": "SI=F",
    "dxy": "DX-Y.NYB",
    "vix": "^VIX",
    "oil": "CL=F",
    "sp500": "SPY",
    "nasdaq": "QQQ",
    "us10y": "^TNX",
    "us2y": "^IRX",
}

RETURN_LOOKBACKS = [1, 5, 10]


def download_close(name, ticker, period, interval):
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Close"]].rename(columns={"Close": f"{name}_close"})
    df = df.dropna().sort_index()

    # normalise to a tz-naive calendar DATE so daily series from different
    # exchanges line up exactly on an inner join.
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    df.index = df.index.normalize()
    df = df[~df.index.duplicated(keep="last")]
    df.index.name = "date"

    return df


def build(period="10y", interval="1d", horizon=1, big_move_q=0.5):
    print(f"Building dataset: period={period} interval={interval} horizon={horizon}")

    frames = []
    for name, ticker in TICKERS.items():
        print(f"  downloading {name:7s} {ticker}")
        frames.append(download_close(name, ticker, period, interval))

    # exact inner join on the trading date -> no stale forward-fills
    merged = pd.concat(frames, axis=1, join="inner").sort_index()
    merged = merged.dropna()

    for name in TICKERS:
        col = f"{name}_close"
        for lb in RETURN_LOOKBACKS:
            merged[f"{name}_return_{lb}d"] = merged[col].pct_change(lb)

    merged["gold_sma_20"] = merged["gold_close"].rolling(20).mean()
    merged["gold_sma_50"] = merged["gold_close"].rolling(50).mean()
    merged["gold_above_sma20"] = (merged["gold_close"] > merged["gold_sma_20"]).astype(int)
    merged["gold_above_sma50"] = (merged["gold_close"] > merged["gold_sma_50"]).astype(int)

    # ---- targets -------------------------------------------------------
    fwd = merged["gold_close"].shift(-horizon) / merged["gold_close"] - 1
    # IMPORTANT: the forward-looking target is named with a `future_` prefix
    # and NO `_<n>d` suffix, so it can never be picked up by a feature filter
    # that selects columns ending in `_return_1d` etc.
    merged["future_return"] = fwd
    merged["future_horizon_days"] = horizon
    merged["target_up"] = (fwd > 0).astype(int)

    # balanced volatility target: above-median absolute move
    thr = fwd.abs().median()
    merged["big_move_threshold"] = thr
    merged["target_big_move"] = (fwd.abs() > thr).astype(int)

    merged = merged.dropna()

    os.makedirs("results", exist_ok=True)
    out = "results/dataset_daily.csv"
    merged.to_csv(out)

    print(f"\nSaved {out}")
    print(f"Rows:    {len(merged)}")
    print(f"Columns: {len(merged.columns)}")
    print(f"Date range: {merged.index.min().date()} -> {merged.index.max().date()}")
    print(f"target_up balance:       {merged['target_up'].mean():.3f} up")
    print(f"target_big_move balance: {merged['target_big_move'].mean():.3f} big")
    return merged


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="10y")
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--horizon", type=int, default=1)
    args = ap.parse_args()
    build(period=args.period, interval=args.interval, horizon=args.horizon)
