"""
Cost-aware, walk-forward backtest of the XGBoost direction signal.

This is the "does the (weak) edge actually trade?" test:

  * Signals are pure out-of-sample: each day's position comes from a model
    trained only on prior data via the same purged walk-forward splitter.
  * Position sizing: go long when P(up) > 0.5 + band, short when
    P(up) < 0.5 - band, else flat. The band is a conviction threshold.
  * TRANSACTION COSTS: a round-trip cost (bps) is charged whenever the
    position changes. The original backtests ignored costs entirely.
  * Reported: net/gross return, annualised Sharpe, max drawdown, hit rate,
    number of trades, all benchmarked against buy & hold.

Run after build_dataset.py.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from xgboost import XGBClassifier

from evaluation import purged_walkforward_splits

COST_BPS = 2.0     # cost per SIDE in basis points (round trip = 2x on a flip)
BAND = 0.02        # conviction band around 0.5
PERIODS_PER_YEAR = 252

df = pd.read_csv("results/dataset_daily.csv")
features = [c for c in df.columns
            if c.endswith(("_return_1d", "_return_5d", "_return_10d"))
            and not c.startswith("future")]
features += ["gold_above_sma20", "gold_above_sma50"]
X = df[features].to_numpy()
y = df["target_up"].to_numpy()
fwd = df["future_return"].to_numpy()


def make_model():
    return XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         eval_metric="logloss", random_state=42, n_jobs=1)


# ---- generate pooled out-of-sample probabilities ----------------------
proba = np.full(len(y), np.nan)
for tr, te in purged_walkforward_splits(len(y), n_splits=5, horizon=1, embargo=5):
    m = clone(make_model()).fit(X[tr], y[tr])
    proba[te] = m.predict_proba(X[te])[:, 1]

mask = ~np.isnan(proba)
p = proba[mask]
r = fwd[mask]

# ---- positions & returns ----------------------------------------------
pos = np.where(p > 0.5 + BAND, 1.0, np.where(p < 0.5 - BAND, -1.0, 0.0))

cost = COST_BPS / 1e4
turnover = np.abs(np.diff(np.concatenate([[0.0], pos])))
costs = turnover * cost

gross = pos * r
net = gross - costs


def stats(returns, label):
    eq = np.cumprod(1 + returns)
    total = eq[-1] - 1
    ann = (1 + total) ** (PERIODS_PER_YEAR / len(returns)) - 1
    vol = returns.std() * np.sqrt(PERIODS_PER_YEAR)
    sharpe = (returns.mean() / returns.std() * np.sqrt(PERIODS_PER_YEAR)
              if returns.std() > 0 else 0.0)
    dd = (eq / np.maximum.accumulate(eq) - 1).min()
    print(f"  {label:14s} total={total:+.2%}  ann={ann:+.2%}  "
          f"vol={vol:.2%}  Sharpe={sharpe:+.2f}  maxDD={dd:.2%}")
    return sharpe


print("=" * 70)
print(f"COST-AWARE WALK-FORWARD BACKTEST  (cost={COST_BPS}bps/side, band={BAND})")
print("=" * 70)
print(f"OOS days traded     : {mask.sum()}")
print(f"Long / Short / Flat : {(pos>0).sum()} / {(pos<0).sum()} / {(pos==0).sum()}")
print(f"Position flips      : {int((turnover>0).sum())}")
active = pos != 0
hit = (np.sign(gross[active]) > 0).mean() if active.any() else float("nan")
print(f"Hit rate (active)   : {hit:.3f}")
print()
stats(net, "Strategy NET")
stats(gross, "Strategy GROSS")
stats(r, "Buy & Hold")
print()
print(f"Cost drag           : {(gross.sum() - net.sum()):+.4f} cumulative return")
print("Note: Sharpe near 0 / negative net => no tradable edge after costs.")
