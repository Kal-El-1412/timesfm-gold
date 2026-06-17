"""
The decisive volatility test: REALIZED vs IMPLIED.

A volatility "edge" only exists if we can forecast future realized vol BETTER
than the market's implied vol (GVZ) already does, OR if there is a systematic
variance risk premium we can harvest after costs. Predicting realized vol from
its own history (Phase-2 R^2=0.14) is NOT enough on its own, because volatility
clustering is already priced into options.

This script answers three questions, all out-of-sample / purged:

  1. Is there a variance risk premium? (is implied systematically above the
     realized vol that follows?)
  2. Can our model beat implied at FORECASTING forward realized vol?
     (compare purged-CV R^2: implied-only vs our-features vs combined)
  3. Does a cost-aware vol strategy make money? (harvest the premium
     unconditionally vs trade only when our model disagrees with implied)

GVZ is ~30-calendar-day implied vol, so the realized horizon is H=21 trading
days. Vol is annualised to match GVZ's units (annualised %).

Run after build_dataset.py (which now includes gvz_close).
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

from evaluation import purged_walkforward_splits

H = 21                 # trading days ~ GVZ 30-calendar-day horizon
ANN = np.sqrt(252) * 100
COST_VOLPTS = 0.5      # round-trip vol-point cost for a vol trade
EMBARGO = H

df = pd.read_csv("results/dataset_daily.csv", parse_dates=["date"], index_col="date")
r = df["gold_return_1d"]

# past realized vol (features), annualised
df["rv_past_10"] = r.rolling(10).std() * ANN
df["rv_past_21"] = r.rolling(21).std() * ANN
df["rv_past_63"] = r.rolling(63).std() * ANN

# forward realized vol over next H days, annualised (explicit forward window)
arr = r.to_numpy()
n = len(arr)
fwd_rv = np.full(n, np.nan)
for t in range(n - H):
    fwd_rv[t] = np.std(arr[t + 1: t + 1 + H]) * ANN
df["rv_future"] = fwd_rv

df = df.dropna()
implied = df["gvz_close"].to_numpy()
y = df["rv_future"].to_numpy()

macro = [c for c in df.columns
         if c.endswith(("_return_1d", "_return_5d", "_return_10d"))
         and not c.startswith("future") and not c.startswith("gvz")]
vol_past = ["rv_past_10", "rv_past_21", "rv_past_63"]


# ---------------------------------------------------------------------------
# Q1. Variance risk premium
# ---------------------------------------------------------------------------
vrp = implied - y
print("#" * 70)
print("Q1. VARIANCE RISK PREMIUM (implied GVZ - forward realized, vol points)")
print("#" * 70)
print(f"Rows: {len(df)}   H={H} trading days")
print(f"Mean implied (GVZ)     : {implied.mean():.2f}")
print(f"Mean forward realized  : {y.mean():.2f}")
print(f"Mean VRP               : {vrp.mean():+.2f} vol pts")
print(f"% days implied>realized: {(vrp > 0).mean():.1%}")
print("-> a persistently positive VRP means short-vol is paid on average")


# ---------------------------------------------------------------------------
# Q2. Can we beat implied at forecasting forward realized vol?
# ---------------------------------------------------------------------------
def purged_oos_pred(model, X):
    X = np.asarray(X)
    pred = np.full(len(y), np.nan)
    for tr, te in purged_walkforward_splits(len(y), 5, H, EMBARGO):
        m = clone(model).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def report_r2(pred, label):
    mask = ~np.isnan(pred)
    print(f"  {label:34s} R^2 = {r2_score(y[mask], pred[mask]):+.4f}")
    return pred


def lr():  return LinearRegression()
def xgb(): return XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=42, n_jobs=1)

print("\n" + "#" * 70)
print("Q2. FORECASTING forward realized vol (purged walk-forward OOS R^2)")
print("#" * 70)
p_imp = report_r2(purged_oos_pred(lr(), df[["gvz_close"]]), "implied (GVZ) only")
report_r2(purged_oos_pred(xgb(), df[vol_past]), "our past-vol features only")
report_r2(purged_oos_pred(xgb(), df[macro]), "macro features only")
p_comb = report_r2(purged_oos_pred(xgb(), df[["gvz_close"] + vol_past + macro]),
                   "implied + vol + macro (combined)")
print("-> if combined R^2 > implied-only R^2, we add information beyond the market")


# ---------------------------------------------------------------------------
# Q3. Cost-aware vol strategy (non-overlapping trades, step = H)
# ---------------------------------------------------------------------------
print("\n" + "#" * 70)
print(f"Q3. COST-AWARE VOL STRATEGY (cost={COST_VOLPTS} vol pts/trade, "
      f"non-overlapping)")
print("#" * 70)

idx = np.arange(0, len(y), H)          # independent, non-overlapping windows
forecast = p_comb                       # our OOS realized-vol forecast
edge = implied - forecast               # >0 => model says implied too high -> short vol

def summarise(pnl, label):
    pnl = pnl[~np.isnan(pnl)]
    if len(pnl) == 0:
        print(f"  {label:28s} no trades"); return
    sharpe = pnl.mean() / pnl.std() * np.sqrt(252 / H) if pnl.std() > 0 else 0
    print(f"  {label:28s} trades={len(pnl):3d}  mean={pnl.mean():+.2f}  "
          f"win%={(pnl>0).mean():.0%}  Sharpe(ann)={sharpe:+.2f}  "
          f"total={pnl.sum():+.1f}")

# A) always short vol (harvest premium): pnl = (implied - realized) - cost
pnl_uncond = (implied[idx] - y[idx]) - COST_VOLPTS
summarise(pnl_uncond, "Always short vol")

# B) model-conditional: short when edge>+band, long when edge<-band, else flat
BAND = 1.0
pos = np.where(edge[idx] > BAND, 1.0, np.where(edge[idx] < -BAND, -1.0, 0.0))
pnl_cond = pos * (implied[idx] - y[idx]) - np.abs(pos) * COST_VOLPTS
summarise(pnl_cond, f"Model-conditional (band={BAND})")

print("\nInterpretation:")
print("  * Positive VRP + profitable 'always short vol' => a premium exists,")
print("    but that is a known risk premium, not a forecasting edge.")
print("  * Model-conditional beating always-short (esp. higher Sharpe) is the")
print("    real test of whether OUR forecast adds tradable value over implied.")
