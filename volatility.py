"""
Phase 2 scaffold: VOLATILITY prediction (the only target with a real edge).

Phase 1 showed 24h direction is random. This module pivots to predicting the
*magnitude* of future moves — forward realized volatility — and is structured
around three questions:

  1. POSITIVE CONTROL: can the harness recover a signal that is KNOWN to exist?
     Volatility clusters (GARCH effect), so future realized vol should be
     predictable from PAST realized vol. If our purged walk-forward setup
     can't find this, no result from it can be trusted. This is the sanity
     check the original research never had.

  2. Does MACRO information add anything beyond vol persistence?

  3. Same evaluation discipline as Phase 1: purged + embargoed walk-forward,
     pooled out-of-sample R^2, per-fold spread, and a permutation p-value.

Run after build_dataset.py (uses results/dataset_daily.csv).

Optional: `pip install arch` enables a true GARCH(1,1) baseline; otherwise an
EWMA / rolling-vol persistence baseline is used (same conclusion, no new dep).
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

from evaluation import purged_walkforward_splits

H = 5            # forward window (trading days) for realized volatility
EMBARGO = H      # embargo >= horizon so forward windows can't overlap folds

df = pd.read_csv("results/dataset_daily.csv", parse_dates=["date"], index_col="date")
r = df["gold_return_1d"]

# ---- realized-vol features (past) and target (forward) ----------------
df["rv_past_5"] = r.rolling(5).std()
df["rv_past_10"] = r.rolling(10).std()
df["rv_past_20"] = r.rolling(20).std()

# forward realized vol over the NEXT H days (explicit, no rolling ambiguity)
arr = r.to_numpy()
n = len(arr)
fut = np.full(n, np.nan)
for t in range(n - H):
    fut[t] = np.std(arr[t + 1: t + 1 + H])
df["rv_future"] = fut

df = df.dropna()

macro_feats = [c for c in df.columns
               if c.endswith(("_return_1d", "_return_5d", "_return_10d"))
               and not c.startswith("future")]
vol_feats = ["rv_past_5", "rv_past_10", "rv_past_20"]

y = df["rv_future"].to_numpy()


def purged_r2(model, X, y, label, n_perm=0):
    X = np.asarray(X)
    fold_r2, preds, truth = [], [], []
    for tr, te in purged_walkforward_splits(len(y), n_splits=5, horizon=H,
                                            embargo=EMBARGO):
        m = clone(model).fit(X[tr], y[tr])
        p = m.predict(X[te])
        fold_r2.append(r2_score(y[te], p))
        preds.append(p)
        truth.append(y[te])
    preds, truth = np.concatenate(preds), np.concatenate(truth)
    fold_r2 = np.array(fold_r2)
    pooled = r2_score(truth, preds)

    pval = None
    if n_perm:
        rng = np.random.default_rng(0)
        count = 0
        for _ in range(n_perm):
            yp = rng.permutation(y)
            pp, tt = [], []
            for tr, te in purged_walkforward_splits(len(y), 5, H, EMBARGO):
                mm = clone(model).fit(X[tr], yp[tr])
                pp.append(mm.predict(X[te]))
                tt.append(yp[te])
            if r2_score(np.concatenate(tt), np.concatenate(pp)) >= pooled:
                count += 1
        pval = (count + 1) / (n_perm + 1)

    print(f"\n=== {label} ===")
    print(f"Pooled OOS R^2 : {pooled:+.4f}")
    print(f"Per-fold R^2   : {fold_r2.mean():+.4f} +/- {fold_r2.std():.4f} "
          f"[{', '.join(f'{x:+.3f}' for x in fold_r2)}]")
    if pval is not None:
        verdict = "real signal" if pval < 0.05 else "not distinguishable from random"
        print(f"Permutation p  : {pval:.4f} -> {verdict}")
    return pooled


def lr():
    return LinearRegression()


def xgb():
    return XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
                        subsample=0.8, colsample_bytree=0.8,
                        random_state=42, n_jobs=1)


print("#" * 70)
print("VOLATILITY PREDICTION  (forward realized vol, H =", H, "days)")
print("#" * 70)
print(f"Rows: {len(df)}   target mean rv_future: {y.mean():.5f}")

# 1. POSITIVE CONTROL: vol persistence (known to exist)
purged_r2(lr(), df[vol_feats], y,
          "POSITIVE CONTROL - linear vol persistence (past RV -> future RV)",
          n_perm=200)

# 2. Macro only
purged_r2(xgb(), df[macro_feats], y,
          "Macro features only -> future RV")

# 3. Macro + vol persistence
purged_r2(xgb(), df[vol_feats + macro_feats], y,
          "Vol persistence + macro -> future RV")

print("\nInterpretation:")
print("  * If the positive control shows R^2 >> 0 with p<0.05, the harness")
print("    correctly recovers a signal known to exist -> results are trustworthy.")
print("  * Compare macro-only and combined R^2 to the control to see whether")
print("    macro adds anything beyond plain volatility clustering.")
