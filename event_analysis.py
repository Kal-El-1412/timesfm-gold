"""
Phase 3 (A1): does gold's behaviour concentrate around scheduled events?

Three questions, all out-of-sample where a model is involved:

  1. MOVE SIZE: are absolute gold moves bigger on event days (NFP/FOMC)?
     If yes -> options/straddle TIMING around events is informative even if
     direction is not. This is a usable result on its own.

  2. DIRECTION: is next-day direction any more predictable inside event windows
     than outside? (purged walk-forward XGB, accuracy stratified by window).

  3. Do event flags add anything to the direction model overall?

Run after build_dataset.py.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from xgboost import XGBClassifier

from evaluation import purged_walkforward_splits
from event_calendar import build_event_flags

df = pd.read_csv("results/dataset_daily.csv", parse_dates=["date"], index_col="date")
flags = build_event_flags(df.index, window=1)
df = df.join(flags)

absmove = df["gold_return_1d"].abs() * 100  # % absolute daily move

# ---------------------------------------------------------------------------
# Q1. Move size on event vs non-event days
# ---------------------------------------------------------------------------
print("#" * 70)
print("Q1. ABSOLUTE DAILY MOVE on event vs non-event days (%)")
print("#" * 70)


def move_compare(label, mask):
    a = absmove[mask == 1]
    b = absmove[mask == 0]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    print(f"  {label:12s} event={a.mean():.3f}%  non-event={b.mean():.3f}%  "
          f"ratio={a.mean()/b.mean():.2f}x  p={p:.2e}  (n_evt={len(a)})")


move_compare("NFP", df["is_nfp"])
move_compare("FOMC", df["is_fomc"])
move_compare("Any event", df["is_event"])
print("-> ratio>1 with small p => events reliably enlarge moves (vol timing).")

# ---------------------------------------------------------------------------
# Q2/Q3. Direction predictability inside vs outside event windows
# ---------------------------------------------------------------------------
base_feats = [c for c in df.columns
              if c.endswith(("_return_1d", "_return_5d", "_return_10d"))
              and not c.startswith(("future", "gvz"))]
base_feats += ["gold_above_sma20", "gold_above_sma50"]
event_feats = base_feats + ["is_nfp", "is_fomc", "is_event"]
y = df["target_up"].to_numpy()


def oos_pred(features):
    X = df[features].to_numpy()
    pred = np.full(len(y), np.nan)
    for tr, te in purged_walkforward_splits(len(y), 5, horizon=1, embargo=5):
        m = clone(XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8,
                                eval_metric="logloss", random_state=42, n_jobs=1))
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def acc(pred, mask):
    m = (~np.isnan(pred)) & (mask == 1)
    return (pred[m] == y[m]).mean(), int(m.sum())


print("\n" + "#" * 70)
print("Q2/Q3. DIRECTION accuracy (purged walk-forward), stratified by window")
print("#" * 70)
pred_base = oos_pred(base_feats)
pred_event = oos_pred(event_feats)

tested = ~np.isnan(pred_base)
all_mask = tested.astype(int)
for label, mask in [("All days", all_mask),
                    ("Event-window days", df["is_event"].to_numpy() * tested),
                    ("Non-event days", (1 - df["is_event"].to_numpy()) * tested)]:
    a_b, n = acc(pred_base, mask)
    a_e, _ = acc(pred_event, mask)
    print(f"  {label:20s} base={a_b:.3f}  +eventfeats={a_e:.3f}  (n={n})")

print("\nInterpretation:")
print("  * Direction ~0.50 in every bucket => events do NOT make direction")
print("    predictable; the surprise (A2) data would be the only remaining shot.")
print("  * If Q1 shows big move-size ratios, the usable result is VOL timing")
print("    (buy options/straddles into events), not directional forecasting.")
