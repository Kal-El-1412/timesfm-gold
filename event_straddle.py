"""
Event vol-timing backtest: is being LONG volatility into scheduled events paid?

Background:
  * Phase 2b: UNCONDITIONALLY, implied vol (GVZ) > realized 77% of the time, so
    long volatility loses on average (the variance risk premium favours sellers).
  * Phase 3 A1: gold moves are ~18% bigger on event days.

So the precise, tradable question is: are scheduled events the POCKETS where
realized vol exceeds what implied priced in -- i.e. where LONG straddles pay --
even though long vol loses on average? If yes, the strategy is "buy a straddle
into NFP/FOMC, stay flat otherwise".

Proxy (we have GVZ but no option chain):
  * A 1-day straddle held over the event day, priced off GVZ at entry (t-1).
  * implied 1-day move  = GVZ_{t-1} / 100 / sqrt(252)        (breakeven)
  * realized 1-day move = |gold_return_1d_t|
  * long-straddle payoff ~ realized - implied - cost          (in return units)
  * cost is a round-trip option cost expressed in underlying-return points.

This is a first-order proxy (ignores gamma path / vega term structure) but is
the right sign test for an event vol-timing edge. A negative result kills the
idea; a positive one justifies a real option-chain backtest.

Run after build_dataset.py.
"""

import numpy as np
import pandas as pd
from scipy import stats

from event_calendar import build_event_flags

COST_RET = 0.0010   # round-trip straddle cost in underlying return points (~10bps)
ANN = np.sqrt(252)

df = pd.read_csv("results/dataset_daily.csv", parse_dates=["date"], index_col="date")
df = df.join(build_event_flags(df.index, window=0))   # event DAY itself (window=0)

implied_daily = df["gvz_close"].shift(1) / 100.0 / ANN   # priced at prior close
realized_daily = df["gold_return_1d"].abs()

# long-straddle payoff proxy (return units), net of cost
payoff = (realized_daily - implied_daily - COST_RET)
df["payoff"] = payoff
df = df.dropna(subset=["payoff", "is_event"])

print("#" * 70)
print(f"EVENT VOL-TIMING / LONG-STRADDLE PROXY  (cost={COST_RET*1e4:.0f}bps)")
print("#" * 70)
print(f"Rows: {len(df)}   mean implied daily move: {implied_daily.mean():.4f}  "
      f"mean realized: {realized_daily.mean():.4f}")


def bucket(label, mask):
    p = df.loc[mask == 1, "payoff"]
    if len(p) == 0:
        print(f"  {label:16s} no obs"); return None
    sharpe = p.mean() / p.std() * np.sqrt(252) if p.std() > 0 else 0
    print(f"  {label:16s} n={len(p):4d}  mean_payoff={p.mean()*1e4:+6.1f}bps  "
          f"win%={(p>0).mean():.0%}  Sharpe(ann)={sharpe:+.2f}  "
          f"total={p.sum()*100:+.1f}%")
    return p


print("\nLong-straddle payoff by day type (positive => long vol pays):")
p_evt = bucket("Event days", df["is_event"])
p_nfp = bucket("  NFP days", df["is_nfp"])
p_fomc = bucket("  FOMC days", df["is_fomc"])
p_non = bucket("Non-event days", 1 - df["is_event"])
p_all = bucket("All days", pd.Series(1, index=df.index))

# is the event-day payoff significantly better than non-event?
if p_evt is not None and p_non is not None:
    t, pval = stats.ttest_ind(p_evt, p_non, equal_var=False)
    print(f"\nEvent vs non-event payoff: diff="
          f"{(p_evt.mean()-p_non.mean())*1e4:+.1f}bps  p={pval:.3f}")

print("\nInterpretation:")
print("  * Unconditional/all-days payoff should be NEGATIVE (variance premium).")
print("  * If EVENT-day payoff is positive (and > non-event, small p), long")
print("    straddles into events are an edge -> justify a real option backtest.")
print("  * If event payoff is also negative, vol-selling (not buying) is the")
print("    only edge, and events do not flip it -> A-track is exhausted.")
