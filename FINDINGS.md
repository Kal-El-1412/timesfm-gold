# Gold Forecast Engine — Final Findings

**Status: research concluded.** Phases 1–3 are complete. This document is the
executive summary; `RESEARCH_NOTES.md` has the full methodology and per-phase
numbers.

## The question

Can a profitable gold price-forecasting / trading engine be built from market
data using modern AI/ML (TimesFM, gradient-boosted macro models, implied vol,
event windows)?

## The answer

**No — not from market-price-derived data.** Across every dimension we could test
with rigorous, leak-free, out-of-sample evaluation, gold is efficiently priced.
The market already contains the predictable information; no model in this study
added tradable value over it.

## What was tested, and what we found

| Dimension | Method | Result | Verdict |
|-----------|--------|--------|---------|
| 24h direction | TimesFM (200M) | 50.5–51.1%, CIs bracket 50 | random |
| 24h direction | XGBoost macro | 50.9%, permutation p=0.51 | random |
| direction tradable | cost-aware backtest | +4.7% vs buy-and-hold +212% | untradable |
| 3-class regime | XGBoost | 32.4% vs 33.3% | random |
| return level | linear | OOS R² = −0.70 | worse than mean |
| volatility (own history) | XGBoost | R²=0.14, p=0.005 | real but… |
| …vs the market | realized vs **implied (GVZ)** | implied R²=0.30; ours adds nothing | subsumed |
| variance risk premium | implied vs realized | implied > realized 77%, short-vol Sharpe 0.83 | premium, not a forecast |
| event direction | NFP/FOMC windows | ~50% | random |
| event move size | NFP/FOMC windows | ~18% larger, p=0.001 | real |
| long vol into events | straddle proxy | −44.6 bps/event, p=0.66 vs non-event | no edge |

## The three things worth remembering

1. **Direction is dead.** Two independent model families (a time-series
   foundation model and gradient-boosted macro features) plus event-window
   conditioning all land at ~50% with overlapping confidence intervals. This is
   the strongest form of a null result.

2. **Volatility looked alive until benchmarked against the market.** Realized vol
   is predictable from its own history (R²=0.14), but the market's implied vol
   (GVZ) forecasts it far better (R²=0.30), and our features add nothing once
   implied is included. The apparent edge was already priced in.

3. **The only money is a known risk premium, not a forecast.** Selling volatility
   is paid (implied > realized 77% of the time), but that is generic short-vol
   beta with crash-tail risk — available to anyone, and not what this project set
   out to build. Events enlarge moves, yet options over-price that enlargement, so
   even buying vol with perfect event timing loses.

## What was NOT tested (the one open door)

A **directional reaction to economic *surprises*** (actual − consensus on
CPI/NFP/FOMC). This requires a paid/keyed economic-calendar feed that was not
available, and its expected payoff was judged too uncertain to justify the cost.
The framework is ready to ingest such data (`event_calendar.py` accepts a CSV
override) if it is ever obtained — see the "A2" notes in `RESEARCH_NOTES.md`.

## What this project delivered

Not a trading strategy — a **rigorous answer to whether one is achievable**, plus
a reusable quantitative-research framework that:

- builds clean multi-year, multi-market datasets with no resolution-misalignment,
- evaluates hypotheses with purged/embargoed walk-forward CV, bootstrap CIs,
  permutation tests, and proper baselines,
- benchmarks any signal against the market's own forecast (implied vol),
- charges realistic transaction costs, and
- caught a real label-leakage bug on its first run.

The same framework can be pointed at any other asset to answer the same question
in days, not months. **The negative result is the result** — and it was reached
without fooling ourselves, which was the entire point.
