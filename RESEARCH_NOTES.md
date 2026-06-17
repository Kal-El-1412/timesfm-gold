# Gold Forecast Engine — Rigor Upgrades (Phase 1 re-evaluation)

This pass applied the review recommendations to the codebase and re-ran the
experiments with a statistically defensible setup. The headline Phase-1
conclusion ("no price-only edge for 24h gold direction") **survives** — but is
now backed by 2,049 out-of-sample observations instead of ~157, and one **new,
statistically significant** finding emerged: short-horizon *volatility* is
mildly predictable even though *direction* is not.

## What changed in the code

| Area | Before | After |
|------|--------|-------|
| Data | `6mo` hourly, ~782 rows | `10y` daily, **2,458 rows** (`build_dataset.py`) |
| Macro alignment | `merge_asof` 3h tol → stale 24h-futures vs equity joins | exact inner join on trading **date**, same resolution |
| Validation | single 80/20 split | **purged + embargoed walk-forward CV** (`evaluation.py`) |
| Uncertainty | none | bootstrap 95% CI, per-fold spread, **permutation p-value** |
| Baseline | none | majority-class `DummyClassifier` under same splits |
| Reporting bug | AUC printed as "Accuracy"; target mislabeled | accuracy **and** AUC reported separately; targets named correctly |
| Label leakage | — | found & fixed: `future_return_1d` was being selected as a feature (gave fake 99.8% acc); target renamed `future_return` + `not startswith("future")` guard |
| Costs | ignored | `strategy_backtest.py` charges `COST_BPS` per side on every flip |
| TimesFM quantiles | discarded (used point endpoint only) | `forecast_engine.py` now exposes the quantile band as forecast uncertainty / confidence |
| Direction test | forecast horizon 24h vs eval horizon 8h (mismatch) | `direction_accuracy.py` forces forecast horizon == eval horizon + binomial CI |

## Results (leak-free, purged walk-forward, 2,049 OOS rows)

| Experiment | Metric | Result | Baseline | Verdict |
|-----------|--------|--------|----------|---------|
| Linear regression (return) | OOS R² | **−0.70** | 0 (mean) | worse than mean |
| XGB **direction** (up/down) | accuracy | **50.9%** (CI 48.8–53.1) | 54.2% majority | p=0.51 → **random** |
| XGB direction | AUC | 0.489 | 0.50 | no edge |
| XGB multiclass DOWN/MID/UP | accuracy | **32.4%** (±2.7) | 33.3% | random |
| XGB **big-move** (volatility) | accuracy | **52.9%** (CI 50.7–55.1) | 48.6% | **p=0.0099 → real** |
| XGB big-move | AUC | **0.545** | 0.50 | modest edge |

### Cost-aware backtest of the direction signal (`strategy_backtest.py`)

```
Hit rate (active)   : 0.506
Strategy NET   total=+4.71%   ann=+0.57%   Sharpe=+0.12   maxDD=-41%
Strategy GROSS total=+42.62%  ann=+4.46%   Sharpe=+0.35   maxDD=-40%
Buy & Hold     total=+211.94% ann=+15.02%  Sharpe=+0.89   maxDD=-23%
```

Trading the direction signal underperforms simply holding gold and has a worse
drawdown; transaction costs alone erase ~38 points of gross return. **The
direction signal is not tradable.**

### TimesFM direction test (`direction_accuracy.py`, horizon == eval horizon)

Full foundation-model run, hourly bars, forecast horizon matched to evaluation
horizon, with binomial 95% CIs:

| Window | 4h | 8h | 24h |
|--------|----|----|-----|
| 2y (n≈1108) | 50.5% [47.6, 53.5] | 50.5% [47.5, 53.4] | 51.1% [48.1, 54.0] |
| 1y (n≈534)  | 52.1% [47.8, 56.3] | 53.9% [49.7, 58.2] | 50.8% [46.5, 55.0] |

Every CI brackets 50%. The 1y/8h point estimate (53.9%) looks directional but
its CI includes 50 **and** it does not replicate in the 2y window (50.5%) — and
the sign of the deviation even flips versus the original report (which was below
50% on all horizons). That is the signature of noise, not edge. TimesFM (200M
foundation model) and the leak-free XGBoost macro model (50.9%, p=0.51) now
**independently agree** that 24h gold direction is unpredictable.

## Interpretation

1. **Direction forecasting from price/macro is dead** — now proven rigorously,
   not just suggested. p=0.51, AUC<0.5, loses to buy-and-hold after costs.
2. **Volatility/magnitude is mildly predictable** (p=0.0099). This directly
   validates the report's own Phase-2 instinct: predict *volatility regime and
   large-move probability*, not direction. That is the thread worth pulling.
3. The framework now produces honest, error-barred answers and caught a real
   leakage bug on its first run — exactly what a research harness should do.

## Phase 2 (started) — volatility prediction (`volatility.py`)

Pivot from direction to *magnitude*. Forward realized volatility over the next
H=5 days, evaluated with the same purged walk-forward discipline.

| Model | Pooled OOS R² | Note |
|-------|---------------|------|
| **Positive control** — past RV → future RV (linear) | **+0.14**, perm p=0.005 | signal known to exist is recovered → harness is trustworthy |
| Macro features only → future RV | −0.04 | macro does not predict vol |
| Vol persistence + macro (XGB) | +0.04 | macro adds overfit noise; worse than persistence alone |

Takeaways:
- The **positive control passes** — the framework correctly finds volatility
  clustering, which validates that the Phase-1 null results are real, not an
  artifact of an over-conservative pipeline.
- **Volatility is predictable from its own history**; macro factors do not help.
- Next: realized-vol / large-move *trading* construct (option-style payoffs),
  event-window features (CPI/NFP/FOMC), and an optional GARCH(1,1) baseline
  (`pip install arch`) alongside the EWMA persistence control.

## How to run

```bash
python build_dataset.py --period 10y --interval 1d --horizon 1
python ml_baseline.py            # linear R² baseline
python xgboost_classifier.py     # direction + volatility, CI + permutation
python xgboost_multiclass.py     # DOWN/MID/UP regime
python strategy_backtest.py      # cost-aware walk-forward PnL
python direction_accuracy.py     # TimesFM direction test (downloads model)
```
