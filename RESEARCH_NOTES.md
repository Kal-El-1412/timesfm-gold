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

## Interpretation

1. **Direction forecasting from price/macro is dead** — now proven rigorously,
   not just suggested. p=0.51, AUC<0.5, loses to buy-and-hold after costs.
2. **Volatility/magnitude is mildly predictable** (p=0.0099). This directly
   validates the report's own Phase-2 instinct: predict *volatility regime and
   large-move probability*, not direction. That is the thread worth pulling.
3. The framework now produces honest, error-barred answers and caught a real
   leakage bug on its first run — exactly what a research harness should do.

## How to run

```bash
python build_dataset.py --period 10y --interval 1d --horizon 1
python ml_baseline.py            # linear R² baseline
python xgboost_classifier.py     # direction + volatility, CI + permutation
python xgboost_multiclass.py     # DOWN/MID/UP regime
python strategy_backtest.py      # cost-aware walk-forward PnL
python direction_accuracy.py     # TimesFM direction test (downloads model)
```
