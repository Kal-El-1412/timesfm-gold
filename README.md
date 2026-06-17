# timesfm-gold

A research framework for testing whether modern AI / ML techniques can forecast
gold prices — built around **falsification and measurable criteria**, not
convincing charts.

> **Status: research concluded.** Across direction, volatility, and event
> windows, no tradable forecasting edge was found in gold from market-derived
> data; the market (notably implied vol) already prices the predictable
> information. See **[`FINDINGS.md`](FINDINGS.md)** for the executive summary and
> [`RESEARCH_NOTES.md`](RESEARCH_NOTES.md) for full methodology and per-phase
> results.

## TL;DR findings (leak-free, purged walk-forward)

| Hypothesis | Method | Result | Verdict |
|------------|--------|--------|---------|
| 24h direction | TimesFM (200M) | 50.5–51.1%, CIs bracket 50 | random |
| 24h direction | XGBoost macro | 50.9%, p=0.51 | random |
| direction tradable | cost-aware backtest | +4.7% vs B&H +212% | untradable |
| 3-class regime | XGBoost | 32.4% vs 33.3% | random |
| return level | Linear | R² −0.70 | worse than mean |
| **volatility / big move** | XGBoost | 52.9%, AUC 0.545, **p=0.0099** | **mild real edge** |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Pipeline

```bash
python build_dataset.py --period 10y --interval 1d --horizon 1  # build dataset
python ml_baseline.py          # linear R² baseline (purged CV)
python xgboost_classifier.py   # direction + volatility, CIs + permutation test
python xgboost_multiclass.py   # DOWN/MID/UP regime
python strategy_backtest.py    # cost-aware walk-forward PnL vs buy & hold
python direction_accuracy.py   # TimesFM direction test (downloads ~200M model)
```

## Key modules

| File | Purpose |
|------|---------|
| `build_dataset.py` | Multi-year daily dataset, clean same-resolution macro alignment, documented targets |
| `evaluation.py` | Purged + embargoed walk-forward CV, bootstrap CIs, permutation p-values, baselines |
| `forecast_engine.py` | TimesFM wrapper; exposes the quantile band as forecast uncertainty / confidence |
| `strategy_backtest.py` | Cost-aware backtest (charges bps per side on every position flip) |
| `xgboost_classifier.py` | Direction & volatility classifiers with honest metrics |

## Methodology notes

- **No single train/test split.** Every metric comes from purged, embargoed
  walk-forward folds so overlapping forward-return labels cannot leak.
- **Error bars on everything.** Point estimates near 50% are reported with 95%
  CIs and a permutation p-value, because a bare "47%" is indistinguishable from
  a coin flip without them.
- **Costs are charged.** A signal that looks good gross can be worthless net.

## Status

**Concluded (Phases 1–3).** Direction is random across two independent model
families; volatility is predictable only to the extent the market's implied vol
(GVZ) already does it; scheduled-event windows enlarge moves but options
over-price the enlargement. The only positive expectancy is the short-vol
variance risk premium — a known risk premium, not a forecasting edge. The single
untested path (directional reaction to economic *surprises*) needs an external
consensus data feed and was not pursued. Full write-up in [`FINDINGS.md`](FINDINGS.md).
