"""
Rigorous evaluation utilities for forecasting experiments.

This module fixes the statistical weaknesses identified in the Phase 1 review:

  * Purged + embargoed walk-forward CV (Lopez de Prado style) so that
    overlapping forward-return labels do not leak between train and test.
  * Confidence intervals on every metric via the across-fold distribution
    AND a bootstrap on the pooled out-of-sample predictions.
  * A permutation test: shuffle the labels many times and see how often a
    model does at least as well by chance -> an honest p-value for "is this
    better than random?".
  * Baseline comparison against a DummyClassifier (majority / stratified),
    because a model that only predicts the majority class can look great on
    accuracy while learning nothing.

Use evaluate_classifier() for the whole package.
"""

import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score


# ---------------------------------------------------------------------------
# Purged walk-forward splitter
# ---------------------------------------------------------------------------
def purged_walkforward_splits(n, n_splits=5, horizon=1, embargo=0):
    """
    Expanding-window walk-forward splits with purge+embargo.

    The last `n_splits` contiguous blocks become successive test folds; each
    training set is everything strictly before the fold, minus a gap of
    (horizon - 1 + embargo) rows so that a training label's forward window
    cannot overlap the test block.
    """
    fold_size = n // (n_splits + 1)
    gap = (horizon - 1) + embargo

    for k in range(1, n_splits + 1):
        test_start = fold_size * k
        test_end = fold_size * (k + 1) if k < n_splits else n
        train_end = max(0, test_start - gap)
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        if len(train_idx) and len(test_idx):
            yield train_idx, test_idx


# ---------------------------------------------------------------------------
# Metrics with uncertainty
# ---------------------------------------------------------------------------
def _bootstrap_accuracy_ci(y_true, y_pred, n_boot=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    accs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        accs[b] = accuracy_score(y_true[idx], y_pred[idx])
    lo = np.quantile(accs, alpha / 2)
    hi = np.quantile(accs, 1 - alpha / 2)
    return float(lo), float(hi)


def _permutation_pvalue(model, X, y, observed_acc, splitter_kwargs,
                        n_perm=200, seed=0):
    """
    Fraction of label-shuffled runs whose pooled OOS accuracy >= observed.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    count = 0
    for p in range(n_perm):
        y_perm = rng.permutation(y)
        preds, truth = _pooled_oos(model, X, y_perm, **splitter_kwargs)
        acc = accuracy_score(truth, preds)
        if acc >= observed_acc:
            count += 1
    # +1 smoothing
    return (count + 1) / (n_perm + 1)


def _pooled_oos(model, X, y, n_splits=5, horizon=1, embargo=0):
    X = np.asarray(X)
    y = np.asarray(y)
    preds, truth = [], []
    for tr, te in purged_walkforward_splits(len(y), n_splits, horizon, embargo):
        m = clone(model)
        m.fit(X[tr], y[tr])
        preds.append(m.predict(X[te]))
        truth.append(y[te])
    return np.concatenate(preds), np.concatenate(truth)


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------
def evaluate_classifier(model, X, y, name="model", n_splits=5, horizon=1,
                        embargo=5, n_perm=200, run_permutation=True):
    X = np.asarray(X)
    y = np.asarray(y)
    splitter_kwargs = dict(n_splits=n_splits, horizon=horizon, embargo=embargo)

    fold_accs = []
    pooled_pred, pooled_true = [], []
    for tr, te in purged_walkforward_splits(len(y), **splitter_kwargs):
        m = clone(model)
        m.fit(X[tr], y[tr])
        p = m.predict(X[te])
        fold_accs.append(accuracy_score(y[te], p))
        pooled_pred.append(p)
        pooled_true.append(y[te])

    pooled_pred = np.concatenate(pooled_pred)
    pooled_true = np.concatenate(pooled_true)

    acc = accuracy_score(pooled_true, pooled_pred)
    ci_lo, ci_hi = _bootstrap_accuracy_ci(pooled_true, pooled_pred)
    fold_accs = np.array(fold_accs)

    # baselines fit/evaluated under the SAME purged splits
    dummy_major = DummyClassifier(strategy="most_frequent")
    dp, dt = _pooled_oos(dummy_major, X, y, **splitter_kwargs)
    dummy_acc = accuracy_score(dt, dp)

    pval = None
    if run_permutation:
        pval = _permutation_pvalue(model, X, y, acc, splitter_kwargs,
                                   n_perm=n_perm)

    print(f"\n=== {name} : purged walk-forward ({n_splits} folds, "
          f"horizon={horizon}, embargo={embargo}) ===")
    print(f"Pooled OOS rows      : {len(pooled_true)}")
    print(f"Accuracy             : {acc:.4f}  (95% CI {ci_lo:.3f}-{ci_hi:.3f})")
    print(f"Per-fold accuracy    : "
          f"{fold_accs.mean():.4f} +/- {fold_accs.std():.4f}  "
          f"[{', '.join(f'{a:.3f}' for a in fold_accs)}]")
    print(f"Majority-class base  : {dummy_acc:.4f}")
    if pval is not None:
        verdict = "NOT distinguishable from random" if pval > 0.05 else "better than random"
        print(f"Permutation p-value  : {pval:.4f}  -> {verdict}")

    return {
        "name": name,
        "accuracy": acc,
        "ci": (ci_lo, ci_hi),
        "fold_accs": fold_accs,
        "dummy_acc": dummy_acc,
        "p_value": pval,
        "pooled_true": pooled_true,
        "pooled_pred": pooled_pred,
    }
