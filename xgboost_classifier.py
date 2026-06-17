"""
XGBoost direction / volatility experiments on the daily macro dataset.

Fixes vs the original:
  * Reports ACCURACY and AUC as SEPARATE numbers (the original printed AUC
    under an "Accuracy" label).
  * Names the target correctly. We run TWO clearly-labelled experiments:
      1. target_up        -> "can we predict 24h+ DIRECTION?"
      2. target_big_move  -> "can we predict a large MOVE (volatility)?"
  * Uses purged walk-forward CV with confidence intervals, a permutation
    p-value, and a majority-class baseline instead of a single 80/20 split.
"""

import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from evaluation import evaluate_classifier, purged_walkforward_splits

HORIZON = 1  # forward horizon used when the dataset was built

df = pd.read_csv("results/dataset_daily.csv")

feature_cols = [c for c in df.columns
                if c.endswith(("_return_1d", "_return_5d", "_return_10d"))
                and not c.startswith("future")]
feature_cols += ["gold_above_sma20", "gold_above_sma50"]

X = df[feature_cols]


def make_model():
    return XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42, n_jobs=1,
    )


def pooled_auc(model, X, y):
    """AUC over pooled purged-CV out-of-sample probabilities."""
    import numpy as np
    from sklearn.base import clone
    Xv, yv = np.asarray(X), np.asarray(y)
    probs, truth = [], []
    for tr, te in purged_walkforward_splits(len(yv), n_splits=5,
                                            horizon=HORIZON, embargo=5):
        m = clone(model)
        m.fit(Xv[tr], yv[tr])
        probs.append(m.predict_proba(Xv[te])[:, 1])
        truth.append(yv[te])
    probs, truth = np.concatenate(probs), np.concatenate(truth)
    return roc_auc_score(truth, probs)


print("#" * 70)
print("EXPERIMENT 1 - DIRECTION (target_up): can we predict gold up/down?")
print("#" * 70)
res_dir = evaluate_classifier(make_model(), X, df["target_up"],
                              name="XGB direction", horizon=HORIZON, embargo=5,
                              n_perm=100)
print(f"Pooled OOS AUC       : {pooled_auc(make_model(), X, df['target_up']):.4f}")

print("\n" + "#" * 70)
print("EXPERIMENT 2 - VOLATILITY (target_big_move): can we predict a big move?")
print("#" * 70)
res_vol = evaluate_classifier(make_model(), X, df["target_big_move"],
                              name="XGB big-move", horizon=HORIZON, embargo=5,
                              n_perm=100)
print(f"Pooled OOS AUC       : {pooled_auc(make_model(), X, df['target_big_move']):.4f}")

# feature importance from a full-sample fit (descriptive only)
m = make_model().fit(X, df["target_up"])
importance = (pd.DataFrame({"feature": feature_cols,
                            "importance": m.feature_importances_})
              .sort_values("importance", ascending=False))
print("\nTop 10 features (direction model, descriptive):")
print(importance.head(10).to_string(index=False))
importance.to_csv("results/xgboost_feature_importance.csv", index=False)
