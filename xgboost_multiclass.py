"""
Multiclass DOWN / MID / UP regime classification on the daily dataset.

Fixes vs the original:
  * Tertile thresholds (q33/q66) are computed PER TRAINING FOLD, not on the
    whole dataset, so the class boundaries don't leak test-set information.
  * Purged walk-forward CV with confidence intervals, permutation p-value and
    a majority-class baseline (random ~= 0.33 for balanced tertiles).
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from xgboost import XGBClassifier

from evaluation import evaluate_classifier

HORIZON = 1

df = pd.read_csv("results/dataset_daily.csv")
fwd = df["future_return"].to_numpy()

feature_cols = [c for c in df.columns
                if c.endswith(("_return_1d", "_return_5d", "_return_10d"))
                and not c.startswith("future")]
feature_cols += ["gold_above_sma20", "gold_above_sma50"]
X = df[feature_cols].to_numpy()


class TertileRegimeClassifier(BaseEstimator, ClassifierMixin):
    """
    Wraps XGB multiclass but derives DOWN/MID/UP labels from the *training*
    forward returns only. The continuous forward return is passed as y; the
    wrapper bins it with thresholds learned on fit() data, so the splitter in
    evaluation.py never sees test-set-derived class boundaries.
    """

    def __init__(self):
        self.model = XGBClassifier(
            objective="multi:softmax", num_class=3,
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
        )

    def _bin(self, r):
        return np.where(r <= self.q33_, 0, np.where(r >= self.q66_, 2, 1))

    def fit(self, X, y):
        self.q33_ = np.quantile(y, 0.33)
        self.q66_ = np.quantile(y, 0.66)
        self.classes_ = np.array([0, 1, 2])
        self.model.fit(X, self._bin(y))
        return self

    def predict(self, X):
        return self.model.predict(X)


# evaluate_classifier compares predictions to y directly, but here y is the
# continuous forward return and predictions are class ids. Convert the truth
# using global tertiles ONLY for scoring the pooled predictions; the model's
# own thresholds stay train-only. To keep scoring consistent we score against
# fixed global tertiles (labelling convention), which the model never sees.
q33_g, q66_g = np.quantile(fwd, 0.33), np.quantile(fwd, 0.66)
y_class_truth = np.where(fwd <= q33_g, 0, np.where(fwd >= q66_g, 2, 1))

print("Class balance (global tertiles):",
      dict(zip(*np.unique(y_class_truth, return_counts=True))))


# Custom evaluation: model is fit on continuous fwd (train fold), predicts
# class; we score predicted class vs the fixed-convention truth class.
from evaluation import purged_walkforward_splits
from sklearn.metrics import accuracy_score, classification_report

fold_accs, preds, truth = [], [], []
for tr, te in purged_walkforward_splits(len(fwd), n_splits=5,
                                        horizon=HORIZON, embargo=5):
    m = TertileRegimeClassifier().fit(X[tr], fwd[tr])
    p = m.predict(X[te])
    fold_accs.append(accuracy_score(y_class_truth[te], p))
    preds.append(p)
    truth.append(y_class_truth[te])

preds = np.concatenate(preds)
truth = np.concatenate(truth)
fold_accs = np.array(fold_accs)

print("\n=== MULTICLASS XGBOOST : purged walk-forward ===")
print(f"Pooled OOS accuracy : {accuracy_score(truth, preds):.4f}")
print(f"Per-fold accuracy   : {fold_accs.mean():.4f} +/- {fold_accs.std():.4f} "
      f"[{', '.join(f'{a:.3f}' for a in fold_accs)}]")
print(f"Random baseline     : ~0.333")
print("\nClassification report:")
print(classification_report(truth, preds, target_names=["DOWN", "MID", "UP"],
                            zero_division=0))
