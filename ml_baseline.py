"""
Linear-regression baseline on the daily dataset, scored with purged
walk-forward CV. Reports out-of-sample R^2 (pooled) and per-fold spread, so a
single lucky/unlucky split can't drive the conclusion. R^2 < 0 means the model
is worse than predicting the training-mean return.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from evaluation import purged_walkforward_splits

df = pd.read_csv("results/dataset_daily.csv")

features = [c for c in df.columns
            if c.endswith(("_return_1d", "_return_5d", "_return_10d"))
            and not c.startswith("future")]
X = df[features].to_numpy()
y = df["future_return"].to_numpy()

model = LinearRegression()

fold_r2, preds, truth = [], [], []
for tr, te in purged_walkforward_splits(len(y), n_splits=5, horizon=1, embargo=5):
    m = clone(model).fit(X[tr], y[tr])
    p = m.predict(X[te])
    fold_r2.append(r2_score(y[te], p))
    preds.append(p)
    truth.append(y[te])

preds, truth = np.concatenate(preds), np.concatenate(truth)
fold_r2 = np.array(fold_r2)

print("\n=== LINEAR BASELINE : purged walk-forward ===")
print(f"Pooled OOS R^2 : {r2_score(truth, preds):.5f}")
print(f"Per-fold R^2   : {fold_r2.mean():.5f} +/- {fold_r2.std():.5f} "
      f"[{', '.join(f'{r:.3f}' for r in fold_r2)}]")
print("(R^2 <= 0  =>  no better than predicting the mean return)")
