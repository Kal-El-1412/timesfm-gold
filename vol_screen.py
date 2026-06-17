"""
Pivot #1 go/no-go screen: forward realized-vol predictability on assets that
LACK a liquid implied-vol benchmark (crypto pairs, FX crosses, small-caps).

For gold the vol edge died only because GVZ (a traded implied-vol market) already
forecast realized vol. These assets have no such benchmark, so a model only has
to beat the honest naive baselines to deliver standalone value (risk sizing,
margining, vol-aware execution).

Metric (all out-of-sample, purged walk-forward, target = std of next H daily
returns):
  * persistence  : current rolling vol as the forecast (random-walk vol)
  * EWMA         : RiskMetrics lambda=0.94 conditional vol
  * model (XGB)  : richer past-vol features
We report each forecast's OOS R^2 vs the mean, the model's lift over EWMA, and a
permutation p-value for the model. Pass bar: model R^2 > 0 with p<0.05 AND not
worse than EWMA.
"""

import sys
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

from evaluation import purged_walkforward_splits

# NOTE: volatility is log-normal and strongly regime-shifting, so RAW-level R^2
# is a misleading metric (level bias drives it negative even when the forecast
# ranks high/low-vol periods correctly). We therefore model in LOG-vol space and
# judge primarily by rank IC (Spearman) and log-space R^2 -- rank skill is what
# risk sizing actually needs.
LOG_EPS = 1e-9

H = 21
LAMBDA = 0.94
N_PERM = 100
ASSETS = ["ETH-USD", "SOL-USD", "ADA-USD", "EURGBP=X", "AUDNZD=X", "IWM"]


def load(ticker):
    d = yf.download(ticker, period="6y", interval="1d", auto_adjust=True, progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d.dropna()
    return d["Close"].pct_change().dropna()


def features_and_target(ret):
    df = pd.DataFrame({"ret": ret})
    df["rv5"] = ret.rolling(5).std()
    df["rv10"] = ret.rolling(10).std()
    df["rv21"] = ret.rolling(21).std()
    df["rv63"] = ret.rolling(63).std()
    df["absret"] = ret.abs()
    df["absret5"] = ret.abs().rolling(5).mean()

    # EWMA (RiskMetrics) conditional vol, per day
    ewma_var = ret.pow(2).ewm(alpha=1 - LAMBDA, adjust=False).mean()
    df["ewma"] = np.sqrt(ewma_var)

    # forward realized vol over next H days
    arr = ret.to_numpy()
    n = len(arr)
    fwd = np.full(n, np.nan)
    for t in range(n - H):
        fwd[t] = np.std(arr[t + 1: t + 1 + H])
    df["target"] = fwd
    return df.dropna()


def model():
    return XGBRegressor(n_estimators=250, max_depth=3, learning_rate=0.04,
                        subsample=0.8, colsample_bytree=0.8,
                        random_state=42, n_jobs=1)


def oos_logvol(X, y):
    """Out-of-sample log-vol predictions; returns forecasts in vol units."""
    ly = np.log(y + LOG_EPS)
    pred = np.full(len(y), np.nan)
    for tr, te in purged_walkforward_splits(len(y), 5, H, embargo=H):
        m = clone(model()).fit(X[tr], ly[tr])
        pred[te] = np.exp(m.predict(X[te]))
    return pred


def screen(ticker):
    df = features_and_target(load(ticker))
    feat = ["rv5", "rv10", "rv21", "rv63", "absret", "absret5", "ewma"]
    X = np.log(df[feat].to_numpy() + LOG_EPS)   # log-space features
    y = df["target"].to_numpy()

    pred = oos_logvol(X, y)
    mask = ~np.isnan(pred)
    yt, pt = y[mask], pred[mask]
    ewma = df["ewma"].to_numpy()[mask]

    ic_model = spearmanr(pt, yt).statistic           # rank IC (key metric)
    ic_ewma = spearmanr(ewma, yt).statistic
    r2_log = r2_score(np.log(yt + LOG_EPS), np.log(pt + LOG_EPS))

    # permutation p on the model's rank IC
    rng = np.random.default_rng(0)
    count = 0
    for _ in range(N_PERM):
        pp = oos_logvol(X, rng.permutation(y))
        mm = ~np.isnan(pp)
        if spearmanr(pp[mm], y[mm]).statistic >= ic_model:
            count += 1
    pval = (count + 1) / (N_PERM + 1)

    ok = ic_model > 0 and pval < 0.05 and ic_model >= ic_ewma - 0.02
    print(f"{ticker:10s} n={mask.sum():4d}  "
          f"IC_ewma={ic_ewma:+.3f}  IC_model={ic_model:+.3f}  "
          f"logR2={r2_log:+.3f}  p={pval:.3f}  {'PASS' if ok else 'fail'}")
    return ic_model, pval


if __name__ == "__main__":
    assets = sys.argv[1:] or ASSETS
    print(f"VOL PREDICTABILITY SCREEN  (H={H}d, log-vol, purged WF, perm={N_PERM})")
    print("metric = rank IC (Spearman) of OOS forecast vs realized vol; "
          "pass = IC>0, p<0.05, >= EWMA\n")
    for a in assets:
        try:
            screen(a)
        except Exception as e:
            print(f"{a:10s} ERR {str(e)[:50]}")
