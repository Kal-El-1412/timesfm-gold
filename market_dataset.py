import yfinance as yf
import pandas as pd

gold = yf.download(
    "GC=F",
    period="6mo",
    interval="1h",
    auto_adjust=True,
)

dxy = yf.download(
    "DX-Y.NYB",
    period="6mo",
    interval="1h",
    auto_adjust=True,
)

vix = yf.download(
    "^VIX",
    period="6mo",
    interval="1h",
    auto_adjust=True,
)

for df in [gold, dxy, vix]:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

gold = gold[["Close"]].rename(
    columns={"Close": "gold_close"}
)

dxy = dxy[["Close"]].rename(
    columns={"Close": "dxy_close"}
)

vix = vix[["Close"]].rename(
    columns={"Close": "vix_close"}
)

df = gold.join(
    dxy,
    how="inner"
).join(
    vix,
    how="inner"
)

df["gold_return_1h"] = (
    df["gold_close"].pct_change()
)

df["dxy_return_1h"] = (
    df["dxy_close"].pct_change()
)

df["vix_return_1h"] = (
    df["vix_close"].pct_change()
)

df["gold_return_24h"] = (
    df["gold_close"]
    .pct_change(24)
)

df["dxy_return_24h"] = (
    df["dxy_close"]
    .pct_change(24)
)

df["vix_return_24h"] = (
    df["vix_close"]
    .pct_change(24)
)

df["future_gold_return_24h"] = (
    df["gold_close"].shift(-24)
    / df["gold_close"]
    - 1
)

df = df.dropna()

df.to_csv(
    "results/market_dataset.csv"
)

print(df.head())
print()
print(df.columns)
print()
print("Rows:", len(df))
