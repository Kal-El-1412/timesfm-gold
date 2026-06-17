import pandas as pd
import yfinance as yf

dxy = yf.download(
    "DX-Y.NYB",
    period="6mo",
    interval="1h",
    auto_adjust=True,
)

gold = yf.download(
    "GC=F",
    period="6mo",
    interval="1h",
    auto_adjust=True,
)

# Flatten columns if needed
if isinstance(dxy.columns, pd.MultiIndex):
    dxy.columns = dxy.columns.get_level_values(0)

if isinstance(gold.columns, pd.MultiIndex):
    gold.columns = gold.columns.get_level_values(0)

dxy = dxy[["Close"]]
gold = gold[["Close"]]

merged = pd.merge(
    dxy,
    gold,
    left_index=True,
    right_index=True,
    how="inner",
    suffixes=("_dxy", "_gold")
)

merged["dxy_change"] = (
    merged["Close_dxy"].pct_change()
)

merged["gold_change"] = (
    merged["Close_gold"].pct_change()
)

merged = merged.dropna()

same_direction = (
    (
        (merged["dxy_change"] > 0)
        &
        (merged["gold_change"] > 0)
    )
    |
    (
        (merged["dxy_change"] < 0)
        &
        (merged["gold_change"] < 0)
    )
)

opposite_direction = (
    (
        (merged["dxy_change"] > 0)
        &
        (merged["gold_change"] < 0)
    )
    |
    (
        (merged["dxy_change"] < 0)
        &
        (merged["gold_change"] > 0)
    )
)

same_pct = same_direction.mean() * 100
opp_pct = opposite_direction.mean() * 100

print("\n=== DXY vs GOLD ===")
print(f"Rows: {len(merged)}")
print(f"Same Direction: {same_pct:.2f}%")
print(f"Opposite Direction: {opp_pct:.2f}%")

corr = merged["dxy_change"].corr(
    merged["gold_change"]
)

print(f"Correlation: {corr:.4f}")
