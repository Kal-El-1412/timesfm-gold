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

merged["dxy_change"] = merged["Close_dxy"].pct_change()

lookahead = 24

correct = 0
wrong = 0

for i in range(1, len(merged) - lookahead):

    dxy_move = merged.iloc[i]["dxy_change"]

    current_gold = merged.iloc[i]["Close_gold"]
    future_gold = merged.iloc[i + lookahead]["Close_gold"]

    gold_change = (
        (future_gold - current_gold)
        / current_gold
    )

    # DXY up -> Gold down
    if dxy_move > 0 and gold_change < 0:
        correct += 1

    elif dxy_move < 0 and gold_change > 0:
        correct += 1

    else:
        wrong += 1

accuracy = (
    correct / (correct + wrong)
    * 100
)

print("\n=== DXY SIGNAL TEST ===")
print(f"Correct: {correct}")
print(f"Wrong: {wrong}")
print(f"Accuracy: {accuracy:.2f}%")
