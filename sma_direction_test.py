import pandas as pd
import yfinance as yf

gold = yf.download(
    "GC=F",
    period="6mo",
    interval="1h",
    auto_adjust=True,
)

if isinstance(gold.columns, pd.MultiIndex):
    gold.columns = gold.columns.get_level_values(0)

gold["sma50"] = gold["Close"].rolling(50).mean()

gold = gold.dropna()

correct = 0
wrong = 0

lookahead = 24

for i in range(50, len(gold) - lookahead):

    current = gold.iloc[i]["Close"]
    sma = gold.iloc[i]["sma50"]

    future = gold.iloc[i + lookahead]["Close"]

    actual_change = future - current

    if current > sma and actual_change > 0:
        correct += 1

    elif current < sma and actual_change < 0:
        correct += 1

    else:
        wrong += 1

accuracy = (
    correct / (correct + wrong)
    * 100
)

print("\n=== SMA DIRECTION TEST ===")
print(f"Correct: {correct}")
print(f"Wrong: {wrong}")
print(f"Accuracy: {accuracy:.2f}%")
