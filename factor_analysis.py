import pandas as pd

df = pd.read_csv(
    "results/market_dataset.csv"
)

target = "future_gold_return_24h"

print("\n=== FACTOR CORRELATION ===")

for col in [
    "gold_return_1h",
    "gold_return_24h",
    "dxy_return_1h",
    "dxy_return_24h",
    "vix_return_1h",
    "vix_return_24h",
]:

    corr = df[col].corr(
        df[target]
    )

    print(
        f"{col}: {corr:.4f}"
    )
