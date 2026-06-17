import pandas as pd

df = pd.read_csv("results/timesfm_backtest_results.csv")

closed = df[df["outcome"].isin(["WIN", "LOSS"])]

closed["abs_forecast"] = (
    closed["forecast_change_pct"]
    .abs()
)

strong = closed[
    closed["abs_forecast"] >= 0.75
]

weak = closed[
    closed["abs_forecast"] < 0.75
]

print("\n=== STRONG FORECASTS ===")
print(len(strong))
print(strong["outcome"].value_counts())

print("\n=== WEAK FORECASTS ===")
print(len(weak))
print(weak["outcome"].value_counts())
