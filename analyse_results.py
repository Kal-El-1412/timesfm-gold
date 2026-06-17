import pandas as pd

df = pd.read_csv("results/timesfm_backtest_results.csv")

closed = df[df["outcome"].isin(["WIN", "LOSS"])]

print("\n=== OVERALL ===")
print(closed["outcome"].value_counts())

print("\n=== LONGS ===")
longs = closed[closed["trade_idea"] == "POSSIBLE LONG"]
print(longs["outcome"].value_counts())

print("\n=== SHORTS ===")
shorts = closed[closed["trade_idea"] == "POSSIBLE SHORT"]
print(shorts["outcome"].value_counts())

print("\n=== CONFIDENCE GROUPS ===")
print(
    closed.groupby("confidence")["outcome"]
    .value_counts()
)

print("\n=== FORECAST STRENGTH ===")
print(
    closed["forecast_change_pct"]
    .describe()
)
