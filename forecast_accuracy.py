import pandas as pd

df = pd.read_csv("results/timesfm_backtest_results.csv")

closed = df[df["outcome"].isin(["WIN", "LOSS"])]

correct_direction = 0
total = 0

for _, row in closed.iterrows():

    forecast_change = row["forecast_change_pct"]

    if row["trade_idea"] == "POSSIBLE LONG":

        total += 1

        if row["outcome"] == "WIN":
            correct_direction += 1

    elif row["trade_idea"] == "POSSIBLE SHORT":

        total += 1

        if row["outcome"] == "WIN":
            correct_direction += 1

accuracy = (
    correct_direction / total * 100
    if total else 0
)

print("\n=== FORECAST ACCURACY ===")
print(f"Signals: {total}")
print(f"Correct Direction: {correct_direction}")
print(f"Accuracy: {accuracy:.2f}%")
