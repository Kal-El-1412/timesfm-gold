import pandas as pd

df = pd.read_csv("results/timesfm_backtest_results.csv")

risk_per_trade = 1.0

pnl = 0

wins = 0
losses = 0

for _, row in df.iterrows():

    if row["outcome"] == "WIN":

        reward = abs(
            row["take_profit"] - row["entry"]
        )

        risk = abs(
            row["entry"] - row["stop_loss"]
        )

        rr = reward / risk

        pnl += rr

        wins += 1

    elif row["outcome"] == "LOSS":

        pnl -= risk_per_trade

        losses += 1

print("\n=== PNL ANALYSIS ===")
print(f"Wins      : {wins}")
print(f"Losses    : {losses}")
print(f"Net R     : {pnl:.2f}")
print(f"Avg R     : {pnl / (wins + losses):.3f}")
