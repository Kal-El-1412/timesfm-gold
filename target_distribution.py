import pandas as pd

df = pd.read_csv(
    "results/macro_dataset.csv"
)

target = (
    df["future_gold_return_24h"]
    * 100
)

print("\n=== TARGET DISTRIBUTION ===")

print(target.describe())

print("\nMove > 0.5%:")
print((target.abs() > 0.5).mean())

print("\nMove > 1.0%:")
print((target.abs() > 1.0).mean())

print("\nMove > 1.5%:")
print((target.abs() > 1.5).mean())

print("\nMove > 2.0%:")
print((target.abs() > 2.0).mean())
