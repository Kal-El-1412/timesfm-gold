import pandas as pd

df = pd.read_csv(
    "results/macro_dataset.csv"
)

target = (
    df["future_gold_return_24h"] * 100
)

q33 = target.quantile(0.33)
q66 = target.quantile(0.66)

print("\n=== PERCENTILES ===")
print("33%:", q33)
print("66%:", q66)

down = (target <= q33).sum()
mid = ((target > q33) & (target < q66)).sum()
up = (target >= q66).sum()

print("\nDOWN:", down)
print("MID :", mid)
print("UP  :", up)
