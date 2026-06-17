import os
import pandas as pd
import yfinance as yf


TICKERS = {
    "gold": "GC=F",
    "silver": "SI=F",
    "dxy": "DX-Y.NYB",
    "vix": "^VIX",
    "oil": "CL=F",
    "sp500": "SPY",
    "nasdaq": "QQQ",
    "us10y": "^TNX",
    "us2y": "^IRX",
}


def download_close(name, ticker):
    df = yf.download(
        ticker,
        period="6mo",
        interval="1h",
        auto_adjust=True,
        progress=True,
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Close"]].rename(columns={"Close": f"{name}_close"})
    df = df.dropna()
    df = df.sort_index()

    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC")
    else:
        df.index = df.index.tz_localize("UTC")

    df = df.reset_index().rename(columns={"Datetime": "timestamp", "Date": "timestamp"})

    return df


def main():
    os.makedirs("results", exist_ok=True)

    print("Downloading gold base data...")
    base = download_close("gold", TICKERS["gold"])

    merged = base

    for name, ticker in TICKERS.items():
        if name == "gold":
            continue

        print(f"Downloading {name}: {ticker}")
        other = download_close(name, ticker)

        merged = pd.merge_asof(
            merged.sort_values("timestamp"),
            other.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
            tolerance=pd.Timedelta("3h"),
        )

    merged = merged.set_index("timestamp")
    merged = merged.dropna()

    for name in TICKERS.keys():
        close_col = f"{name}_close"

        merged[f"{name}_return_1h"] = merged[close_col].pct_change(1)
        merged[f"{name}_return_4h"] = merged[close_col].pct_change(4)
        merged[f"{name}_return_24h"] = merged[close_col].pct_change(24)

    merged["gold_sma_20"] = merged["gold_close"].rolling(20).mean()
    merged["gold_sma_50"] = merged["gold_close"].rolling(50).mean()

    merged["gold_above_sma20"] = (
        merged["gold_close"] > merged["gold_sma_20"]
    ).astype(int)

    merged["gold_above_sma50"] = (
        merged["gold_close"] > merged["gold_sma_50"]
    ).astype(int)

    merged["future_gold_return_24h"] = (
        merged["gold_close"].shift(-24) / merged["gold_close"] - 1
    )

    merged["target_big_move"] = (
        merged["future_gold_return_24h"].abs() > 0.005
    ).astype(int)

    merged = merged.dropna()

    merged.to_csv("results/macro_dataset.csv")

    print("\nSaved to results/macro_dataset.csv")
    print("Rows:", len(merged))
    print("Columns:", len(merged.columns))
    print(merged.tail())


if __name__ == "__main__":
    main()
