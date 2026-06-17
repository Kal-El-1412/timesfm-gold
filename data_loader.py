import pandas as pd
import yfinance as yf


def load_gold_data(period="6mo", interval="1h"):
    df = yf.download(
        "GC=F",
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=True,
    )

    df = df.dropna()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Volume"]
    df = df[[col for col in required if col in df.columns]]

    return df
