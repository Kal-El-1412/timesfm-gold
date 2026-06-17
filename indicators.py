from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.trend import ADXIndicator
from ta.volatility import AverageTrueRange


def add_indicators(df):
    df = df.copy()

    df["rsi"] = RSIIndicator(
        close=df["Close"],
        window=14,
    ).rsi()

    macd = MACD(close=df["Close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    atr = AverageTrueRange(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=14,
    )
    df["atr"] = atr.average_true_range()

    lookback = 50
    df["support"] = df["Low"].rolling(lookback).min()
    df["resistance"] = df["High"].rolling(lookback).max()

    adx = ADXIndicator(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=14
    )

    df["adx"] = adx.adx()

    return df.dropna()
