import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange

# 1. Load gold data
df = yf.download("GC=F", period="6mo", interval="1h", auto_adjust=True)
df = df.dropna()

# Fix yfinance multi-index issue if present
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# 2. Indicators
df["rsi"] = RSIIndicator(close=df["Close"], window=14).rsi()

macd = MACD(close=df["Close"])
df["macd"] = macd.macd()
df["macd_signal"] = macd.macd_signal()
df["macd_hist"] = macd.macd_diff()

atr = AverageTrueRange(
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    window=14
)
df["atr"] = atr.average_true_range()

# 3. Basic support / resistance
lookback = 50
df["support"] = df["Low"].rolling(lookback).min()
df["resistance"] = df["High"].rolling(lookback).max()

latest = df.iloc[-1]
price = latest["Close"]

# 4. Replace this with your TimesFM forecast result
# Example: forecast is slightly above current price
timesfm_forecast_price = price * 1.003

forecast_change_pct = (timesfm_forecast_price - price) / price * 100

# 5. Rules
timesfm_direction = "bullish" if forecast_change_pct > 0.15 else "bearish" if forecast_change_pct < -0.15 else "neutral"

rsi_bullish = 45 <= latest["rsi"] <= 70
rsi_bearish = 30 <= latest["rsi"] <= 55

macd_bullish = latest["macd"] > latest["macd_signal"]
macd_bearish = latest["macd"] < latest["macd_signal"]

near_support = price <= latest["support"] + latest["atr"]
near_resistance = price >= latest["resistance"] - latest["atr"]

# 6. Trade idea logic
idea = "NO TRADE"
entry = price
stop_loss = None
take_profit = None

if timesfm_direction == "bullish" and rsi_bullish and macd_bullish and not near_resistance:
    idea = "POSSIBLE LONG"
    stop_loss = entry - (1.5 * latest["atr"])
    take_profit = entry + (2.5 * latest["atr"])

elif timesfm_direction == "bearish" and rsi_bearish and macd_bearish and not near_support:
    idea = "POSSIBLE SHORT"
    stop_loss = entry + (1.5 * latest["atr"])
    take_profit = entry - (2.5 * latest["atr"])

print("\n--- GOLD TRADE IDEA ---")
print(f"Price: {price:.2f}")
print(f"TimesFM forecast price: {timesfm_forecast_price:.2f}")
print(f"Forecast change: {forecast_change_pct:.2f}%")
print(f"TimesFM direction: {timesfm_direction}")
print(f"RSI: {latest['rsi']:.2f}")
print(f"MACD: {latest['macd']:.4f}")
print(f"MACD Signal: {latest['macd_signal']:.4f}")
print(f"ATR: {latest['atr']:.2f}")
print(f"Support: {latest['support']:.2f}")
print(f"Resistance: {latest['resistance']:.2f}")
print(f"Idea: {idea}")

if stop_loss and take_profit:
    print(f"Entry: {entry:.2f}")
    print(f"Stop loss: {stop_loss:.2f}")
    print(f"Take profit: {take_profit:.2f}")
