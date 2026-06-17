import pandas as pd

from data_loader import load_gold_data
from indicators import add_indicators
from strategy import generate_trade_idea


def fake_forecast_from_momentum(row):
    """
    Temporary forecast substitute for backtesting.
    Later we replace this with TimesFM walk-forward forecasts.
    """
    if row["macd"] > row["macd_signal"]:
        direction = "bullish"
        forecast_price = row["Close"] * 1.003
    elif row["macd"] < row["macd_signal"]:
        direction = "bearish"
        forecast_price = row["Close"] * 0.997
    else:
        direction = "neutral"
        forecast_price = row["Close"]

    return {
        "current_price": float(row["Close"]),
        "forecast_price": float(forecast_price),
        "forecast_change_pct": ((forecast_price - row["Close"]) / row["Close"]) * 100,
        "direction": direction,
    }


def evaluate_trade(future_df, trade):
    """
    Checks whether take-profit or stop-loss was hit first.
    """
    if trade["idea"] == "NO TRADE":
        return None

    stop_loss = trade["stop_loss"]
    take_profit = trade["take_profit"]

    for _, candle in future_df.iterrows():
        high = candle["High"]
        low = candle["Low"]

        if trade["idea"] == "POSSIBLE LONG":
            if low <= stop_loss:
                return "LOSS"
            if high >= take_profit:
                return "WIN"

        if trade["idea"] == "POSSIBLE SHORT":
            if high >= stop_loss:
                return "LOSS"
            if low <= take_profit:
                return "WIN"

    return "OPEN"


def main():
    print("Loading data...")
    df = load_gold_data(period="6mo", interval="1h")

    print("Adding indicators...")
    df = add_indicators(df)

    results = []

    lookahead_candles = 24  # next 24 hours on 1h chart

    print("Running backtest...")

    for i in range(100, len(df) - lookahead_candles):
        historical_slice = df.iloc[: i + 1]
        current_row = historical_slice.iloc[-1]

        forecast = fake_forecast_from_momentum(current_row)
        trade = generate_trade_idea(historical_slice, forecast)

        outcome = evaluate_trade(
            df.iloc[i + 1 : i + 1 + lookahead_candles],
            trade,
        )

        if outcome is None:
            continue

        results.append({
            "timestamp": df.index[i],
            "idea": trade["idea"],
            "entry": trade["entry"],
            "stop_loss": trade["stop_loss"],
            "take_profit": trade["take_profit"],
            "confidence": trade["confidence"],
            "rsi": trade["rsi"],
            "macd": trade["macd"],
            "macd_signal": trade["macd_signal"],
            "atr": trade["atr"],
            "outcome": outcome,
        })

    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("No trades found.")
        return

    wins = len(results_df[results_df["outcome"] == "WIN"])
    losses = len(results_df[results_df["outcome"] == "LOSS"])
    open_trades = len(results_df[results_df["outcome"] == "OPEN"])
    total_closed = wins + losses

    win_rate = (wins / total_closed * 100) if total_closed > 0 else 0

    print("\n==============================")
    print(" BACKTEST RESULTS")
    print("==============================")
    print(f"Total trades      : {len(results_df)}")
    print(f"Wins              : {wins}")
    print(f"Losses            : {losses}")
    print(f"Open/Unresolved   : {open_trades}")
    print(f"Win rate          : {win_rate:.2f}%")

    results_df.to_csv("results/backtest_results.csv", index=False)
    print("\nSaved to: results/backtest_results.csv")


if __name__ == "__main__":
    main()
