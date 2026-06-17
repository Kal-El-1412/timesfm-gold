import os
from datetime import datetime

import pandas as pd

from data_loader import load_gold_data
from indicators import add_indicators
from forecast_engine import forecast_direction
from strategy import generate_trade_idea


def save_trade_idea(forecast, trade):
    os.makedirs("results", exist_ok=True)

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "current_price": forecast["current_price"],
        "forecast_price": forecast["forecast_price"],
        "forecast_change_pct": forecast["forecast_change_pct"],
        "timesfm_direction": forecast["direction"],
        "trade_idea": trade["idea"],
        "confidence": trade["confidence"],
        "entry": trade["entry"],
        "stop_loss": trade["stop_loss"],
        "take_profit": trade["take_profit"],
        "rsi": trade["rsi"],
        "macd": trade["macd"],
        "macd_signal": trade["macd_signal"],
        "atr": trade["atr"],
        "support": trade["support"],
        "resistance": trade["resistance"],
    }

    output_file = "results/trade_ideas.csv"

    if os.path.exists(output_file):
        existing = pd.read_csv(output_file)
        existing = pd.concat(
            [existing, pd.DataFrame([row])],
            ignore_index=True
        )
        existing.to_csv(output_file, index=False)
    else:
        pd.DataFrame([row]).to_csv(output_file, index=False)

    return output_file


def display_results(forecast, trade):
    print("\n==============================")
    print(" GOLD AI TRADE IDEA")
    print("==============================")

    print(f"\nCurrent Price      : {forecast['current_price']:.2f}")
    print(f"Forecast Price     : {forecast['forecast_price']:.2f}")
    print(f"Forecast Change %  : {forecast['forecast_change_pct']:.2f}%")
    print(f"TimesFM Direction  : {forecast['direction']}")

    print("\n----- Indicators -----")

    print(f"RSI                : {trade['rsi']:.2f}")
    print(f"MACD               : {trade['macd']:.4f}")
    print(f"MACD Signal        : {trade['macd_signal']:.4f}")
    print(f"ATR                : {trade['atr']:.2f}")

    print(f"Support            : {trade['support']:.2f}")
    print(f"Resistance         : {trade['resistance']:.2f}")

    print("\n----- Trade Decision -----")

    print(f"Idea               : {trade['idea']}")
    print(f"Confidence         : {trade['confidence']}%")

    if trade["stop_loss"] is not None:
        print(f"Entry              : {trade['entry']:.2f}")
        print(f"Stop Loss          : {trade['stop_loss']:.2f}")
        print(f"Take Profit        : {trade['take_profit']:.2f}")


def main():
    print("Loading gold data...")
    df = load_gold_data(
        period="6mo",
        interval="1h"
    )

    print("Adding indicators...")
    df = add_indicators(df)

    print("Running TimesFM forecast...")
    forecast = forecast_direction(
        df,
        horizon=12
    )

    print("Generating trade idea...")
    trade = generate_trade_idea(
        df,
        forecast
    )

    display_results(
        forecast,
        trade
    )

    csv_file = save_trade_idea(
        forecast,
        trade
    )

    print(f"\nTrade idea saved to: {csv_file}")


if __name__ == "__main__":
    main()
