import pandas as pd

from data_loader import load_gold_data
from indicators import add_indicators
from forecast_engine import forecast_direction
from strategy import generate_trade_idea
from backtest import evaluate_trade


def main():
    print("Loading gold data...")
    df = load_gold_data(period="6mo", interval="1h")

    print("Adding indicators...")
    df = add_indicators(df)

    results = []

    lookahead_candles = 24
    step = 10
    min_history = 300

    print("Running real TimesFM walk-forward backtest...")

    for i in range(min_history, len(df) - lookahead_candles, step):
        print(f"Testing candle {i} of {len(df)}...")

        historical_slice = df.iloc[: i + 1]

        forecast = forecast_direction(
            historical_slice,
            horizon=24,
        )

        trade = generate_trade_idea(
            historical_slice,
            forecast,
        )

        outcome = evaluate_trade(
            df.iloc[i + 1 : i + 1 + lookahead_candles],
            trade,
        )

        if outcome is None:
            continue

        results.append({
            "timestamp": df.index[i],
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
            "outcome": outcome,
        })

    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("\nNo TimesFM trades found.")
        return

    results_df.to_csv(
        "results/timesfm_backtest_results.csv",
        index=False,
    )

    wins = len(results_df[results_df["outcome"] == "WIN"])
    losses = len(results_df[results_df["outcome"] == "LOSS"])
    open_trades = len(results_df[results_df["outcome"] == "OPEN"])
    total_closed = wins + losses

    win_rate = (wins / total_closed * 100) if total_closed else 0

    print("\n==============================")
    print(" TIMESFM WALK-FORWARD RESULTS")
    print("==============================")
    print(f"Total trades      : {len(results_df)}")
    print(f"Wins              : {wins}")
    print(f"Losses            : {losses}")
    print(f"Open/Unresolved   : {open_trades}")
    print(f"Win rate          : {win_rate:.2f}%")
    print("\nSaved to: results/timesfm_backtest_results.csv")


if __name__ == "__main__":
    main()
