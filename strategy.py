
def generate_trade_idea(df, forecast_result):
    latest = df.iloc[-1]
    adx = float(latest["adx"])

    price = float(latest["Close"])
    atr = float(latest["atr"])
    rsi = float(latest["rsi"])
    macd = float(latest["macd"])
    macd_signal = float(latest["macd_signal"])
    support = float(latest["support"])
    resistance = float(latest["resistance"])

    direction = forecast_result["direction"]

    rsi_bullish = 45 <= rsi <= 70
    rsi_bearish = 30 <= rsi <= 55

    macd_bullish = macd > macd_signal
    macd_bearish = macd < macd_signal

    near_support = price <= support + atr
    near_resistance = price >= resistance - atr

    idea = "NO TRADE"
    entry = price
    stop_loss = None
    take_profit = None
    confidence = 0

    if adx < 25:
        return {
            "idea": "NO TRADE",
            "confidence": 0,
            "entry": price,
            "stop_loss": None,
            "take_profit": None,
            "rsi": rsi,
            "macd": macd,
            "macd_signal": macd_signal,
            "atr": atr,
            "support": support,
            "resistance": resistance,
        }

    if direction == "bullish":
        confidence += 30
        if rsi_bullish:
            confidence += 20
        if macd_bullish:
            confidence += 20
        if not near_resistance:
            confidence += 20

        if confidence >= 90:
            idea = "POSSIBLE LONG"
            stop_loss = entry - (1.5 * atr)
            take_profit = entry + (2.5 * atr)

    elif direction == "bearish":
        confidence += 30
        if rsi_bearish:
            confidence += 20
        if macd_bearish:
            confidence += 20
        if not near_support:
            confidence += 20

        if confidence >= 90:
            idea = "POSSIBLE SHORT"
            stop_loss = entry + (1.5 * atr)
            take_profit = entry - (2.5 * atr)

    return {
        "idea": idea,
        "confidence": confidence,
        "entry": entry,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "rsi": rsi,
        "macd": macd,
        "macd_signal": macd_signal,
        "atr": atr,
        "support": support,
        "resistance": resistance,
    }
