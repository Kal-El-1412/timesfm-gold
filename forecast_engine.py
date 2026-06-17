import numpy as np
import torch
import timesfm


_model = None


def get_model():
    global _model

    if _model is None:
        torch.set_float32_matmul_precision("high")

        _model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch"
        )

        _model.compile(
            timesfm.ForecastConfig(
                max_context=1024,
                max_horizon=60,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )

    return _model


def forecast_direction(df, horizon=12):
    model = get_model()

    series = df["Close"].to_numpy(dtype=np.float32).reshape(-1)
    series = series[-1024:]

    point_forecast, quantile_forecast = model.forecast(
        inputs=[series],
        horizon=horizon,
    )

    forecast = np.array(point_forecast[0]).reshape(-1)

    current_price = float(series[-1])
    forecast_price = float(forecast[-1])
    change_pct = ((forecast_price - current_price) / current_price) * 100

    if change_pct > 0.25:
        direction = "bullish"
    elif change_pct < -0.25:
        direction = "bearish"
    else:
        direction = "neutral"

    # --- use the quantile head, not just the point endpoint -------------
    # TimesFM returns quantiles per horizon step; the spread of the terminal
    # step is a model-implied forecast uncertainty / volatility estimate, and
    # the sign-agreement of the quantile band is a confidence proxy. The
    # original code discarded this entirely.
    forecast_low = forecast_high = forecast_price
    band_pct = 0.0
    confident = False
    try:
        q = np.array(quantile_forecast[0])  # shape: [horizon, n_quantiles]
        q_terminal = q[-1]
        forecast_low = float(q_terminal.min())
        forecast_high = float(q_terminal.max())
        band_pct = (forecast_high - forecast_low) / current_price * 100
        # confident only when the whole band sits on one side of spot
        confident = (forecast_low > current_price) or (forecast_high < current_price)
    except Exception:
        pass

    return {
        "current_price": current_price,
        "forecast_price": forecast_price,
        "forecast_change_pct": change_pct,
        "direction": direction,
        "forecast_series": forecast,
        "forecast_low": forecast_low,
        "forecast_high": forecast_high,
        "forecast_band_pct": band_pct,
        "quantile_confident": confident,
    }
