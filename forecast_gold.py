import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import torch
import timesfm

# Download gold futures daily data
df = yf.download("GC=F", period="2y", interval="1d",auto_adjust=True)
df = df.dropna()

series = df["Close"].to_numpy(dtype=np.float32).reshape(-1)
series = series[-1024:]

torch.set_float32_matmul_precision("high")

model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
)

model.compile(
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

horizon = 30

point_forecast, quantile_forecast = model.forecast(
    horizon=horizon,
    inputs=[series]
)

forecast = np.array(point_forecast[0]).reshape(-1)

timesfm_forecast_price = forecast[-1]

future_dates = pd.date_range(
    start=df.index[-1] + pd.Timedelta(days=1),
    periods=horizon,
    freq="D"
)

plt.figure(figsize=(12, 6))
plt.plot(df.index[-120:], series[-120:], label="Actual gold close")
plt.plot(future_dates, forecast, label="TimesFM forecast")
plt.title("Gold forecast using TimesFM")
plt.legend()
plt.grid(True)
plt.show()
