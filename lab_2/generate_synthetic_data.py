"""
Generate synthetic time series data for the Forecasting Lab (ARIMA).

The series simulates a fictional monthly economic indicator ("Widget
Sales Index") with trend, seasonality, and autocorrelated noise --
structure realistic enough to require differencing and support a
genuine ACF/PACF-based model selection exercise, while being fully
synthetic (no data licensing/attribution concerns).
"""
import numpy as np
import pandas as pd

np.random.seed(42)

# Monthly data, 8 years
n_periods = 96
dates = pd.date_range(start="2018-01-01", periods=n_periods, freq="MS")

# Trend component (gradual upward drift)
trend = np.linspace(100, 180, n_periods)

# Seasonal component (annual cycle, peak in Nov/Dec - holiday shopping)
month_effect = {
    1: -8, 2: -10, 3: -5, 4: 0, 5: 3, 6: 5,
    7: 2, 8: -2, 9: 4, 10: 10, 11: 18, 12: 22
}
seasonal = np.array([month_effect[d.month] for d in dates])

# Autocorrelated noise (AR(1) process to make it feel like real data)
noise = np.zeros(n_periods)
phi = 0.4
for t in range(1, n_periods):
    noise[t] = phi * noise[t-1] + np.random.normal(0, 4)

# Combine
values = trend + seasonal + noise
values = np.round(values, 2)

df = pd.DataFrame({
    "date": dates,
    "widget_sales_index": values
})

df.to_csv("data/widget_sales.csv", index=False)
print(f"Generated {n_periods} monthly observations.")
print(df.head(10))
print("...")
print(df.tail(5))