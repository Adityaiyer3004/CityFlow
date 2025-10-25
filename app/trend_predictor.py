# app/trend_predictor.py

import pandas as pd
import sqlite3
from prophet import Prophet
from datetime import datetime

def forecast_congestion(days=1):
    """Predicts disruption count for the next N days using Prophet."""
    conn = sqlite3.connect("cityflow.db")
    df = pd.read_sql("SELECT date, disruptions FROM daily_trends ORDER BY date", conn)
    conn.close()

    # ✅ Handle cases with insufficient data
    if len(df) < 3:
        return None, "📉 Not enough historical data to forecast yet."

    # ✅ Prophet expects columns ds (date) and y (value)
    df.rename(columns={"date": "ds", "disruptions": "y"}, inplace=True)
    df["ds"] = pd.to_datetime(df["ds"])

    # ✅ Fit simple Prophet model
    model = Prophet(daily_seasonality=True)
    model.fit(df)

    # ✅ Generate forecast for the next N days
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    # ✅ Extract next-day prediction
    next_day = forecast.iloc[-1]
    prediction = int(next_day["yhat"])

    # ✅ Compare with most recent actual value
    last_actual = df["y"].iloc[-1]
    trend = "increase" if prediction > last_actual else "decrease"
    pct_change = ((prediction - last_actual) / last_actual) * 100

    # ✅ Clean narrative summary
    summary = (
        f"Based on recent trends, disruptions are expected to {trend} "
        f"by {abs(pct_change):.1f}% tomorrow — approximately {prediction} total incidents."
    )

    return forecast, summary
