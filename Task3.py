# Load necessary libraries

import os
import pandas as pd
import requests
import matplotlib.pyplot as plt
from prophet import Prophet
from dotenv import load_dotenv

# ─── Configuration ───────────────────────────────────────────────────────
load_dotenv()
SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK_URL")

CSV_FILE = "ai_news_sentiment.csv"
POSITIVE_THRESHOLD = 0.15
NEGATIVE_THRESHOLD = -0.15

# ─── Load CSV ────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_FILE)
df.columns = df.columns.str.strip().str.lower()

if "score" not in df.columns:
    raise ValueError("CSV must contain a 'score' column")

# ─── Function to Send Slack Alert ────────────────────────────────────────
def send_slack_alert(message: str):
    if not SLACK_WEBHOOK:
        print("⚠ Slack Webhook URL not set.")
        return
    payload = {"text": message}
    try:
        r = requests.post(SLACK_WEBHOOK, json=payload)
        if r.status_code == 200:
            print(" Slack alert sent:", message)
        else:
            print(f"⚠ Failed to send Slack alert: {r.status_code}, {r.text}")
    except Exception as e:
        print("⚠ Slack alert error:", e)

# ─── Send Alerts Based on Sentiment ──────────────────────────────────────
for i, row in df.iterrows():
    score = row["score"]
    title = row.get("title", "No Title")

    if score > POSITIVE_THRESHOLD:
        send_slack_alert(f" Positive Alert: Sentiment score {score:.2f} for '{title}'")
    elif score < NEGATIVE_THRESHOLD:
        send_slack_alert(f" Negative Alert: Sentiment score {score:.2f} for '{title}'")

print(" All alerts processed.")

# ─── Forecasting with Prophet ────────────────────────────────────────────
if "published_at" not in df.columns:
    raise ValueError("CSV must contain a 'published_at' column with dates")

df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce").dt.tz_localize(None)
df = df.dropna(subset=["published_at"])

# Aggregate daily sentiment
daily_df = df.resample("D", on="published_at")["score"].mean().reset_index()
daily_df = daily_df.rename(columns={"published_at": "ds", "score": "y"})

# Optional: smooth slightly
daily_df["y"] = daily_df["y"].rolling(window=3, min_periods=1).mean()

# Fit Prophet model
model = Prophet(daily_seasonality=True, weekly_seasonality=True)
model.fit(daily_df)

# Forecast 7 days
future = model.make_future_dataframe(periods=7)
forecast = model.predict(future)

# Plot results
plt.figure(figsize=(12, 5))
plt.scatter(daily_df["ds"], daily_df["y"], color="blue", label="Actual Sentiment", s=15)
plt.plot(forecast["ds"], forecast["yhat"], color="black", label="Predicted Trend")
plt.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"],
                 color="lightblue", alpha=0.4, label="Confidence Interval")
plt.title("Sentiment Forecast (7 Days)")
plt.xlabel("Date")
plt.ylabel("Sentiment Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("sentiment_forecast.png")
plt.show()

print(" Task 3 Complete! Forecast saved as sentiment_forecast.png")